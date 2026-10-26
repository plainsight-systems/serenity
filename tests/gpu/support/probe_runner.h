#pragma once

// Runs one kernel of the tests' probe library (tests/gpu/kernels, compiled
// in as serenity::metallib::probes) the way every probe test does (ES.3,
// F.10), on Metal; the CUDA backend's runner has the same name and
// interface (tests/cuda/support/probe_runner.h), so a probe test is one
// source for both, each test target putting its own runner's directory on
// its include path (tests/gpu/CMakeLists.txt): each binding a buffer of its
// own, or an address the caller keeps resident; one dispatch; a wait; and what
// the kernel wrote read back. One device, queue and library per runner, so a
// test case builds them once and runs its probes on them (P.9, GPU.6).
//
// Bindings are positional: binding i is the kernel's [[buffer(i)]]. run()
// binds its output at 0 and its inputs from 1, in the order given; each
// probe kernel's comment lists its bindings in that order, its records in
// its own header beside it (kernels/<kernel>.h).
//
// Failures are the test's (P.7): a buffer, argument table or pipeline the
// device does not make fails the test case at once, naming Metal's reason
// where it gave one; a GPU fault is thrown by Submission's settling
// (metal/device/submission.h).

#include <doctest/doctest.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/scene/scene.h"
#include "gpu/support/shared_resources.h"
#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/submission.h"
#include "metal/scene/scene_buffers.h"
#include "serenity/metallib/probes.h"

namespace serenity::tests {

// Why Metal refused to make something, from the error it gave, or that it
// gave none (I.10).
inline std::string reason(const NS::Error* error) {
  return error != nullptr ? error->localizedDescription()->utf8String()
                          : "Metal gave no reason";
}

// A type whose bytes are its value, so it may be copied to and from the GPU
// as bytes (SL.con.4, T.10).
template <typename T>
concept GpuBytes = std::is_trivially_copyable_v<T>;

// One binding of a probe's dispatch: bytes copied into a buffer of its own,
// or the address of memory the caller keeps resident through the runner's
// submission.
class Binding {
 public:
  template <GpuBytes T>
  static Binding of(const T& value) {
    return Binding(std::as_bytes(std::span(&value, 1)));
  }

  template <GpuBytes T>
  static Binding of(const std::vector<T>& values) {
    return Binding(std::as_bytes(std::span(values)));
  }

  // `size` zeroed bytes, for a kernel to write.
  static Binding zeroed(std::size_t size) {
    return Binding(std::vector<std::byte>(size));
  }

  static Binding at(MTL::GPUAddress address) {
    Binding b(std::vector<std::byte>{});
    b.address_ = address;
    return b;
  }

  const std::vector<std::byte>& bytes() const noexcept { return bytes_; }
  MTL::GPUAddress address() const noexcept { return address_; }

 private:
  explicit Binding(std::span<const std::byte> bytes)
      : bytes_(bytes.begin(), bytes.end()) {}
  explicit Binding(std::vector<std::byte> bytes) : bytes_(std::move(bytes)) {}

  std::vector<std::byte> bytes_;
  MTL::GPUAddress address_ = 0;  // 0: a buffer of bytes_ of its own
};

class ProbeRunner {
 public:
  // Every buffer a dispatch bound, read back after it completed.
  class Buffers {
   public:
    // Binding `binding`'s contents, as the T it holds, as many as fit
    // in the bytes it was given.
    template <GpuBytes T>
    std::vector<T> read(std::size_t binding) const {
      REQUIRE(binding < buffers_.size());
      REQUIRE(buffers_[binding]);
      std::vector<T> values(sizes_[binding] / sizeof(T));
      std::memcpy(values.data(), buffers_[binding]->contents(),
                  values.size() * sizeof(T));
      return values;
    }

   private:
    friend class ProbeRunner;
    // null where a binding was an address
    std::vector<NS::SharedPtr<MTL::Buffer>> buffers_;
    std::vector<std::size_t> sizes_;
  };

  ProbeRunner() = default;
  ProbeRunner(const ProbeRunner&) = delete;
  ProbeRunner& operator=(const ProbeRunner&) = delete;
  ProbeRunner(ProbeRunner&&) = delete;
  ProbeRunner& operator=(ProbeRunner&&) = delete;
  ~ProbeRunner() = default;

  // Dispatches `threads` threads of `kernel` over `bindings`, its pipeline
  // built as `options` say (metal/device/library.h), waits, and returns
  // every buffer it bound.
  Buffers dispatch(const char* kernel, std::size_t threads,
                   std::initializer_list<Binding> bindings,
                   metal::PipelineOptions options = {}) {
    return dispatch(kernel, threads,
                    std::span<const Binding>(bindings.begin(), bindings.size()),
                    options);
  }

  Buffers dispatch(const char* kernel, std::size_t threads,
                   std::span<const Binding> bindings,
                   metal::PipelineOptions options = {}) {
    const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    MTL::Device* mtl = device_.handle();
    const auto pipeline = library_.compute_pipeline(kernel, options);
    Buffers out;
    for (const Binding& binding : bindings) {
      if (binding.address() != 0) {
        out.buffers_.emplace_back();
        out.sizes_.push_back(0);
        continue;
      }
      out.buffers_.push_back(
          shared_buffer(device_, submission_, binding.bytes()));
      out.sizes_.push_back(binding.bytes().size());
    }

    auto descriptor =
        NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
    descriptor->setMaxBufferBindCount(bindings.size());
    NS::Error* error = nullptr;
    auto table =
        NS::TransferPtr(mtl->newArgumentTable(descriptor.get(), &error));
    INFO("argument table: " << tests::reason(error));
    REQUIRE(table);
    std::size_t index = 0;
    for (const Binding& binding : bindings) {
      table->setAddress(binding.address() != 0
                            ? binding.address()
                            : out.buffers_[index]->gpuAddress(),
                        index);
      ++index;
    }

    const metal::FrameSlot slot = submission_.begin();
    MTL4::ComputeCommandEncoder* encoder =
        slot.commands->computeCommandEncoder();
    REQUIRE(encoder != nullptr);
    encoder->setArgumentTable(table.get());
    encoder->setComputePipelineState(pipeline.get());
    encoder->dispatchThreads(MTL::Size(threads, 1, 1),
                             MTL::Size(pipeline->threadExecutionWidth(), 1, 1));
    encoder->endEncoding();
    submission_.commit();
    (void)submission_.wait_until_complete(slot.sequence);
    return out;
  }

  // Dispatches `threads` threads of `kernel`, its output, one Out a
  // thread, at binding 0 and `inputs` from 1, its pipeline built as
  // `options` say, and returns the output.
  template <GpuBytes Out>
  std::vector<Out> run(const char* kernel, std::size_t threads,
                       std::initializer_list<Binding> inputs,
                       metal::PipelineOptions options = {}) {
    std::vector<Binding> bindings;
    bindings.reserve(inputs.size() + 1);
    bindings.push_back(Binding::zeroed(threads * sizeof(Out)));
    bindings.insert(bindings.end(), inputs.begin(), inputs.end());
    return dispatch(kernel, threads, std::span<const Binding>(bindings),
                    options)
        .template read<Out>(0);
  }

  metal::Device& device() noexcept { return device_; }
  metal::Submission& submission() noexcept { return submission_; }

 private:
  metal::Device device_;
  metal::Submission submission_{device_};
  metal::Library library_{device_, metallib::probes};
};

// A scene's arrays and block on the GPU, as the passes read them
// (metal/scene/scene_buffers.h), for a probe that reads a scene through its
// block, under the name both backends' runners give it.
class ProbeScene {
 public:
  ProbeScene(ProbeRunner& runner, const scene::SceneDescription& scene)
      : buffers_(runner.device(), runner.submission(), scene) {}

  std::uint64_t block_address() const noexcept {
    return buffers_.block_address();
  }

 private:
  metal::SceneBuffers buffers_;
};

}  // namespace serenity::tests
