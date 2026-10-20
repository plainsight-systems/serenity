#pragma once

// Runs one kernel of the tests' probe module (tests/cuda/kernels, compiled
// in as serenity::cuda_modules::probes) the way every probe test does, on
// CUDA: Metal's runner (tests/gpu/support/probe_runner.h) under the same
// name and with the same interface, so a probe test of tests/gpu/ is one
// source for both backends, built with each backend's runner (each test
// target puts its own support directory on the include path; tests/
// cuda/CMakeLists.txt). One device, submission and module per runner, so a
// test case makes them once and runs its probes on them (P.9).
//
// Bindings are positional, as Metal's: binding i is the kernel's parameter
// i, a pointer to device memory holding the binding's bytes, each in an
// allocation of its own, or an address the caller keeps alive. run() binds
// its output at 0 and its inputs from 1, in the order given; each probe
// kernel's comment lists its bindings in that order, its records in its own
// header (tests/gpu/kernels/<kernel>.h), shared with Metal's.
//
// A dispatch of n threads is a launch of n threads in blocks of
// probe_block, the last block cut short by each kernel's own guard on its
// count, which every probe has. Failures are the test's (P.7): the driver's
// errors are thrown as cuda::CudaError, which fails the case; a fault is
// reported at the wait.

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/scene/scene.h"
#include "cuda/device/buffer.h"
#include "cuda/device/device.h"
#include "cuda/device/error.h"
#include "cuda/device/module.h"
#include "cuda/device/submission.h"
#include "cuda/scene/scene_buffers.h"
#include "serenity/cuda_modules/probes.h"

namespace serenity::tests {

// A type whose bytes are its value, so it may be copied to and from the GPU
// as bytes (SL.con.4, T.10).
template <typename T>
concept GpuBytes = std::is_trivially_copyable_v<T>;

// One binding of a probe's launch: bytes copied into an allocation of its
// own, or the address of memory the caller keeps alive.
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

  static Binding at(std::uint64_t address) {
    Binding b(std::vector<std::byte>{});
    b.address_ = address;
    return b;
  }

  const std::vector<std::byte>& bytes() const noexcept { return bytes_; }
  std::uint64_t address() const noexcept { return address_; }

 private:
  explicit Binding(std::span<const std::byte> bytes)
      : bytes_(bytes.begin(), bytes.end()) {}
  explicit Binding(std::vector<std::byte> bytes) : bytes_(std::move(bytes)) {}

  std::vector<std::byte> bytes_;
  std::uint64_t address_ = 0;  // 0: an allocation of bytes_ of its own
};

class ProbeRunner {
 public:
  // Every allocation a launch bound, read back after it completed.
  class Buffers {
   public:
    // Binding `binding`'s contents, as the T it holds, as many as fit in
    // the bytes it was given.
    template <GpuBytes T>
    std::vector<T> read(std::size_t binding) const {
      REQUIRE(binding < buffers_.size());
      REQUIRE(buffers_[binding].address() != 0);
      std::vector<T> values(sizes_[binding] / sizeof(T));
      cuda::check(cuMemcpyDtoH(values.data(), buffers_[binding].address(),
                               values.size() * sizeof(T)),
                  "ProbeRunner: cuMemcpyDtoH");
      return values;
    }

   private:
    friend class ProbeRunner;
    // empty where a binding was an address
    std::vector<cuda::DeviceBuffer> buffers_;
    std::vector<std::size_t> sizes_;
  };

  ProbeRunner() = default;
  ProbeRunner(const ProbeRunner&) = delete;
  ProbeRunner& operator=(const ProbeRunner&) = delete;
  ProbeRunner(ProbeRunner&&) = delete;
  ProbeRunner& operator=(ProbeRunner&&) = delete;
  ~ProbeRunner() = default;

  // Launches `threads` threads of `kernel` over `bindings`, waits, and
  // returns every allocation it bound.
  Buffers dispatch(const char* kernel, std::size_t threads,
                   std::initializer_list<Binding> bindings) {
    return dispatch(
        kernel, threads,
        std::span<const Binding>(bindings.begin(), bindings.size()));
  }

  Buffers dispatch(const char* kernel, std::size_t threads,
                   std::span<const Binding> bindings) {
    const CUfunction function = module_.kernel(kernel);
    Buffers out;
    std::vector<CUdeviceptr> addresses;
    addresses.reserve(bindings.size());
    for (const Binding& binding : bindings) {
      if (binding.address() != 0) {
        out.buffers_.emplace_back();
        out.sizes_.push_back(0);
        addresses.push_back(binding.address());
        continue;
      }
      // At least one byte: an empty input still binds an address.
      out.buffers_.emplace_back(
          std::max<std::size_t>(binding.bytes().size(), 1));
      out.sizes_.push_back(binding.bytes().size());
      if (!binding.bytes().empty()) {
        cuda::upload_now(out.buffers_.back().address(), binding.bytes(),
                         "ProbeRunner");
      }
      addresses.push_back(out.buffers_.back().address());
    }
    // Each parameter a pointer to its address, which the driver copies at
    // the launch (cuda/device/module.h).
    std::vector<void*> parameters;
    parameters.reserve(addresses.size());
    for (CUdeviceptr& address : addresses) {
      parameters.push_back(&address);
    }
    const auto blocks =
        static_cast<unsigned>((threads + probe_block - 1) / probe_block);
    const cuda::FrameSlot slot = submission_.begin();
    if (threads > 0) {
      cuda::check(cuLaunchKernel(function, blocks, 1, 1, probe_block, 1, 1, 0,
                                 slot.stream, parameters.data(), nullptr),
                  std::string("ProbeRunner: cuLaunchKernel ") + kernel);
    }
    submission_.commit();
    submission_.wait_until_complete(slot.sequence);
    return out;
  }

  // Launches `threads` threads of `kernel`, its output, one Out a thread,
  // at binding 0 and `inputs` from 1, and returns the output.
  template <GpuBytes Out>
  std::vector<Out> run(const char* kernel, std::size_t threads,
                       std::initializer_list<Binding> inputs) {
    std::vector<Binding> bindings;
    bindings.reserve(inputs.size() + 1);
    bindings.push_back(Binding::zeroed(threads * sizeof(Out)));
    bindings.insert(bindings.end(), inputs.begin(), inputs.end());
    return dispatch(kernel, threads, std::span<const Binding>(bindings))
        .template read<Out>(0);
  }

  cuda::Submission& submission() noexcept { return submission_; }

 private:
  // A warp's threads, twice: any size divides the work; the probes are not
  // timed.
  static constexpr unsigned probe_block = 64;

  cuda::Device device_;
  cuda::Submission submission_{device_};
  cuda::Module module_{cuda_modules::probes};
};

// A scene's arrays and block on the GPU, as the passes read them
// (cuda/scene/scene_buffers.h), for a probe that reads a scene through its
// block: Metal's ProbeScene's counterpart (tests/gpu/support/
// probe_runner.h).
class ProbeScene {
 public:
  ProbeScene(ProbeRunner& runner, const scene::SceneDescription& scene)
      : buffers_(runner.submission(), scene) {}

  std::uint64_t block_address() const noexcept {
    return buffers_.block_address();
  }

 private:
  cuda::SceneBuffers buffers_;
};

}  // namespace serenity::tests
