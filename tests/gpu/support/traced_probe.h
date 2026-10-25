#pragma once

// Runs one probe kernel that traces the scene's acceleration structure, the
// way every such probe test does (ES.3, F.10): the kernel's output at
// buffer 0, its inputs from 1 in the order given, then the structure and the
// shapes' records, transforms and boxes; one dispatch; a wait; and the
// output read back. Each tracing kernel's comment lists its bindings in that
// order (kernels/trace_probe.metal, kernels/leave_probe.metal).
//
// Its own dispatch, not the probe runner's (probe_runner.h): it binds the
// structures' block (metal/acceleration/structure_block.h), and records
// their update in the same encoder as the trace when the shapes are moved,
// as a frame does. The scene and the probes' library are a TracedScene's,
// made once for all its dispatches (P.9, GPU.6), as the probe runner's are.
// The CUDA backend's harness has the same names and interface
// (tests/cuda/support/traced_probe.h), so a test that traces is one source
// for both, each test target putting its own harness on its include path.
//
// Failures are the test's (P.7), as the probe runner's are.

#include <doctest/doctest.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <optional>
#include <span>
#include <utility>
#include <vector>

// the backend's runner, by the include path (SF.12)
#include <probe_runner.h>

#include "core/contracts/transform.h"
#include "core/scene/scene.h"
#include "core/shapes/mesh_levels.h"
#include "core/shapes/shape_meshes.h"
#include "gpu/support/shared_resources.h"
#include "metal/acceleration/scene_acceleration.h"
#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/submission.h"
#include "metal/passes/restir_pt_initial/restir_pt_initial.h"
#include "metal/scene/scene_buffers.h"
#include "metal/scene/shape_transforms.h"

namespace serenity::tests {

// A scene on the GPU, as a frame traces it, made once for a test's probes:
// its arrays, its shapes' transforms (one copy, or one a frame slot where
// shapes move) and its structures, which learn the scene's lights from it
// and its moving shapes, `moving`, from the test; and the probes' library
// (metallib::probes). A fixture: what it holds refers to what it holds, so
// it is neither copied nor moved (C.8).
class TracedScene {
 public:
  TracedScene(const scene::SceneDescription& description,
              std::span<const std::uint32_t> moving)
      : buffers_(device_, submission_, description),
        transforms_(device_, submission_, description.shapes.transforms,
                    moving.empty() ? metal::FrameArray::Copies::one
                                   : metal::FrameArray::Copies::per_frame),
        lights_(scene::light_shapes(description)),
        levels_(shapes::sphere_levels(description)),
        acceleration_(
            device_, submission_, description.shapes,
            {.moving = moving, .lights = lights_, .levels = levels_}) {
    for (std::size_t i = 0; i < levels_.size(); ++i) {
      gaps_.push_back(
          levels_[i] == shapes::no_level
              ? 0.0
              : shapes::icosphere_gaps.at(levels_[i]) *
                    contracts::scale(description.shapes.transforms[i]));
    }
  }
  TracedScene(const TracedScene&) = delete;
  TracedScene& operator=(const TracedScene&) = delete;
  TracedScene(TracedScene&&) = delete;
  TracedScene& operator=(TracedScene&&) = delete;
  ~TracedScene() = default;

  metal::Device& device() noexcept { return device_; }
  metal::Submission& submission() noexcept { return submission_; }
  const metal::Library& library() const noexcept { return library_; }
  const metal::SceneBuffers& buffers() const noexcept { return buffers_; }
  metal::ShapeTransforms& transforms() noexcept { return transforms_; }
  metal::SceneAcceleration& acceleration() noexcept { return acceleration_; }

  // How far inside shape `shape`'s own surface, along its normal, a traced
  // hit may lie: a sphere's mesh's gap (core/shapes/mesh_levels.h); 0 for a
  // box, whose triangles lie on its faces. The CUDA backend's harness says
  // the same (tests/cuda/support/traced_probe.h).
  double hit_gap(std::uint32_t shape) const { return gaps_.at(shape); }

 private:
  metal::Device device_;
  metal::Submission submission_{device_};
  metal::SceneBuffers buffers_;
  metal::ShapeTransforms transforms_;
  std::vector<std::uint32_t> lights_;
  std::vector<std::uint8_t> levels_;
  metal::SceneAcceleration acceleration_;
  std::vector<double> gaps_;  // a shape each, for hit_gap()
  metal::Library library_{device_, metallib::probes};
};

// Whether a probe seeds manifold next event estimation, as the initial pass
// does: here, its pipeline then links the seed's intersection function and
// is handed its table, bound after the shapes' boxes
// (metal/passes/restir_pt_initial/restir_pt_initial.h). The CUDA harness
// takes the same, its module holding the seed's program for every probe.
enum class Seeding { none, mnee };

// Dispatches `threads` threads of `kernel` over `traced`'s structure for
// the next frame's slot, its output, one Out a thread, at binding 0 and
// `inputs` from 1, in one command buffer; if `moved` is given, first writes
// them as the slot's transforms and records the structure's update.
template <GpuBytes Out>
std::vector<Out> run_traced(
    TracedScene& traced, const char* kernel, std::size_t threads,
    std::initializer_list<Binding> inputs,
    const std::optional<std::vector<contracts::Transform>>& moved =
        std::nullopt,
    Seeding seeding = Seeding::none) {
  const bool seeds = seeding == Seeding::mnee;
  const auto pipeline = traced.library().compute_pipeline(
      kernel,
      seeds ? metal::PipelineOptions{.intersection_functions =
                                         metal::seed_intersection_functions}
            : metal::PipelineOptions{});
  NS::SharedPtr<MTL::IntersectionFunctionTable> seed_functions;
  std::optional<metal::Resident> seed_resident;
  if (seeds) {
    seed_functions = traced.library().intersection_table(
        pipeline.get(), metal::seed_intersection_functions);
    seed_resident.emplace(
        traced.submission().keep_resident(seed_functions.get()));
  }
  const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  MTL::Device* mtl = traced.device().handle();
  const auto out = shared_buffer(traced.device(), traced.submission(), {},
                                 threads * sizeof(Out));
  std::vector<NS::SharedPtr<MTL::Buffer>> buffers;
  for (const Binding& input : inputs) {
    buffers.push_back(
        shared_buffer(traced.device(), traced.submission(), input.bytes()));
  }
  auto descriptor =
      NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
  descriptor->setMaxBufferBindCount(1 + buffers.size() + 4 + (seeds ? 1 : 0));
  NS::Error* error = nullptr;
  auto table = NS::TransferPtr(mtl->newArgumentTable(descriptor.get(), &error));
  INFO("argument table: " << reason(error));
  REQUIRE(table);

  const metal::FrameSlot slot = traced.submission().begin();
  MTL4::ComputeCommandEncoder* encoder = slot.commands->computeCommandEncoder();
  REQUIRE(encoder != nullptr);
  if (moved) {
    const std::span<contracts::Transform> placed =
        traced.transforms().transforms(slot.slot);
    std::ranges::copy(*moved, placed.begin());
    traced.acceleration().update(encoder, slot.slot, placed);
  }
  std::size_t index = 0;
  table->setAddress(out->gpuAddress(), index++);
  for (const auto& buffer : buffers) {
    table->setAddress(buffer->gpuAddress(), index++);
  }
  table->setAddress(traced.acceleration().address(slot.slot),
                    index++);  // the structures by mask
  table->setAddress(traced.buffers().block().shapes, index++);
  table->setAddress(traced.transforms().address(slot.slot), index++);
  table->setAddress(traced.buffers().block().boxes, index++);
  if (seeds) {
    table->setResource(seed_functions->gpuResourceID(), index);
  }
  encoder->setArgumentTable(table.get());
  encoder->setComputePipelineState(pipeline.get());
  encoder->dispatchThreads(MTL::Size(threads, 1, 1),
                           MTL::Size(pipeline->threadExecutionWidth(), 1, 1));
  encoder->endEncoding();
  traced.submission().commit();
  (void)traced.submission().wait_until_complete(slot.sequence);

  std::vector<Out> results(threads);
  std::memcpy(results.data(), out->contents(), threads * sizeof(Out));
  return results;
}

}  // namespace serenity::tests
