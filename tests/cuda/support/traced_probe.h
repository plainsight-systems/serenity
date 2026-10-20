#pragma once

// Runs one probe that traces the scene's structures, the way every such
// probe test does, on CUDA: Metal's harness (tests/gpu/support/
// traced_probe.h) under the same names and interface, so a test that
// traces is one source for both backends. The probe is a ray generation
// program of the tracing probes' module (tests/cuda/kernels/
// traced_probes.cu), its bindings Metal's, positional, each a device
// address in its launch parameters (kernels/probe_launch.h): its output at
// 0, its inputs from 1, then the structures' place (the structures are in
// the launch parameters by value), the shapes' records, transforms and
// boxes. One launch; a wait; the output read back. When the shapes are
// moved, the slot's transforms are written and the structures rebuilt on
// the stream before the launch, as a frame does.
//
// Failures are the test's (P.7): the driver's and OptiX's errors are thrown
// as cuda::CudaError, which fails the case.

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "core/contracts/transform.h"
#include "core/scene/scene.h"
#include "cuda/acceleration/scene_acceleration.h"
#include "cuda/device/buffer.h"
#include "cuda/device/device.h"
#include "cuda/device/error.h"
#include "cuda/device/optix_context.h"
#include "cuda/device/pipeline.h"
#include "cuda/device/submission.h"
#include "cuda/scene/scene_buffers.h"
#include "cuda/scene/shape_transforms.h"
#include "kernels/probe_launch.h"
#include "serenity/optix_modules/traced_probes.h"

// the backend's runner, by the include path (SF.12): Binding, GpuBytes
#include <probe_runner.h>

namespace serenity::tests {

// A scene on the GPU, as a frame traces it, made once for a test's probes:
// its arrays, its shapes' transforms and its structures, which learn the
// scene's lights from it and its moving shapes, `moving`, from the test;
// and a pipeline for each probe asked for, made at its first run. A
// fixture: what it holds refers to what it holds (C.8).
class TracedScene {
 public:
  TracedScene(const scene::SceneDescription& description,
              std::span<const std::uint32_t> moving)
      : buffers_(submission_, description),
        transforms_(submission_, description.shapes.transforms),
        acceleration_(
            optix_, submission_, description.shapes,
            {.moving = moving, .lights = scene::light_shapes(description)}) {}
  TracedScene(const TracedScene&) = delete;
  TracedScene& operator=(const TracedScene&) = delete;
  TracedScene(TracedScene&&) = delete;
  TracedScene& operator=(TracedScene&&) = delete;
  ~TracedScene() { submission_.wait_idle(); }

  cuda::Submission& submission() noexcept { return submission_; }
  const cuda::SceneBuffers& buffers() const noexcept { return buffers_; }
  cuda::ShapeTransforms& transforms() noexcept { return transforms_; }
  cuda::SceneAcceleration& acceleration() noexcept { return acceleration_; }

  // The pipeline of the probe `name`, its ray generation program
  // "__raygen__<name>", made at its first use.
  const cuda::Pipeline& pipeline(const std::string& name) {
    auto found = pipelines_.find(name);
    if (found == pipelines_.end()) {
      static constexpr std::array<std::string_view, 2> intersections{
          "__intersection__shape", "__intersection__seed"};
      const std::string ray_generation = "__raygen__" + name;
      const std::array<cuda::Pipeline::HitRecord, cuda::structure_records>
          records = acceleration_.hit_records(buffers_.block(),
                                              transforms_.address());
      found = pipelines_
                  .emplace(name,
                           std::make_unique<cuda::Pipeline>(
                               optix_, optix_modules::traced_probes,
                               cuda::Pipeline::Programs{
                                   .ray_generation = ray_generation,
                                   .intersections = intersections,
                                   .launch_parameters = "probe",
                                   .payload_values = cuda::trace_payload_values,
                                   .hit_records = records,
                               }))
                  .first;
    }
    return *found->second;
  }

 private:
  cuda::Device device_;
  cuda::Submission submission_{device_};
  cuda::OptixContext optix_{device_};
  cuda::SceneBuffers buffers_;
  cuda::ShapeTransforms transforms_;
  cuda::SceneAcceleration acceleration_;
  std::map<std::string, std::unique_ptr<cuda::Pipeline>> pipelines_;
};

// Launches `threads` threads of probe `kernel` over `traced`'s structures,
// its output, one Out a thread, at binding 0 and `inputs` from 1, as one
// frame; if `moved` is given, first writes them as the slot's transforms and
// rebuilds the structures.
template <GpuBytes Out>
std::vector<Out> run_traced(
    TracedScene& traced, const char* kernel, std::size_t threads,
    std::initializer_list<Binding> inputs,
    const std::optional<std::vector<contracts::Transform>>& moved =
        std::nullopt) {
  const cuda::Pipeline& pipeline = traced.pipeline(kernel);
  std::vector<cuda::DeviceBuffer> buffers;
  buffers.emplace_back(std::max<std::size_t>(threads * sizeof(Out), 1));
  for (const Binding& input : inputs) {
    buffers.emplace_back(std::max<std::size_t>(input.bytes().size(), 1));
    if (!input.bytes().empty()) {
      cuda::upload_now(buffers.back().address(), input.bytes(), "run_traced");
    }
  }
  REQUIRE(buffers.size() + 4 <= probe_bindings);

  cuda::Submission& submission = traced.submission();
  const cuda::FrameSlot slot = submission.begin();
  if (moved) {
    const std::span<contracts::Transform> placed =
        traced.transforms().transforms(slot.slot);
    std::ranges::copy(*moved, placed.begin());
    traced.transforms().upload(slot);
    traced.acceleration().update(slot, placed);
  }
  ProbeLaunch launch{};
  const std::span<std::uint64_t> bindings(launch.bindings);
  auto next = std::ranges::transform(buffers, bindings.begin(),
                                     [](const cuda::DeviceBuffer& buffer) {
                                       return buffer.address();
                                     })
                  .out;
  ++next;  // the structures' place: in the launch parameters, by value
  *next++ = traced.buffers().block().shapes;
  *next++ = traced.transforms().address();
  *next = traced.buffers().block().boxes;
  launch.structures = traced.acceleration().block();
  const cuda::DeviceBuffer parameters(sizeof(launch));
  cuda::check(cuMemcpyHtoDAsync(parameters.address(), &launch, sizeof(launch),
                                slot.stream),
              "run_traced: cuMemcpyHtoDAsync");
  pipeline.launch(
      slot.stream, parameters.address(), sizeof(launch),
      {static_cast<std::uint32_t>(std::max<std::size_t>(threads, 1)), 1});
  submission.commit();
  submission.wait_until_complete(slot.sequence);

  std::vector<Out> results(threads);
  cuda::check(
      cuMemcpyDtoH(results.data(), buffers[0].address(), threads * sizeof(Out)),
      "run_traced: cuMemcpyDtoH");
  return results;
}

}  // namespace serenity::tests
