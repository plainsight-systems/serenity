#include "cuda/acceleration/scene_acceleration.h"

#include <optix.h>
#include <optix_stubs.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>
#include <utility>

#include "cuda/device/error.h"

namespace serenity::cuda {

namespace {

// The core's box is OptiX's: two packed triples, min then max
// (scene_acceleration.h).
static_assert(sizeof(shapes::Bounds) == sizeof(OptixAabb));
static_assert(offsetof(shapes::Bounds, min) == offsetof(OptixAabb, minX));
static_assert(offsetof(shapes::Bounds, max) == offsetof(OptixAabb, maxX));

// OptiX's one build input for a tree's boxes. `boxes` must outlive the
// input's use (OptiX reads the address from it): the caller's local.
OptixBuildInput box_input(const CUdeviceptr& boxes, std::uint32_t count,
                          const unsigned& flags) {
  OptixBuildInput input{};
  input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
  // The C API's descriptor is a union, filled by the member its type names,
  // as OptiX's and CUDA's documentation fill it.
  // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
  input.customPrimitiveArray.aabbBuffers = &boxes;
  input.customPrimitiveArray.numPrimitives = count;
  input.customPrimitiveArray.strideInBytes = sizeof(shapes::Bounds);
  input.customPrimitiveArray.flags = &flags;
  input.customPrimitiveArray.numSbtRecords = 1;
  // NOLINTEND(cppcoreguidelines-pro-type-union-access)
  return input;
}

// Any-hit off for every box (trace.cuh): the intersection program decides
// each hit, and nothing else runs.
constexpr unsigned box_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

OptixAccelBuildOptions build_options() {
  OptixAccelBuildOptions options{};
  options.buildFlags = OPTIX_BUILD_FLAG_NONE;  // OptiX's defaults (above)
  options.operation = OPTIX_BUILD_OPERATION_BUILD;
  return options;
}

}  // namespace

SceneAcceleration::Tree SceneAcceleration::make_tree(
    const OptixContext& context, Submission& submission,
    shapes::TreePlan plan) {
  Tree tree;
  tree.plan = std::move(plan);
  const std::span<const shapes::Bounds> at_rest = tree.plan.at_rest;
  tree.boxes =
      std::make_unique<FrameUpload>(submission, std::as_bytes(at_rest));

  const CUdeviceptr boxes = tree.boxes->address();
  const OptixBuildInput input =
      box_input(boxes, static_cast<std::uint32_t>(at_rest.size()), box_flags);
  const OptixAccelBuildOptions options = build_options();
  OptixAccelBufferSizes sizes{};
  check_optix(optixAccelComputeMemoryUsage(context.handle(), &options, &input,
                                           1, &sizes),
              "SceneAcceleration: optixAccelComputeMemoryUsage");
  tree.structure = DeviceBuffer(sizes.outputSizeInBytes);
  tree.scratch = DeviceBuffer(std::max<std::size_t>(sizes.tempSizeInBytes, 1));
  return tree;
}

void SceneAcceleration::build(Tree& tree, CUstream stream) const {
  const CUdeviceptr boxes = tree.boxes->address();
  const OptixBuildInput input = box_input(
      boxes, static_cast<std::uint32_t>(tree.plan.members.size()), box_flags);
  const OptixAccelBuildOptions options = build_options();
  check_optix(optixAccelBuild(context_, stream, &options, &input, 1,
                              tree.scratch.address(), tree.scratch.bytes(),
                              tree.structure.address(), tree.structure.bytes(),
                              &tree.handle, nullptr, 0),
              "SceneAcceleration: optixAccelBuild");
}

SceneAcceleration::SceneAcceleration(const OptixContext& context,
                                     Submission& submission,
                                     const shapes::Shapes& shapes, Roles roles)
    : context_(context.handle()) {
  if (const std::optional<std::string> reason =
          shapes::structure_refusal(shapes, roles.moving, roles.lights)) {
    throw CudaError("SceneAcceleration: " + *reason);
  }
  shapes::StructurePlan plan =
      shapes::plan_structures(shapes, roles.moving, roles.lights);
  everything_ = make_tree(context, submission, std::move(plan.everything));
  // Its own tree only when it differs from everything and holds something
  // (structure_block.h; the core's plan).
  if (plan.occluders) {
    occluders_ = make_tree(context, submission, std::move(*plan.occluders));
  }
  no_surfaces_ = !plan.occluders && plan.occluder_table.empty();
  occluder_count_ = static_cast<std::uint32_t>(plan.occluder_table.size());
  // At least one entry: memory must have bytes; with no surface it is
  // never read.
  const std::span<const std::uint32_t> table = plan.occluder_table;
  occluder_table_ = DeviceBuffer(std::max<std::size_t>(table.size_bytes(), 4));
  if (!table.empty()) {
    check(cuMemcpyHtoD(occluder_table_.address(), table.data(),
                       table.size_bytes()),
          "SceneAcceleration: cuMemcpyHtoD");
  }

  // The still trees' structures, built now, in one submission.
  const bool everything_still = !everything_.moves();
  const bool occluders_still = occluders_ && !occluders_->moves();
  if (!everything_still && !occluders_still) {
    return;  // each is built by every frame's update()
  }
  const FrameSlot built = submission.begin();
  for (Tree* tree : {&everything_, occluders_ ? &*occluders_ : nullptr}) {
    if (tree != nullptr && !tree->moves()) {
      build(*tree, built.stream);
    }
  }
  submission.commit();
  // Start-up work, not a frame: waited for here, so the structures are
  // whole before any frame traces them (GPU.8), and settled here, so it is
  // never measured as a frame (submission.h).
  submission.wait_until_complete(built.sequence);
  // The scratch was the builds' alone, and they are done (GPU.9).
  for (Tree* tree : {&everything_, occluders_ ? &*occluders_ : nullptr}) {
    if (tree != nullptr && !tree->moves()) {
      tree->scratch = DeviceBuffer();
    }
  }
}

void SceneAcceleration::update(
    const FrameSlot& frame, std::span<const contracts::Transform> transforms) {
  if (!everything_.moves()) {
    throw CudaError("SceneAcceleration::update: nothing in the scene moves");
  }
  for (Tree* tree : {&everything_, occluders_ ? &*occluders_ : nullptr}) {
    if (tree == nullptr || !tree->moves()) {
      continue;
    }
    // Step 1: the tree's moving boxes, where this frame places their
    // shapes, uploaded.
    const std::span<shapes::Bounds> boxes =
        tree->boxes->view<shapes::Bounds>(frame.slot);
    if (tree == &everything_ && transforms.size() != boxes.size()) {
      throw CudaError("SceneAcceleration::update: not one transform per shape");
    }
    shapes::place_moving(tree->plan, transforms, boxes);
    tree->boxes->upload(frame);
    // Step 2: the structure, over every box.
    build(*tree, frame.stream);
  }
}

StructureBlock SceneAcceleration::block() const noexcept {
  OptixTraversableHandle occluding = everything_.handle;
  if (occluders_) {
    occluding = occluders_->handle;
  } else if (no_surfaces_) {
    occluding = 0;
  }
  return StructureBlock{everything_.handle, occluding, occluder_count_, 0u};
}

std::array<Pipeline::HitData, structure_records> SceneAcceleration::hit_records(
    const gpu::SceneBlock& scene, CUdeviceptr transforms) const noexcept {
  const auto record = [&](std::uint64_t table) {
    const StructureRecord data{table, scene.shapes, scene.boxes, transforms};
    Pipeline::HitData words{};
    static_assert(sizeof(data) == sizeof(words));
    std::memcpy(words.data(), &data, sizeof(data));
    return words;
  };
  std::array<Pipeline::HitData, structure_records> records{};
  records[everything_record] = record(0);  // box i is shape i
  records[occluders_record] = record(occluder_table_.address());
  return records;
}

}  // namespace serenity::cuda
