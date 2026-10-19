#include "metal/acceleration/scene_acceleration.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <string>
#include <type_traits>

#include "metal/acceleration/structure_block.h"
#include "metal/device/error.h"
#include "metal/device/support.h"

namespace serenity::metal {

// The core's bounds are Metal's bounding boxes, byte for byte, so they are
// written as they are: the same size, each corner at the same offset, and
// bytes that are their value (SL.con.4).
static_assert(sizeof(shapes::Bounds) == sizeof(MTL::AxisAlignedBoundingBox));
static_assert(offsetof(shapes::Bounds, min) ==
              offsetof(MTL::AxisAlignedBoundingBox, min));
static_assert(offsetof(shapes::Bounds, max) ==
              offsetof(MTL::AxisAlignedBoundingBox, max));
static_assert(std::is_trivially_copyable_v<shapes::Bounds>);
// A structure's resource ID is the 64 bits the block holds for it
// (structure_block.h).
static_assert(sizeof(MTL::ResourceID) == sizeof(gpu::Structure));
static_assert(std::is_trivially_copyable_v<gpu::StructureBlock>);

namespace {

// Whether `indices` are shapes' indices, of `count`, in increasing order.
bool increasing_shapes(std::span<const std::uint32_t> indices,
                       std::size_t count) {
  for (std::size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] >= count || (i > 0 && indices[i] <= indices[i - 1])) {
      return false;
    }
  }
  return true;
}

}  // namespace

SceneAcceleration::Tree SceneAcceleration::make_tree(
    const Device& device, Submission& submission, const shapes::Shapes& shapes,
    std::span<const std::uint32_t> members,
    std::span<const std::uint32_t> moving, MTL::ResidencySet* build_only) {
  Tree tree;
  // Its moving shapes, by the box each is in: both lists increasing, so one
  // walk pairs them.
  std::size_t next = 0;
  for (std::size_t box = 0; box < members.size(); ++box) {
    while (next < moving.size() && moving[next] < members[box]) {
      ++next;
    }
    if (next < moving.size() && moving[next] == members[box]) {
      tree.moving_boxes.push_back(static_cast<std::uint32_t>(box));
      tree.moving_shapes.push_back(members[box]);
      tree.moving_bounds.push_back(
          shapes::object_bounds(shapes, shapes.records[members[box]]));
    }
  }
  const bool moves = tree.moves();

  // Every member's box at rest: its geometry's bounds placed by its transform.
  std::vector<shapes::Bounds> at_rest(members.size());
  for (std::size_t box = 0; box < members.size(); ++box) {
    const std::uint32_t shape = members[box];
    at_rest[box] = shapes::world_bounds(
        shapes::object_bounds(shapes, shapes.records[shape]),
        shapes.transforms[shape]);
  }
  tree.boxes = std::make_unique<FrameArray>(
      device, submission, std::as_bytes(std::span(at_rest)),
      moves ? FrameArray::Copies::per_frame : FrameArray::Copies::one);

  MTL::Device* metal_device = device.handle();
  const std::uint32_t structures = moves ? frames_in_flight : 1u;
  for (std::uint32_t slot = 0; slot < structures; ++slot) {
    Built& built = tree.structures.at(slot);
    auto geometry = NS::TransferPtr(
        MTL4::AccelerationStructureBoundingBoxGeometryDescriptor::alloc()
            ->init());
    geometry->setBoundingBoxBuffer(MTL4::BufferRange(
        tree.boxes->address(slot), tree.boxes->bytes(slot).size()));
    geometry->setBoundingBoxCount(members.size());
    geometry->setBoundingBoxStride(sizeof(shapes::Bounds));
    geometry->setOpaque(true);
    built.descriptor = NS::TransferPtr(
        MTL4::PrimitiveAccelerationStructureDescriptor::alloc()->init());
    const std::array<const NS::Object*, 1> geometries{geometry.get()};
    built.descriptor->setGeometryDescriptors(
        NS::Array::array(geometries.data(), geometries.size()));

    const MTL::AccelerationStructureSizes sizes =
        metal_device->accelerationStructureSizes(built.descriptor.get());
    built.structure = NS::TransferPtr(metal_device->newAccelerationStructure(
        sizes.accelerationStructureSize));
    built.scratch = NS::TransferPtr(metal_device->newBuffer(
        sizes.buildScratchBufferSize, MTL::ResourceStorageModePrivate));
    if (!built.structure || !built.scratch) {
      throw MetalError(
          "SceneAcceleration: the device made no acceleration structure or "
          "scratch buffer");
    }
    // Traced every frame: resident until this object is destroyed.
    built.structure_resident = submission.keep_resident(built.structure.get());
    if (moves) {
      built.scratch_resident = submission.keep_resident(
          built.scratch.get());  // each frame's build writes it
    } else {
      build_only->addAllocation(built.scratch.get());
    }
  }
  return tree;
}

SceneAcceleration::SceneAcceleration(const Device& device,
                                     Submission& submission,
                                     const shapes::Shapes& shapes,
                                     Roles roles) {
  const std::span<const std::uint32_t> moving = roles.moving;
  const std::span<const std::uint32_t> lights = roles.lights;
  const std::size_t count = shapes.records.size();
  if (count == 0 || shapes.transforms.size() != count) {
    throw MetalError(
        "SceneAcceleration: no shapes, or not one transform per shape");
  }
  if (!increasing_shapes(moving, count)) {
    throw MetalError(
        "SceneAcceleration: the moving shapes are not shapes' indices in "
        "increasing order");
  }
  if (!increasing_shapes(lights, count)) {
    throw MetalError(
        "SceneAcceleration: the lights are not shapes' indices in increasing "
        "order");
  }

  // Everything's members are every shape, so its box i is shape i; the
  // occluders', the shapes that are not lights, in order.
  std::vector<std::uint32_t> every(count);
  for (std::size_t i = 0; i < count; ++i) {
    every[i] = static_cast<std::uint32_t>(i);
  }
  std::vector<std::uint32_t> surfaces;
  std::ranges::set_difference(every, lights, std::back_inserter(surfaces));

  const auto pool = scoped_pool();
  MTL::Device* metal_device = device.handle();
  // What only the start-up build reads, resident for its command buffer
  // alone: the still trees' scratch. A moving tree builds every frame, and
  // its scratch is resident for good.
  auto set_descriptor =
      NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
  NS::Error* error = nullptr;
  auto build_only = NS::TransferPtr(
      metal_device->newResidencySet(set_descriptor.get(), &error));
  if (!build_only) {
    throw MetalError("SceneAcceleration: the device made no residency set: " +
                     describe(error));
  }

  everything_ =
      make_tree(device, submission, shapes, every, moving, build_only.get());
  // Its own tree only when it differs from everything and holds something
  // (structure_block.h): with no light it is everything; with no surface,
  // nothing.
  if (!lights.empty() && !surfaces.empty()) {
    occluders_ = make_tree(device, submission, shapes, surfaces, moving,
                           build_only.get());
  }
  const std::span<const std::uint32_t> table =
      lights.empty() ? std::span<const std::uint32_t>(every)
                     : std::span<const std::uint32_t>(surfaces);
  const std::array<std::span<const std::byte>, 1> arrays{std::as_bytes(table)};
  occluder_table_ = std::make_unique<StaticArrays>(device, submission, arrays);

  // The block each slot binds, written now and never again: each slot's
  // structures are the same objects every frame.
  const bool any_moves =
      everything_.moves() || (occluders_ && occluders_->moves());
  std::array<gpu::StructureBlock, frames_in_flight> blocks{};
  for (std::uint32_t slot = 0; slot < frames_in_flight; ++slot) {
    const MTL::ResourceID all = everything_.in(slot).structure->gpuResourceID();
    const MTL::ResourceID occluding =
        occluders_ ? occluders_->in(slot).structure->gpuResourceID() : all;
    blocks.at(slot) = gpu::StructureBlock{
        all._impl, occluding._impl, occluder_table_->address(0),
        static_cast<std::uint32_t>(table.size()), 0u};
  }
  // Every copy starts as slot 0's; with a copy a slot, each is then its
  // own slot's, before any frame (frame_array.h).
  blocks_ = std::make_unique<FrameArray>(
      device, submission, std::as_bytes(std::span(blocks.data(), 1)),
      any_moves ? FrameArray::Copies::per_frame : FrameArray::Copies::one);
  if (any_moves) {
    for (std::uint32_t slot = 1; slot < frames_in_flight; ++slot) {
      blocks_->view<gpu::StructureBlock>(slot)[0] = blocks.at(slot);
    }
  }

  // The still trees' structures, built now, in one submission.
  const bool everything_still = !everything_.moves();
  const bool occluders_still = occluders_ && !occluders_->moves();
  if (!everything_still && !occluders_still) {
    return;  // each structure is built by its slot's first update()
  }
  build_only->commit();
  const FrameSlot build = submission.begin();
  build.commands->useResidencySet(build_only.get());
  MTL4::ComputeCommandEncoder* encoder =
      build.commands->computeCommandEncoder();
  if (encoder == nullptr) {
    throw MetalError(
        "SceneAcceleration: the command buffer made no compute encoder");
  }
  for (Tree* tree : {&everything_, occluders_ ? &*occluders_ : nullptr}) {
    if (tree == nullptr || tree->moves()) {
      continue;
    }
    Built& built = tree->structures[0];
    encoder->buildAccelerationStructure(
        built.structure.get(), built.descriptor.get(),
        MTL4::BufferRange(built.scratch->gpuAddress(),
                          built.scratch->length()));
  }
  encoder->endEncoding();
  submission.commit();
  // Start-up work, not a frame: waited for here, so the structures are
  // whole before any frame traces them (GPU.8: the builds' writes are
  // complete before any read), and settled here, so it is never measured
  // as a frame (submission.h).
  submission.wait_until_complete(build.sequence);
  // The scratch was the builds' alone, and they are done: let it go rather
  // than hold it for the run (GPU.9). The set that made it resident is
  // released with it, at the end of this scope.
  for (Tree* tree : {&everything_, occluders_ ? &*occluders_ : nullptr}) {
    if (tree != nullptr && !tree->moves()) {
      tree->structures[0].scratch.reset();
    }
  }
}

void SceneAcceleration::update(
    MTL4::ComputeCommandEncoder* encoder, std::uint32_t slot,
    std::span<const contracts::Transform> transforms) {
  if (!everything_.moves()) {
    throw MetalError("SceneAcceleration::update: nothing in the scene moves");
  }
  if (encoder == nullptr || slot >= frames_in_flight) {
    throw MetalError(
        "SceneAcceleration::update: no encoder, or no frame slot " +
        std::to_string(slot));
  }
  if (transforms.size() !=
      everything_.boxes->view<shapes::Bounds>(slot).size()) {
    throw MetalError("SceneAcceleration::update: not one transform per shape");
  }
  for (Tree* tree : {&everything_, occluders_ ? &*occluders_ : nullptr}) {
    if (tree == nullptr || !tree->moves()) {
      continue;
    }
    // Step 1: the tree's moving boxes, where this frame places their shapes.
    // made from boxes
    const std::span<shapes::Bounds> boxes =
        tree->boxes->view<shapes::Bounds>(slot);
    for (std::size_t i = 0; i < tree->moving_boxes.size(); ++i) {
      boxes[tree->moving_boxes[i]] = shapes::world_bounds(
          tree->moving_bounds[i], transforms[tree->moving_shapes[i]]);
    }
    // Step 2: the slot's structure, over every box.
    const Built& built = tree->structures.at(slot);
    encoder->buildAccelerationStructure(
        built.structure.get(), built.descriptor.get(),
        MTL4::BufferRange(built.scratch->gpuAddress(),
                          built.scratch->length()));
  }
  // Step 3: every pass after it traces the finished structures.
  encoder->barrierAfterEncoderStages(MTL::StageAccelerationStructure,
                                     MTL::StageDispatch,
                                     MTL4::VisibilityOptionDevice);
}

MTL::GPUAddress SceneAcceleration::address(std::uint32_t slot) const {
  if (slot >= frames_in_flight) {
    throw MetalError("SceneAcceleration::address: no frame slot " +
                     std::to_string(slot));
  }
  return blocks_->address(slot);
}

}  // namespace serenity::metal
