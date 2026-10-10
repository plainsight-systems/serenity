#include "metal/acceleration/scene_acceleration.h"

#include <array>
#include <cstddef>
#include <string>
#include <type_traits>

#include "metal/device/error.h"
#include "metal/device/support.h"

namespace serenity::metal {

// The core's bounds are Metal's bounding boxes, byte for byte, so they are
// written as they are: the same size, each corner at the same offset, and
// bytes that are their value (SL.con.4).
static_assert(sizeof(shapes::Bounds) == sizeof(MTL::AxisAlignedBoundingBox));
static_assert(offsetof(shapes::Bounds, min) == offsetof(MTL::AxisAlignedBoundingBox, min));
static_assert(offsetof(shapes::Bounds, max) == offsetof(MTL::AxisAlignedBoundingBox, max));
static_assert(std::is_trivially_copyable_v<shapes::Bounds>);

SceneAcceleration::SceneAcceleration(const Device& device, Submission& submission, const shapes::Shapes& shapes,
                                     std::span<const std::uint32_t> moving) {
    const std::size_t count = shapes.records.size();
    if (count == 0 || shapes.transforms.size() != count) {
        throw MetalError("SceneAcceleration: no shapes, or not one transform per shape");
    }
    for (std::size_t i = 0; i < moving.size(); ++i) {
        if (moving[i] >= count || (i > 0 && moving[i] <= moving[i - 1])) {
            throw MetalError("SceneAcceleration: the moving shapes are not shapes' indices in increasing order");
        }
    }
    moving_.assign(moving.begin(), moving.end());
    for (std::uint32_t shape : moving_) {
        moving_bounds_.push_back(shapes::object_bounds(shapes, shapes.records[shape]));
    }
    const bool moves = !moving_.empty();

    // Every shape's box at rest: its geometry's bounds placed by its transform.
    std::vector<shapes::Bounds> at_rest(count);
    for (std::size_t i = 0; i < count; ++i) {
        at_rest[i] = shapes::world_bounds(shapes::object_bounds(shapes, shapes.records[i]), shapes.transforms[i]);
    }
    boxes_ = std::make_unique<FrameArray>(device, submission, std::as_bytes(std::span(at_rest)),
                                          moves ? FrameArray::Copies::per_frame : FrameArray::Copies::one);

    const auto pool = scoped_pool();
    MTL::Device* metal_device = device.handle();
    // What only the start-up build reads, resident for its command buffer
    // alone: a still scene's scratch. A moving scene builds every frame, and
    // its scratch is resident for good; it has no start-up build.
    NS::SharedPtr<MTL::ResidencySet> build_only;
    if (!moves) {
        auto set_descriptor = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
        NS::Error* error = nullptr;
        build_only = NS::TransferPtr(metal_device->newResidencySet(set_descriptor.get(), &error));
        if (!build_only) {
            throw MetalError("SceneAcceleration: the device made no residency set: " + describe(error));
        }
    }

    const std::uint32_t structures = moves ? frames_in_flight : 1u;
    for (std::uint32_t slot = 0; slot < structures; ++slot) {
        Built& built = structures_[slot];
        auto geometry = NS::TransferPtr(MTL4::AccelerationStructureBoundingBoxGeometryDescriptor::alloc()->init());
        geometry->setBoundingBoxBuffer(MTL4::BufferRange(boxes_->address(slot), boxes_->bytes(slot).size()));
        geometry->setBoundingBoxCount(count);
        geometry->setBoundingBoxStride(sizeof(shapes::Bounds));
        geometry->setOpaque(true);
        built.descriptor = NS::TransferPtr(MTL4::PrimitiveAccelerationStructureDescriptor::alloc()->init());
        const std::array<const NS::Object*, 1> geometries{geometry.get()};
        built.descriptor->setGeometryDescriptors(NS::Array::array(geometries.data(), geometries.size()));

        const MTL::AccelerationStructureSizes sizes = metal_device->accelerationStructureSizes(built.descriptor.get());
        built.structure = NS::TransferPtr(metal_device->newAccelerationStructure(sizes.accelerationStructureSize));
        built.scratch =
            NS::TransferPtr(metal_device->newBuffer(sizes.buildScratchBufferSize, MTL::ResourceStorageModePrivate));
        if (!built.structure || !built.scratch) {
            throw MetalError("SceneAcceleration: the device made no acceleration structure or scratch buffer");
        }
        // Traced every frame: resident until this object is destroyed.
        built.structure_resident = submission.keep_resident(built.structure.get());
        if (moves) {
            built.scratch_resident = submission.keep_resident(built.scratch.get());  // each frame's build writes it
        } else {
            build_only->addAllocation(built.scratch.get());
        }
    }
    if (moves) {
        return;  // each slot's structure is built by its first frame's update()
    }
    build_only->commit();

    // A still scene's one structure, built now.
    const FrameSlot build = submission.begin();
    build.commands->useResidencySet(build_only.get());
    MTL4::ComputeCommandEncoder* encoder = build.commands->computeCommandEncoder();
    if (encoder == nullptr) {
        throw MetalError("SceneAcceleration: the command buffer made no compute encoder");
    }
    Built& built = structures_[0];
    encoder->buildAccelerationStructure(built.structure.get(), built.descriptor.get(),
                                        MTL4::BufferRange(built.scratch->gpuAddress(), built.scratch->length()));
    encoder->endEncoding();
    submission.commit();
    // Start-up work, not a frame: waited for here, so the structure is whole
    // before any frame traces it (GPU.8: the build's writes are complete
    // before any read), and settled here, so it is never measured as a frame
    // (submission.h).
    submission.wait_until_complete(build.sequence);
    // The scratch was the build's alone, and the build is done: let it go
    // rather than hold it for the run (GPU.9). The set that made it resident
    // is released with it, at the end of this scope.
    built.scratch.reset();
}

void SceneAcceleration::update(MTL4::ComputeCommandEncoder* encoder, std::uint32_t slot,
                               std::span<const contracts::Transform> transforms) {
    if (moving_.empty()) {
        throw MetalError("SceneAcceleration::update: nothing in the scene moves");
    }
    if (encoder == nullptr || slot >= frames_in_flight) {
        throw MetalError("SceneAcceleration::update: no encoder, or no frame slot " + std::to_string(slot));
    }
    const std::span<shapes::Bounds> boxes = boxes_->view<shapes::Bounds>(slot);  // made from boxes
    if (transforms.size() != boxes.size()) {
        throw MetalError("SceneAcceleration::update: not one transform per shape");
    }
    // Step 1: the moving shapes' boxes, where this frame places them.
    for (std::size_t i = 0; i < moving_.size(); ++i) {
        const std::uint32_t shape = moving_[i];
        boxes[shape] = shapes::world_bounds(moving_bounds_[i], transforms[shape]);
    }
    // Step 2: the slot's structure, over every box.
    const Built& built = structures_[slot];
    encoder->buildAccelerationStructure(built.structure.get(), built.descriptor.get(),
                                        MTL4::BufferRange(built.scratch->gpuAddress(), built.scratch->length()));
    // Step 3: every pass after it traces the finished structure.
    encoder->barrierAfterEncoderStages(MTL::StageAccelerationStructure, MTL::StageDispatch,
                                       MTL4::VisibilityOptionDevice);
}

MTL::ResourceID SceneAcceleration::resource(std::uint32_t slot) const {
    if (slot >= frames_in_flight) {
        throw MetalError("SceneAcceleration::resource: no frame slot " + std::to_string(slot));
    }
    const Built& built = moving_.empty() ? structures_[0] : structures_[slot];
    return built.structure->gpuResourceID();
}

}  // namespace serenity::metal
