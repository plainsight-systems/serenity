#include "metal/acceleration/scene_acceleration.h"

#include <string>

#include "metal/device/error.h"

namespace serenity::metal {

// The core's bounds are Metal's bounding boxes, byte for byte, so they are
// written as they are.
static_assert(sizeof(shapes::Bounds) == sizeof(MTL::AxisAlignedBoundingBox));

namespace {

std::string describe(const NS::Error* error) {
    if (error == nullptr || error->localizedDescription() == nullptr) {
        return "Metal gave no description";
    }
    return error->localizedDescription()->utf8String();
}

std::span<shapes::Bounds> as_boxes(std::span<std::byte> bytes) {
    // The bytes were made from boxes (the constructor), in a buffer whose
    // copies start 256-byte aligned: they are boxes.
    return {reinterpret_cast<shapes::Bounds*>(bytes.data()), bytes.size() / sizeof(shapes::Bounds)};
}

}  // namespace

SceneAcceleration::SceneAcceleration(const Device& device, Submission& submission, const shapes::Shapes& shapes,
                                     std::span<const std::uint32_t> moving) {
    const std::size_t count = shapes.records.size();
    if (count == 0 || shapes.transforms.size() != count) {
        throw Error("SceneAcceleration: no shapes, or not one transform per shape");
    }
    for (std::size_t i = 0; i < moving.size(); ++i) {
        if (moving[i] >= count || (i > 0 && moving[i] <= moving[i - 1])) {
            throw Error("SceneAcceleration: the moving shapes are not shapes' indices in increasing order");
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
                                          moves ? frames_in_flight : 1u);

    auto drained = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    MTL::Device* mtl = device.handle();
    auto set_descriptor = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
    NS::Error* error = nullptr;
    // What only the start-up build reads, resident for its command buffer
    // alone: a still scene's scratch.
    auto build_only = NS::TransferPtr(mtl->newResidencySet(set_descriptor.get(), &error));
    if (!build_only) {
        throw Error("SceneAcceleration: the device made no residency set: " + describe(error));
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
        const NS::Object* list[] = {geometry.get()};
        built.descriptor->setGeometryDescriptors(NS::Array::array(list, 1));

        const MTL::AccelerationStructureSizes sizes = mtl->accelerationStructureSizes(built.descriptor.get());
        built.structure = NS::TransferPtr(mtl->newAccelerationStructure(sizes.accelerationStructureSize));
        built.scratch =
            NS::TransferPtr(mtl->newBuffer(sizes.buildScratchBufferSize, MTL::ResourceStorageModePrivate));
        if (!built.structure || !built.scratch) {
            throw Error("SceneAcceleration: the device made no acceleration structure or scratch buffer");
        }
        // Traced every frame: resident for good.
        submission.make_resident(built.structure.get());
        if (moves) {
            submission.make_resident(built.scratch.get());  // each frame's build writes it
        } else {
            build_only->addAllocation(built.scratch.get());
        }
    }
    build_only->commit();
    if (moves) {
        return;  // each slot's structure is built by its first frame's update()
    }

    // A still scene's one structure, built now.
    const FrameSlot build = submission.begin();
    build.commands->useResidencySet(build_only.get());
    MTL4::ComputeCommandEncoder* encoder = build.commands->computeCommandEncoder();
    const Built& built = structures_[0];
    encoder->buildAccelerationStructure(built.structure.get(), built.descriptor.get(),
                                        MTL4::BufferRange(built.scratch->gpuAddress(), built.scratch->length()));
    encoder->endEncoding();
    submission.commit();
    // Start-up work, not a frame: waited for here, so the structure is whole
    // before any frame traces it, and settled here, so it is never measured
    // as a frame (submission.h).
    (void)submission.wait_until_complete(build.sequence);
}

void SceneAcceleration::update(MTL4::ComputeCommandEncoder* encoder, std::uint32_t slot,
                               std::span<const contracts::Transform> transforms) const {
    if (moving_.empty()) {
        throw Error("SceneAcceleration::update: nothing in the scene moves");
    }
    const std::span<shapes::Bounds> boxes = as_boxes(boxes_->bytes(slot));
    if (transforms.size() != boxes.size()) {
        throw Error("SceneAcceleration::update: not one transform per shape");
    }
    // Step 1: the moving shapes' boxes, where this frame places them.
    for (std::size_t i = 0; i < moving_.size(); ++i) {
        const std::uint32_t shape = moving_[i];
        boxes[shape] = shapes::world_bounds(moving_bounds_[i], transforms[shape]);
    }
    // Step 2: the slot's structure, over every box.
    const Built& built = structures_[slot % frames_in_flight];
    encoder->buildAccelerationStructure(built.structure.get(), built.descriptor.get(),
                                        MTL4::BufferRange(built.scratch->gpuAddress(), built.scratch->length()));
    // Step 3: every pass after it traces the finished structure.
    encoder->barrierAfterEncoderStages(MTL::StageAccelerationStructure, MTL::StageDispatch,
                                       MTL4::VisibilityOptionDevice);
}

MTL::ResourceID SceneAcceleration::resource(std::uint32_t slot) const {
    const Built& built = moving_.empty() ? structures_[0] : structures_[slot % frames_in_flight];
    return built.structure->gpuResourceID();
}

}  // namespace serenity::metal
