#include "metal/acceleration/primitives.h"

#include <cstring>

#include <Metal/MTL4AccelerationStructure.hpp>  // not in Metal.hpp's umbrella

#include "metal/device/error.h"

namespace serenity::metal {

// The core's bounds are Metal's bounding boxes, byte for byte, so the boxes
// are copied as they are.
static_assert(sizeof(shapes::Bounds) == sizeof(MTL::AxisAlignedBoundingBox));

namespace {

std::string describe(const NS::Error* error) {
    if (error == nullptr || error->localizedDescription() == nullptr) {
        return "Metal gave no description";
    }
    return error->localizedDescription()->utf8String();
}

}  // namespace

PrimitiveAcceleration::PrimitiveAcceleration(const Device& device, Submission& submission,
                                             std::span<const shapes::Bounds> bounds) {
    if (bounds.empty()) {
        throw Error("PrimitiveAcceleration: no boxes to build over");
    }
    auto drained = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    MTL::Device* mtl = device.handle();

    boxes_ = NS::TransferPtr(mtl->newBuffer(bounds.size_bytes(), MTL::ResourceStorageModeShared));
    if (!boxes_) {
        throw Error("PrimitiveAcceleration: the device made no buffer for the boxes");
    }
    std::memcpy(boxes_->contents(), bounds.data(), bounds.size_bytes());

    auto geometry = NS::TransferPtr(MTL4::AccelerationStructureBoundingBoxGeometryDescriptor::alloc()->init());
    geometry->setBoundingBoxBuffer(MTL4::BufferRange(boxes_->gpuAddress(), boxes_->length()));
    geometry->setBoundingBoxCount(bounds.size());
    geometry->setBoundingBoxStride(sizeof(shapes::Bounds));
    geometry->setOpaque(true);

    auto descriptor = NS::TransferPtr(MTL4::PrimitiveAccelerationStructureDescriptor::alloc()->init());
    const NS::Object* geometries[] = {geometry.get()};
    descriptor->setGeometryDescriptors(NS::Array::array(geometries, 1));

    const MTL::AccelerationStructureSizes sizes = mtl->accelerationStructureSizes(descriptor.get());
    structure_ = NS::TransferPtr(mtl->newAccelerationStructure(sizes.accelerationStructureSize));
    auto scratch = NS::TransferPtr(mtl->newBuffer(sizes.buildScratchBufferSize, MTL::ResourceStorageModePrivate));
    if (!structure_ || !scratch) {
        throw Error("PrimitiveAcceleration: the device made no acceleration structure or scratch buffer");
    }
    // The structure is traced every frame, so it joins the queue's residency
    // for good. The boxes and the scratch are read only by the build, so they
    // are resident for its command buffer alone and released after it.
    submission.make_resident(structure_.get());
    auto set_descriptor = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
    NS::Error* error = nullptr;
    auto build_only = NS::TransferPtr(mtl->newResidencySet(set_descriptor.get(), &error));
    if (!build_only) {
        throw Error("PrimitiveAcceleration: the device made no residency set: " + describe(error));
    }
    build_only->addAllocation(boxes_.get());
    build_only->addAllocation(scratch.get());
    build_only->commit();

    const FrameSlot build = submission.begin();
    build.commands->useResidencySet(build_only.get());
    MTL4::ComputeCommandEncoder* encoder = build.commands->computeCommandEncoder();
    encoder->buildAccelerationStructure(structure_.get(), descriptor.get(),
                                        MTL4::BufferRange(scratch->gpuAddress(), scratch->length()));
    encoder->endEncoding();
    submission.commit();
    // Start-up work, not a frame: waited for here, so the structure is whole
    // before any frame traces it, and settled here, so it is never measured
    // as a frame (submission.h).
    (void)submission.wait_until_complete(build.sequence);
}

}  // namespace serenity::metal
