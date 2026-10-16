#include "metal/device/readback.h"

#include <string>

#include "metal/device/error.h"
#include "metal/device/support.h"

namespace serenity::metal {

NS::SharedPtr<MTL::Buffer> read_back(MTL::Device* device, Submission& submission, frame::Extent size,
                                     std::size_t bytes, std::span<const ImageCopy> copies, std::string_view what) {
    const auto pool = scoped_pool();
    auto buffer = NS::TransferPtr(device->newBuffer(bytes, MTL::ResourceStorageModeShared));
    if (!buffer) {
        throw MetalError(std::string(what) + ": the device made no buffer to read into");
    }
    const Resident resident = submission.keep_resident(buffer.get());
    const FrameSlot begun = submission.begin();
    MTL4::ComputeCommandEncoder* const encoder = begun.commands->computeCommandEncoder();
    if (encoder == nullptr) {
        throw MetalError(std::string(what) + ": the command buffer made no encoder");
    }
    // Every dispatch committed before, finished and visible (above).
    encoder->barrierAfterQueueStages(MTL::StageDispatch, MTL::StageBlit, MTL4::VisibilityOptionDevice);
    const MTL::Size extent{size.width, size.height, 1};
    for (const ImageCopy& copy : copies) {
        encoder->copyFromTexture(copy.texture, 0, 0, MTL::Origin{0, 0, 0}, extent, buffer.get(), copy.offset,
                                 copy.row_bytes, copy.row_bytes * size.height);
    }
    encoder->endEncoding();
    submission.commit();
    (void)submission.wait_until_complete(begun.sequence);
    return buffer;
}

}  // namespace serenity::metal
