#include "metal/frame/accumulation.h"

#include <string>

#include "metal/device/error.h"

namespace serenity::metal {

Accumulation::Accumulation(const Device& device, Submission& submission)
    : device_(NS::RetainPtr(device.handle())), submission_(submission) {}

std::uint32_t Accumulation::prepare(const frame::FrameInputs& inputs, frame::Extent size, bool scene_moves) {
    frame::Joined joined;
    try {
        joined = history_.join(inputs, size, scene_moves);
    } catch (const frame::HistoryError& refused) {
        throw Error(refused.what());
    }
    if (!joined.remade) {
        return joined.held;
    }
    // Starting over at a new size: the image's contents are not read (the
    // pass treats every pixel as empty), so only its size matters.
    if (texture_) {
        // Frames in flight may still read the old image. What the drain
        // settles goes untimed: a resize's frame or two.
        (void)submission_.drain();
        submission_.release_resident(texture_.get());
        texture_.reset();
    }
    auto drained = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto descriptor = NS::TransferPtr(MTL::TextureDescriptor::alloc()->init());
    descriptor->setTextureType(MTL::TextureType2D);
    descriptor->setPixelFormat(MTL::PixelFormatRGBA32Float);
    descriptor->setWidth(size.width);
    descriptor->setHeight(size.height);
    descriptor->setStorageMode(MTL::StorageModePrivate);
    descriptor->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    texture_ = NS::TransferPtr(device_->newTexture(descriptor.get()));
    if (!texture_) {
        throw Error("the device made no " + std::to_string(size.width) + " x " + std::to_string(size.height) +
                    " accumulated image");
    }
    submission_.make_resident(texture_.get());
    return joined.held;
}

}  // namespace serenity::metal
