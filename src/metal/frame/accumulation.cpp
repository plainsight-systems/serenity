#include "metal/frame/accumulation.h"

#include <string>

#include "metal/device/error.h"

namespace serenity::metal {

Accumulation::Accumulation(const Device& device, Submission& submission)
    : device_(NS::RetainPtr(device.handle())), submission_(submission) {}

std::uint32_t Accumulation::prepare(const frame::FrameInputs& inputs, frame::Extent size) {
    if (inputs.accumulated_since > inputs.index) {
        throw Error("frame " + std::to_string(inputs.index) + " accumulates since frame " +
                    std::to_string(inputs.accumulated_since) + ", after itself");
    }
    const std::uint64_t held = inputs.index - inputs.accumulated_since;
    if (held > frame::max_accumulated_frames) {
        throw Error("frame " + std::to_string(inputs.index) + " would join " + std::to_string(held) +
                    " accumulated frames; an image holds at most " +
                    std::to_string(frame::max_accumulated_frames) + " (start over sooner)");
    }

    if (held == 0) {
        // Starting over: the image's contents are not read (the pass treats
        // every pixel as empty), so only its size matters.
        if (!texture_ || !(size == size_)) {
            if (texture_) {
                // Frames in flight may still read the old image. What the
                // drain settles goes untimed: a resize's frame or two.
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
                throw Error("the device made no " + std::to_string(size.width) + " x " +
                            std::to_string(size.height) + " accumulated image");
            }
            submission_.make_resident(texture_.get());
            size_ = size;
        }
        since_ = inputs.accumulated_since;
        next_ = inputs.index + 1;
        return 0;
    }

    if (!texture_ || inputs.accumulated_since != since_ || inputs.index != next_ || !(size == size_)) {
        throw Error("frame " + std::to_string(inputs.index) + " claims an image accumulated since frame " +
                    std::to_string(inputs.accumulated_since) + " at " + std::to_string(size.width) + " x " +
                    std::to_string(size.height) + ", which is not what the image holds" +
                    (texture_ ? " (frames " + std::to_string(since_) + " .. " + std::to_string(next_ - 1) +
                                    " at " + std::to_string(size_.width) + " x " + std::to_string(size_.height) + ")"
                              : " (nothing yet)") +
                    "; a frame skipped, or a change without starting over");
    }
    next_ = inputs.index + 1;
    return static_cast<std::uint32_t>(held);
}

}  // namespace serenity::metal
