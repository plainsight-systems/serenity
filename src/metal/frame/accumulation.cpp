#include "metal/frame/accumulation.h"

#include <string>

#include "metal/device/error.h"

namespace serenity::metal {

Accumulation::Accumulation(const Device& device, Submission& submission)
    : device_(NS::RetainPtr(device.handle())), submission_(submission) {}

std::uint32_t Accumulation::prepare(const frame::FrameInputs& inputs, frame::Extent size, bool scene_changes) {
    // The rule is asked on a copy, kept only once the image it describes
    // exists, so a failure leaves the history saying what the image holds
    // (E.4).
    frame::History next = history_;
    const frame::Joined joined = [&] {
        try {
            return next.join(inputs, size, scene_changes);
        } catch (const frame::HistoryError& refused) {
            throw Error(refused.what());
        }
    }();
    if (!joined.remade) {
        history_ = next;
        return joined.held;
    }
    // Starting over at a new size: the image's contents are not read (the
    // pass treats every pixel as empty), so only its size matters. Checked
    // before the old image goes, so a size Metal cannot make changes nothing.
    check_texture_size(size, "the accumulated image");
    if (texture_) {
        // Frames in flight may still read the old image: settled here, so a
        // failure among them is reported, and untimed, a resize's frame or
        // two. From here the old image is gone, so until a new one is made
        // the history holds nothing.
        submission_.drain();
        resident_.reset();
        texture_.reset();
        history_ = frame::History{};
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
    resident_ = submission_.keep_resident(texture_.get());
    history_ = next;
    return joined.held;
}

}  // namespace serenity::metal
