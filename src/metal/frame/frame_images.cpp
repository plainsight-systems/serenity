#include "metal/frame/frame_images.h"

#include <algorithm>
#include <string>

#include "metal/device/error.h"

namespace serenity::metal {

namespace {

NS::SharedPtr<MTL::Texture> make_image(MTL::Device* device, Submission& submission, MTL::PixelFormat format,
                                       frame::Extent size, const char* what) {
    auto descriptor = NS::TransferPtr(MTL::TextureDescriptor::alloc()->init());
    descriptor->setTextureType(MTL::TextureType2D);
    descriptor->setPixelFormat(format);
    descriptor->setWidth(size.width);
    descriptor->setHeight(size.height);
    descriptor->setStorageMode(MTL::StorageModePrivate);
    descriptor->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    auto texture = NS::TransferPtr(device->newTexture(descriptor.get()));
    if (!texture) {
        throw Error("the device made no " + std::to_string(size.width) + " x " + std::to_string(size.height) + " " +
                    what);
    }
    submission.make_resident(texture.get());
    return texture;
}

// Half of `size` on each axis, rounded up, at least 1 (core/passes/tone_map.h,
// step 2).
frame::Extent half(frame::Extent size) {
    return {std::max(1u, (size.width + 1) / 2), std::max(1u, (size.height + 1) / 2)};
}

}  // namespace

FrameImages::FrameImages(const Device& device, Submission& submission, bool radiance, bool pyramid)
    : device_(NS::RetainPtr(device.handle())),
      submission_(submission),
      wants_radiance_(radiance),
      wants_pyramid_(pyramid) {}

void FrameImages::prepare(frame::Extent size) {
    if (size.width == 0 || size.height == 0) {
        throw Error("FrameImages: an empty frame");
    }
    if (size == size_) {
        return;
    }
    // Frames in flight may still read the old images, whole or, after a
    // failed prepare(), in part. What the drain settles goes untimed: a
    // resize's frame or two.
    size_ = {};
    bool drained_queue = false;
    const auto release = [&](NS::SharedPtr<MTL::Texture>& texture) {
        if (!texture) {
            return;
        }
        if (!drained_queue) {
            (void)submission_.drain();
            drained_queue = true;
        }
        submission_.release_resident(texture.get());
        texture.reset();
    };
    release(radiance_);
    for (auto& level : pyramid_) {
        release(level);
    }
    auto drained = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    if (wants_radiance_) {
        radiance_ = make_image(device_.get(), submission_, MTL::PixelFormatRGBA32Float, size, "radiance image");
    }
    if (wants_pyramid_) {
        frame::Extent level = size;
        for (auto& texture : pyramid_) {
            level = half(level);
            texture = make_image(device_.get(), submission_, MTL::PixelFormatRGBA16Float, level, "bloom level");
        }
    }
    size_ = size;
}

}  // namespace serenity::metal
