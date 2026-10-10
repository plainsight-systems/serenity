#include "metal/frame/frame_images.h"

#include <algorithm>
#include <string>

#include "metal/device/error.h"

namespace serenity::metal {

namespace {

NS::SharedPtr<MTL::Texture> make_image(MTL::Device* device, MTL::PixelFormat format, frame::Extent size,
                                       const char* what) {
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

MTL::Texture* FrameImages::bloom(std::uint32_t level) const {
    if (level >= pyramid_.size()) {
        throw Error("FrameImages: no bloom level " + std::to_string(level));
    }
    return pyramid_[level].texture.get();
}

void FrameImages::prepare(frame::Extent size) {
    check_texture_size(size, "FrameImages");
    if (size == size_) {
        return;
    }
    // Frames in flight may still read the old images, whole or, after a
    // failed prepare(), in part. What the drain settles goes untimed: a
    // resize's frame or two.
    size_ = {};
    bool drained_queue = false;
    const auto release = [&](Image& image) {
        if (!image.texture) {
            return;
        }
        if (!drained_queue) {
            // Settled here, so a failure in a frame in flight is reported.
            submission_.drain();
            drained_queue = true;
        }
        image.resident.reset();
        image.texture.reset();
    };
    release(radiance_);
    for (auto& level : pyramid_) {
        release(level);
    }
    auto drained = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    const auto make = [&](Image& image, MTL::PixelFormat format, frame::Extent at, const char* what) {
        image.texture = make_image(device_.get(), format, at, what);
        image.resident = submission_.keep_resident(image.texture.get());
    };
    if (wants_radiance_) {
        make(radiance_, MTL::PixelFormatRGBA32Float, size, "radiance image");
    }
    if (wants_pyramid_) {
        frame::Extent level = size;
        for (auto& image : pyramid_) {
            level = half(level);
            make(image, MTL::PixelFormatRGBA16Float, level, "bloom level");
        }
    }
    size_ = size;
}

}  // namespace serenity::metal
