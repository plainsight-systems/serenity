#include "metal/frame/frame_images.h"

#include <algorithm>
#include <string>

#include "metal/device/error.h"
#include "metal/device/support.h"

namespace serenity::metal {

namespace {

// Half of `size` on each axis, rounded up, at least 1 (core/passes/tone_map.h,
// step 2).
constexpr frame::Extent half(frame::Extent size) noexcept {
    return {std::max(1u, (size.width + 1) / 2), std::max(1u, (size.height + 1) / 2)};
}

}  // namespace

FrameImages::FrameImages(const Device& device, Submission& submission, Bloom bloom)
    : device_(NS::RetainPtr(device.handle())), submission_(submission), bloom_(bloom) {}

MTL::Texture* FrameImages::bloom(std::uint32_t level) const {
    if (level >= pyramid_.size()) {
        throw MetalError("FrameImages: no bloom level " + std::to_string(level));
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
    bool settled_in_flight = false;
    const auto release = [&](Image& image) {
        if (!image.texture) {
            return;
        }
        if (!settled_in_flight) {
            // Settled here, so a failure in a frame in flight is reported.
            submission_.drain();
            settled_in_flight = true;
        }
        image.resident.reset();
        image.texture.reset();
    };
    release(radiance_);
    for (auto& level : pyramid_) {
        release(level);
    }
    const auto pool = scoped_pool();
    const auto make = [&](Image& image, MTL::PixelFormat format, frame::Extent at, const char* what) {
        image.texture = make_private_texture(device_.get(), format, at, what);
        image.resident = submission_.keep_resident(image.texture.get());
    };
    make(radiance_, MTL::PixelFormatRGBA32Float, size, "radiance image");
    if (bloom_ == Bloom::pyramid) {
        frame::Extent level = size;
        for (auto& image : pyramid_) {
            level = half(level);
            make(image, MTL::PixelFormatRGBA16Float, level, "bloom level");
        }
    }
    size_ = size;
}

}  // namespace serenity::metal
