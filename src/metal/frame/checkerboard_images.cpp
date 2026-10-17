#include "metal/frame/checkerboard_images.h"

#include "metal/device/support.h"

namespace serenity::metal {

CheckerboardImages::CheckerboardImages(const Device& device, Submission& submission)
    : device_(NS::RetainPtr(device.handle())), submission_(submission) {}

void CheckerboardImages::prepare(frame::Extent size) {
    check_texture_size(size, "checkerboard fill's history");
    if (size == size_) {
        return;
    }
    // Frames in flight may still read the old images (feature_images.h,
    // FeatureImages::prepare, the same reasons).
    size_ = {};
    if (light_[0]) {
        submission_.drain();
    }
    const auto pool = scoped_pool();
    for (ResidentTexture& image : light_) {
        image.reset();
        image = make_resident_texture(device_.get(), submission_, MTL::PixelFormatRGBA32Float, size,
                                      "checkerboard fill's light and depth");
    }
    size_ = size;
}

}  // namespace serenity::metal
