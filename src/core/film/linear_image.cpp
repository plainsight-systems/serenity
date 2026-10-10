// Contract 13, the linear image (core/contracts/linear_image.h): defined by
// Film, its owner. Every refusal is made before anything is allocated
// (I.5): an extent's sides are bounded first, so the counts below are at
// most 16384 x 16384 x 4 floats and cannot overflow a size_t (ES.103).

#include "core/contracts/linear_image.h"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace serenity::contracts {

namespace {

constexpr std::size_t rgb_channels = 3;   // what a LinearImage keeps a pixel
constexpr std::size_t rgba_channels = 4;  // what the GPU's accumulated image holds a pixel

// The extent's pixels; throws std::invalid_argument, starting with `what`,
// for a side of 0 or past max_image_side.
std::size_t checked_pixels(frame::Extent extent, std::string_view what) {
    if (extent.width == 0 || extent.height == 0 || extent.width > max_image_side ||
        extent.height > max_image_side) {
        throw std::invalid_argument(std::string{what} + ": an image of " + std::to_string(extent.width) + " x " +
                                    std::to_string(extent.height) + ", where each side is from 1 to " +
                                    std::to_string(max_image_side));
    }
    return std::size_t{extent.width} * extent.height;
}

}  // namespace

LinearImage make_linear_image(frame::Extent extent) {
    return LinearImage{extent,
                       std::vector<float>(checked_pixels(extent, "make_linear_image") * rgb_channels, 0.0f)};
}

LinearImage from_rgba(frame::Extent extent, std::span<const float> rgba) {
    const std::size_t pixels = checked_pixels(extent, "from_rgba");
    if (rgba.size() != pixels * rgba_channels) {
        throw std::invalid_argument("from_rgba: " + std::to_string(rgba.size()) + " floats given for a " +
                                    std::to_string(extent.width) + " x " + std::to_string(extent.height) +
                                    " RGBA image, which holds " + std::to_string(pixels * rgba_channels));
    }
    LinearImage image = make_linear_image(extent);
    // An index loop over pixels, not a range-for: one index addresses both
    // layouts, 4 floats a pixel in and 3 out (ES.71).
    for (std::size_t p = 0; p < pixels; ++p) {
        for (std::size_t c = 0; c < rgb_channels; ++c) {
            image.rgb[p * rgb_channels + c] = rgba[p * rgba_channels + c];
        }
    }
    return image;
}

std::size_t checked_values(const LinearImage& image, std::string_view what) {
    const std::size_t values = checked_pixels(image.extent, what) * rgb_channels;
    if (image.rgb.size() != values) {
        throw std::invalid_argument(std::string{what} + ": " + std::to_string(image.rgb.size()) + " floats for a " +
                                    std::to_string(image.extent.width) + " x " +
                                    std::to_string(image.extent.height) + " image, which holds " +
                                    std::to_string(values));
    }
    return values;
}

std::string value_position(frame::Extent extent, std::size_t i) {
    if (extent.width == 0) {
        throw std::invalid_argument("value_position: an image of width 0 has no positions");  // ES.105
    }
    const std::size_t pixel = i / rgb_channels;
    return "pixel (" + std::to_string(pixel % extent.width) + ", " + std::to_string(pixel / extent.width) +
           ") channel " + std::to_string(i % rgb_channels);
}

}  // namespace serenity::contracts
