#include "core/output/png.h"

#include <cstdint>
#include <limits>
#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include <stb_image_write.h>

namespace serenity::output {

namespace {

constexpr int channels = 4;  // R, G, B, A: a byte each (png.h)

}  // namespace

void write_png(const std::filesystem::path& path, frame::Extent extent, std::span<const std::uint8_t> rgba) {
    if (extent.width == 0 || extent.height == 0) {
        throw PngError("cannot write " + path.string() + ": the image is empty");
    }
    constexpr std::uint64_t int_max = static_cast<std::uint64_t>(std::numeric_limits<int>::max());
    const std::uint64_t row_bytes = std::uint64_t{extent.width} * channels;
    if (extent.width > int_max || extent.height > int_max || row_bytes > int_max) {
        throw PngError("cannot write " + path.string() + ": the image is too large for the encoder");
    }
    const std::uint64_t expected = row_bytes * extent.height;
    if (rgba.size() != expected) {
        throw PngError("cannot write " + path.string() + ": " + std::to_string(rgba.size()) +
                    " bytes given for a " + std::to_string(extent.width) + " x " + std::to_string(extent.height) +
                    " RGBA image, which needs " + std::to_string(expected));
    }
    const int ok = stbi_write_png(path.c_str(), static_cast<int>(extent.width), static_cast<int>(extent.height),
                                  channels, rgba.data(), static_cast<int>(row_bytes));
    if (ok == 0) {
        throw PngError("cannot write " + path.string());
    }
}

}  // namespace serenity::output
