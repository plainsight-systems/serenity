#include "metal/device/offscreen.h"

#include <cstddef>
#include <string>

#include "metal/device/error.h"
#include "metal/device/support.h"

namespace serenity::metal {

namespace {

constexpr std::size_t bytes_per_pixel = 4;  // RGBA8Unorm

}  // namespace

Offscreen::Offscreen(const Device& device, Submission& submission, frame::Extent size) : size_(size) {
    check_texture_size(size, "Offscreen");
    const auto pool = scoped_pool();
    MTL::TextureDescriptor* descriptor =
        MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA8Unorm, size.width, size.height, false);
    descriptor->setStorageMode(MTL::StorageModeShared);
    descriptor->setUsage(MTL::TextureUsageShaderWrite | MTL::TextureUsageShaderRead);
    texture_ = NS::TransferPtr(device.handle()->newTexture(descriptor));
    if (!texture_) {
        throw MetalError("Offscreen: the device made no " + std::to_string(size.width) + " x " +
                    std::to_string(size.height) + " texture");
    }
    resident_ = submission.keep_resident(texture_.get());
}

std::size_t Offscreen::rgba_size() const {
    return std::size_t{size_.width} * size_.height * bytes_per_pixel;
}

void Offscreen::read_rgba(std::span<std::uint8_t> out) const {
    const std::size_t row_bytes = std::size_t{size_.width} * bytes_per_pixel;
    const std::size_t expected = rgba_size();
    if (out.size() != expected) {
        throw MetalError("read_rgba: " + std::to_string(out.size()) + " bytes given for an image of " +
                    std::to_string(expected));
    }
    texture_->getBytes(out.data(), row_bytes, MTL::Region{0, 0, size_.width, size_.height}, 0);
}

}  // namespace serenity::metal
