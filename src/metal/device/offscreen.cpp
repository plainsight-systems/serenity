#include "metal/device/offscreen.h"

#include "metal/device/error.h"

namespace serenity::metal {

Offscreen::Offscreen(const Device& device, Submission& submission, frame::Extent size) : size_(size) {
    if (size.width == 0 || size.height == 0) {
        throw Error("Offscreen: an image needs a width and a height");
    }
    auto drained = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    MTL::TextureDescriptor* descriptor =
        MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA8Unorm, size.width, size.height, false);
    descriptor->setStorageMode(MTL::StorageModeShared);
    descriptor->setUsage(MTL::TextureUsageShaderWrite | MTL::TextureUsageShaderRead);
    texture_ = NS::TransferPtr(device.handle()->newTexture(descriptor));
    if (!texture_) {
        throw Error("Offscreen: the device made no " + std::to_string(size.width) + " x " +
                    std::to_string(size.height) + " texture");
    }
    resident_ = submission.keep_resident(texture_.get());
}

void Offscreen::read_rgba(std::span<std::uint8_t> out) const {
    const std::size_t row_bytes = std::size_t{size_.width} * 4;
    const std::size_t expected = row_bytes * size_.height;
    if (out.size() != expected) {
        throw Error("read_rgba: " + std::to_string(out.size()) + " bytes given for an image of " +
                    std::to_string(expected));
    }
    texture_->getBytes(out.data(), row_bytes, MTL::Region(0, 0, size_.width, size_.height), 0);
}

}  // namespace serenity::metal
