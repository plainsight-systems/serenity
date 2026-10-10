#include "metal/device/support.h"

#include "metal/device/device.h"
#include "metal/device/error.h"

namespace serenity::metal {

std::string describe(const NS::Error* error) {
    if (error == nullptr || error->localizedDescription() == nullptr) {
        return "Metal gave no description";
    }
    return error->localizedDescription()->utf8String();
}

NS::SharedPtr<NS::AutoreleasePool> scoped_pool() {
    return NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
}

NS::SharedPtr<MTL::Texture> make_private_texture(MTL::Device* device, MTL::PixelFormat format, frame::Extent size,
                                                 const char* what) {
    check_texture_size(size, what);
    auto descriptor = NS::TransferPtr(MTL::TextureDescriptor::alloc()->init());
    descriptor->setTextureType(MTL::TextureType2D);
    descriptor->setPixelFormat(format);
    descriptor->setWidth(size.width);
    descriptor->setHeight(size.height);
    descriptor->setStorageMode(MTL::StorageModePrivate);
    descriptor->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    auto texture = NS::TransferPtr(device->newTexture(descriptor.get()));
    if (!texture) {
        throw Error(std::string{"the device made no "} + std::to_string(size.width) + " x " +
                    std::to_string(size.height) + " " + what);
    }
    return texture;
}

void dispatch_per_pixel(MTL4::ComputeCommandEncoder* encoder, const MTL::ComputePipelineState* pipeline,
                        frame::Extent size) {
    const NS::UInteger width = pipeline->threadExecutionWidth();
    const NS::UInteger rows = pipeline->maxTotalThreadsPerThreadgroup() / width;
    encoder->dispatchThreads(MTL::Size{size.width, size.height, 1}, MTL::Size{width, rows, 1});
}

}  // namespace serenity::metal
