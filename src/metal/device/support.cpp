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

NS::SharedPtr<MTL::Texture> make_private_texture(MTL::Device* device,
                                                 MTL::PixelFormat format,
                                                 frame::Extent size,
                                                 std::string_view what) {
  check_texture_size(size, what);
  auto descriptor = NS::TransferPtr(MTL::TextureDescriptor::alloc()->init());
  descriptor->setTextureType(MTL::TextureType2D);
  descriptor->setPixelFormat(format);
  descriptor->setWidth(size.width);
  descriptor->setHeight(size.height);
  descriptor->setStorageMode(MTL::StorageModePrivate);
  descriptor->setUsage(MTL::TextureUsageShaderRead |
                       MTL::TextureUsageShaderWrite);
  auto texture = NS::TransferPtr(device->newTexture(descriptor.get()));
  if (!texture) {
    throw MetalError("the device made no " + frame::to_string(size) + " " +
                     std::string(what));
  }
  return texture;
}

void dispatch_per_pixel(MTL4::ComputeCommandEncoder* encoder,
                        const MTL::ComputePipelineState* pipeline,
                        frame::Extent size) {
  // The largest power of two whose square the pipeline allows, and as
  // many rows of it as fit.
  const NS::UInteger threads = pipeline->maxTotalThreadsPerThreadgroup();
  NS::UInteger side = 1;
  while ((2 * side) * (2 * side) <= threads) {
    side *= 2;
  }
  encoder->dispatchThreads(MTL::Size{size.width, size.height, 1},
                           MTL::Size{side, threads / side, 1});
}

}  // namespace serenity::metal
