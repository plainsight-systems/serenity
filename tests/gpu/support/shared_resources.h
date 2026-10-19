#pragma once

// The shared-storage buffers and textures the GPU tests make for the kernels
// they run, made one way (ES.3, F.10): the probe runner's, the tracing
// probe's, the ReSTIR PT probes', the tone map's and the denoiser fixture's.
// No doctest here, since denoise_check (a tool, not a test) makes its
// textures with it too: a device that makes none throws
// std::runtime_error, which a test case reports as its failure (P.7).

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <span>
#include <stdexcept>

#include "core/frame/extent.h"
#include "metal/device/device.h"
#include "metal/device/submission.h"

namespace serenity::tests {

// A shared buffer of at least `size` bytes, and at least 16 so an empty
// array still has an address: zeroed, then `bytes` copied to its start.
// Resident for as long as `submission` lives (Submission::make_resident,
// for a test's probe buffers).
inline NS::SharedPtr<MTL::Buffer> shared_buffer(
    const metal::Device& device, metal::Submission& submission,
    std::span<const std::byte> bytes, std::size_t size = 0) {
  const std::size_t made = std::max({bytes.size(), size, std::size_t{16}});
  auto buffer = NS::TransferPtr(
      device.handle()->newBuffer(made, MTL::ResourceStorageModeShared));
  if (!buffer) {
    throw std::runtime_error("the device made no buffer");
  }
  std::memset(buffer->contents(), 0, made);
  if (!bytes.empty()) {
    std::memcpy(buffer->contents(), bytes.data(), bytes.size());
  }
  submission.make_resident(buffer.get());
  return buffer;
}

// A 2D texture of `format` at `size` in shared storage, for `usage`; the
// caller makes it resident where it is bound.
inline NS::SharedPtr<MTL::Texture> shared_texture(const metal::Device& device,
                                                  MTL::PixelFormat format,
                                                  frame::Extent size,
                                                  MTL::TextureUsage usage) {
  auto descriptor = NS::TransferPtr(MTL::TextureDescriptor::alloc()->init());
  descriptor->setTextureType(MTL::TextureType2D);
  descriptor->setPixelFormat(format);
  descriptor->setWidth(size.width);
  descriptor->setHeight(size.height);
  descriptor->setStorageMode(MTL::StorageModeShared);
  descriptor->setUsage(usage);
  auto texture = NS::TransferPtr(device.handle()->newTexture(descriptor.get()));
  if (!texture) {
    throw std::runtime_error("the device made no texture");
  }
  return texture;
}

}  // namespace serenity::tests
