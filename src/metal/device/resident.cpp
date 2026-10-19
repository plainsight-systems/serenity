#include "metal/device/resident.h"

#include <cstring>
#include <string>

#include "metal/device/error.h"
#include "metal/device/support.h"

namespace serenity::metal {

ResidentBuffer make_resident_buffer(MTL::Device* device, Submission& submission,
                                    std::size_t bytes,
                                    MTL::ResourceOptions storage,
                                    std::string_view what, Zeroed zeroed) {
  ResidentBuffer made;
  made.buffer = NS::TransferPtr(device->newBuffer(bytes, storage));
  if (!made.buffer) {
    throw MetalError("the device made no buffer for " + std::string(what));
  }
  if (zeroed == Zeroed::yes) {
    std::memset(made.buffer->contents(), 0, bytes);
  }
  made.resident = submission.keep_resident(made.buffer.get());
  return made;
}

ResidentTexture make_resident_texture(MTL::Device* device,
                                      Submission& submission,
                                      MTL::PixelFormat format,
                                      frame::Extent size,
                                      std::string_view what) {
  ResidentTexture made;
  made.texture = make_private_texture(device, format, size, what);
  made.resident = submission.keep_resident(made.texture.get());
  return made;
}

}  // namespace serenity::metal
