#include "cuda/device/images.h"

#include <string>

#include "cuda/device/error.h"

namespace serenity::cuda {

namespace {

constexpr unsigned channels = 4;  // RGBA

void check_size(frame::Extent size, const char* what) {
  if (size.width == 0 || size.height == 0) {
    throw CudaError(std::string(what) + ": an image of no pixels");
  }
}

// Metal's sampler (images.h): bilinear, clamped to the edge, normalized
// coordinates.
CUDA_TEXTURE_DESC filtered() {
  CUDA_TEXTURE_DESC texture{};
  for (CUaddress_mode& mode : texture.addressMode) {
    mode = CU_TR_ADDRESS_MODE_CLAMP;
  }
  texture.filterMode = CU_TR_FILTER_MODE_LINEAR;
  texture.flags = CU_TRSF_NORMALIZED_COORDINATES;
  return texture;
}

}  // namespace

PitchedImage::PitchedImage(const Device& device, Submission& submission,
                           frame::Extent size)
    : submission_(submission), size_(size) {
  check_size(size, "PitchedImage");
  constexpr std::size_t pixel_bytes = channels * sizeof(float);
  check(cuMemAllocPitch(&address_, &pitch_, size.width * pixel_bytes,
                        size.height, pixel_bytes),
        "cuMemAllocPitch");
  try {
    int alignment = 0;
    check(cuDeviceGetAttribute(&alignment,
                               CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT,
                               device.handle()),
          "cuDeviceGetAttribute");
    if (alignment <= 0 || pitch_ % static_cast<std::size_t>(alignment) != 0) {
      throw CudaError("PitchedImage: a pitch of " + std::to_string(pitch_) +
                      " bytes, which a texture cannot read (its alignment " +
                      std::to_string(alignment) + ")");
    }
    CUDA_RESOURCE_DESC resource{};
    resource.resType = CU_RESOURCE_TYPE_PITCH2D;
    // The C API's descriptor is a union, filled by the member its type
    // names, as CUDA's documentation fills it.
    // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
    resource.res.pitch2D.devPtr = address_;
    resource.res.pitch2D.format = CU_AD_FORMAT_FLOAT;
    resource.res.pitch2D.numChannels = channels;
    resource.res.pitch2D.width = size.width;
    resource.res.pitch2D.height = size.height;
    resource.res.pitch2D.pitchInBytes = pitch_;
    // NOLINTEND(cppcoreguidelines-pro-type-union-access)
    const CUDA_TEXTURE_DESC texture = filtered();
    check(cuTexObjectCreate(&texture_, &resource, &texture, nullptr),
          "cuTexObjectCreate");
  } catch (...) {
    (void)cuMemFree(address_);
    throw;
  }
}

PitchedImage::~PitchedImage() {
  (void)cuTexObjectDestroy(texture_);
  (void)cuMemFree(address_);
}

void PitchedImage::read(std::span<float> out) const {
  const std::size_t floats = std::size_t{size_.width} * size_.height * channels;
  if (out.size() != floats) {
    throw CudaError("PitchedImage::read: " + std::to_string(out.size()) +
                    " floats for an image of " + std::to_string(floats));
  }
  submission_.finish();
  CUDA_MEMCPY2D copy{};
  copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  copy.srcDevice = address_;
  copy.srcPitch = pitch_;
  copy.dstMemoryType = CU_MEMORYTYPE_HOST;
  copy.dstHost = out.data();
  copy.dstPitch = std::size_t{size_.width} * channels * sizeof(float);
  copy.WidthInBytes = copy.dstPitch;
  copy.Height = size_.height;
  check(cuMemcpy2D(&copy), "PitchedImage::read: cuMemcpy2D");
}

LevelImage::LevelImage(frame::Extent size) : size_(size) {
  check_size(size, "LevelImage");
  CUDA_ARRAY3D_DESCRIPTOR array{};
  array.Width = size.width;
  array.Height = size.height;
  array.Depth = 0;  // two-dimensional
  array.Format = CU_AD_FORMAT_HALF;
  array.NumChannels = channels;
  array.Flags = CUDA_ARRAY3D_SURFACE_LDST;
  check(cuArray3DCreate(&array_, &array), "cuArray3DCreate");
  try {
    CUDA_RESOURCE_DESC resource{};
    resource.resType = CU_RESOURCE_TYPE_ARRAY;
    // The descriptor's union, by the member its type names (above).
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)
    resource.res.array.hArray = array_;
    check(cuSurfObjectCreate(&surface_, &resource), "cuSurfObjectCreate");
    const CUDA_TEXTURE_DESC texture = filtered();
    if (const CUresult made =
            cuTexObjectCreate(&texture_, &resource, &texture, nullptr);
        made != CUDA_SUCCESS) {
      (void)cuSurfObjectDestroy(surface_);
      check(made, "cuTexObjectCreate");
    }
  } catch (...) {
    (void)cuArrayDestroy(array_);
    throw;
  }
}

LevelImage::~LevelImage() {
  (void)cuTexObjectDestroy(texture_);
  (void)cuSurfObjectDestroy(surface_);
  (void)cuArrayDestroy(array_);
}

}  // namespace serenity::cuda
