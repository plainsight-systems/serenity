#include "cuda/passes/tone_map/tone_map.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>

#include "core/frame/schedule.h"
#include "cuda/device/error.h"
#include "serenity/cuda_modules/tone_map.h"

namespace serenity::cuda {

namespace {

// A size as the kernels take it (tone_map.cu): CUDA's uint2, in the
// layout the kernel's parameter has.
struct Size {
  std::uint32_t width;
  std::uint32_t height;
};

Size size_of(frame::Extent extent) { return {extent.width, extent.height}; }

passes::ToneMap checked(const passes::ToneMap& settings) {
  if (const std::optional<std::string> reason = frame::invalid(settings)) {
    throw CudaError("ToneMapPass: " + *reason);
  }
  return settings;
}

}  // namespace

ToneMapPass::ToneMapPass(const passes::ToneMap& settings)
    : settings_(checked(settings)),
      module_(cuda_modules::tone_map),
      down_first_(module_.kernel("tone_map_down_first")),
      down_(module_.kernel("tone_map_down")),
      up_(module_.kernel("tone_map_up")),
      finish_(module_.kernel("tone_map_finish")) {}

void ToneMapPass::record(CUstream stream, const PitchedImage& radiance,
                         const FrameImages& images, CUdeviceptr target) const {
  // Kernel parameters are copied by the driver at each launch (module.h),
  // so these locals may change between launches.
  passes::ToneMap settings = settings_;

  // Steps 1 and 2 for B_0.
  {
    CUtexObject input = radiance.texture();
    Size input_size = size_of(radiance.size());
    CUsurfObject level = images.level(0).surface();
    Size level_size = size_of(images.level(0).size());
    auto arguments = std::to_array<void*>(
        {&settings, &input, &input_size, &level, &level_size});
    launch_per_pixel(stream, down_first_, images.level(0).size(), arguments);
  }
  // Step 2 for B_1 .. B_5, each from the level before.
  for (std::uint32_t k = 1; k < passes::bloom_levels; ++k) {
    CUtexObject input = images.level(k - 1).texture();
    Size input_size = size_of(images.level(k - 1).size());
    CUsurfObject level = images.level(k).surface();
    Size level_size = size_of(images.level(k).size());
    auto arguments =
        std::to_array<void*>({&input, &input_size, &level, &level_size});
    launch_per_pixel(stream, down_, images.level(k).size(), arguments);
  }
  // Step 3 for B_4 .. B_0, each from the level below it: `below` counts
  // down from the last level to B_1.
  for (std::uint32_t below = passes::bloom_levels - 1; below > 0; --below) {
    const std::uint32_t k = below - 1;
    CUtexObject input = images.level(below).texture();
    Size input_size = size_of(images.level(below).size());
    CUsurfObject level = images.level(k).surface();
    Size level_size = size_of(images.level(k).size());
    auto arguments =
        std::to_array<void*>({&input, &input_size, &level, &level_size});
    launch_per_pixel(stream, up_, images.level(k).size(), arguments);
  }
  // Steps 1 and 4 to 6, into the target.
  {
    CUdeviceptr input = radiance.address();
    std::uint64_t pitch = radiance.pitch();
    CUtexObject bloom = images.level(0).texture();
    Size bloom_size = size_of(images.level(0).size());
    CUdeviceptr out = target;
    Size size = size_of(radiance.size());
    auto arguments = std::to_array<void*>(
        {&settings, &input, &pitch, &bloom, &bloom_size, &out, &size});
    launch_per_pixel(stream, finish_, radiance.size(), arguments);
  }
}

}  // namespace serenity::cuda
