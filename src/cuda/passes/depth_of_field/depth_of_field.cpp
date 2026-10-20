#include "cuda/passes/depth_of_field/depth_of_field.h"

#include <array>
#include <cstdint>

#include "core/passes/depth_of_field.h"
#include "cuda/device/error.h"
#include "cuda/device/images.h"
#include "serenity/cuda_modules/depth_of_field.h"

namespace serenity::cuda {

DepthOfFieldPass::DepthOfFieldPass()
    : module_(cuda_modules::depth_of_field),
      presort_(module_.kernel("dof_presort")),
      tiles_(module_.kernel("dof_tiles")),
      dilate_(module_.kernel("dof_dilate")),
      gather_(module_.kernel("dof_gather")),
      median_(module_.kernel("dof_median")),
      composite_(module_.kernel("dof_composite")) {}

void DepthOfFieldPass::record(CUstream stream,
                              const DepthOfFieldInputs& inputs) const {
  const DepthOfFieldImages* images = inputs.images;
  if (inputs.radiance == 0 || inputs.full_depth == 0 || images == nullptr ||
      images->presort() == nullptr) {
    throw CudaError(
        "DepthOfFieldPass: the frame lacks the radiance image, the full-size "
        "depth or the pass's images");
  }
  const frame::Extent output = inputs.size;
  if (images->size() != output || inputs.constants.width != output.width ||
      inputs.constants.height != output.height) {
    throw CudaError(
        "DepthOfFieldPass: the pass's images or constants are not at the "
        "output's size");
  }
  // Kernel parameters are copied by the driver at each launch (module.h);
  // each kernel's are the constants, then Metal's bindings in the order
  // its kernel declares them (depth_of_field.cu).
  contracts::DepthOfFieldConstants k = inputs.constants;
  CUdeviceptr radiance = inputs.radiance;
  std::uint64_t pitch = inputs.radiance_pitch;
  CUdeviceptr depth = inputs.full_depth;
  CUdeviceptr tiles = images->tiles();
  CUdeviceptr dilated = images->dilated();
  CUsurfObject presort = images->presort()->surface();
  CUsurfObject far = images->far()->surface();
  CUsurfObject near = images->near()->surface();
  CUsurfObject far_filtered = images->far_filtered()->surface();
  CUsurfObject near_filtered = images->near_filtered()->surface();
  CUtexObject far_sampled = images->far_filtered()->texture();
  CUtexObject near_sampled = images->near_filtered()->texture();

  // The grids (contract 21): the output's, half of it, and its tiles.
  const frame::Extent half = passes::depth_of_field_half(output);
  const frame::Extent tile_grid = passes::depth_of_field_tiles(output);
  // Each step after the one before, in the stream's order (depth_of_field.h).
  {
    auto arguments =
        std::to_array<void*>({&k, &radiance, &pitch, &depth, &presort});
    launch_per_pixel(stream, presort_, half, arguments);
  }
  {
    auto arguments = std::to_array<void*>({&k, &tiles, &presort});
    launch_per_pixel(stream, tiles_, tile_grid, arguments);
  }
  {
    auto arguments = std::to_array<void*>({&k, &tiles, &dilated});
    launch_per_pixel(stream, dilate_, tile_grid, arguments);
  }
  {
    auto arguments =
        std::to_array<void*>({&k, &dilated, &presort, &far, &near});
    launch_per_pixel(stream, gather_, half, arguments);
  }
  {
    auto arguments =
        std::to_array<void*>({&k, &far, &near, &far_filtered, &near_filtered});
    launch_per_pixel(stream, median_, half, arguments);
  }
  {
    auto arguments = std::to_array<void*>(
        {&k, &radiance, &pitch, &depth, &far_sampled, &near_sampled});
    launch_per_pixel(stream, composite_, output, arguments);
  }
}

}  // namespace serenity::cuda
