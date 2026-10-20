// ReSTIR PT's initial sampling's kernels (restir_pt_initial.h), those that
// trace nothing: the frame's power pyramid and its light tiles, Metal's
// restir_pt_light_power and restir_pt_light_tiles (metal/passes/
// restir_pt_initial/restir_pt_initial.metal), each taking the frame's
// launch parameters by address (frame/frame_launch.h).

#include "core/contracts/light_tile.h"
#include "cuda/frame/frame_launch.h"
#include "cuda/light_selection/power_light.cuh"
#include "cuda/passes/restir_pt/restir_pt.cuh"

namespace serenity {
namespace shaders {

// The frame's power pyramid (contracts/light_tile.h), by one block of
// light_power_threads: every light's power at this frame's glows, and its
// sums (light_selection/power_light.cuh).
__device__ inline void light_power_block(
    const serenity::cuda::FrameLaunch& launch) {
  const restir_pt::Scene scene{&launch.structures, launch.scene,
                               launch.transforms, launch.glows};
  build_power_pyramid(scene.lights(), launch.scene->light_counts->lights,
                      launch.light_power, threadIdx.x, blockDim.x);
}

// The frame's light tiles (contracts/light_tile.h), one thread an entry:
// each a light drawn by power from the frame's pyramid and packed as this
// frame places and lights it.
__device__ inline void light_tile_entry(
    const serenity::cuda::FrameLaunch& launch) {
  const uint n = blockIdx.x * blockDim.x + threadIdx.x;
  const restir_pt::Scene scene{&launch.structures, launch.scene,
                               launch.transforms, launch.glows,
                               launch.light_power};
  if (n >= serenity::contracts::light_tiles *
               serenity::contracts::light_tile_size ||
      scene.selection().count == 0u) {
    return;
  }
  launch.light_tiles[n] =
      restir_pt::tile_entry(scene, n, launch.frame.frame_index);
}

}  // namespace shaders
}  // namespace serenity

extern "C" __global__ void restir_pt_light_power(
    const serenity::cuda::FrameLaunch* launch) {
  serenity::shaders::light_power_block(*launch);
}

extern "C" __global__ void restir_pt_light_tiles(
    const serenity::cuda::FrameLaunch* launch) {
  serenity::shaders::light_tile_entry(*launch);
}
