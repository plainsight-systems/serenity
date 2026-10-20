// ReSTIR PT's duplication map (restir_pt_duplication.h): step 13 of
// passes/restir_pt/restir_pt.cuh, Metal's restir_pt_duplication
// (metal/passes/restir_pt_duplication/restir_pt_duplication.metal), a
// kernel, one thread a pixel in blocks of 24 x 24 (reuse/duplication.cuh),
// the block's keys in shared memory, Metal's threadgroup memory; with the
// checkerboard, one a slot of the packed grid, its window 17 x 17 slots
// (restir_pt.cuh, "Checkerboard rendering"). Every thread reaches the
// block's barrier: none returns before it.

#include "core/contracts/frame_constants.h"
#include "core/contracts/path_reservoir.h"
#include "core/contracts/path_reuse_settings.h"
#include "cuda/frame/frame_launch.h"
#include "cuda/passes/restir_pt/restir_pt.cuh"
#include "cuda/reuse/duplication.cuh"

namespace serenity {
namespace shaders {

__device__ inline void duplication_block(
    const serenity::cuda::FrameLaunch& launch) {
  __shared__ uint2 keys[duplication::span * duplication::span];
  const uint2 local = make_uint2(threadIdx.x, threadIdx.y);
  const uint2 group = make_uint2(blockIdx.x, blockIdx.y);
  const uint2 pixel = make_uint2(group.x * duplication::group + local.x,
                                 group.y * duplication::group + local.y);
  // The arrays' grid: the image, or the packed one, whose slots are its
  // row-major indices (restir_pt.cuh, Traced).
  const uint2 size =
      restir_pt::traced(launch.frame, launch.path_settings).grid();
  duplication::load_keys(
      keys, launch.path_finals, size,
      make_uint2(group.x * duplication::group, group.y * duplication::group),
      local.y * duplication::group + local.x);
  __syncthreads();
  if (pixel.x < size.x && pixel.y < size.y) {
    launch.duplication[pixel.y * size.x + pixel.x] =
        duplication::score(keys, local);
  }
}

}  // namespace shaders
}  // namespace serenity

extern "C" __global__ void restir_pt_duplication(
    const serenity::cuda::FrameLaunch* launch) {
  serenity::shaders::duplication_block(*launch);
}
