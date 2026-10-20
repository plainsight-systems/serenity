#pragma once

// Axis: GPU backend (device code): a thread's place in its warp.
//
// The lane and the mask of the lanes below it, read from PTX's special
// registers (%laneid, %lanemask_lt): in an OptiX program threadIdx is the
// launch index, not the lane (the OptiX 9.1 Programming Guide's table of
// special registers, 6.2), and only threadIdx and its kin have CUDA
// intrinsics, so the lanes are PTX's, by inline assembly, as the guide
// says. The same in a plain CUDA kernel. For the warp-aggregated counts
// and appends (film/accumulate.cuh, device/append.cuh).

#include "cuda/math/vector.cuh"

namespace serenity {
namespace shaders {

__device__ inline uint lane_id() {
  uint id = 0;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(id));
  return id;
}

__device__ inline uint lanes_below() {
  uint mask = 0;
  asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
  return mask;
}

}  // namespace shaders
}  // namespace serenity
