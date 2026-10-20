#pragma once

// Axis: Film (what a pixel accumulates).
//
// Folding a frame's radiance for a pixel into the mean of its samples so
// far, Metal's (metal/film/accumulate.metal.h, which states the running
// mean, why a sample that is not finite is left out and counted, and why
// the count is exact): each pixel four floats, the mean radiance in rgb
// and in w the count of samples it is the mean of.
//
//   mean' = mean + (sample - mean) / (n + 1),   n' = n + 1
//
// The count of samples left out is added to once a warp, not once a sample
// (GDSA.5: one atomic a lane on one address serializes in the frame that
// most needs reporting): count_non_finite() counts the failing lanes among
// those that call it with one ballot, and one of them adds the count. The
// OptiX guide allows a ballot in a ray generation program for an algorithm
// that does not depend on which threads share a warp (6.2, its example the
// warp-aggregated atomic add; docs/research/2026-10-19-cuda-backend.md,
// "Phase 3's design"): the count is the same whichever do. Unlike Metal's
// simd_sum, no thread need call it that has no pixel. Which lane adds is
// read from the lane's own register, %laneid: in an OptiX program
// threadIdx is the launch index, not the lane (the guide's table of
// special registers, 6.2), and only threadIdx and its kin have CUDA
// intrinsics, so the lane is PTX's (inline assembly, as the guide says).

#include "cuda/math/vector.cuh"

namespace serenity {
namespace shaders {

__device__ inline bool finite(float3 sample) { return all(isfinite(sample)); }

// The pixel after folding in `sample`: `pixel` as the image held it, rgb
// the mean and w the count; `starting_over` when the frame holds no frames
// before it. A sample that is not finite leaves the pixel as it was (or
// empty, starting over); the caller reports it with count_non_finite().
__device__ inline float4 accumulate(float4 pixel, float3 sample,
                                    bool starting_over) {
  const float4 held = starting_over ? make_float4(0.0f) : pixel;
  if (!finite(sample)) {
    return held;
  }
  const float n = held.w;
  return make_float4(rgb(held) + (sample - rgb(held)) / (n + 1.0f), n + 1.0f);
}

// The calling thread's lane in its warp, 0 to 31.
__device__ inline uint lane() {
  uint id = 0;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(id));
  return id;
}

// Adds to `counter` the number of the calling threads of the warp whose
// `failed` is true, with one atomic for them.
__device__ inline void count_non_finite(bool failed, uint* counter) {
  const uint active = __activemask();
  const uint failures = __ballot_sync(active, failed);
  // The lowest active lane adds for them all.
  if (failures != 0u && lane() == uint(__ffs(active) - 1)) {
    atomicAdd(counter, uint(__popc(failures)));
  }
}

}  // namespace shaders
}  // namespace serenity
