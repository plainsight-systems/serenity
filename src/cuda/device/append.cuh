#pragma once

// Axis: GPU backend (device code): appending survivors to a list.
//
// Divergent work is grouped by being work (GDSA.13): the threads that have
// some append it to a list, and a later launch runs one thread an item,
// Metal's (metal/device/append.metal.h), which two lists use: ReSTIR PT's
// replays (passes/restir_pt/restir_pt.cuh) and MNEE's requests
// (integrator/mnee.cuh). Metal's indirect grid has no counterpart here: no
// CUDA or OptiX launch takes its size from the GPU, so a list's reader
// launches at the list's capacity and its threads past the count return
// (docs/research/2026-10-19-cuda-backend.md, "Phase 4's design", "No
// indirect launch").
//
// The reserve is one atomic for the warp's lanes that call together, the
// active ones (GDSA.10), as Metal's simd_reserve is for a SIMD group: each
// lane's count's bits gathered by a ballot apiece, a lane's place the sum of
// the counts of the active lanes below it, the warp's total added to the
// counter by its lowest active lane and handed to the others by one
// shuffle. The OptiX guide allows warp-wide instructions in its programs
// for an algorithm that does not depend on which threads share a warp
// (6.2): the places differ with the grouping, the items appended do not.
// Checked on the RTX 3090 Ti in a 2D ray generation launch, divergent
// lanes appending 0 to 2 items each: every item appended once (the plan's
// "Phase 4's design"). The lanes are PTX's (device/lanes.cuh).

#include "core/contracts/path_reuse_settings.h"
#include "cuda/device/lanes.cuh"
#include "cuda/math/vector.cuh"

namespace serenity {
namespace shaders {

// The most items one thread appends at once: a pixel's paired neighbors,
// whose count's two bits the reserve gathers.
inline constexpr uint most_appended = serenity::contracts::max_path_neighbors;
static_assert(most_appended < 4u,
              "warp_reserve gathers two bits of a lane's count");

// Reserves `count` places, at most most_appended, in a list whose length is
// `counter`: one atomic for the warp's active lanes. The lane's first place;
// each lane's follow the lanes' below it. Every active lane must call. The
// counter passes the list's capacity where it fills, by what did not fit:
// the caller writes only below it.
__device__ inline uint warp_reserve(uint* counter, uint count) {
  const uint active = __activemask();
  const uint ones = __ballot_sync(active, (count & 1u) != 0u);
  const uint twos = __ballot_sync(active, (count & 2u) != 0u);
  const uint below = lanes_below();
  const uint before =
      uint(__popc(ones & below)) + 2u * uint(__popc(twos & below));
  const uint total = uint(__popc(ones)) + 2u * uint(__popc(twos));
  const int first_lane = __ffs(int(active)) - 1;
  uint base = 0u;
  if (lane_id() == uint(first_lane) && total > 0u) {
    base = atomicAdd(counter, total);
  }
  return __shfl_sync(active, base, first_lane) + before;
}

}  // namespace shaders
}  // namespace serenity
