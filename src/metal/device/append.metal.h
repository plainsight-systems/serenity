#pragma once

// Axis: GPU backend (shaders): appending survivors to a list, and the
// indirect grid that dispatches them.
//
// Divergent work is grouped by being work (GDSA.13): the threads that have
// some append it to a list, and a later dispatch runs one thread an item,
// its grid written on the GPU from the list's count. Two lists do this,
// ReSTIR PT's replays (metal/passes/restir_pt/restir_pt.metal.h) and MNEE's
// requests (metal/integrator/mnee.metal.h); each mechanism is written once,
// here (ES.3, F.10).

#include <metal_stdlib>

#include "metal/device/indirect_grid.h"

namespace serenity {
namespace shaders {

// Reserves `count` places in a list whose length is `counter`: one atomic
// for the SIMD group's lanes that call together, the active ones (GDSA.10).
// The lane's first place; each lane's follow the lanes' before it in the
// group. Every active lane must call: the scan and the broadcast are the
// group's. The counter passes the list's capacity where it fills, by what
// did not fit: the caller writes only below it.
inline uint simd_reserve(device metal::atomic_uint* counter, uint count) {
  const uint before = metal::simd_prefix_exclusive_sum(count);
  const uint total = metal::simd_sum(count);
  uint base = 0u;
  if (metal::simd_is_first() && total > 0u) {
    base = metal::atomic_fetch_add_explicit(counter, total,
                                            metal::memory_order_relaxed);
  }
  return metal::simd_broadcast_first(base) + before;
}

// The grid for `count` items, a thread each, in threadgroups of `group`.
inline serenity::gpu::IndirectGrid indirect_grid(uint count, uint group) {
  // At least one thread: a grid of none is not one Metal takes.
  return serenity::gpu::IndirectGrid{
      {metal::max(count, 1u), 1u, 1u}, {group, 1u, 1u}, count};
}

}  // namespace shaders
}  // namespace serenity
