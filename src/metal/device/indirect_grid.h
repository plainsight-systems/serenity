#pragma once

// Axis: GPU backend (an indirect dispatch's arguments, written on the GPU).
//
// The grid a one-thread kernel writes for a dispatch whose size a count on
// the GPU decides: survivors appended to a list (append.metal.h,
// simd_reserve()), then one thread a survivor (GDSA.13). Metal reads its
// first six words as MTLDispatchThreadsIndirectArguments, the grid's
// threads, then the threadgroup's; the count it was made from follows, for
// the dispatched kernel's threads to compare their index with, since the
// grid holds at least one thread. ReSTIR PT's replays and MNEE's solve use
// it (metal/passes/restir_pt/restir_pt.metal.h; metal/passes/mnee/
// mnee.metal), one layout for both. Written by the GPU alone
// (indirect_grid(), append.metal.h); the CPU only places it in a small
// buffer beside its count (frame_images.h, mnee_list.h), asserting it fits,
// and hands its address to dispatchThreads.

#include "core/contracts/shared_layout.h"

namespace serenity {
namespace gpu {

struct IndirectGrid {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays): MSL has no std::array
  uint32_t threads[3];  // the grid's, at least one
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays): MSL has no std::array
  uint32_t group[3];  // the threadgroup's
  uint32_t count;     // the survivors, which the grid's first threads take
};

static_assert(
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers): Metal's layout
    sizeof(IndirectGrid) == 28,
    "IndirectGrid must be the same 28 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<IndirectGrid>,
              "IndirectGrid is written by the GPU as bytes");
#endif

}  // namespace gpu
}  // namespace serenity
