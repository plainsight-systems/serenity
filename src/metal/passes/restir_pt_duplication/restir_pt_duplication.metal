// ReSTIR PT's duplication map (restir_pt_duplication.h): step 13 of
// metal/passes/restir_pt/restir_pt.metal.h, one thread a pixel in groups of
// 24 x 24 (metal/reuse/duplication.metal.h). Every thread reaches the
// group's barrier: none returns before it.

#include <metal_stdlib>

#include "core/contracts/frame_constants.h"
#include "core/contracts/path_reservoir.h"
#include "metal/passes/bindings.h"
#include "metal/reuse/duplication.metal.h"

using namespace metal;
using namespace serenity::shaders;
namespace binding = serenity::bindings::restir_pt;

kernel void restir_pt_duplication(constant serenity::contracts::FrameConstants& frame [[buffer(binding::constants)]],
                                  const device serenity::contracts::PathReservoir* finals [[buffer(binding::finals)]],
                                  device float* map [[buffer(binding::duplication)]],
                                  uint2 pixel [[thread_position_in_grid]],
                                  uint2 local [[thread_position_in_threadgroup]],
                                  uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup uint2 keys[duplication::span * duplication::span];
    const uint2 size = uint2(frame.width, frame.height);
    duplication::load_keys(keys, finals, size, group * duplication::group, local.y * duplication::group + local.x);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (pixel.x < size.x && pixel.y < size.y) {
        map[pixel.y * size.x + pixel.x] = duplication::score(keys, local);
    }
}
