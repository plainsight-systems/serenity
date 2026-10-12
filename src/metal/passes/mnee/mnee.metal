// The MNEE pass's kernels (mnee.h): the work list's length as an indirect
// grid, its requests solved one thread each, and, for the path pass, the
// light they found folded into the accumulated image.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/frame_constants.h"
#include "core/contracts/mnee_request.h"
#include "core/contracts/transform.h"
#include "metal/film/accumulate.metal.h"
#include "metal/integrator/mnee.metal.h"
#include "metal/integrator/path.metal.h"
#include "metal/passes/bindings.h"
#include "metal/scene/scene_block.metal.h"

using namespace metal;
using namespace metal::raytracing;
using namespace serenity::shaders;
namespace binding = serenity::bindings::mnee;

// The solve's indirect grid, as Metal reads MTLDispatchThreadsIndirectArguments,
// and the count it was made from (contract 16's header, at
// mnee_grid_offset).
struct SolveGrid {
    uint grid[3];
    uint group[3];
    uint count;
};

constant constexpr uint solve_group = 64u;  // mnee.h's

// One thread: the list's length as the solve's grid, and the count set back
// to 0 for the next frame's list, which a barrier orders after the solve
// that reads the grid.
kernel void mnee_grid(device atomic_uint* header [[buffer(binding::header)]],
                      device SolveGrid* grid [[buffer(binding::grid)]]) {
    const uint written = atomic_load_explicit(header, memory_order_relaxed);
    const uint capacity = atomic_load_explicit(header + 1, memory_order_relaxed);
    atomic_store_explicit(header, 0u, memory_order_relaxed);
    const uint count = min(written, capacity);  // past it, none was written (contract 16)
    // At least one thread: a grid of none is not one Metal takes.
    *grid = SolveGrid{{max(count, 1u), 1u, 1u}, {solve_group, 1u, 1u}, count};
}

// Each request solved, its light added to its pixel's: three atomic adds,
// since the requests of one path's vertices share a pixel. A light not
// finite is counted (metal/film/non_finite.h) and left out, as the path
// pass leaves out a sample.
kernel void mnee_solve(primitive_acceleration_structure structure [[buffer(binding::structure)]],
                       constant serenity::gpu::SceneBlock& block [[buffer(binding::scene)]],
                       constant float* sphere_glows [[buffer(binding::glows)]],
                       constant serenity::contracts::Transform* transforms [[buffer(binding::transforms)]],
                       device atomic_uint* non_finite [[buffer(binding::non_finite)]],
                       const device serenity::contracts::MneeRequest* requests [[buffer(binding::requests)]],
                       const device SolveGrid* grid [[buffer(binding::grid)]],
                       device atomic_float* light [[buffer(binding::light)]],
                       uint n [[thread_position_in_grid]]) {
    bool failed = false;
    if (n < grid->count) {
        const path::Scene scene{structure, &block, transforms, sphere_glows};
        const serenity::contracts::MneeRequest request = requests[n];
        const float3 found = mnee::solve_request(scene, request);
        failed = !all(isfinite(found));
        if (!failed && any(found != 0.0f)) {
            for (uint c = 0u; c < 3u; ++c) {
                atomic_fetch_add_explicit(light + request.pixel * 4u + c, found[c], memory_order_relaxed);
            }
        }
    }
    count_non_finite(failed, non_finite);
}

// The path pass's: each pixel's MNEE light, this frame's sample's share,
// added to the accumulated image's mean (its count, n, in a: the mean
// moves by it over n, as if the path's sample had held it), and set back to
// 0 for the next frame. A pixel that holds no sample (its only one not
// finite, and left out) takes none of it either.
kernel void mnee_fold(constant serenity::contracts::FrameConstants& frame [[buffer(binding::constants)]],
                      device float4* light [[buffer(binding::light)]],
                      texture2d<float, access::read_write> accumulated [[texture(binding::accumulated)]],
                      uint2 pixel [[thread_position_in_grid]]) {
    if (pixel.x >= frame.width || pixel.y >= frame.height) {
        return;
    }
    const uint i = pixel.y * frame.width + pixel.x;
    const float3 found = light[i].xyz;
    if (any(found != 0.0f)) {
        const float4 held = accumulated.read(pixel);
        if (held.a > 0.0f) {
            accumulated.write(float4(held.rgb + found / held.a, held.a), pixel);
        }
        light[i] = float4(0.0f);
    }
}
