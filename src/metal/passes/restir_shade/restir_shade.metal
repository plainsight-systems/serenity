// ReSTIR DI's shade (restir_shade.h): step 6 of
// metal/passes/restir_di/restir_di.metal.h, one thread a pixel. The direct
// light from the final reservoir and the path's continuation, the path's
// numbers continued from the counter the initial pass recorded, folded into
// the accumulated image as the path pass folds its own
// (passes/path/path.metal), whose mean is the frame's radiance.
//
// In a graph with a temporal merge (contracts/reuse_settings.h: history is
// read, and so not 0, exactly there), where the shade traced its
// survivor's ray, it writes the reservoir back with that test, and W = 0
// where it was hidden, for the next frame's merge (restir_di.metal.h,
// Shaded); its own pixel's alone, so in place is safe.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/frame_constants.h"
#include "core/contracts/reservoir.h"
#include "core/contracts/reuse_constants.h"
#include "core/contracts/surface_record.h"
#include "core/contracts/transform.h"
#include "metal/film/accumulate.metal.h"
#include "metal/passes/bindings.h"
#include "metal/passes/restir_di/restir_di.metal.h"

using namespace metal;
using namespace metal::raytracing;
using namespace serenity::shaders;
namespace binding = serenity::bindings::restir_shade;

kernel void restir_shade(constant serenity::contracts::FrameConstants& frame [[buffer(binding::constants)]],
                         primitive_acceleration_structure structure [[buffer(binding::structure)]],
                         constant serenity::gpu::SceneBlock& block [[buffer(binding::scene)]],
                         constant float* sphere_glows [[buffer(binding::glows)]],
                         constant serenity::contracts::Transform* transforms [[buffer(binding::transforms)]],
                         constant serenity::contracts::ReuseConstants& reuse [[buffer(binding::reuse)]],
                         const device serenity::contracts::SurfaceRecord* records [[buffer(binding::records)]],
                         const device serenity::contracts::PathState* path_states [[buffer(binding::path_states)]],
                         device serenity::contracts::Reservoir* finals [[buffer(binding::final_reservoirs)]],
                         device atomic_uint* non_finite [[buffer(binding::non_finite)]],
                         texture2d<float, access::read_write> accumulated [[texture(binding::accumulated)]],
                         uint2 pixel [[thread_position_in_grid]]) {
    // Every thread reaches count_non_finite() below, which sums over its
    // SIMD group: none returns early.
    const bool inside = pixel.x < frame.width && pixel.y < frame.height;
    bool failed = false;
    if (inside) {
        const uint i = pixel.y * frame.width + pixel.x;
        const restir::Scene scene{structure, &block, transforms, sphere_glows};
        const serenity::contracts::PathState state = path_states[i];
        PathNumbers numbers = path_numbers(pixel, frame.frame_index);
        numbers.dimension = state.counts & serenity::contracts::path_counter_mask;  // where step 1 left it
        const restir::Shaded shaded = restir::shade(scene, records[i], state, finals[i], reuse.settings, numbers);
        if (shaded.traced && reuse.settings.history != 0u) {
            finals[i] = shaded.reservoir;
        }

        failed = !finite(shaded.radiance);
        const float4 held = accumulate(accumulated.read(pixel), shaded.radiance, frame.accumulated_frames == 0u);
        accumulated.write(held, pixel);
    }
    count_non_finite(failed, non_finite);
}
