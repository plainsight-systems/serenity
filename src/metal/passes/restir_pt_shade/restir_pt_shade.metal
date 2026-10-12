// ReSTIR PT's shade (restir_pt_shade.h): step 12 of metal/passes/restir_pt/
// restir_pt.metal.h, one thread a pixel. The frame's light buffer, plus the
// initial reservoir's own estimate where no spatial pass merged one in,
// folded into the accumulated image as the path pass folds its own
// (passes/path/path.metal), whose mean is the frame's radiance.

#include <metal_stdlib>

#include "core/contracts/frame_constants.h"
#include "core/contracts/path_reservoir.h"
#include "core/contracts/path_reuse_settings.h"
#include "metal/film/accumulate.metal.h"
#include "metal/passes/bindings.h"
#include "metal/passes/restir_pt/restir_pt.metal.h"

using namespace metal;
using namespace serenity::shaders;
namespace binding = serenity::bindings::restir_pt;

kernel void restir_pt_shade(constant serenity::contracts::FrameConstants& frame [[buffer(binding::constants)]],
                            constant serenity::contracts::PathReuseSettings& settings [[buffer(binding::settings)]],
                            const device serenity::contracts::PathReservoir* reservoirs
                            [[buffer(binding::reservoirs)]],
                            const device float4* light [[buffer(binding::light)]],
                            device atomic_uint* non_finite [[buffer(binding::non_finite)]],
                            texture2d<float, access::read_write> accumulated [[texture(binding::accumulated)]],
                            uint2 pixel [[thread_position_in_grid]]) {
    // Every thread reaches count_non_finite() below, which sums over its
    // SIMD group: none returns early.
    const bool inside = pixel.x < frame.width && pixel.y < frame.height;
    bool failed = false;
    if (inside) {
        const uint i = pixel.y * frame.width + pixel.x;
        // neighbors and confidence_cap are 0 exactly in a graph without
        // restir_pt_spatial and restir_pt_temporal, whose merges add their
        // estimates to the light buffer (contracts/path_reuse_settings.h).
        const bool merged = settings.neighbors != 0u || settings.confidence_cap > 0.0f;
        const float3 sample = light[i].xyz + (merged ? float3(0.0f) : restir_pt::estimate(reservoirs[i]));
        failed = !finite(sample);
        const float4 held = accumulate(accumulated.read(pixel), sample, frame.accumulated_frames == 0u);
        accumulated.write(held, pixel);
    }
    count_non_finite(failed, non_finite);
}
