// ReSTIR PT's shade (restir_pt_shade.h): step 12 of metal/passes/restir_pt/
// restir_pt.metal.h, one thread a traced pixel (restir_pt.metal.h,
// Traced), in a graph without the spatial pass (whose merge shades
// otherwise). The frame's light buffer, plus the initial reservoir's own
// estimate where no temporal pass merged one in, folded into the
// accumulated image (restir_pt/shade.metal.h).

#include <metal_stdlib>

#include "core/contracts/frame_constants.h"
#include "core/contracts/path_reservoir.h"
#include "core/contracts/path_reuse_settings.h"
#include "metal/film/accumulate.metal.h"
#include "metal/passes/bindings.h"
#include "metal/passes/restir_pt/restir_pt.metal.h"
#include "metal/passes/restir_pt/reuse_constants.metal.h"
#include "metal/passes/restir_pt/shade.metal.h"

using namespace metal;
using namespace serenity::shaders;
namespace binding = serenity::bindings::restir_pt;

kernel void restir_pt_shade(
    constant serenity::contracts::FrameConstants& frame
    [[buffer(binding::constants)]],
    const device serenity::contracts::PathReservoir* reservoirs
    [[buffer(binding::reservoirs)]],
    const device float4* light [[buffer(binding::light)]],
    device atomic_uint* non_finite [[buffer(binding::non_finite)]],
    texture2d<float, access::read_write> accumulated
    [[texture(binding::accumulated)]],
    uint2 grid_place [[thread_position_in_grid]]) {
  // the run's, the pipeline's constants (reuse_constants.h)
  const serenity::contracts::PathReuseSettings settings = reuse_settings();
  // Every thread reaches count_non_finite() below, which sums over its
  // SIMD group: none returns early.
  const restir_pt::Traced traced = restir_pt::traced(frame, settings);
  const uint2 pixel = traced.pixel(grid_place);
  const bool inside = traced.inside(pixel);
  bool failed = false;
  if (inside) {
    const uint i = traced.slot(pixel);
    // neighbors and confidence_cap are 0 exactly in a graph without
    // restir_pt_spatial and restir_pt_temporal, whose merges add their
    // estimates to the light buffer (contracts/path_reuse_settings.h).
    const bool merged =
        settings.neighbors != 0u || settings.confidence_cap > 0.0f;
    const float3 sample =
        light[i].xyz +
        (merged ? float3(0.0f) : restir_pt::estimate(reservoirs[i]));
    failed = restir_pt::shade(accumulated, frame, traced, pixel, sample);
  }
  count_non_finite(failed, non_finite);
}
