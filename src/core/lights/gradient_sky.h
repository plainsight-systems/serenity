#pragma once

// Axis: Light (gradient sky).
//
// An environment at infinity: the radiance arriving from direction d blends
// from `horizon` at d.y = 0 to `zenith` at d.y = 1, as horizon + (zenith -
// horizon) * smoothstep(0, 1, d.y), and is `horizon` below the horizon. The
// first scene's night sky. Until there is light transport it is what a ray
// that leaves the scene sees (metal/passes/preview/preview.h). The shader
// half is metal/lights/gradient_sky.metal.h.

#include "core/contracts/float3.h"

namespace serenity {
namespace lights {

struct GradientSkyData {
    contracts::Float3 zenith;
    float padding0;
    contracts::Float3 horizon;
    float padding1;
};

static_assert(sizeof(GradientSkyData) == 32, "GradientSkyData must be the same 32 bytes on the host and in shaders");

}  // namespace lights
}  // namespace serenity
