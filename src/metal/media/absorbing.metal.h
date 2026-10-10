#pragma once

// Axis: Medium (shader half of media/absorbing.h).
//
// Beer and Lambert: a stretch t of an absorbing medium keeps exp(-absorption
// t) of the light, per channel.

#include <metal_stdlib>

#include "core/media/absorbing.h"
#include "metal/device/layout.metal.h"

namespace serenity {
namespace shaders {

inline float3 absorbing_transmittance(serenity::media::AbsorbingData medium, float t) {
    return metal::exp(-to_float3(medium.absorption) * t);
}

}  // namespace shaders
}  // namespace serenity
