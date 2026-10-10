#pragma once

// Axis: Material (conductor).
//
// A metal: it reflects and does not transmit, and the color of what it
// reflects is the metal's own, by Schlick's approximation to its Fresnel
// reflectance from `f0`, its reflectance at normal incidence, in linear RGB
// (brass is about [0.91, 0.78, 0.42]). The surface's microfacets are
// distributed by GGX with alpha = roughness^2: a roughness near 0 is a
// mirror, 0.3 to 0.5 a soft, satin metal. The shader half is
// metal/materials/conductor.metal.h.

#include "core/contracts/shared_layout.h"
#include "core/contracts/float3.h"

namespace serenity {
namespace materials {

struct ConductorData {
    contracts::Float3 f0;  // each in [0, 1]
    float roughness;       // in (0, 1]
};

static_assert(sizeof(ConductorData) == 16, "ConductorData must be the same 16 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<ConductorData>, "ConductorData is written to the GPU as bytes");
#endif

}  // namespace materials
}  // namespace serenity
