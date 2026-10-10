#pragma once

// Axis: Material (the BSDF contract's shader side, contracts/bsdf.h).
//
// The four questions of contract 2, asked of any resolved Bsdf, answered by
// its kind: the one place a shader dispatches on BsdfKind, so an estimator
// names none. Each switch has no default, so a kind added to BsdfKind and
// not here fails to compile (-Werror).
//
// Reflecting kinds are two-sided: they reflect on whichever side wo is, the
// shading normal turned to face it. Glass decides entering from leaving by
// the unturned normal.

#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "metal/device/layout.metal.h"
#include "metal/materials/coated.metal.h"
#include "metal/materials/conductor.metal.h"
#include "metal/materials/dielectric.metal.h"
#include "metal/materials/rough.metal.h"

namespace serenity {
namespace shaders {

inline float3 facing(serenity::contracts::Bsdf bsdf, float3 wo) {
    const float3 n = to_float3(bsdf.normal);
    return metal::dot(n, wo) < 0.0f ? -n : n;
}

inline float3 bsdf_evaluate(serenity::contracts::Bsdf bsdf, float3 wo, float3 wi) {
    switch (bsdf.kind) {
    case serenity::contracts::BsdfKind::none:
        return float3(0.0f);
    case serenity::contracts::BsdfKind::lambert:
        return lambert_evaluate(bsdf, facing(bsdf, wo), wi);
    case serenity::contracts::BsdfKind::conductor:
        return conductor_evaluate(bsdf, facing(bsdf, wo), wo, wi);
    case serenity::contracts::BsdfKind::dielectric:
        return float3(0.0f);  // delta lobes only
    case serenity::contracts::BsdfKind::coated:
        return coated_evaluate(bsdf, facing(bsdf, wo), wo, wi);
    }
    return float3(0.0f);
}

inline float bsdf_pdf(serenity::contracts::Bsdf bsdf, float3 wo, float3 wi) {
    switch (bsdf.kind) {
    case serenity::contracts::BsdfKind::none:
        return 0.0f;
    case serenity::contracts::BsdfKind::lambert:
        return lambert_pdf(facing(bsdf, wo), wi);
    case serenity::contracts::BsdfKind::conductor:
        return conductor_pdf(bsdf, facing(bsdf, wo), wo, wi);
    case serenity::contracts::BsdfKind::dielectric:
        return 0.0f;  // delta lobes only
    case serenity::contracts::BsdfKind::coated:
        return coated_pdf(bsdf, facing(bsdf, wo), wo, wi);
    }
    return 0.0f;
}

// `u` is three numbers in [0, 1): u.x chooses a lobe, u.yz the direction.
inline serenity::contracts::BsdfSample bsdf_sample(serenity::contracts::Bsdf bsdf, float3 wo, float3 u) {
    switch (bsdf.kind) {
    case serenity::contracts::BsdfKind::none:
        break;
    case serenity::contracts::BsdfKind::lambert:
        return lambert_sample(bsdf, facing(bsdf, wo), u.yz);
    case serenity::contracts::BsdfKind::conductor:
        return conductor_sample(bsdf, facing(bsdf, wo), wo, u.yz);
    case serenity::contracts::BsdfKind::dielectric:
        return dielectric_sample(bsdf, to_float3(bsdf.normal), wo, u.x);
    case serenity::contracts::BsdfKind::coated:
        return coated_sample(bsdf, facing(bsdf, wo), wo, u);
    }
    serenity::contracts::BsdfSample none;
    none.direction = to_packed(float3(0.0f));
    none.pdf = 0.0f;
    none.value = to_packed(float3(0.0f));
    none.lobe = 0u;
    return none;
}

inline uint bsdf_lobes(serenity::contracts::Bsdf bsdf) {
    switch (bsdf.kind) {
    case serenity::contracts::BsdfKind::none:
        return 0u;
    case serenity::contracts::BsdfKind::lambert:
        return serenity::contracts::lobe_reflection | serenity::contracts::lobe_diffuse;
    case serenity::contracts::BsdfKind::conductor:
        return serenity::contracts::lobe_reflection | serenity::contracts::lobe_glossy;
    case serenity::contracts::BsdfKind::dielectric:
        return serenity::contracts::lobe_reflection | serenity::contracts::lobe_transmission |
               serenity::contracts::lobe_delta;
    case serenity::contracts::BsdfKind::coated:
        return serenity::contracts::lobe_reflection | serenity::contracts::lobe_diffuse |
               serenity::contracts::lobe_delta;
    }
    return 0u;
}

}  // namespace shaders
}  // namespace serenity
