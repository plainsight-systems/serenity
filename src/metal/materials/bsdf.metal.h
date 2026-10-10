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
//
// "No sample" is one value everywhere: BsdfSample{}, its pdf 0 (contract
// 2), every other field 0.

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
        return serenity::contracts::BsdfSample{};  // scatters nothing: no sample
    case serenity::contracts::BsdfKind::lambert:
        return lambert_sample(bsdf, facing(bsdf, wo), u.yz);
    case serenity::contracts::BsdfKind::conductor:
        return conductor_sample(bsdf, facing(bsdf, wo), wo, u.yz);
    case serenity::contracts::BsdfKind::dielectric:
        return dielectric_sample(bsdf, to_float3(bsdf.normal), wo, u.x);
    case serenity::contracts::BsdfKind::coated:
        return coated_sample(bsdf, facing(bsdf, wo), wo, u);
    }
    return serenity::contracts::BsdfSample{};
}

// eta_t of a transmission sample drawn for wo: the index of refraction on
// wi's side over wo's, by which the sample's value carries radiance's
// 1 / eta_t^2 (dielectric.metal.h). 1 for a kind that does not transmit.
// pbrt-v4's BSDFSample::eta, here a question of the Bsdf and wo because
// contract 2's BsdfSample does not carry it; the path tracer's roulette
// undoes it (integrator/path.metal.h, step 7).
inline float bsdf_eta(serenity::contracts::Bsdf bsdf, float3 wo) {
    switch (bsdf.kind) {
    case serenity::contracts::BsdfKind::none:
    case serenity::contracts::BsdfKind::lambert:
    case serenity::contracts::BsdfKind::conductor:
    case serenity::contracts::BsdfKind::coated:
        return 1.0f;
    case serenity::contracts::BsdfKind::dielectric:
        return transmitted_eta(bsdf.ior, metal::dot(wo, to_float3(bsdf.normal)) > 0.0f);
    }
    return 1.0f;
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
