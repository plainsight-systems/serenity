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
#include "metal/math/warp.metal.h"

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

// Contract 2's packed form (contracts/bsdf.h states both functions and
// what packing loses): Material's one definition of how a Bsdf is stored in
// ReSTIR PT's G-buffer and MNEE's requests and read back, by every pass
// that writes or reads one. A switch on the kind with no default, as above, so a kind
// added and not packed fails to compile.
//
// A half is rounded from a float to nearest, ties to even, by the shading
// language's conversion; a positive value that rounds to 0 is raised to
// half's least, 2^-24. Precondition: each value finite and at most 65504
// in magnitude, which a renderer that packs has checked of the scene
// (contracts/bsdf.h states both, and the tolerance). Cost: pack, one
// encode_unit and four conversions; unpack, one decode_unit and four
// widenings (GDSA.18: decoded where it is loaded, never stored decoded).
// Pure (F.8).

// `value` as a half's bits, a positive value never 0.
constant constexpr uint half_magnitude = 0x7fffu;  // a half's bits but its sign
constant constexpr uint half_least = 1u;           // the least positive half, 2^-24, a subnormal

inline uint pack_half(float value) {
    const ushort bits = as_type<ushort>(half(value));
    return value > 0.0f && (bits & half_magnitude) == 0u ? half_least : uint(bits);
}

inline float unpack_half(uint bits) {
    return float(as_type<half>(ushort(bits & serenity::contracts::packed_half_mask)));
}

// The scalar in color_b_scalar's high half: the one of alpha and ior the
// kind reads, 0 for a kind that reads neither (contracts/bsdf.h).
inline float packed_scalar(serenity::contracts::Bsdf bsdf) {
    switch (bsdf.kind) {
    case serenity::contracts::BsdfKind::none:
    case serenity::contracts::BsdfKind::lambert:
        return 0.0f;
    case serenity::contracts::BsdfKind::conductor:
        return bsdf.alpha;
    case serenity::contracts::BsdfKind::dielectric:
    case serenity::contracts::BsdfKind::coated:
        return bsdf.ior;
    }
    return 0.0f;
}

inline serenity::contracts::PackedBsdf pack_bsdf(serenity::contracts::Bsdf bsdf) {
    const bool escapes = bsdf.kind == serenity::contracts::BsdfKind::coated;
    return serenity::contracts::PackedBsdf{
        encode_unit(to_float3(bsdf.normal)),
        pack_half(bsdf.color.x) | (pack_half(bsdf.color.y) << serenity::contracts::packed_high_shift),
        pack_half(bsdf.color.z) | (pack_half(packed_scalar(bsdf)) << serenity::contracts::packed_high_shift),
        (escapes ? pack_half(bsdf.escape) : 0u) | (uint(bsdf.kind) << serenity::contracts::packed_kind_shift),
    };
}

using serenity::contracts::packed_kind;  // contracts/bsdf.h

inline serenity::contracts::Bsdf unpack_bsdf(serenity::contracts::PackedBsdf packed) {
    serenity::contracts::Bsdf bsdf{};
    bsdf.normal = to_packed(decode_unit(packed.normal));
    bsdf.kind = packed_kind(packed);
    bsdf.color = serenity::contracts::Float3{unpack_half(packed.color_rg),
                                             unpack_half(packed.color_rg >> serenity::contracts::packed_high_shift),
                                             unpack_half(packed.color_b_scalar)};
    const float scalar = unpack_half(packed.color_b_scalar >> serenity::contracts::packed_high_shift);
    switch (bsdf.kind) {
    case serenity::contracts::BsdfKind::none:
    case serenity::contracts::BsdfKind::lambert:
        break;
    case serenity::contracts::BsdfKind::conductor:
        bsdf.alpha = scalar;
        break;
    case serenity::contracts::BsdfKind::dielectric:
        bsdf.ior = scalar;
        break;
    case serenity::contracts::BsdfKind::coated:
        bsdf.ior = scalar;
        bsdf.escape = unpack_half(packed.escape_kind);
        break;
    }
    return bsdf;
}

}  // namespace shaders
}  // namespace serenity
