#pragma once

// Axis: Integrator (ReSTIR PT's reconnection criteria).
//
// Where ReSTIR PT's hybrid shift may stop replaying a path's numbers and
// reconnect to the path's own vertex instead (metal/integrator/
// shift.metal.h): the first pair of vertices (x_{k-1}, x_k), k >= 2, that
// passes ReSTIR PT Enhanced's criteria (Lin, Kettunen and Wyman 2026, §4):
//
//   - the roughness of the lobe sampled at x_{k-1} is at least alpha_min
//     (the single-vertex test, §4.2): a diffuse lobe's is 1, a glossy
//     lobe's its material's roughness (a conductor's sqrt(GGX alpha), as
//     the 2022 code compares a material's roughness), a delta lobe's 0;
//   - both footprints are at least c / 100 of the primary ray's (the dual
//     footprint test, eq. 5):
//
//       min((p_{k-1}(w_{k-1}) G(x_{k-1} -> x_k))^-1,
//           (p_k(w_k) G(x_k -> x_{k-1}))^-1)
//         >= (c / 100) ||x0 - x1||^2 / (|<n_x1, x1 -> x0>| / (4 pi))
//
//     the first the area x_k stands for, drawn from x_{k-1} (the ray
//     footprint), the second the area x_{k-1} stands for, reached from x_k
//     backward (the inverse footprint); G(x -> y) = |cos at y| / ||x - y||^2.
//     The inverse test is skipped where x_k's density does not change with
//     x_{k-1} (Enhanced footnote 6): a diffuse lobe sampled at x_k, a
//     direction NEE drew at x_k, and an emitter x_k. A delta lobe's
//     footprint is 0: no reconnection to or from one.
//
// Densities are the marginal directional ones (Enhanced supplemental §3),
// which for every kind here is the density of its one lobe that is not
// delta (contract 2: bsdf_pdf()). The test scales with the primary ray's
// footprint, so it needs no tuning to a scene's size: the 2022 code's fixed
// 0.1 m would refuse nearly every pair among the marbles, 1.2 to 2.5 cm
// across (docs/research/2026-10-11-restir-pt-layout.md, "The scale
// problem"). tests/gpu/restir_pt_test.cpp holds a pair a centimeter apart
// accepted under a near camera and refused under one ten times farther.
//
// Pure functions of two vertices and their lobes (F.8). An infinite density
// or threshold is never formed: a delta lobe's footprint is 0 by its bit, a
// skipped test's is the largest float, and a grazing primary hit's threshold
// the largest float too, so the comparisons hold under the shaders' fast
// math, which assumes no infinities.

#include <metal_stdlib>

#include "core/contracts/bsdf.h"

namespace serenity {
namespace shaders {
namespace reconnection {

constant constexpr float largest = metal::numeric_limits<float>::max();

// The floors a footprint divides by, so a surface seen edge on gives a
// large footprint, not an infinity or a NaN: a cosine's, and the geometry
// term's (|cos| / d^2) where reconnect() writes a forward footprint out
// (shift.metal.h).
constant constexpr float least_cosine = 1.0e-20f;
constant constexpr float least_geometry = 1.0e-30f;

// The right-hand side of eq. 5: (c / 100) times the primary ray's footprint,
// from the camera's point x0 and the primary hit x1 with its normal n1. The
// largest float for a primary hit seen edge on, whose footprint has no
// bound.
inline float footprint_threshold(float c, float3 x0, float3 x1, float3 n1) {
    const float3 to_camera = x0 - x1;
    const float d2 = metal::dot(to_camera, to_camera);
    const float cosine = metal::abs(metal::dot(n1, to_camera)) * metal::rsqrt(d2);
    const float spread = d2 * (4.0f * M_PI_F) / metal::max(cosine, least_cosine);
    return cosine > 0.0f ? (c / 100.0f) * spread : largest;
}

// The roughness of `lobe` (contract 2's bits) sampled at a surface of `bsdf`
// (above). A lobe of 0, a direction NEE drew, is taken as the surface's own
// lobe that is not delta.
inline float lobe_roughness(uint lobe, serenity::contracts::Bsdf bsdf) {
    if ((lobe & serenity::contracts::lobe_delta) != 0u) {
        return 0.0f;
    }
    if ((lobe & serenity::contracts::lobe_glossy) != 0u ||
        (lobe == 0u && bsdf.kind == serenity::contracts::BsdfKind::conductor)) {
        return metal::sqrt(bsdf.alpha);
    }
    return 1.0f;  // diffuse
}

// The footprint of the vertex `to`, normal `to_normal`, reached from `from`
// along a direction drawn with solid-angle density `density` (> 0): the
// area it stands for, ||from - to||^2 / (density |cos at to|). 0 for a
// direction a delta lobe drew.
inline float footprint(bool delta, float density, float3 from, float3 to, float3 to_normal) {
    const float3 d = to - from;
    const float d2 = metal::dot(d, d);
    const float cosine = metal::abs(metal::dot(to_normal, d)) * metal::rsqrt(d2);
    return delta || !(density > 0.0f) ? 0.0f : d2 / (density * metal::max(cosine, least_cosine));
}

// A pair's three quantities, by name (I.24).
struct Pair {
    float roughness_before;  // of the lobe sampled at x_{k-1}
    float forward;           // the ray footprint of x_k
    float inverse;           // the inverse footprint of x_{k-1}; `largest` where skipped
};

// Whether the pair passes, against the primary footprint's `threshold`
// (footprint_threshold()) and alpha_min.
inline bool connectable(Pair pair, float threshold, float min_roughness) {
    return pair.roughness_before >= min_roughness && metal::min(pair.forward, pair.inverse) >= threshold;
}

// Whether x_k's inverse test is skipped for the lobe sampled there
// (above): a diffuse one, or NEE's (lobe 0).
inline bool inverse_skipped(uint lobe) {
    return lobe == 0u || (lobe & serenity::contracts::lobe_diffuse) != 0u;
}

}  // namespace reconnection
}  // namespace shaders
}  // namespace serenity
