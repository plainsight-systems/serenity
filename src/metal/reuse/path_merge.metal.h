#pragma once

// Axis: Sample reuse (ReSTIR PT's merges: spatial and temporal).
//
// The generalized RIS merge of a pixel's path reservoir with its paired
// neighbors' (Lin et al. 2022, GRIS; ReSTIR PT Enhanced §3, §6.3), on
// paths already shifted: it takes each neighbor's path as the Integrator's
// shift left it at this pixel, F and its Jacobian, and this pixel's path as
// shifted to each neighbor, and never calls the Integrator itself
// (tools/check_boundaries.sh rule 5; metal/integrator/shift.metal.h shifts,
// metal/passes/restir_pt/restir_pt.metal.h composes).
//
// Pairwise MIS (Bitterli 2022, §9.1.3; the 2022 code's Pairwise branch of
// SpatialReuse.cs.slang), for the canonical pixel c, whose reservoir holds
// x_c with confidence c_c, and k valid neighbors n, each holding x_n with
// confidence c_n; p̂_q(y) the luminance of F at pixel q, and J the shift's
// Jacobian:
//
//   neighbor n's path, shifted here as y_n = T(x_n):
//     m_n = p̂_n(x_n) c_n / (p̂_n(x_n) c_n + p̂_c(y_n) J_{n->c} c_c / k)
//   the canonical path, weighed against every neighbor it was shifted to:
//     m_c = 1 + sum over n of [1 - p̂_n(T(x_c)) J_{c->n} c_n
//                                  / (p̂_n(T(x_c)) J_{c->n} c_n + p̂_c(x_c) c_c /
//                                  k)]
//
// The weights sum to k + 1 over the inputs, so each is divided by k + 1
// where it is used (the 2022 code divides W once at the end). The estimate
// for shading is the vector-valued sum (Enhanced §6.3):
//
//   (1 / (k + 1)) [ m_c F_c W_c + sum over n of m_n F(y_n) J_{n->c} W_n ]
//
// the weights' own resampling distribution's average, rather than one path
// kept and its F times W, which shades every input's color, not only the
// kept one's: the color noise ReSTIR's scalar target leaves (it resamples by
// luminance), averaged over the neighbors at no cost, since F(y_n) is
// computed for the merge anyway. The path kept, for the next frame's
// temporal merge, is drawn by the same weights' scalar form, w_i = m_i
// p̂_c(y_i) J_i W_i / (k + 1), through the one softmax pick
// (metal/reuse/reservoir.metal.h, stream_pick); its W = sum w_i / p̂_c(Y),
// its confidence the inputs' sum, so the kept path's F W has the vector
// sum's luminance.
//
// The temporal merge (Lin et al. 2022, TemporalReuse.cs.slang's Talbot
// branch; ReSTIR PT Enhanced §5, §7): two inputs, the pixel's own reservoir
// c (this frame's tree, confidence c_c) and last frame's final reservoir t
// at where the pixel's surface was (confidence c_t, capped below), each
// weighed by Talbot MIS, the generalized balance heuristic over the two
// domains:
//
//   m_c = c_c p̂_c(x_c) / (c_c p̂_c(x_c) + c_t p̂_t(T(x_c)) J_{c->t})
//   m_t = c_t p̂_t(x_t) / (c_t p̂_t(x_t) + c_c p̂_c(T(x_t)) J_{t->c})
//
// p̂_t last frame's target at its pixel, as last frame lit it (the shift
// of x_c there is evaluated in last frame's scene: metal/passes/restir_pt/
// restir_pt.metal.h). m_c + m_t = 1 for every path both domains can hold,
// which is what keeps the merge unbiased: two shifts a pixel, where pairwise
// MIS would need the same two. The cap: c_t = min(M_t, c_cap c_c), c_cap
// lowered from confidence_cap toward cap_floor by the duplication map at
// t's pixel (metal/reuse/duplication.metal.h; Enhanced §5: c_cap =
// lerp(c_cap^Default, c_cap^min, D^alpha)), which trades the copies'
// correlation for a bias where they are many, as Enhanced states; with no
// duplication map, c_cap is confidence_cap. A confidence weight is a count
// of trees, kept in contract 15's 8 bits rounded to the nearest
// (with_count()); the merge weighs with the unrounded one.
//
// k counts the neighbors whose surface was similar enough to try (the
// merge's own test, before any shift), not the shifts that succeeded: a
// failed shift is a neighbor whose path does not exist at this pixel, and
// its weights say so (I.1; metal/integrator/shift.metal.h, Shifted). Pure
// (F.8); every weight is defined where its terms are 0 (a term of 0 over 0
// is taken as 0), so no NaN reaches the image.

#include <metal_stdlib>

#include "core/contracts/path_reservoir.h"

namespace serenity {
namespace shaders {
namespace path_merge {

// m_n (above): `own` p̂_n(x_n) and `own_count` c_n, the neighbor's;
// `shifted` p̂_c(y_n) J_{n->c}; `canonical_count` c_c; k neighbors.
inline float neighbor_weight(float own, float own_count, float shifted,
                             float canonical_count, uint k) {
  const float a = own * own_count;
  const float sum = a + shifted * canonical_count / float(k);
  return sum > 0.0f ? a / sum : 0.0f;
}

// One neighbor's term of m_c (above): `canonical` p̂_c(x_c) and
// `canonical_count` c_c; `shifted` p̂_n(T(x_c)) J_{c->n}; `neighbor_count`
// c_n; k neighbors.
inline float canonical_share(float canonical, float canonical_count,
                             float shifted, float neighbor_count, uint k) {
  const float b = shifted * neighbor_count;
  const float sum = b + canonical * canonical_count / float(k);
  return 1.0f - (sum > 0.0f ? b / sum : 0.0f);
}

// m_c or m_t (above): `own` c_i p̂_i(x_i), `other` c_j p̂_j(T(x_i)) J.
inline float talbot_weight(float own, float other) {
  const float sum = own + other;
  return sum > 0.0f ? own / sum : 0.0f;
}

// c_cap (above): `cap` confidence_cap, lowered toward `floor` by the
// duplication score `duplication`, D in [0, 1], at sensitivity `exponent`;
// an exponent of 0 (no duplication map in the graph) leaves it.
inline float duplication_cap(float cap, float floor, float exponent,
                             float duplication) {
  return exponent > 0.0f
             ? metal::mix(cap, floor,
                          metal::pow(metal::saturate(duplication), exponent))
             : cap;
}

// c_t (above): last frame's count, at most `cap` times the own count.
inline float temporal_confidence(float last_count, float own_count, float cap) {
  return metal::min(last_count, cap * own_count);
}

// `flags` with its count M set to `count` rounded, held to its 8 bits.
inline uint with_count(uint flags, float count) {
  const uint m = metal::min(uint(metal::round(metal::max(count, 0.0f))),
                            serenity::contracts::path_max_count);
  return (flags & ~serenity::contracts::path_count_mask) | m;
}

// `source`'s path as a shift left it at another pixel (metal/integrator/
// shift.metal.h): F the shifted one; the cached Jacobian denominator the
// shifted path's own, the source's times the shift's Jacobian, which is
// supplemental eq. 4's numerator (contract 15, `jacobian`); every other
// field the source's (its numbers' stream, its reconnection vertex and
// suffix, its lobes), which is what the shifted path is made of.
inline serenity::contracts::PathReservoir shifted_reservoir(
    serenity::contracts::PathReservoir source, float3 integrand,
    float jacobian) {
  serenity::contracts::PathReservoir shifted = source;
  shifted.integrand =
      serenity::contracts::Float3{integrand.x, integrand.y, integrand.z};
  shifted.jacobian = source.jacobian * jacobian;
  return shifted;
}

}  // namespace path_merge
}  // namespace shaders
}  // namespace serenity
