#pragma once

// Axis: Sample reuse (ReSTIR PT's spatial merge).
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
//                                  / (p̂_n(T(x_c)) J_{c->n} c_n + p̂_c(x_c) c_c / k)]
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
// computed for the merge anyway. Phase 1 shades with it alone; the path kept
// for the next frame, by the same weights' scalar form through the one
// softmax pick (metal/reuse/reservoir.metal.h, stream_pick), is what
// temporal reuse (Phase 2) reads, and is drawn when that pass is built.
//
// k counts the neighbors whose surface was similar enough to try (the
// merge's own test, before any shift), not the shifts that succeeded: a
// failed shift is a neighbor whose path does not exist at this pixel, and
// its weights say so (I.1; metal/integrator/shift.metal.h, Shifted). Pure
// (F.8); every weight is defined where its terms are 0 (a term of 0 over 0
// is taken as 0), so no NaN reaches the image.

#include <metal_stdlib>

namespace serenity {
namespace shaders {
namespace path_merge {

// m_n (above): `own` p̂_n(x_n) and `own_count` c_n, the neighbor's;
// `shifted` p̂_c(y_n) J_{n->c}; `canonical_count` c_c; k neighbors.
inline float neighbor_weight(float own, float own_count, float shifted, float canonical_count, uint k) {
    const float a = own * own_count;
    const float sum = a + shifted * canonical_count / float(k);
    return sum > 0.0f ? a / sum : 0.0f;
}

// One neighbor's term of m_c (above): `canonical` p̂_c(x_c) and
// `canonical_count` c_c; `shifted` p̂_n(T(x_c)) J_{c->n}; `neighbor_count`
// c_n; k neighbors.
inline float canonical_share(float canonical, float canonical_count, float shifted, float neighbor_count, uint k) {
    const float b = shifted * neighbor_count;
    const float sum = b + canonical * canonical_count / float(k);
    return 1.0f - (sum > 0.0f ? b / sum : 0.0f);
}

}  // namespace path_merge
}  // namespace shaders
}  // namespace serenity
