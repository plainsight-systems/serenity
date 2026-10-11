#pragma once

// Axis: Sample reuse (the merges' normalization, Z, in its two modes).
//
// Depends on contracts 4, 5 and 14, the settings (core/passes/restir_di.h)
// and this family's reservoir alone. It asks a frame's questions (the
// target at a surface, a shadow ray) of a Frame its caller passes (below),
// so it includes nothing of the Integrator, the lights or the acceleration
// structure (change-axes.md: families meet at contracts); the composition
// that passes one is metal/passes/restir_di/restir_di.metal.h.
//
// What a merge of reservoirs from other surfaces and other frames divides
// by (Bitterli et al. 2020, section 4.3 and Algorithm 6): finishing a merge
// at surface q with survivor y from input s, W = w_sum / (Z p̂_q(y))
// (reservoir.metal.h, step 4), where Z is not the merge's M but the sum of
// the counts M_i of the inputs counted for y:
//
//   the survivor's own input s   counted, M_s: it produced y;
//   any other input i            counted when its surface q_i could have
//                                produced y, as the mode below tests it.
//
// Why. Input r_i came from surface q_i, and a reservoir at q_i holds a
// sample with W_i > 0 only within its domain Omega_i, the samples q_i could
// have kept. Streamed into q's merge, a y outside Omega_i could never have
// come from r_i, yet dividing by the merge's whole M would count r_i's
// candidates as if it could, and q's estimate of y's light would be too
// small. Dividing by the candidates that could have produced y counts every
// sample by the domains that could have made it, and E[f(y) W] is then the
// integral of f over the union of the domains (section 4.3). The derivation
// treats the inputs as independent; spatiotemporal reuse correlates them,
// which ReSTIR's literature treats as costing noise, not the mean.
//
// Visibility reuse (restir_di.metal.h, step 3) puts visibility into what a
// reservoir holds, so a written reservoir's true domain is
//
//   Omega_i = { y : p̂_{q_i}(y) > 0 and y is visible from q_i },
//
// p̂_{q_i} the target at q_i's surface with its lights as its frame lit
// them, and "visible" a shadow ray from q_i's record (contracts/
// surface_record.h) to y's point, against its frame's structure. The two
// modes (core/passes/restir_di.h, bias_correction) differ in one thing:
// whether a merge tests the visibility half of that domain, with rays.
//
// basic: RTXDI's Basic, the default.
//
//   An input i other than the source is counted when p̂_{q_i}(y) > 0: one
//   target evaluation, at q_i's surface as its frame lit it, and no ray.
//   The pixel's own input, when it is not the source, is counted without
//   one: p̂_q(y) > 0, since the merge weighed y so to keep it. No merge
//   traces a ray; the survivor's light at q is decided by the shade's ray,
//   or by a recent test it reuses (restir_di.metal.h, step 6).
//
//   The bias it carries, stated. Z counts every input whose target at y is
//   positive, including those whose surface cannot see y and so could not
//   have produced it: Z is too large exactly where y is visible from q and
//   hidden from another input's surface, and W, so q's light from y, too
//   small. That is the lit side of a shadow's edge, where a neighbor within
//   the radius, or this surface last frame (a moving occluder or a moving
//   firefly), is in that light's shadow: those pixels come out darker than
//   the reference, by up to the share of the merge's count those inputs
//   carry, and nowhere else. Second, a merged reservoir may keep a sample
//   hidden from q with W > 0, which q's shade zeroes (its ray, or a test it
//   reuses), but which the next frame's merges and this frame's spatial
//   merge read as q's: a sample outside the domain they count q for, more
//   of the same darkening at edges. Production ships this (RTXDI's
//   default): the bias is confined to shadow edges and fades as the edge's
//   neighbors' samples agree, and the rays that remove it would be the
//   frame's largest added cost. The final visibility reuse adds its own,
//   stated where it is (restir_di.metal.h, step 6).
//
//   Cost per merge: no ray; one target evaluation for each input not the
//   source and not the pixel's own (the temporal merge's: at most 1; the
//   spatial merge's: at most k).
//
// raytraced: the paper's unbiased normalization, for checking ReSTIR
// against the reference, not for shipping.
//
//   An input i other than the source is counted when y is in Omega_i: its
//   target evaluation and, where positive, one shadow ray from q_i, against
//   q_i's frame's structure. The pixel's own input, when it is not the
//   source, is tested by one ray from q: y is visible from q, or the output
//   is hide()'s, W = 0 with its count kept. That ray decides the
//   survivor's visibility at q too, so every reservoir a merge writes keeps
//   the invariant visibility reuse opened: it holds a sample visible from
//   its own surface, or W = 0. Then a written reservoir's domain is exactly
//   Omega above, every Z test is exact for it, no merge streams a sample
//   its source could not have kept, and the shade traces no ray: the final
//   reservoir's sample is visible from q or weighs 0. (Without the own
//   ray, as RTXDI's Raytraced mode is written, Z counts M_q for a sample q
//   cannot see: harmless to q's own image, which its final ray zeroes, but
//   the merged reservoir then holds samples outside the domain a neighbor's
//   test grants it, a bias in that neighbor's image.)
//
//   Cost: k + 2 rays a pixel in all, one for visibility reuse, one for the
//   temporal merge, k for the spatial, and none for the shade
//   (restir_di.metal.h counts them); the same target evaluations as basic.
//   The temporal input's rays are traced against last frame's structure,
//   which a graph in this mode keeps a frame longer
//   (metal/acceleration/scene_acceleration.h).
//
//   The postcondition (I.7): E[f(y) W] equals the integral of f over the
//   samples visible from q with p̂_q > 0, for the f the shade evaluates, and
//   every ReSTIR graph's mean over frames is the reference's image, held by
//   the GPU test that it converges there (logical-overview.md, principle 3;
//   `make convergence`), to within the tolerance the light point's and the
//   record's quantization leave (contracts/light_sample.h, surface_record.h;
//   the tolerance the implementation measures). A test the path pass shares
//   no code with. Basic mode is held to the same reference by the same
//   test, with its bias stated above as what separates it, and measured.
//
// Both modes, the rest. The temporal input's domain is last frame's: its
// surface as last frame's records hold it, its lights at last frame's
// transforms and glows (metal/scene/shape_transforms.h, light_glows.h),
// and in raytraced mode its rays against last frame's structure. A firefly
// asleep last frame (glow 0) gave last frame's surface nothing, so a sample
// on it is outside that domain however bright it is now, and is counted
// so. Each mode keeps its reservoir's `tested` (contracts/reservoir.h) as
// its tests make it: in raytraced mode every survivor a merge writes was
// tested from q this frame, so its offset and age are 0; in basic mode a
// survivor keeps its own, the merge adding the offset it was reused across.
//
// The weights within Z, decided here: an input is counted by its whole
// count, 1/Z. RTXDI normalizes both its modes another way, by MIS-like
// weights, each input weighted by its target at y times its count (pi over
// piSum, the balance heuristic over the inputs), with the same evaluations
// and rays and less noise; here both modes divide by Z, so they differ in
// rays alone and the raytraced mode checks the code basic mode ships.
// Moving both to RTXDI's weights is a later change, measured by error x
// time (docs/research/2026-10-10-restir-di-layout.md).
//
// Memory: the temporal merge's test reads nothing more, last frame's
// record being in registers from its similarity test. The spatial merge's
// does: its mask keeps only which neighbors passed, and its walk for Z
// reads each passing neighbor's record and reservoir again, (32 + 24) k
// bytes a pixel, counted in spatial.metal.h.

#include <metal_stdlib>

#include "core/contracts/light_sample.h"
#include "core/contracts/reservoir.h"
#include "core/contracts/surface_record.h"
#include "core/passes/restir_di.h"
#include "metal/reuse/reservoir.metal.h"

namespace serenity {
namespace shaders {

// The pixel a merge finishes at (temporal.metal.h, spatial.metal.h): its
// surface record, q, and its own input reservoir, whose target is q's this
// frame (contracts/reservoir.h).
struct Merging {
    serenity::contracts::SurfaceRecord here;
    serenity::contracts::Reservoir own;
};

// A Frame: one frame's answers, which the merges ask and the composition
// gives (metal/passes/restir_di/restir_di.metal.h, FrameView), as that
// frame's lights, structure and camera have them:
//
//   float target(SurfaceRecord q, LightPoint y) const;
//       p̂_q(y), the Integrator's target (metal/integrator/target.metal.h);
//   bool visible(SurfaceRecord q, LightPoint y) const;
//       y unblocked from q, one shadow ray against that frame's structure;
//       asked only in raytraced mode;
//   float depth(float3 point) const;
//       the point's depth along that frame's camera's forward axis.
//
// Stated here, not as a concept: the shading language has none (I.9, T.10,
// as far as the language allows). A type that has them is passed by value;
// no merge names the arrays behind it.

// Whether `there`, the surface of an input that did not produce `y`, is
// counted for it in Z, as `frame`, that surface's own frame, saw it, by
// `mode` (above): its target at y positive and, in raytraced mode, y
// visible from it, the ray asked only where the target is positive.
template <typename Frame>
inline bool supports(Frame frame, serenity::contracts::SurfaceRecord there, serenity::contracts::LightPoint y,
                     serenity::passes::BiasCorrection mode);

// `reservoir` with its sample weighed 0 and its count kept: what a test
// that finds its sample hidden makes of it, in every pass that tests one
// (above; contracts/reservoir.h).
inline serenity::contracts::Reservoir hide(serenity::contracts::Reservoir reservoir);

}  // namespace shaders
}  // namespace serenity
