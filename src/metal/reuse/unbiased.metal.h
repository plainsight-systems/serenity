#pragma once

// Axis: Sample reuse (the merges' normalization, in its two modes).
//
// Depends on contracts 4, 5 and 14, the settings (core/passes/restir_di.h)
// and this family's reservoir alone. It asks a frame's questions (the
// target at a surface, a shadow ray) of a Frame its caller passes (below),
// so it includes nothing of the Integrator, the lights or the acceleration
// structure (change-axes.md: families meet at contracts); the composition
// that passes one is metal/passes/restir_di/restir_di.metal.h.
//
// What a merge of reservoirs from other surfaces and other frames divides
// by: RTXDI's normalization, in both its modes, MIS-like weights over the
// inputs (the balance heuristic, each input's candidates weighed by how
// likely its surface was to have produced the survivor). A merge at
// surface q streams its inputs r_i, from surfaces q_i with counts M_i, each
// with weight p̂_q(y_i) W_i M_i (reservoir.metal.h, step 3), and keeps y
// from input s. Then, with
//
//   d_i(y)  input i's domain target at y: p̂_{q_i}(y), the target at q_i's
//           surface as q_i's own frame lit it, times, in raytraced mode,
//           y's visibility from q_i (below);
//   pi      d_s(y), the survivor's own input's;
//   piSum   the sum over every input i, the pixel's own included, of
//           M_i d_i(y);
//
// the merge finishes with W = w_sum pi / (p̂_q(y) piSum) (reservoir.metal.h,
// step 4); 0 where p̂_q(y) or piSum is 0, or nothing was kept.
//
// The counts are the ones the merge streamed: the pixel's own input's M as
// it is, the temporal input's capped at H times the own input's M (RTXDI's
// history cap, applied before the input is streamed, so the same capped
// count weighs it in w_sum and in piSum), a neighbor's as stored, uncapped.
// The pixel's own input, the merge's canonical one, is streamed first and
// weighed by its stored target, q's this frame (contracts/reservoir.h); its
// domain target is p̂_q(y), which the merge has from streaming y.
//
// Why it is unbiased where the domain targets are exact (the balance
// heuristic, Veach 1997, over the inputs, as Bitterli et al. 2020 offer it
// beside 1/Z and RTXDI implements it). Input i stands for M_i candidates
// whose density at y is proportional to d_i(y) when y is in its domain
// Omega_i, the samples q_i could have kept, and 0 outside it. Given y
// through input s, the merge weighs it by d_s(y) / piSum, and over every
// input whose domain holds y those weights, each times its count, sum to
// sum_i M_i d_i(y) / piSum = 1. Taking the expectation over which input is
// kept (each with probability its streamed weight over w_sum) and over
// each input's own sample (E[W_i g(y_i)] is the integral of g over Omega_i,
// W_i being its contribution weight), E[f(y) W] is then the integral of f
// over the union of the domains: each sample counted once in all, shared
// among the domains that could have made it in proportion to how likely
// each was to. Dividing instead by the counts of the inputs that
// could have produced y, 1/Z (Bitterli et al. 2020, Algorithm 6), is also
// unbiased and is set aside: it shares each sample equally among those
// domains, however unlikely one was to produce it, which costs noise where
// the inputs' targets differ, and RTXDI does not ship it (docs/research/
// 2026-10-10-restir-di-layout.md). The derivation treats the inputs as
// independent; spatiotemporal reuse correlates them, which ReSTIR's
// literature treats as costing noise, not the mean.
//
// Visibility reuse (restir_di.metal.h, step 3) puts visibility into what a
// reservoir holds, so a written reservoir's true domain is
//
//   Omega_i = { y : p̂_{q_i}(y) > 0 and y is visible from q_i },
//
// "visible" a shadow ray from q_i's record (contracts/surface_record.h) to
// y's point, against its frame's structure. The two modes
// (core/passes/restir_di.h, bias_correction) differ in one thing: whether
// d_i holds that visibility, at the price of the rays that test it. Both
// compute the same weights by the same code, so the raytraced mode checks
// what basic mode ships.
//
// basic: RTXDI's Basic, the default.
//
//   d_i(y) = p̂_{q_i}(y), unshadowed. No ray in a merge; the survivor's
//   light at q is decided by the shade's ray, or by a recent test it reuses
//   (restir_di.metal.h, step 6).
//
//   The bias it carries, stated. piSum takes M_i p̂_{q_i}(y) from every
//   input whose target at y is positive, including those whose surface
//   cannot see y and so could not have produced it: the weights over the
//   inputs that could sum to less than 1, by the share those inputs carry,
//   exactly where y is visible from q and hidden from another input's
//   surface, and q's light from y is too small there. That is the lit side
//   of a shadow's edge, where a neighbor within the radius, or this surface
//   last frame (a moving occluder or a moving firefly), is in that light's
//   shadow: those pixels come out darker than the reference, and nowhere
//   else. How much darker is the hidden inputs' share of piSum: at its
//   worst, a light just revealed to a surface that history, capped at
//   C = H M, still counts, shows M / (M + C) of its light the first frame
//   (1/21 at H = 20), recovering as the history follows it. A firefly
//   waking is not that case: asleep last frame (glow 0), it gave last
//   frame's surface nothing, so history does not count it. Second, a merged reservoir may keep a sample hidden from q with W
//   > 0, which q's shade zeroes (its ray, or a test it reuses), but which
//   the next frame's merges and this frame's spatial merge read as q's: a
//   sample outside the domain they weigh q for, more of the same darkening
//   at edges. Production ships this (RTXDI's default): the bias is confined
//   to shadow edges, and the rays that remove it would be the frame's
//   largest added cost. The final visibility reuse adds its own, stated
//   where it is (restir_di.metal.h, step 6).
//
// raytraced: the same weights with visibility in the domain targets, for
// checking ReSTIR against the reference, not for shipping.
//
//   d_i(y) = p̂_{q_i}(y) V_{q_i}(y), the visibility one shadow ray from q_i
//   against q_i's frame's structure, traced only where p̂_{q_i}(y) > 0. For
//   the survivor's own input s, V_{q_s}(y) = 1 with no ray, by the
//   invariant below (RTXDI traces it too, as its own reservoirs do not keep
//   that invariant). For the pixel's own input, when it is not the source,
//   one ray from q: y visible from q, and d_q(y) = p̂_q(y); or hidden, and
//   the output is hide()'s, W = 0 with its count kept, which is what the
//   weights would give f(y) at q, and keeps the invariant. That ray decides
//   the survivor's visibility at q too (RTXDI does not trace the canonical
//   input's: its own visibility enters its weights unshadowed), so every
//   reservoir a merge writes keeps the invariant visibility reuse opened:
//   it holds a sample visible from its own surface, or W = 0. Then a
//   written reservoir's domain is exactly Omega above, every d_i is exact
//   for it, no merge streams a sample its source could not have kept, and
//   the shade traces no ray: the final reservoir's sample is visible from q
//   or weighs 0. (Without the own ray, a merged reservoir would hold
//   samples outside the domain a neighbor's weights grant it, a bias in
//   that neighbor's image.)
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
// Cost per merge at one pixel, the same evaluations in both modes:
//
//   pi      no evaluation: the source's domain target is the target stored
//           in its reservoir, p̂_{q_s}(y) as its own frame lit it
//           (contracts/reservoir.h), or, for the pixel's own input, p̂_q(y)
//           from streaming; and no ray, by the invariant.
//   piSum   one evaluation for each input not the source and not the
//           pixel's own, and in raytraced mode one ray for each of those
//           whose target is positive, and one from q when the source is not
//           the pixel's own. The temporal merge: at most 1 evaluation, and
//           in raytraced mode at most 1 ray. The spatial merge: at most k
//           evaluations and k rays.
//   basic   no ray at all.
//
// The temporal input's domain is last frame's: its surface as last frame's
// records hold it, its lights at last frame's transforms and glows
// (metal/scene/shape_transforms.h, light_glows.h), and in raytraced mode
// its rays against last frame's structure. A firefly asleep last frame
// (glow 0) gave last frame's surface nothing, so a sample on it has d = 0
// there however bright it is now. Each mode keeps its reservoir's `tested`
// (contracts/reservoir.h) as its tests make it: in raytraced mode every
// survivor a merge writes was tested from q this frame, so its offset and
// age are 0; in basic mode a survivor keeps its own, the merge adding the
// offset it was reused across.
//
// Memory: the temporal merge's piSum reads nothing more, last frame's
// record being in registers from its similarity test. The spatial merge's
// does: its mask keeps only which neighbors passed, and its walk for piSum
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

// d_i(y) (above) for `there`, the surface of an input that did not produce
// `y`, as `frame`, that surface's own frame, saw it, by `mode`: its target
// at y, and in raytraced mode 0 where y is hidden from it, the ray asked
// only where the target is positive. Declared, and defined with the merges
// that call it (restir_temporal, restir_spatial; reservoir.metal.h,
// stream_reservoir).
template <typename Frame>
inline float domain_target(Frame frame, serenity::contracts::SurfaceRecord there, serenity::contracts::LightPoint y,
                           serenity::passes::BiasCorrection mode);

// `reservoir` with its sample weighed 0 and its count kept: what a test
// that finds its sample hidden makes of it, in every pass that tests one
// (above; contracts/reservoir.h). Its sample, target and test are kept as
// they are: the test is the one that found it hidden.
inline serenity::contracts::Reservoir hide(serenity::contracts::Reservoir reservoir) {
    reservoir.weight = 0.0f;
    return reservoir;
}

}  // namespace shaders
}  // namespace serenity
