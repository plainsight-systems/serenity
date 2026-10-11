#pragma once

// Axis: Sample reuse (the unbiased normalization, Z).
//
// What makes a merge of reservoirs from other surfaces and other frames
// unbiased (Bitterli et al. 2020, section 4.3 and Algorithm 6): finishing a
// merge at surface q with survivor y, W = w_sum / (Z p̂_q(y))
// (reservoir.metal.h, step 4), where Z is not the merge's M but the sum of
// the counts M_i of only the inputs whose surface could have produced y.
//
// Why. Input r_i came from surface q_i, and a reservoir at q_i holds a
// sample with W_i > 0 only within its domain Omega_i, the samples q_i could
// have kept. Streamed into q's merge, a y outside Omega_i could never have
// come from r_i, yet dividing by the merge's whole M would count r_i's
// candidates as if it could: where some neighbor cannot see a light that q
// can, q's estimate of that light would be too small, and so too dark in
// exactly the shadows the reuse is for. Dividing by Z, the candidates that
// could have produced y, counts every sample by the domains that could
// have made it, and E[f(y) W] is then the integral of f over the union of
// the domains, unbiased for any f that is 0 outside it (section 4.3). The
// derivation treats the inputs as independent; spatiotemporal reuse makes
// them correlated, which ReSTIR's literature treats as costing noise, not
// the mean, and which the convergence test below is there to catch if it
// does not.
//
// The domains. Visibility reuse (metal/integrator/restir_di.metal.h, step
// 3) puts visibility into what a reservoir holds, so a domain is
//
//   Omega_i = { y : p̂_{q_i}(y) > 0 and y is visible from q_i },
//
// p̂_{q_i} the target at q_i's surface with its lights as its frame lit
// them, and "visible" a shadow ray from q_i's record (contracts/
// surface_record.h) to y's point, against its frame's structure. Testing
// y in Omega_i for an input that did not produce y is one target
// evaluation and, where it is positive, one shadow ray: the merges' rays.
// For an input that did produce y, nothing: y in Omega_i because W_i > 0,
// by the invariant below.
//
// The invariant every pass keeps: a reservoir any ReSTIR pass writes holds a
// sample visible from its own surface, or has W = 0 (M kept). Then a
// written reservoir's domain is exactly Omega above, the test of the
// paper's Algorithm 6 with visibility in the target is exact for it, and
// no merge streams a sample its source could not have kept.
//
// The pixel's own domain (decided here: the paper's Algorithm 6 tests
// every input alike, and RTXDI's temporal and spatial resampling count the
// pixel's own input by its target alone, with no ray). The
// pixel's own input, its candidates or its temporal merge's output, has
// domain Omega_q, so Z counts M_q exactly when p̂_q(y) > 0 and y is visible
// from q:
//
//   - y from the own input: in Omega_q by the invariant. Counted, no ray.
//   - y from another input: p̂_q(y) > 0, since the merge weighed it so to
//     choose it; y's visibility from q is traced, one ray from q's record.
//     Visible: M_q counted. Hidden: W = 0, M kept; Z's value then does not
//     matter for this pixel, since f(y) W is 0, and the zero is what keeps
//     the invariant for every later merge that reads this reservoir.
//
// So the ray from q tests the own domain and decides the survivor's
// visibility at once, and the shade traces none: by the invariant, the
// final reservoir's sample is visible from q or weighs 0
// (restir_di.metal.h, step 6). Without that ray, as RTXDI's temporal and
// spatial resampling are written, Z counts M_q for a sample q cannot see:
// harmless to q's own image, which its final shadow ray zeroes, but the
// merged reservoir, read by a neighbor's spatial merge, then holds samples
// outside the domain that neighbor's test grants it, and that neighbor's
// image is biased, slightly. Hence the ray, and the shade's ray it
// replaces: k + 2 rays a pixel in all, one for visibility reuse, one for
// the temporal merge, k for the spatial (restir_di.metal.h counts them).
//
// The temporal input's domain is last frame's: its surface as last frame's
// records hold it, its lights at last frame's transforms and glows, its
// rays against last frame's structure (metal/scene/shape_transforms.h,
// light_glows.h, metal/acceleration/scene_acceleration.h). A firefly asleep
// last frame (glow 0) gave last frame's surface nothing, so a sample on it
// is outside that domain however bright it is now, and is counted so.
//
// The postcondition (I.7): E[f(y) W] equals the integral of f over the
// samples visible from q with p̂_q > 0, for the f the shade evaluates. Held,
// with everything else ReSTIR does, by the GPU test that every ReSTIR graph
// converges to the reference's image (logical-overview.md, principle 3;
// `make convergence`), a test the path pass does not share code with.
//
// Cost, per merge at one pixel: one target evaluation for each input that
// did not produce y, and one shadow ray for each of those whose target is
// positive; the temporal merge's two inputs make one of each, the spatial
// merge's 1 + k make k. No memory beyond what the merge already read: each
// input's surface record and count were read to stream it, and the spatial
// merge keeps which neighbors passed its similarity test in a mask, not by
// reading them again (spatial.metal.h).

#include <metal_stdlib>

#include "core/contracts/light_sample.h"
#include "core/contracts/reservoir.h"
#include "core/contracts/surface_record.h"
#include "metal/reuse/reservoir.metal.h"

namespace serenity {
namespace shaders {

// The pixel a merge finishes at (temporal.metal.h, spatial.metal.h): its
// surface record, q, and its own input reservoir.
struct Merging {
    serenity::contracts::SurfaceRecord here;
    serenity::contracts::Reservoir own;
};

// Whether `there`, one input's surface as its own frame saw it, could have
// produced `y`: its target at y is positive and y is visible from it, the
// visibility asked only where the target is positive. `Surface` is the
// caller's (temporal.metal.h, spatial.metal.h), a type with
//
//   float target(LightPoint y) const;  p̂ there, by its frame's lights
//   bool visible(LightPoint y) const;  a shadow ray, by its frame's structure
//
// so this rule names neither frame nor any family's arrays. Stated here,
// not as a concept: the shading language has none (I.9, T.10, as far as
// the language allows).
template <typename Surface>
inline bool supports(Surface there, serenity::contracts::LightPoint y);

// `reservoir` with its sample weighed 0 and its count kept: what visibility
// reuse and a hidden survivor make of it (above).
inline serenity::contracts::Reservoir hide(serenity::contracts::Reservoir reservoir);

}  // namespace shaders
}  // namespace serenity
