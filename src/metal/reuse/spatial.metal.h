#pragma once

// Axis: Sample reuse (spatial reuse).
//
// ReSTIR DI's step 5 (metal/passes/restir_di/restir_di.metal.h): the
// reservoirs of k neighbors within a radius merged into this pixel's, by
// the rule of the graph's mode (reservoir.metal.h, unbiased.metal.h). The
// only step that reads other pixels of this frame (logical-overview.md,
// principle 5), and so a pass of its own: every pixel's input reservoir is
// written, by the pass before, before any neighbor reads it. RTXDI ships
// k = 1 within 32 pixels, with its material test (core/passes/
// restir_di.h).
//
// Depends on contracts 5 and 14, the settings (core/passes/restir_di.h),
// this family's other headers, and the Sampler's numbers (the shared thing
// every stage draws from: change-axes.md). This frame's answers come from
// the Frame its caller passes (unbiased.metal.h), so it includes nothing of
// the Integrator, the Camera, the lights or the acceleration structure.
//
// The steps, for pixel x with record q (contracts/surface_record.h):
//
//   1. Own input: the reservoir the pass before wrote at x (the temporal
//      merge's, or the initial pass's in a graph without one), count M_q,
//      its stored target q's this frame.
//   2. Neighbors: for n = 0 .. k - 1, a point drawn uniformly in the disc
//      of the graph's radius about x's center (two numbers, the square
//      root's warp for the radius), rounded to pixel x_n. x_n = x, or
//      outside the image: not an input (RTXDI draws from a table of offsets
//      and clamps to the image; a draw and a refusal here need no table).
//      Else its record q_n, whose kind must be not none and which must be
//      similar to q (similar.metal.h: the geometric test, by the Frame's
//      depths, and the material test); if so, bit n of a 32-bit mask is
//      set (k <= 32, core/passes/restir_di.h) and its reservoir r_n is an
//      input, its test moved here, x_n - x added to its offset
//      (reservoir.metal.h, reused_from). The mask keeps which neighbors
//      passed, and nothing else.
//   3. Stream: q's own reservoir, weighed by its stored target, then each
//      input r_n in n's order, weighed by its sample's target at q (the
//      Frame's target()).
//   4. Normalize (unbiased.metal.h), for the survivor y from source s, by
//      the graph's mode: piSum = M_q p̂_q(y) + the sum over the neighbors
//      n in the mask of M_n d_n(y), each M_n the neighbor's count as
//      stored (uncapped, as RTXDI's), and pi the source's term's domain
//      target:
//        own:       d_q(y) = p̂_q(y), from streaming. If s is own, pi is
//                   it. Else, in raytraced mode, one shadow ray from q: if
//                   y is hidden the output is hide()'s, the ray being the
//                   survivor's test here and now;
//        neighbor n in the mask, n != s: d_n(y), its target at q_n, and in
//                   raytraced mode, where positive, times y's visibility,
//                   one shadow ray from q_n;
//        neighbor s: d_s(y) = its reservoir's stored target, p̂_{q_s}(y)
//                   this frame, visible from q_s by the invariant in
//                   raytraced mode: no evaluation, no ray; pi is it.
//      The neighbors are drawn again from the same numbers for this walk
//      (the numbers are a function of their position in the stream,
//      metal/sampler/sampler.metal.h), and their records and counts read
//      again, rather than kept: at k = 1 that is one record from a line the
//      first walk has just read, and at k up to 32, which the settings
//      allow, keeping them would hold 32 k bytes a thread across the ray
//      queries of raytraced mode (GPU.3).
//   5. Finish, and write this frame's final reservoir.
//
// All neighbors are this frame's: their surfaces, lights and structure are
// the frame's own, and the rays are traced against it.
//
// In raytraced mode, step 4 is one loop of k iterations in every lane:
// iteration n tests neighbor n, except at the survivor's source, where it
// tests q's own visibility instead; the same visibility function, a
// different origin. So lanes whose survivors came from different inputs
// run the same loop and the same ray query in step, not a branch for the
// own test beside the neighbors' (GPU.4); a lane whose neighbor n is not in
// the mask, or whose target there is 0, skips its query. In basic mode the
// loop is the same with no query.
//
// Cost, per pixel, at k neighbors: reads q's record and reservoir (32 + 24
// = 56 bytes) and, per neighbor that passes, its record and reservoir (56)
// in step 2 and again in step 4: up to (1 + 2k) x 56 bytes, 168 at k = 1,
// of which step 4's k x 56 come from cache where step 2's lines are still
// there, which is what measurement will tell; writes 24. The neighbors'
// reads are gathers within the radius, not the coalesced reads of a
// pixel's own (SIMD.5, GPU.2): neighbouring lanes draw unrelated offsets,
// so a SIMD group's 32 reads of one neighbor touch up to 32 lines. Kept in
// row-major order in the first version; a tiled order (GDSA.15's
// space-filling curve) is the follow-up if the counters show the spatial
// pass bound by those lines. At most 2k target evaluations (k streamed, k
// for piSum; q's own and the source neighbor's read from their
// reservoirs). Shadow rays: none in basic mode;
// at most k in raytraced, one per input that did not produce y. Numbers:
// 2k for the offsets and 1 + k for the choices, from the pass's own stream.

#include <metal_stdlib>

#include "core/contracts/reservoir.h"
#include "core/contracts/surface_record.h"
#include "core/passes/restir_di.h"
#include "metal/reuse/reservoir.metal.h"
#include "metal/reuse/similar.metal.h"
#include "metal/reuse/unbiased.metal.h"
#include "metal/sampler/sampler.metal.h"

namespace serenity {
namespace shaders {

// Step 2's draw: a neighbor of the pixel at `at` in an image of `size`
// pixels, from two numbers `u` and the graph's radius; `found` false when
// it rounds to the pixel itself or falls outside the image.
struct Neighbor {
    bool found;
    uint2 pixel;
};
inline Neighbor neighbor_pixel(uint2 at, float2 u, float radius, uint2 size);

// Where the neighbors are read from: this frame's surface records and the
// reservoirs the pass before wrote, one a pixel in row-major order, of an
// image of `size` pixels.
struct NeighborArrays {
    const device serenity::contracts::SurfaceRecord* records;
    const device serenity::contracts::Reservoir* inputs;
    uint2 size;
};

// Steps 1 to 5 for `pixel`, at image coordinates `at`, its neighbors read
// from `arrays`; `now` this frame's Frame (unbiased.metal.h states what a
// Frame answers). `numbers` is the pass's stream.
template <typename Frame>
inline serenity::contracts::Reservoir merge_spatial(Frame now, Merging pixel, uint2 at, NeighborArrays arrays,
                                                    serenity::passes::RestirDi settings,
                                                    thread PathNumbers& numbers);

}  // namespace shaders
}  // namespace serenity
