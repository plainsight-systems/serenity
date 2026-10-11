#pragma once

// Axis: Sample reuse (spatial reuse).
//
// ReSTIR DI's step 5 (metal/passes/restir_di/restir_di.metal.h): the
// reservoirs of k neighbors within a radius merged into this pixel's, by
// the unbiased rule (reservoir.metal.h, unbiased.metal.h). The only step
// that reads other pixels of this frame (logical-overview.md, principle 5),
// and so a pass of its own: every pixel's input reservoir is written, by
// the pass before, before any neighbor reads it.
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
//      merge's, or the initial pass's in a graph without one), count M_q.
//   2. Neighbors: for n = 0 .. k - 1, a point drawn uniformly in the disc
//      of the graph's radius about x's center (two numbers, the square
//      root's warp for the radius), rounded to pixel x_n. x_n = x, or
//      outside the image: not an input. Else its record q_n, whose bsdf
//      must be of kind not none and which must be similar to q
//      (similar.metal.h), by the Frame's depths; if so, bit n of a 32-bit
//      mask is set (k <= 32, core/passes/restir_di.h) and its reservoir r_n
//      is an input. The mask keeps which neighbors passed, and nothing
//      else.
//   3. Stream: q's own reservoir, then each input r_n in n's order, each
//      weighed by its sample's target at q (the Frame's target()).
//   4. Z (unbiased.metal.h), for the survivor y from source s:
//        own:       M_q if s is own; else M_q if y is visible from q, one
//                   shadow ray from q; if hidden, the output weighs y 0 and
//                   keeps its count;
//        neighbor n in the mask, n != s: M_n if q_n supports y, its target
//                   and, where positive, one shadow ray from q_n;
//        neighbor s: M_s.
//      The neighbors are drawn again from the same numbers for this walk
//      (the numbers are a function of their position in the stream,
//      metal/sampler/sampler.metal.h), and their records and counts read
//      again, rather than kept: k records are 80 k bytes a thread, which
//      registers would hold only by giving up the threads that hide the ray
//      queries' latency (GPU.3).
//   5. Finish, and write this frame's final reservoir.
//
// All neighbors are this frame's: their surfaces, lights and structure are
// the frame's own, and the rays are traced against it.
//
// Step 4 is one loop of k iterations in every lane: iteration n tests
// neighbor n, except at the survivor's source, where it tests q's own
// visibility instead; the same visibility function, a different origin.
// So lanes whose survivors came from different inputs run the same loop
// and the same ray query in step, not a branch for the own test beside the
// neighbors' (GPU.4); a lane whose neighbor n is not in the mask, or whose
// target there is 0, skips its query.
//
// Cost, per pixel, at k neighbors: reads q's record and reservoir (80 + 32
// = 112 bytes) and, per neighbor that passes, its record and reservoir
// (112) twice, in steps 2 and 4: up to (1 + 2k) x 112 bytes, 1232 at k = 5,
// of which the second walk's k x 112 come from cache where the first
// walk's lines are still there, which is what measurement will tell; writes
// 32. The reads are gathers within the
// radius, not the coalesced reads of a pixel's own (SIMD.5, GPU.2):
// neighbouring lanes draw unrelated offsets, so a SIMD group's 32 reads of
// one neighbor touch up to 32 lines. Kept in row-major order in the first
// version; a tiled order (GDSA.15's space-filling curve) is the follow-up
// if the counters show the spatial pass bound by those lines. At most
// 1 + 2k target evaluations (the k streamed and k for Z, and q's own),
// and at most k shadow rays, one per input that did not produce y.
// Numbers: 2k for the offsets and 1 + k for the choices, from the pass's
// own stream.

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
