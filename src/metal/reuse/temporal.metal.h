#pragma once

// Axis: Sample reuse (temporal reuse).
//
// ReSTIR DI's step 4 (metal/passes/restir_di/restir_di.metal.h): the
// reservoir this surface held last frame, found where the surface was then,
// merged into this frame's initial reservoir by the rule of the graph's
// mode (reservoir.metal.h, unbiased.metal.h). A reservoir belongs to a
// surface, not a pixel (logical-overview.md, "Reservoirs are state keyed on
// surfaces"), so it is looked up where the surface was.
//
// Depends on contracts 5 and 14, the settings (contracts/reuse_settings.h),
// this family's other headers, and the Sampler's numbers (the shared thing
// every stage draws from: change-axes.md). This frame's and last frame's
// answers come from the Frames its caller passes (unbiased.metal.h), so it
// includes nothing of the Integrator, the Camera, the lights or the
// acceleration structure.
//
// The steps, for pixel x with record q (contracts/surface_record.h):
//
//   1. History: if the reuse constants say this frame starts the history
//      over (contracts/reuse_constants.h, previous = 0), the output is the
//      initial reservoir as it is, and nothing below runs. The caller's.
//   2. Where it was: q's previous position (the initial pass wrote it: the
//      hit's point on its shape, placed by last frame's transform), through
//      last frame's camera, is a point of last frame's image, rounded to
//      its nearest pixel p (RTXDI's rounding). Outside the image, or
//      behind the camera: no temporal input. The caller's, which has the
//      camera (metal/passes/restir_di/restir_di.metal.h,
//      FrameView::project).
//   3. Is it the same surface. The first pixel x' tried is p, or, with
//      permutation on (contracts/reuse_settings.h), p's image under the
//      frame's permutation: p's coordinates with their two low bits each
//      exclusive-or'ed with two numbers in [0, 3] the frame draws, a
//      permutation of each aligned 4 x 4 block, the same for every pixel of
//      the frame (step 2's numbers, below), so no two pixels of a block
//      read the same reservoir. Then, while none has passed, up to 8 more:
//      the 8 pixels around p, in a fixed order (RTXDI tries up to 9 pixels
//      in a small neighborhood of the reprojection, the further ones at
//      random; a fixed ring here draws no numbers). A pixel passes when it
//      is inside the image and last frame's record q' there has a kind
//      not none and is similar to q (similar.metal.h, the geometric test):
//      q's depth taken at its previous position by last frame's camera, so
//      a surface that moved is compared where it was. The first that
//      passes is the input's pixel x'; none: no temporal input
//      (find_last_frame()).
//   4. Stream: q's own reservoir r (count M_r, its stored target, which is
//      q's this frame), then last frame's final reservoir r' at x', its
//      count capped at C = min(M_r', H M_r), H times the own reservoir's
//      count (RTXDI's history cap; contracts/reuse_settings.h): a sample kept
//      for many frames is weighed as at most H frames' candidates, which
//      bounds how long an old choice outlives the light that made it a
//      good one; weighed by its sample's target at q as this frame lights
//      it, and its test moved here, x' - p added to its offset and 1 to its
//      age (reservoir.metal.h, reused_from). The capped C weighs r' both
//      here and in step 5.
//   5. Normalize (unbiased.metal.h), for the survivor y, by the graph's
//      mode, d_{q'} the domain target at q' as last frame saw it (its
//      target by last frame's transforms and glows, and in raytraced mode,
//      where positive, times y's visibility from q', one shadow ray against
//      last frame's structure):
//        y from r:  pi = p̂_q(y), from streaming;
//                   piSum = M_r p̂_q(y) + C d_{q'}(y): one evaluation, and
//                   in raytraced mode at most one ray;
//        y from r': pi = r''s stored target, p̂_{q'}(y) as last frame lit
//                   it, visible from q' by the invariant in raytraced mode:
//                   no evaluation, no ray at q';
//                   piSum = M_r p̂_q(y) + C pi. In raytraced mode, one
//                   shadow ray from q against this frame's structure: if y
//                   is hidden the output is hide()'s, and the ray is the
//                   survivor's test, here and now.
//   6. Finish (reservoir.metal.h, step 4); the caller writes it where the
//      graph's next pass reads (metal/frame/frame_images.h): the working
//      reservoir, in place, when a spatial merge follows, which reads it at
//      neighbors; this frame's final reservoir otherwise.
//
// The frame's permutation numbers: the Sampler's stream for pixel (0, 0)
// and the frame, at this pass's purpose for them (restir_di.metal.h,
// permutation_numbers), which every pixel draws alike, so the permutation
// is the frame's, a function of its index alone (principle 2), with no
// value the CPU computes.
//
// The previous surface is last frame's in every respect: its record from
// last frame's records, its lights at last frame's transforms and glows,
// and, in raytraced mode, its rays against last frame's structure, all of
// which the `then` Frame answers from. A surface seen through glass or in a
// mirror reprojects as if seen straight, which is wrong when the glass
// moves or the camera does; the similarity test then rejects it, or lets
// through a surface whose reservoir the normalization weighs by how
// likely it was to produce the survivor: noise in raytraced mode, and in
// basic mode the edge-bounded bias unbiased.metal.h states.
//
// Step 5's ray, in raytraced mode, is one ray query whichever frame it asks
// of: each lane chooses its structure, its ray's origin and the light's
// position, last frame's or this one's, as data, and the SIMD group runs
// one query, not a branch for each frame, so lanes whose survivors came
// from different inputs do not serialize two queries (GPU.4). The
// composition's Frame makes that possible: `now` and `then` are the same
// type, differing in data. In basic mode there is no query.
//
// Cost, per pixel: reads q's record (32 bytes) and previous position (12),
// q's reservoir (24), q' (32) and r' (24): 124 bytes where the first pixel
// tried passes, which is every pixel the camera and the scene leave seen,
// and 32 more for each further pixel tried, up to 380 at a disocclusion
// where all nine are tried; the reads at x' gathered from wherever the
// surface was, which for a still or slow camera is near x, so neighbouring
// lanes read neighbouring addresses (GPU.2). Writes 24. Target evaluations:
// one (r''s sample at q) and at most one more (piSum's, at q'), r's own
// target and r''s at q' read from the reservoirs. Rays: none in basic mode;
// at most one in raytraced. Numbers: two, the choices of step 4, from the
// pass's own stream (metal/sampler/sampler.metal.h), and the frame's two for
// the permutation.

#include <metal_stdlib>

#include "core/contracts/reservoir.h"
#include "core/contracts/surface_record.h"
#include "core/contracts/reuse_settings.h"
#include "metal/reuse/reservoir.metal.h"
#include "metal/reuse/similar.metal.h"
#include "metal/reuse/unbiased.metal.h"
#include "metal/sampler/sampler.metal.h"

namespace serenity {
namespace shaders {

// Pixels tried for the temporal input (step 3): the reprojected one and
// the 8 around it.
constant constexpr uint temporal_tries = 9u;

// The frame's permutation of each aligned 4 x 4 block (step 3): the two
// numbers in [0, 3] a pixel's x and y have their two low bits
// exclusive-or'ed with; (0, 0), the identity, with permutation off. A type,
// not a bare uint2, so it cannot be passed for a pixel (I.24).
struct BlockPermutation {
    uint2 bits;
};

// Step 3's first pixel: `reprojected` under `permutation`.
inline uint2 permuted_pixel(uint2 reprojected, BlockPermutation permutation);

// Step 2's result, the caller's: where q was last frame, and the pixel p
// of last frame's image that point rounds to.
struct Reprojection {
    float3 previous_position;
    uint2 pixel;
};

// Where step 3 reads: last frame's surface records and final reservoirs,
// one a pixel in row-major order, of an image of `size` pixels.
struct LastArrays {
    const device serenity::contracts::SurfaceRecord* records;
    const device serenity::contracts::Reservoir* finals;
    uint2 size;
};

// What last frame left where the pixel's surface was: where the surface
// was, the offset x' - p of the pixel step 3 found, and last frame's record
// and final reservoir there.
struct LastFrame {
    float3 previous_position;
    int2 offset;
    serenity::contracts::SurfaceRecord there;
    serenity::contracts::Reservoir last_final;
};

// Step 3's answer: whether a pixel passed, and if one did, what it left.
struct Found {
    bool found;
    LastFrame last;  // read only where found
};

// Step 3 for `pixel`, from `reprojection`, reading `arrays`; `then` last
// frame's Frame, for the depth at q's previous position; `permutation` the
// frame's.
template <typename Frame>
inline Found find_last_frame(Frame then, Merging pixel, Reprojection reprojection, LastArrays arrays,
                             BlockPermutation permutation, serenity::contracts::ReuseSettings settings);

// Steps 4 to 6 for `pixel`, given what `last` frame left. `now` and `then`
// are this frame's and last frame's Frames (unbiased.metal.h states what a
// Frame answers). `numbers` is the pass's stream. Callers that found no
// temporal input (steps 1 to 3) do not call it.
template <typename Frame>
inline serenity::contracts::Reservoir merge_temporal(Frame now, Frame then, Merging pixel, LastFrame last,
                                                     serenity::contracts::ReuseSettings settings,
                                                     thread PathNumbers& numbers);

}  // namespace shaders
}  // namespace serenity
