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
//      the 8 pixels around p, in ring order from a place the frame draws
//      (RTXDI tries up to 9 pixels in a small neighborhood of the
//      reprojection, the further ones at random). Not from a fixed place:
//      then a pixel whose first try fails takes, frame after frame, the
//      neighbor on the same side, and its history drifts one way across
//      the image, which on a grazing floor measurably moved the image's
//      mean (docs/research/2026-10-11-restir-temporal.md); a start the
//      frame draws, the same for every pixel, turns that drift into a
//      walk with no side (its number the frame's, as the permutation's:
//      GDSA.3). A pixel passes when it
//      is inside the image and last frame's record q' there has a kind
//      not none and is similar to q (similar.metal.h, the geometric test
//      and the material test, as RTXDI's temporal resampling asks both):
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
// The frame's permutation numbers, three, the permutation's two and the
// ring's start: the Sampler's stream for pixel (0, 0) and the frame, at
// this pass's purpose for them (restir_di.metal.h, permutation_numbers),
// which every pixel draws alike, so they are the frame's, a function of
// its index alone (principle 2), with no value the CPU computes.
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
// pass's own stream (metal/sampler/sampler.metal.h), and the frame's three
// for the permutation and the ring's start.

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
// the ring_size around it. ReSTIR PT's temporal reuse searches the same
// way, by these (metal/passes/restir_pt/restir_pt.metal.h, step 5).
constant constexpr uint temporal_tries = 9u;
constant constexpr uint ring_size = temporal_tries - 1u;

// RTXDI's permutation: of each aligned permutation_range x
// permutation_range block (below).
constant constexpr uint permutation_range = 4u;

// The frame's order of step 3's tries: its permutation of each aligned
// 4 x 4 block (permutation_range), the two numbers in [0, 3] a pixel's x
// and y have their two low bits exclusive-or'ed with, (0, 0), the
// identity, with permutation off; and where in the ring around p the
// further tries start, in [0, ring_size - 1]. A type, not a bare uint2, so it cannot be passed for a pixel
// (I.24).
struct BlockPermutation {
    uint2 bits;
    uint ring_start;
};

// Step 3's first pixel: `reprojected` under `permutation`.
inline uint2 permuted_pixel(uint2 reprojected, BlockPermutation permutation) {
    return reprojected ^ permutation.bits;
}

// Step 3's further pixels, around the reprojected one, in their fixed
// order (ES.27's C array: a shader constant table, as the noise tables).
constant constexpr int2 temporal_ring[ring_size] = {int2(-1, -1), int2(0, -1), int2(1, -1), int2(-1, 0),
                                                    int2(1, 0),   int2(-1, 1), int2(0, 1),  int2(1, 1)};

// Try t > 0's pixel: the ring's (t - 1 + start)th place around `p`.
inline int2 ring_neighbor(int2 p, uint t, uint start) {
    return p + temporal_ring[(t - 1u + start) % ring_size];
}

// Where the ring's tries start, from `u` in [0, 1): each place alike.
inline uint draw_ring_start(float u) {
    return metal::min(uint(u * float(ring_size)), ring_size - 1u);
}

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
                             BlockPermutation permutation, serenity::contracts::ReuseSettings settings) {
    const float3 normal_here = decode_unit(pixel.here.bsdf.normal);
    const float depth_here = then.depth(reprojection.previous_position);
    const int2 reprojected = int2(reprojection.pixel);
    for (uint t = 0; t < temporal_tries; ++t) {
        const int2 x = t == 0u ? int2(permuted_pixel(reprojection.pixel, permutation))
                               : ring_neighbor(reprojected, t, permutation.ring_start);
        if (metal::any(x < int2(0)) || metal::any(x >= int2(arrays.size))) {
            continue;
        }
        const uint j = uint(x.y) * arrays.size.x + uint(x.x);
        const serenity::contracts::SurfaceRecord there = arrays.records[j];
        if (!similar(depth_here, normal_here, there, then.depth(to_float3(there.position)), settings) ||
            !similar_material(pixel.here.bsdf, there.bsdf)) {
            continue;
        }
        return Found{true, LastFrame{reprojection.previous_position, x - reprojected, there, arrays.finals[j]}};
    }
    return Found{false, LastFrame{}};
}

// Steps 4 to 6 for `pixel`, given what `last` frame left. `now` and `then`
// are this frame's and last frame's Frames (unbiased.metal.h states what a
// Frame answers). `numbers` is the pass's stream. Callers that found no
// temporal input (steps 1 to 3) do not call it.
template <typename Frame>
inline serenity::contracts::Reservoir merge_temporal(Frame now, Frame then, Merging pixel, LastFrame last,
                                                     serenity::contracts::ReuseSettings settings,
                                                     thread PathNumbers& numbers) {
    namespace c = serenity::contracts;
    // Step 4: the own reservoir, then last frame's, its count capped at H
    // times the own one's, its sample weighed at q as this frame lights it.
    const uint own_count = pixel.own.count & c::count_mask;
    const uint capped = metal::min(last.last_final.count & c::count_mask, settings.history * own_count);
    const c::LightPoint carried = last.last_final.sample;
    const float carried_here = c::holds_light(carried) ? now.target(pixel.here, carried) : 0.0f;
    Streaming reservoir = empty_reservoir();
    stream_reservoir(reservoir, MergeInput{pixel.own, own_count, pixel.own.target, from_own}, next_number(numbers));
    stream_reservoir(reservoir,
                     MergeInput{reused_from(last.last_final, last.offset, 1u), capped, carried_here, from_temporal},
                     next_number(numbers));

    // Step 5: pi and piSum by the graph's mode (unbiased.metal.h).
    if (reservoir.source == from_temporal) {
        const float pi = last.last_final.target;
        const c::Reservoir merged = finish(
            reservoir, Normalization{pi, float(own_count) * reservoir.target + float(capped) * pi});
        // Raytraced: the own domain's ray from q, which is the survivor's
        // test here and now (unbiased.metal.h).
        if (settings.bias_correction == c::BiasCorrection::raytraced && merged.weight > 0.0f) {
            c::Reservoir tested = merged;
            tested.tested = 0u;
            return now.visible(pixel.here, merged.sample) ? tested : hide(tested);
        }
        return merged;
    }
    const float there = c::holds_light(reservoir.sample)
                            ? domain_target(then, last.there, reservoir.sample, settings.bias_correction)
                            : 0.0f;
    return finish(reservoir,
                  Normalization{reservoir.target, float(own_count) * reservoir.target + float(capped) * there});
}

}  // namespace shaders
}  // namespace serenity
