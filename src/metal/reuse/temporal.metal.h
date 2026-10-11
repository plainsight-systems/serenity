#pragma once

// Axis: Sample reuse (temporal reuse).
//
// ReSTIR DI's step 4 (metal/passes/restir_di/restir_di.metal.h): the
// reservoir this surface held last frame, found where the surface was then,
// merged into this frame's initial reservoir by the unbiased rule
// (reservoir.metal.h, unbiased.metal.h). A reservoir belongs to a surface,
// not a pixel (logical-overview.md, "Reservoirs are state keyed on
// surfaces"), so it is looked up where the surface was.
//
// Depends on contracts 5 and 14, the settings (core/passes/restir_di.h),
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
//      last frame's camera, is pixel x' of last frame's image. Outside the
//      image, or behind the camera: no temporal input. One pixel, rounded
//      down, as the paper takes. The caller's, which has the camera
//      (metal/passes/restir_di/restir_di.metal.h, FrameView::project).
//   3. Is it the same surface: last frame's record q' at x', its bsdf of
//      kind not none, similar to q (similar.metal.h): q's depth taken at its
//      previous position by last frame's camera, so a surface that moved is
//      compared where it was. Else no temporal input.
//   4. Stream: q's own reservoir r (count M_r), then last frame's final
//      reservoir r' at x', its count capped at H x M
//      (core/passes/restir_di.h): a sample kept for many frames is weighed
//      as at most H frames' candidates, which bounds how long an old choice
//      outlives the light that made it a good one. Each weighed by its
//      sample's target at q, as this frame lights it.
//   5. Z (unbiased.metal.h), for the survivor y:
//        y from r:  M_r, plus min(M_r', H M) if q' supports y as last frame
//                   saw it: its target by last frame's transforms and
//                   glows, and, where positive, one shadow ray from q'
//                   against last frame's structure;
//        y from r': min(M_r', H M), plus M_r if y is visible from q, one
//                   shadow ray from q against this frame's structure; if it
//                   is hidden, the output weighs y 0 and keeps its count.
//   6. Finish (reservoir.metal.h, step 4); the caller writes it where the
//      graph's next pass reads (metal/frame/frame_images.h): the working
//      reservoir, in place, when a spatial merge follows, which reads it at
//      neighbors; this frame's final reservoir otherwise.
//
// The previous surface is last frame's in every respect: its record from
// last frame's records, its lights at last frame's transforms and glows,
// its rays against last frame's structure, all of which the `then` Frame
// answers from. A surface seen through glass or in a mirror reprojects as
// if seen straight, which is wrong when the glass moves or the camera
// does; the similarity test then rejects it, or lets through a surface
// whose reservoir Z weighs by what it could have produced. Noise, never
// bias (logical-overview.md).
//
// Step 5's ray is one ray query whichever frame it asks of: each lane
// chooses its structure, its ray's origin and the light's position, last
// frame's or this one's, as data, and the SIMD group runs one query, not a
// branch for each frame, so lanes whose survivors came from different
// inputs do not serialize two queries (GPU.4). The composition's Frame
// makes that possible: `now` and `then` are the same type, differing in
// data.
//
// Cost, per pixel: reads q's record (80 bytes) and previous position (12),
// q's reservoir (32), q' (80) and r' (32), each once: 236 bytes, the two at
// x' gathered from wherever the surface was, which for a still or slow
// camera is near x, so neighbouring lanes read neighbouring addresses
// (GPU.2); writes 32. Three target evaluations (r's and r''s samples at q,
// and Z's one) and at most one shadow ray. Two numbers, the two choices of
// step 4, from the pass's own stream (metal/sampler/sampler.metal.h).

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

// What last frame left where the pixel's surface was: where the surface
// was, and last frame's record and final reservoir at the reprojected
// pixel (step 2, the caller's).
struct LastFrame {
    float3 previous_position;
    serenity::contracts::SurfaceRecord there;
    serenity::contracts::Reservoir last_final;
};

// Steps 3 to 6 for `pixel`, given what `last` frame left. `now` and `then`
// are this frame's and last frame's Frames (unbiased.metal.h states what a
// Frame answers). `numbers` is the pass's stream. Callers that found no
// temporal input (steps 1 and 2) do not call it.
template <typename Frame>
inline serenity::contracts::Reservoir merge_temporal(Frame now, Frame then, Merging pixel, LastFrame last,
                                                     serenity::passes::RestirDi settings,
                                                     thread PathNumbers& numbers);

}  // namespace shaders
}  // namespace serenity
