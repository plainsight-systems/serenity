#pragma once

// Axis: Sample reuse (temporal reuse).
//
// ReSTIR DI's step 4 (metal/integrator/restir_di.metal.h): the reservoir
// this surface held last frame, found where the surface was then, merged
// into this frame's candidates' reservoir by the unbiased rule
// (reservoir.metal.h, unbiased.metal.h). A reservoir belongs to a surface,
// not a pixel (logical-overview.md, "Reservoirs are state keyed on
// surfaces"), so it is looked up where the surface was.
//
// The steps, for pixel x with record q (contracts/surface_record.h):
//
//   1. History: if the reuse constants say this frame starts the history
//      over (contracts/reuse_constants.h, previous = 0), the output is the
//      candidates' reservoir as it is, and nothing below runs.
//   2. Where it was: q's previous position (the surfaces pass wrote it:
//      the hit's point on its shape, placed by last frame's transform),
//      through last frame's camera (metal/camera/thin_lens.metal.h,
//      camera_pixel), is pixel x' of last frame's image. Outside the image,
//      or behind the camera: no temporal input. One pixel, rounded down,
//      as the paper takes.
//   3. Is it the same surface: last frame's record q' at x', of kind not
//      none, similar to q (similar.metal.h): q's depth taken at its
//      previous position through last frame's camera, so a surface that
//      moved is compared where it was. Else no temporal input.
//   4. Stream: q's own reservoir r (the candidates', count M_r), then last
//      frame's final reservoir r' at x', its count capped at H x M
//      (core/passes/restir_di.h): a sample kept for many frames is weighed
//      as at most H frames' candidates, which bounds how long an old choice
//      outlives the light that made it a good one. Each weighed by its
//      sample's target at q, as this frame lights it (target.metal.h).
//   5. Z (unbiased.metal.h), for the survivor y:
//        y from r:  M_r, plus min(M_r', H M) if q' supports y as last frame
//                   saw it: p̂_{q'}(y) by last frame's transforms and
//                   glows, and, where positive, one shadow ray from q'
//                   against last frame's structure;
//        y from r': min(M_r', H M), plus M_r if y is visible from q, one
//                   shadow ray from q against this frame's structure; if it
//                   is hidden, the output weighs y 0 and keeps its count.
//   6. Finish (reservoir.metal.h, step 4), and write it where the graph's
//      next pass reads (metal/frame/frame_images.h): the working reservoir,
//      in place, when a spatial merge follows, which reads it at
//      neighbors; this frame's final reservoir otherwise.
//
// The previous surface is last frame's in every respect: its record from
// last frame's records, its lights at last frame's transforms and glows,
// its rays against last frame's structure (the scene `then` the pass is
// given, as SceneView makes it of those three: metal/scene/
// scene_block.metal.h). A surface seen through glass or in a mirror
// reprojects as if seen straight, which is wrong when the glass moves or
// the camera does; the similarity test then rejects it, or lets through a
// surface whose reservoir Z weighs by what it could have produced. Noise,
// never bias (logical-overview.md).
//
// Step 5's ray is one ray query whichever frame it asks of: each lane
// chooses its structure, its ray's origin and the light's position, last
// frame's or this one's, as data, and the SIMD group runs one query, not a
// branch for each frame, so lanes whose survivors came from different
// inputs do not serialize two queries (GPU.4).
//
// Cost, per pixel: reads q's record (64 bytes) and previous position (12),
// q's reservoir (16), q' (64) and r' (16), each once: 172 bytes, the two at
// x' gathered from wherever the surface was, which for a still or slow
// camera is near x, so neighbouring lanes read neighbouring addresses
// (GPU.2); writes 16. Three target evaluations (r's and r''s samples at q,
// and Z's one) and at most one shadow ray. Two numbers, the two choices of
// step 4, from the pass's own stream (metal/sampler/sampler.metal.h).

#include <metal_stdlib>

#include "core/contracts/camera.h"
#include "core/contracts/reservoir.h"
#include "core/contracts/reuse_constants.h"
#include "core/contracts/surface_record.h"
#include "metal/reuse/reservoir.metal.h"
#include "metal/reuse/similar.metal.h"
#include "metal/reuse/unbiased.metal.h"
#include "metal/sampler/sampler.metal.h"

namespace serenity {
namespace shaders {

// Step 2: where `previous_position` is in last frame's image of `size`
// pixels, through last frame's camera; none outside it.
struct Reprojected {
    bool found;
    uint2 pixel;
};
inline Reprojected reproject(float3 previous_position, constant serenity::contracts::ReuseConstants& reuse,
                             uint2 size);

// What last frame left where the pixel's surface was: where the surface
// was, and last frame's record and final reservoir at the reprojected
// pixel.
struct LastFrame {
    float3 previous_position;
    serenity::contracts::SurfaceRecord there;
    serenity::contracts::Reservoir last_final;
};

// Steps 3 to 6 for `pixel`, given what `last` frame left. `now` and
// `then` are the scene as this frame and last frame place, light and trace
// it: scene_block.metal.h's SceneView, of either frame's arrays and
// structure, which is what `Scene` must be. `numbers` is the pass's stream.
// Callers that found no temporal input (step 2) do not call it.
template <typename Scene>
inline serenity::contracts::Reservoir merge_temporal(Scene now, Scene then, Merging pixel, LastFrame last,
                                                     constant serenity::contracts::ReuseConstants& reuse,
                                                     thread PathNumbers& numbers);

}  // namespace shaders
}  // namespace serenity
