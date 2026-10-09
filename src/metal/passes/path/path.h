#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (path tracing).
//
// The naive path tracer: milestone 1, the baseline every later estimator is
// measured against (README.md). Each frame traces one path per pixel and
// folds its radiance into the accumulated image (metal/frame/accumulation.h),
// so a still image converges toward the scene's full light transport: light
// straight from the fireflies and the sky, light every surface reflects onto
// every other, and, slowly and noisily, light glass focuses (caustics). It
// is unbiased: its mean over frames is the image the reference converges
// to (logical-overview.md, principle 3).
//
// The algorithm is the textbook path tracer with next event estimation and
// multiple importance sampling: Veach's (Robust Monte Carlo Methods for
// Light Transport Simulation, 1997, ch. 9) as pbrt-v4 writes it
// (PathIntegrator::Li, section 13.4). Per pixel, per frame, in
// metal/integrator/path.metal.h, whose sections carry these step numbers:
//
//   L = 0, the radiance found; beta = 1, the path's throughput; the ray
//   from the camera through a point in the pixel drawn anew each frame,
//   so anti-aliasing comes from the average, not from extra rays.
//   For each surface, up to 8:
//
//   Step 1  Trace: the nearest surface along the ray.
//   Step 2  Escape: if there is none, L += beta x sky(direction), and stop.
//           The sky is reached by BSDF sampling alone; it is not aimed at.
//   Step 3  Emission: if the surface glows, L += beta x w x L_e, and stop
//           (a light scatters nothing). w = 1 for the camera's own ray and
//           after a delta lobe, which aiming could not have found; otherwise
//           the power heuristic against aiming,
//             w = p_bsdf^2 / (p_bsdf^2 + p_aim^2),
//           p_bsdf the pdf of the bounce that found the light, p_aim =
//           P(light) x pdf_light(the previous surface -> this direction).
//   Step 4  Resolve the surface's BSDF (contract 2).
//   Step 5  Next event estimation, where the BSDF has a lobe that is not
//           delta (aims_at_lights): choose one light uniformly, P = 1 / N
//           (light_selection/uniform_light.metal.h); draw a direction toward
//           it, pdf p_light (contract 3); trace one shadow ray; if nothing
//           blocks it,
//             L += beta x f(wo, wi) |cos| x L_e x w / (P x p_light),
//             w = (P p_light)^2 / ((P p_light)^2 + p_bsdf(wi)^2).
//           Cost per surface is the same for any number of lights
//           (principle 9).
//   Step 6  Sample the BSDF: wi, f and pdf (contract 2). If pdf is 0, stop.
//           beta *= f |cos| / pdf; keep pdf and whether the lobe was delta
//           for step 3 at the next surface.
//   Step 7  Russian roulette, from the 4th surface: survive with
//           q = min(the largest channel of beta, 0.95), else stop;
//           beta /= q, so the mean is unchanged.
//   Step 8  Continue from the surface along wi.
//
//   Step 9  Accumulate: the image's new mean is (old x n + L) / (n + 1), n
//           the frames it held (contracts/frame_constants.h,
//           accumulated_frames); write it back, and write it for display to
//           the frame's target.
//
// The numbers a path draws are a function of the pixel, the frame's index
// and the dimension, the count of numbers drawn before it on the path
// (metal/sampler/sampler.metal.h): independent between frames, so their mean
// converges, and any frame can be rendered again exactly, alone (principle 2).
//
// For display the mean is encoded as the preview encodes it
// (passes/display.metal.h).
//
// Ordering: the pass reads the image the previous frame wrote. Metal 4 does
// not track hazards, so before its dispatch the pass records a barrier that
// waits for the queue's earlier dispatches (barrierAfterQueueStages, dispatch
// before dispatch): frames in flight overlap their recording, not their
// writes to the image.
//
// Cost: one thread per pixel, in rows of the execution width (GPU.2). Per
// pixel, per frame, at most 8 surfaces, each one ray to the next surface and
// at most one shadow ray: 16 rays at most, about 2 x the mean path length
// in practice. One read and one write of 16 bytes of the accumulated image
// and 4 bytes of the target per pixel.
class PathPass {
public:
    PathPass(const Device& device, const Library& library);

    // Records the pass. Throws Error if `resources` has no scene, no camera
    // or no accumulated image.
    void record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const;

private:
    NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
};

}  // namespace serenity::metal
