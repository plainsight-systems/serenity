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
// Per pixel, per frame, the path (metal/integrator/path.metal.h):
//
//   - starts at a position within the pixel drawn anew each frame, so
//     anti-aliasing comes from the average, not from extra rays;
//   - at each surface, resolves its BSDF (contract 2) and, where it has a
//     lobe that is not delta, aims at one light: chosen uniformly among the
//     lights (metal/light_selection/uniform_light.metal.h), a direction
//     toward it drawn by the emitter (contract 3), and one shadow ray. Cost
//     per vertex is constant in the number of lights (principle 9);
//   - then samples the BSDF for the next direction and continues;
//   - weighs the two ways of reaching a light against each other by the
//     power heuristic (Veach, multiple importance sampling): aiming at the
//     light, and a BSDF-sampled ray that happens to reach it. Both count,
//     neither twice. Light reached only through delta lobes (seen in glass
//     or a mirror) has no aimed counterpart and counts whole;
//   - counts the sky where a ray leaves the scene, by BSDF sampling alone:
//     the sky is not aimed at;
//   - stops after 8 surfaces, or earlier by Russian roulette from the 4th,
//     with a survival probability of the path's throughput, at most 0.95,
//     dividing by it so the mean is unchanged.
//
// The numbers a path draws are a function of the pixel, the frame's index
// and the dimension, the count of numbers drawn before it on the path
// (metal/sampler/sampler.metal.h): independent between frames, so their mean
// converges, and any frame can be rendered again exactly, alone (principle 2).
//
// The new mean is (old x n + this frame) / (n + 1), n the frames the image
// held (contracts/frame_constants.h, accumulated_frames), written back to the
// accumulated image, and also, for display, to the frame's target, encoded
// as the preview encodes it (passes/display.metal.h).
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
