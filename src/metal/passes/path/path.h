#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (path tracing).
//
// The pass that runs the naive path tracer (metal/integrator/path.metal.h,
// which states the algorithm) once per pixel per frame and folds the result
// into the accumulated image (metal/film/accumulate.metal.h), so a still
// image converges toward the scene's full light transport.
//
// Per pixel, per frame:
//
//   - the camera ray passes through a point in the pixel drawn anew each
//     frame (metal/sampler/sampler.metal.h), so anti-aliasing comes from the
//     average, not from extra rays;
//   - the integrator returns the radiance along it;
//   - the accumulated image's mean for the pixel takes it in, n being the
//     frames the image held (contracts/frame_constants.h,
//     accumulated_frames), and is written back;
//   - the mean, encoded for display as the preview encodes it
//     (passes/display.metal.h), is written to the frame's target.
//
// Ordering: the pass reads the image the previous frame wrote. Metal 4 does
// not track hazards, so before its dispatch the pass records a barrier that
// waits for the queue's earlier dispatches (barrierAfterQueueStages, dispatch
// before dispatch): frames in flight overlap their recording, not their
// writes to the image.
//
// Cost: one thread per pixel, in rows of the execution width (GPU.2); per
// pixel, the integrator's path (about twice its length in rays, a handful
// of surfaces in practice; integrator/path.metal.h), and 36 bytes of
// images: the accumulated pixel's 16 read and 16 written, and the target's
// 4.
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
