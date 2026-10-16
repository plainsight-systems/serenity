#pragma once

#include <optional>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"
#include "metal/passes/mnee/mnee.h"

namespace serenity::metal {

// Axis: Pass (path tracing).
//
// The pass that runs the path tracer without reuse, next event estimation
// weighed against BSDF sampling by MIS (metal/integrator/path.metal.h,
// which states the algorithm), once per pixel per frame, and folds the result
// into the accumulated image (metal/film/accumulate.metal.h), so a still
// image converges toward the scene's full light transport.
//
// Per pixel, per frame:
//
//   - the camera ray passes through a point in the pixel drawn anew each
//     frame (metal/sampler/sampler.metal.h), so anti-aliasing comes from the
//     average, not from extra rays; through a lens, from a point on the lens
//     drawn anew each frame too, so depth of field comes from the average
//     as well (contracts/camera.h);
//   - the integrator returns the radiance along it;
//   - the accumulated image's mean for the pixel takes it in, n being the
//     frames the image held (contracts/frame_constants.h,
//     accumulated_frames), and is written back.
//
// The accumulated image is the frame's radiance image: its rgb, the mean,
// linear, is what the presenting pass that follows (core/frame/schedule.h)
// reads, and its a, the count, no presenting pass reads (metal/frame/
// renderer.h binds it so). Written once, not copied to an image of its own
// as well (GDSA.6: the copy cost 16 bytes a pixel of traffic and an image
// of the frame's size, and no frame time the measurement could tell apart
// from noise; docs/research/2026-10-10-path-radiance-write.md).
//
// Ordering: the pass reads the image the previous frame wrote. Metal 4 does
// not track hazards; the renderer records the barrier that waits for the
// queue's earlier dispatches before the first pass of the frame that touches
// state the frames share, this one (metal/frame/renderer.h, GPU.8): frames
// in flight overlap their recording, not their writes to the image.
//
// The graph's mnee setting (core/frame/schedule.h) picks which of two
// kernels runs: the path tracer, or the path tracer with caustics by
// manifold next event estimation. Two kernels rather than a flag the shader
// reads, so the path tracer without MNEE compiles, and costs, what it did.
//
// Cost: one thread per pixel, in rows of the execution width (GPU.2); per
// pixel, the integrator's path (about twice its length in rays, a handful
// of surfaces in practice; integrator/path.metal.h), and 32 bytes of
// image: the accumulated pixel's 16 read and 16 written.
class PathPass {
public:
    // With `mnee`, the path tracer with manifold next event estimation
    // (metal/integrator/path.metal.h, "Manifold next event estimation"), a
    // kernel of its own; without, the kernel the pass always ran.
    PathPass(const Library& library, bool mnee);

    // Records the pass. Throws MetalError if `resources` has no scene, no
    // camera, no accumulated image or no counter of samples not finite.
    void record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const;

private:
    NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
    std::optional<MneePass> mnee_;  // with mnee: the requests' solve and fold
};

}  // namespace serenity::metal
