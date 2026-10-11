#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (ReSTIR DI's shade).
//
// ReSTIR DI's step 6 (metal/passes/restir_di/restir_di.metal.h,
// which its shader includes): one thread a
// pixel; the direct light at its surface from its final reservoir, with no
// shadow ray (the merges' invariant: the sample is visible from the surface
// or weighs 0, metal/reuse/unbiased.metal.h), times the path's throughput,
// plus the light the camera's path found on the way, plus the path's
// continuation from the surface for the indirect light
// (metal/integrator/path.metal.h, continue_path()); folded into the
// accumulated image (metal/film/accumulate.metal.h), as the path pass folds
// its own, and counted where it is not finite (metal/film/non_finite.h).
//
// Reads: the scene and structure, the reuse constants, the records, path
// states and final reservoirs, and the accumulated image. Writes: the
// accumulated image, which is the frame's radiance for the presenting pass
// after it (core/frame/schedule.h), and the frame's count of samples not
// finite. It writes no reservoir: the final ones stay as the last merge
// wrote them, for the next frame.
//
// Ordering: after the last merge (or the candidates), behind the renderer's
// dispatch barrier; the accumulated image is the frame before's history,
// which the renderer's queue barrier at the frame's first pass has waited
// for (metal/frame/renderer.h), as for the path pass.
//
// Cost, per pixel: 176 bytes read (the 80-byte record, the 64-byte path
// state, the 32-byte reservoir) and the accumulated pixel's 16 read and 16
// written; one target evaluation, with its light's 76 bytes of logical
// loads; no ray
// for the direct light, and then the continuation's rays, as the path
// tracer's from the same surface without its shadow ray there. One
// dispatch.
class RestirShadePass {
public:
    explicit RestirShadePass(const Library& library);

    // Records the pass. Throws MetalError if `resources` has no scene, no
    // reuse constants, no records, path states or final reservoirs, no
    // accumulated image or no counter of samples not finite.
    void record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const;

private:
    NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
};

}  // namespace serenity::metal
