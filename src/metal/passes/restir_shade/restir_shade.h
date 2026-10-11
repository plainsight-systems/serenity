#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (ReSTIR DI's shade).
//
// ReSTIR DI's step 6 (metal/passes/restir_di/restir_di.metal.h, which its
// shader includes): one thread a pixel; the direct light at its surface
// from its final reservoir, its visibility from the reservoir's test where
// that test is this pixel's own or, in basic mode with visibility reuse,
// recent and near enough, and otherwise from one shadow ray; times the
// path's throughput, plus the light the camera's path found on the way,
// plus the path's continuation from the surface for the indirect light
// (metal/integrator/path.metal.h, continue_path()); folded into the
// accumulated image (metal/film/accumulate.metal.h), as the path pass folds
// its own, and counted where it is not finite (metal/film/non_finite.h).
//
// Reads: the scene and structure, the reuse constants, the records, path
// states and final reservoirs, and the accumulated image. Writes: the
// accumulated image, which is the frame's radiance for the presenting pass
// after it (core/frame/schedule.h), and the frame's count of samples not
// finite; and, in a graph with a temporal merge, where it traced a ray, its
// own pixel's final reservoir, in place: that ray as the reservoir's test,
// here and now, and W = 0 where the sample was hidden, for the next frame's
// temporal merge. In place is safe: each thread reads and writes its own
// pixel's alone, and nothing after the shade reads them this frame. The
// write-back arrives with the temporal merge that reads it: until then no
// graph the backend runs has one, its kernel binds the final reservoirs
// read-only and writes nothing back, and step 6 returns what it would
// write (restir_di.metal.h, Shaded).
//
// Ordering: after the last merge (or the candidates), behind the renderer's
// dispatch barrier; the accumulated image is the frame before's history,
// which the renderer's queue barrier at the frame's first pass has waited
// for (metal/frame/renderer.h), as for the path pass. The next frame reads
// the reservoirs it wrote back behind that same queue barrier.
//
// Cost, per pixel: 92 bytes read (the 32-byte record, the 36-byte path
// state, the 24-byte reservoir), the accumulated pixel's 16 read and 16
// written, and at most 8 written back (the reservoir's test and W) where it
// traced; one target evaluation, with its light's 76 bytes of logical
// loads; for the direct light, no ray in raytraced mode or a graph with no
// merge, and in basic mode one where the reservoir's test is not taken;
// then the continuation's rays, as the path tracer's from the same surface
// without its shadow ray there. One dispatch.
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
