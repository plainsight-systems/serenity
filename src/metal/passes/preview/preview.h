#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (preview).
//
// A diagnostic view of primary visibility (logical-overview.md), kept for
// good, like the test pattern: what the camera's rays reach, through glass,
// shown unlit. It is not the light transport and makes no claim to it; it
// shows that the geometry, the materials, the textures and the environment
// are where the scene says, before any light is computed.
//
// Per pixel, one ray from the camera through the pixel's center
// (camera/pinhole.h), deterministic, with no random numbers:
//
//   - leaving the scene, it shows the environment in its direction;
//   - at a rough surface, it shows that surface's color, unlit, and stops;
//   - at glass, the Fresnel term F splits it (dielectric.metal.h): the
//     reflected part, weighted by F, is traced once more to the first thing
//     it reaches and shown as that, flat (a rough surface's color, or the
//     environment, glass there showing the environment); the refracted part,
//     weighted by 1 - F, continues. Under total internal reflection the
//     whole ray reflects and continues.
//
// A ray continues for at most 8 surfaces of glass; one that has not reached
// a rough surface or left the scene by then contributes nothing further.
// Reflections inside the glass beyond the first are not shown. Both are
// properties of this preview, not of the renderer.
//
// The kernel is preview.metal, which uses the shared shader halves of the
// kinds: pinhole.metal.h, trace.metal.h (the hardware loop over boxes, and
// each shape kind's exact test: sphere.metal.h, box.metal.h),
// dielectric.metal.h, checker.metal.h and gradient_sky.metal.h.
//
// Cost: one thread per pixel, in rows of the execution width (GPU.2). Per
// pixel, at most 8 + 1 rays for the glass in this scene, typically 1 to 3;
// each ray's hardware traversal visits a handful of boxes. Memory read is
// the scene's few hundred bytes, from cache, and 4 bytes written per pixel.
class PreviewPass {
public:
    PreviewPass(const Device& device, const Library& library);

    // Records the pass. Throws Error if `resources` has no scene.
    void record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const;

private:
    NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
};

}  // namespace serenity::metal
