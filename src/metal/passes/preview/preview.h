#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (preview).
//
// Deterministic ray tracing: what the camera sees, lit by the scene's
// glowing spheres and its sky, with no average across frames. It is not the
// light transport the renderer is for (logical-overview.md): it counts light
// that reaches a surface straight from a light or the sky, and the light
// metal and glass reflect and glass refracts, and leaves out what surfaces
// reflect onto other rough surfaces and what glass focuses (caustics). Those
// arrive with the path tracer and its reuse. Kept for good, like the test
// pattern: a still image that every later estimator's direct light must
// agree with.
//
// Per pixel, four camera rays, at a rotated grid of positions within it
// (anti-aliasing), each followed until it reaches something that ends it:
//
//   - leaving the scene, the sky in its direction;
//   - a glowing sphere, its radiance;
//   - a rough surface: from each light, its irradiance (lights/
//     sphere_light.h) times the fraction of the light visible, by 4 shadow
//     rays drawn uniformly over the cone the light fills, out to its rim
//     (metal/lights/sphere_light.metal.h); and the sky, by 2
//     cosine-distributed rays, each that escapes bringing the sky's
//     radiance; times the surface's color. A shadow is soft where a light is
//     partly hidden.
//   - metal: each light's highlight, by GGX with its roughness widened by
//     the light's size, over the light's exact solid angle, by the fraction
//     visible as above; and the rest of
//     the scene by 4 reflected rays drawn from GGX's visible normals, each
//     shaded as a reflection (below), lights not counted again;
//   - glass: the Fresnel term F splits it (dielectric.metal.h); the part
//     reflected where it enters from outside, weighted by F, is shaded as a
//     reflection; the refracted part, weighted by 1 - F, continues. Under
//     total internal reflection the whole ray reflects and continues. At
//     most 8 surfaces of glass; where a ray leaves glass its reflected part
//     is dropped.
//
// A reflection is shaded more simply: a rough surface by each light whose
// center it sees, one shadow ray each, and the sky above it unblocked; a
// glowing sphere by its glow, seen in glass, not in metal, whose highlights
// already count it; metal and glass by the sky in the ray's direction.
//
// Glass is opaque to shadow rays: the light it would focus is a caustic.
//
// The numbers each estimate draws are a function of the pixel and the
// purpose (metal/sampler/sampler.metal.h), not of the frame, so a still
// scene renders the same image every frame, its error a fine, fixed grain.
//
// Colors are linear (core/scene/scene.h). For display, a color brighter than
// 1 in some channel is scaled by its largest channel, keeping its hue, and
// encoded with sRGB's transfer function, which an 8-bit target and the
// display expect. Tone mapping, when it comes, is a pass of its own and the
// encoding moves to it.
//
// Every light is counted at every pixel: the every-light selection
// (light_selection/every_light.metal.h), so the image has no selection
// noise. Its cost therefore grows with the number of lights, the one place
// that principle 9 of logical-overview.md does not hold, and why the preview
// is for scenes of a few lights. A scene of thousands of fireflies is the
// estimators' to render.
//
// The kernel is preview.metal: it places the four camera rays, launches the
// integrator (metal/integrator/direct.metal.h) for each, and encodes the
// mean for display. The integrator uses the shared shader halves of the
// kinds: trace.metal.h (the hardware loop over boxes, each shape kind's
// exact test in shapes.metal.h), rough.metal.h, conductor.metal.h,
// dielectric.metal.h, textures.metal.h, sphere_light.metal.h,
// gradient_sky.metal.h, sampler.metal.h and warp.metal.h.
//
// Cost: one thread per pixel, in rows of the execution width (GPU.2). On a
// rough surface, with L lights, 4 x (1 + 4L + 2) rays a pixel, growing by
// 16 with each light: 44 for the first scene's two fireflies; more on
// metal. Measured on the M3 Max, the
// first scene takes 78 ms of GPU time at 3456 x 2234 and 20 ms at half that
// in each direction. The window's frame budget is met by rendering below
// the display's resolution and upscaling, which comes later.
class PreviewPass {
public:
    PreviewPass(const Device& device, const Library& library);

    // Records the pass. Throws Error if `resources` has no scene or no
    // camera.
    void record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const;

private:
    NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
};

}  // namespace serenity::metal
