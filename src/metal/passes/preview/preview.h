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
// (anti-aliasing), each followed until it reaches something that ends it.
// What it does at a surface it decides by the surface's BSDF and its lobes
// (contract 2), and every light it reads through the emitter (contract 3),
// naming no material and no light kind (metal/integrator/direct.metal.h):
//
//   - leaving the scene, the sky in its direction;
//   - a light, its glow (emitted); a glowing sphere scatters nothing more;
//   - a surface with a lobe that is not delta (rough, metal): from each
//     light, 4 directions the emitter draws over it (for a sphere,
//     uniformly over the cone it fills), each f |cos| L / pdf where no
//     shape but the light lies before it, so shadows are soft where a light
//     is partly hidden and a metal's highlight is the light's own shape seen
//     through its lobe; and rays its BSDF samples, each weighted value |cos|
//     / pdf: a diffuse lobe's 2 bring the sky where they escape, a glossy
//     lobe's 4 bring what they reach, shaded as a reflection (below),
//     lights not counted again;
//   - a surface whose lobes are all delta (glass): every lobe taken, each
//     with its probability, by sampling the BSDF at both ends of u.x; the
//     reflection, where the ray arrives from outside, shaded as a
//     reflection; the transmitted part continuing. At most 8 such surfaces.
//
// A reflection is shaded more simply: a light by its glow, in glass but not
// in metal, whose highlights already count it; a surface with a diffuse
// lobe by each light toward its middle, one shadow ray each, and the sky
// above it unblocked; any other surface by the sky along the ray.
//
// Glass is opaque to shadow rays: the light it would focus is a caustic.
//
// The numbers each estimate draws are a function of the pixel and the
// purpose (metal/sampler/sampler.metal.h), not of the frame, so a still
// scene renders the same image every frame, its error a fine, fixed grain.
//
// Colors are linear (core/scene/scene.h), and the pass writes them so, into
// the frame's radiance image (metal/frame/frame_images.h): how they are shown
// is the presenting pass's that follows it (core/frame/schedule.h), display
// for the numbers as they are, tone_map for the look.
//
// Every light is counted at every pixel: the every-light selection
// (light_selection/every_light.metal.h), so the image has no selection
// noise. Its cost therefore grows with the number of lights, the one place
// that principle 9 of logical-overview.md does not hold, and why the preview
// is for scenes of a few lights. A scene of thousands of fireflies is the
// estimators' to render.
//
// The kernel is preview.metal: it places the four camera rays, launches the
// integrator (metal/integrator/direct.metal.h) for each, and writes their
// mean to the radiance image.
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

    // Records the pass. Throws Error if `resources` has no scene, no camera
    // or no radiance image.
    void record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const;

private:
    NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
};

}  // namespace serenity::metal
