// The preview (preview.h): the pass that launches the direct-light
// integrator (integrator/direct.metal.h) for each pixel, at four positions
// within it, and writes the mean to the radiance image.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/camera.h"
#include "core/contracts/frame_constants.h"
#include "core/contracts/transform.h"
#include "metal/camera/thin_lens.metal.h"
#include "metal/integrator/direct.metal.h"
#include "metal/passes/bindings.h"
#include "metal/scene/scene_block.metal.h"

using namespace metal;
using namespace metal::raytracing;
using namespace serenity::shaders;

namespace {

// The pixel's sample positions: a rotated grid, the four points that
// separate both rows and columns of near-horizontal and near-vertical edges.
constant constexpr uint pixel_samples = 4;
constant constexpr float2 positions[pixel_samples] = {
    float2(0.375f, 0.125f), float2(0.875f, 0.375f), float2(0.625f, 0.875f), float2(0.125f, 0.625f)};

// Through a lens, position i's point on the lens is position i +
// lens_shift's: the one diagonally across, so no position is its own.
constant constexpr uint lens_shift = 2u;

}  // namespace

kernel void preview(constant serenity::contracts::FrameConstants& frame
                    [[buffer(serenity::bindings::preview::constants)]],
                    constant serenity::contracts::CameraData& camera [[buffer(serenity::bindings::preview::camera)]],
                    primitive_acceleration_structure structure [[buffer(serenity::bindings::preview::structure)]],
                    constant serenity::gpu::SceneBlock& block [[buffer(serenity::bindings::preview::scene)]],
                    constant float* sphere_glows [[buffer(serenity::bindings::preview::glows)]],
                    constant serenity::contracts::Transform* transforms
                    [[buffer(serenity::bindings::preview::transforms)]],
                    texture2d<float, access::write> radiance [[texture(serenity::bindings::preview::radiance)]],
                    uint2 pixel [[thread_position_in_grid]]) {
    if (pixel.x >= frame.width || pixel.y >= frame.height) {
        return;
    }
    // The scene, through its block, and as this frame places and lights it
    // (metal/scene/scene_block.metal.h).
    const direct::Scene scene{structure, &block, transforms, sphere_glows};

    float3 color = float3(0.0f);
    for (uint i = 0; i < pixel_samples; ++i) {
        // Through a lens, each position with another of the four as its
        // point on the lens (contracts/camera.h): fixed, as everything here
        // is, so a thing far from focus shows as up to four copies.
        const CameraRay ray = camera_ray(camera, float2(pixel) + positions[i],
                                         positions[(i + lens_shift) % pixel_samples], uint2(frame.width, frame.height));
        color += direct::radiance(scene, ray, direct::PixelSamples{pixel, i});
    }
    radiance.write(float4(color / float(pixel_samples), 1.0f), pixel);
}
