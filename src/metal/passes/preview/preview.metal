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


}  // namespace

kernel void preview(constant serenity::contracts::FrameConstants& frame [[buffer(0)]],
                    constant serenity::contracts::CameraData& camera [[buffer(1)]],
                    primitive_acceleration_structure structure [[buffer(2)]],
                    constant serenity::gpu::SceneBlock& block [[buffer(3)]],
                    constant float* sphere_glows [[buffer(4)]],
                    constant serenity::contracts::Transform* transforms [[buffer(5)]],
                    texture2d<float, access::write> radiance [[texture(0)]],
                    uint2 pixel [[thread_position_in_grid]]) {
    if (pixel.x >= frame.width || pixel.y >= frame.height) {
        return;
    }
    // The scene, through its block, and as this frame places and lights it
    // (metal/scene/scene_block.metal.h).
    direct::Scene scene;
    scene.structure = structure;
    scene.block = &block;
    scene.transforms = transforms;
    scene.glows = sphere_glows;

    float3 color = float3(0.0f);
    for (uint i = 0; i < pixel_samples; ++i) {
        // Through a lens, each position with another of the four as its
        // point on the lens (contracts/camera.h): fixed, as everything here
        // is, so a thing far from focus shows as up to four copies.
        const CameraRay ray =
            camera_ray(camera, float2(pixel) + positions[i], positions[(i + 2) % pixel_samples], frame.width,
                       frame.height);
        color += direct::radiance(scene, ray.origin, ray.direction, direct::Pixel{pixel, i});
    }
    radiance.write(float4(color / float(pixel_samples), 1.0f), pixel);
}
