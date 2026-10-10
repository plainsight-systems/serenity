// The path pass (path.h): one path per pixel per frame, through a point in
// the pixel drawn anew each frame, folded into the accumulated image, whose
// mean is the frame's radiance.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/camera.h"
#include "core/contracts/frame_constants.h"
#include "core/contracts/transform.h"
#include "metal/camera/thin_lens.metal.h"
#include "metal/film/accumulate.metal.h"
#include "metal/integrator/path.metal.h"
#include "metal/sampler/sampler.metal.h"
#include "metal/scene/scene_block.metal.h"

using namespace metal;
using namespace metal::raytracing;
using namespace serenity::shaders;

kernel void path_trace(constant serenity::contracts::FrameConstants& frame [[buffer(0)]],
                       constant serenity::contracts::CameraData& camera [[buffer(1)]],
                       primitive_acceleration_structure structure [[buffer(2)]],
                       constant serenity::gpu::SceneBlock& block [[buffer(3)]],
                       constant float* sphere_glows [[buffer(4)]],
                       constant serenity::contracts::Transform* transforms [[buffer(5)]],
                       device atomic_uint* non_finite [[buffer(6)]],
                       texture2d<float, access::read_write> accumulated [[texture(0)]],
                       texture2d<float, access::write> radiance [[texture(1)]],
                       uint2 pixel [[thread_position_in_grid]]) {
    // Every thread reaches count_non_finite() below, which sums over its
    // SIMD group: none returns early.
    const bool inside = pixel.x < frame.width && pixel.y < frame.height;
    bool failed = false;
    if (inside) {
        // The scene, through its block, and as this frame places and lights
        // it (metal/scene/scene_block.metal.h).
        path::Scene scene;
        scene.structure = structure;
        scene.block = &block;
        scene.transforms = transforms;
        scene.glows = sphere_glows;

        PathNumbers numbers = path_numbers(pixel, frame.frame_index);
        const float2 within = next_numbers2(numbers);  // the point in the pixel
        // The point on the lens, drawn only through one, so a pinhole's
        // paths draw what they drew before lenses.
        const float2 lens = camera.lens_radius > 0.0f ? next_numbers2(numbers) : float2(0.5f);
        const CameraRay ray = camera_ray(camera, float2(pixel) + within, lens, frame.width, frame.height);
        const float3 sample = path::radiance(scene, ray.origin, ray.direction, numbers);

        failed = !finite(sample);
        const float4 held = accumulate(accumulated.read(pixel), sample, frame.accumulated_frames == 0u);
        accumulated.write(held, pixel);
        radiance.write(float4(held.rgb, 1.0f), pixel);
    }
    count_non_finite(failed, non_finite);
}
