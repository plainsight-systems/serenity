// The path pass (path.h): one path per pixel per frame, through a point in
// the pixel drawn anew each frame, folded into the accumulated image, whose
// mean is the frame's radiance: the passes after it read the image itself.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/camera.h"
#include "core/contracts/frame_constants.h"
#include "core/contracts/mnee_request.h"
#include "core/contracts/transform.h"
#include "metal/camera/thin_lens.metal.h"
#include "metal/film/accumulate.metal.h"
#include "metal/integrator/mnee.metal.h"
#include "metal/integrator/path.metal.h"
#include "metal/passes/bindings.h"
#include "metal/sampler/sampler.metal.h"
#include "metal/scene/scene_block.metal.h"

using namespace metal;
using namespace metal::raytracing;
using namespace serenity::shaders;

// One pixel's path, folded in: the radiance by the path tracer, or with
// MNEE (`manifold`, the graph's mnee setting), its requests written on
// `queue`, whose light the MNEE pass adds after (metal/passes/mnee/mnee.h).
// A kernel for each, below, so the one without compiles the loop it always
// did (path.metal.h) and binds what it always bound.
template <bool manifold>
inline void trace_pixel(constant serenity::contracts::FrameConstants& frame,
                        constant serenity::contracts::CameraData& camera, serenity::shaders::SceneStructure structure,
                        constant serenity::gpu::SceneBlock& block, constant float* sphere_glows,
                        constant serenity::contracts::Transform* transforms, device atomic_uint* non_finite,
                        texture2d<float, access::read_write> accumulated, mnee::Queue queue, uint2 pixel) {
    // Every thread reaches count_non_finite() below, which sums over its
    // SIMD group: none returns early.
    const bool inside = pixel.x < frame.width && pixel.y < frame.height;
    bool failed = false;
    if (inside) {
        // The scene, through its block, and as this frame places and lights
        // it (metal/scene/scene_block.metal.h).
        const path::Scene scene{structure, &block, transforms, sphere_glows};

        // The camera's ray as pixel_ray() draws it (camera/thin_lens.metal.h),
        // written out here: through pixel_ray() the path pass's kernel
        // compiled otherwise and measured 2% slower (docs/research/
        // 2026-10-16-codebase-guidelines-pass.md, C2). The same numbers in
        // the same order; the byte diff of every graph holds them equal.
        PathNumbers numbers = path_numbers(pixel, frame.frame_index);
        const float2 within = next_numbers2(numbers);  // the point in the pixel
        // The point on the lens, drawn only through one, so a pinhole's
        // paths draw what they drew before lenses; a pinhole reads none.
        const float2 lens = has_lens(camera) ? next_numbers2(numbers) : float2(0.0f);
        const CameraRay ray =
            camera_ray(camera, float2(pixel) + within, lens, uint2(frame.width, frame.height));
        const float3 sample = manifold ? path::radiance_manifold(scene, ray, numbers, queue,
                                                                 pixel.y * frame.width + pixel.x)
                                       : path::radiance(scene, ray, numbers);

        failed = !finite(sample);
        const float4 held = accumulate(accumulated.read(pixel), sample, frame.accumulated_frames == 0u);
        accumulated.write(held, pixel);
    }
    count_non_finite(failed, non_finite);
}

kernel void path_trace(constant serenity::contracts::FrameConstants& frame
                       [[buffer(serenity::bindings::path::constants)]],
                       constant serenity::contracts::CameraData& camera [[buffer(serenity::bindings::path::camera)]],
                       serenity::shaders::SceneStructure structure [[buffer(serenity::bindings::path::structure)]],
                       constant serenity::gpu::SceneBlock& block [[buffer(serenity::bindings::path::scene)]],
                       constant float* sphere_glows [[buffer(serenity::bindings::path::glows)]],
                       constant serenity::contracts::Transform* transforms
                       [[buffer(serenity::bindings::path::transforms)]],
                       device atomic_uint* non_finite [[buffer(serenity::bindings::path::non_finite)]],
                       texture2d<float, access::read_write> accumulated
                       [[texture(serenity::bindings::path::accumulated)]],
                       uint2 pixel [[thread_position_in_grid]]) {
    trace_pixel<false>(frame, camera, structure, block, sphere_glows, transforms, non_finite, accumulated,
                       mnee::Queue{nullptr, nullptr}, pixel);
}

kernel void path_trace_mnee(constant serenity::contracts::FrameConstants& frame
                            [[buffer(serenity::bindings::path::constants)]],
                            constant serenity::contracts::CameraData& camera
                            [[buffer(serenity::bindings::path::camera)]],
                            serenity::shaders::SceneStructure structure [[buffer(serenity::bindings::path::structure)]],
                            constant serenity::gpu::SceneBlock& block [[buffer(serenity::bindings::path::scene)]],
                            constant float* sphere_glows [[buffer(serenity::bindings::path::glows)]],
                            constant serenity::contracts::Transform* transforms
                            [[buffer(serenity::bindings::path::transforms)]],
                            device atomic_uint* non_finite [[buffer(serenity::bindings::path::non_finite)]],
                            device serenity::contracts::MneeRequest* requests
                            [[buffer(serenity::bindings::path::mnee_requests)]],
                            device atomic_uint* header [[buffer(serenity::bindings::path::mnee_header)]],
                            texture2d<float, access::read_write> accumulated
                            [[texture(serenity::bindings::path::accumulated)]],
                            uint2 pixel [[thread_position_in_grid]]) {
    trace_pixel<true>(frame, camera, structure, block, sphere_glows, transforms, non_finite, accumulated,
                      mnee::Queue{requests, header}, pixel);
}
