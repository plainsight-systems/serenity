// ReSTIR PT's initial sampling (restir_pt_initial.h): steps 1 to 4 of
// metal/passes/restir_pt/restir_pt.metal.h, one thread a traced pixel
// (restir_pt.metal.h, Traced: every pixel, or the frame's field). The path
// tree along the camera's ray, resampled to one path into the pixel's path
// reservoir; the pixel's primary hit for the merge; the light the camera
// sees directly into the frame's light buffer.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/camera.h"
#include "core/contracts/frame_constants.h"
#include "core/contracts/light_tile.h"
#include "core/contracts/mnee_request.h"
#include "core/contracts/path_reservoir.h"
#include "core/contracts/path_reuse_settings.h"
#include "core/contracts/transform.h"
#include "metal/integrator/mnee.metal.h"
#include "metal/passes/bindings.h"
#include "metal/passes/restir_pt/restir_pt.metal.h"

using namespace metal;
using namespace metal::raytracing;
using namespace serenity::shaders;
namespace binding = serenity::bindings::restir_pt;

// One pixel's tree (steps 1 to 4), with MNEE (`manifold`, the graph's
// mnee: its requests written on `queue`, whose light the MNEE pass adds to
// the light buffer after) or without; a kernel for each, below, so the one
// without compiles the tree it always did and binds what it always bound.
template <bool manifold>
inline void initial_pixel(constant serenity::contracts::FrameConstants& frame,
                          constant serenity::contracts::CameraData& camera, serenity::shaders::SceneStructure structure,
                          constant serenity::gpu::SceneBlock& block, constant float* sphere_glows,
                          constant serenity::contracts::Transform* transforms,
                          constant serenity::contracts::PathReuseSettings& settings,
                          device serenity::contracts::PathReservoir* reservoirs,
                          device serenity::contracts::PrimaryHit* primaries, device float4* light,
                          const device serenity::contracts::LightTileEntry* tiles,
                          device serenity::contracts::Float3* previous_positions,
                          constant serenity::contracts::Transform* last_transforms, mnee::Queue queue, uint2 grid_place) {
    const restir_pt::Traced traced = restir_pt::traced(frame, settings);
    const uint2 pixel = traced.pixel(grid_place);
    if (!traced.inside(pixel)) {
        return;
    }
    const uint i = traced.slot(pixel);
    const restir_pt::Scene scene{structure, &block, transforms, sphere_glows};
    PathNumbers numbers;
    const CameraRay ray = restir_pt::primary_ray(camera, pixel, frame.frame_index, uint2(frame.width, frame.height),
                                                 numbers);
    // confidence_cap is 0 exactly in a graph without restir_pt_temporal,
    // where neither array is bound (contracts/path_reuse_settings.h).
    const bool temporal = settings.confidence_cap > 0.0f;
    restir_pt::TreeGatherOf<manifold> tree = restir_pt::start_tree<manifold>(
        settings, numbers, scene.shapes(), restir_pt::block_tile(tiles, pixel, frame.frame_index), queue, i);
    (void)path::walk_path<path::Start::camera, path::Stop::at_end>(scene, path::walk_from(ray), Shading{}, 0u,
                                                                   numbers, tree);
    reservoirs[i] = restir_pt::finish_tree(tree);
    primaries[i] = tree.primary;
    if (temporal && tree.primary.distance > 0.0f) {
        previous_positions[i] = to_packed(restir_pt::previous_position(tree, transforms, last_transforms));
    }
    light[i] = float4(tree.seen, 0.0f);
}

kernel void restir_pt_initial(constant serenity::contracts::FrameConstants& frame [[buffer(binding::constants)]],
                              constant serenity::contracts::CameraData& camera [[buffer(binding::camera)]],
                              serenity::shaders::SceneStructure structure [[buffer(binding::structure)]],
                              constant serenity::gpu::SceneBlock& block [[buffer(binding::scene)]],
                              constant float* sphere_glows [[buffer(binding::glows)]],
                              constant serenity::contracts::Transform* transforms [[buffer(binding::transforms)]],
                              constant serenity::contracts::PathReuseSettings& settings [[buffer(binding::settings)]],
                              device serenity::contracts::PathReservoir* reservoirs [[buffer(binding::reservoirs)]],
                              device serenity::contracts::PrimaryHit* primaries [[buffer(binding::primaries)]],
                              device float4* light [[buffer(binding::light)]],
                              const device serenity::contracts::LightTileEntry* tiles [[buffer(binding::light_tiles)]],
                              device serenity::contracts::Float3* previous_positions
                              [[buffer(binding::previous_positions)]],
                              constant serenity::contracts::Transform* last_transforms
                              [[buffer(binding::last_transforms)]],
                              uint2 grid_place [[thread_position_in_grid]]) {
    initial_pixel<false>(frame, camera, structure, block, sphere_glows, transforms, settings, reservoirs, primaries,
                         light, tiles, previous_positions, last_transforms, mnee::Queue{nullptr, nullptr}, grid_place);
}

kernel void restir_pt_initial_mnee(constant serenity::contracts::FrameConstants& frame [[buffer(binding::constants)]],
                                   constant serenity::contracts::CameraData& camera [[buffer(binding::camera)]],
                                   serenity::shaders::SceneStructure structure [[buffer(binding::structure)]],
                                   constant serenity::gpu::SceneBlock& block [[buffer(binding::scene)]],
                                   constant float* sphere_glows [[buffer(binding::glows)]],
                                   constant serenity::contracts::Transform* transforms [[buffer(binding::transforms)]],
                                   constant serenity::contracts::PathReuseSettings& settings
                                   [[buffer(binding::settings)]],
                                   device serenity::contracts::PathReservoir* reservoirs
                                   [[buffer(binding::reservoirs)]],
                                   device serenity::contracts::PrimaryHit* primaries [[buffer(binding::primaries)]],
                                   device float4* light [[buffer(binding::light)]],
                                   const device serenity::contracts::LightTileEntry* tiles
                                   [[buffer(binding::light_tiles)]],
                                   device serenity::contracts::Float3* previous_positions
                                   [[buffer(binding::previous_positions)]],
                                   constant serenity::contracts::Transform* last_transforms
                                   [[buffer(binding::last_transforms)]],
                                   device serenity::contracts::MneeRequest* mnee_requests
                                   [[buffer(binding::mnee_requests)]],
                                   device atomic_uint* mnee_header [[buffer(binding::mnee_header)]],
                                   uint2 grid_place [[thread_position_in_grid]]) {
    initial_pixel<true>(frame, camera, structure, block, sphere_glows, transforms, settings, reservoirs, primaries,
                        light, tiles, previous_positions, last_transforms, mnee::Queue{mnee_requests, mnee_header},
                        grid_place);
}

// The frame's light tiles (contracts/light_tile.h), one thread an entry:
// each a light drawn by the light selection and packed as this frame places
// and lights it.
kernel void restir_pt_light_tiles(constant serenity::contracts::FrameConstants& frame [[buffer(binding::constants)]],
                                  serenity::shaders::SceneStructure structure [[buffer(binding::structure)]],
                                  constant serenity::gpu::SceneBlock& block [[buffer(binding::scene)]],
                                  constant float* sphere_glows [[buffer(binding::glows)]],
                                  constant serenity::contracts::Transform* transforms [[buffer(binding::transforms)]],
                                  device serenity::contracts::LightTileEntry* tiles [[buffer(binding::light_tiles)]],
                                  uint n [[thread_position_in_grid]]) {
    const restir_pt::Scene scene{structure, &block, transforms, sphere_glows};
    if (n >= serenity::contracts::light_tiles * serenity::contracts::light_tile_size || scene.selection().count == 0u) {
        return;
    }
    tiles[n] = restir_pt::tile_entry(scene, n, frame.frame_index);
}
