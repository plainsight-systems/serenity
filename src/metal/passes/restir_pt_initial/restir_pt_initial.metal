// ReSTIR PT's initial sampling (restir_pt_initial.h): steps 1 to 4 of
// metal/passes/restir_pt/restir_pt.metal.h, one thread a pixel. The path
// tree along the camera's ray, resampled to one path into the pixel's path
// reservoir; the pixel's primary hit for the merge; the light the camera
// sees directly into the frame's light buffer.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/camera.h"
#include "core/contracts/frame_constants.h"
#include "core/contracts/light_tile.h"
#include "core/contracts/path_reservoir.h"
#include "core/contracts/path_reuse_settings.h"
#include "core/contracts/transform.h"
#include "metal/passes/bindings.h"
#include "metal/passes/restir_pt/restir_pt.metal.h"

using namespace metal;
using namespace metal::raytracing;
using namespace serenity::shaders;
namespace binding = serenity::bindings::restir_pt;

kernel void restir_pt_initial(constant serenity::contracts::FrameConstants& frame [[buffer(binding::constants)]],
                              constant serenity::contracts::CameraData& camera [[buffer(binding::camera)]],
                              primitive_acceleration_structure structure [[buffer(binding::structure)]],
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
                              uint2 pixel [[thread_position_in_grid]]) {
    if (pixel.x >= frame.width || pixel.y >= frame.height) {
        return;
    }
    const uint i = pixel.y * frame.width + pixel.x;
    const restir_pt::Scene scene{structure, &block, transforms, sphere_glows};
    PathNumbers numbers;
    const CameraRay ray = restir_pt::primary_ray(camera, pixel, frame.frame_index, uint2(frame.width, frame.height),
                                                 numbers);
    // confidence_cap is 0 exactly in a graph without restir_pt_temporal,
    // where neither array is bound (contracts/path_reuse_settings.h).
    const bool temporal = settings.confidence_cap > 0.0f;
    restir_pt::TreeGather tree =
        restir_pt::start_tree(settings, numbers, scene.shapes(), restir_pt::block_tile(tiles, pixel, frame.frame_index),
                              temporal ? last_transforms : nullptr);
    (void)path::walk_path<path::Start::camera, path::Stop::at_end>(scene, path::walk_from(ray), Shading{}, 0u,
                                                                   numbers, tree);
    reservoirs[i] = restir_pt::finish_tree(tree);
    primaries[i] = tree.primary;
    if (temporal) {
        previous_positions[i] = to_packed(tree.previous_position);
    }
    light[i] = float4(tree.seen, 0.0f);
}

// The frame's light tiles (contracts/light_tile.h), one thread an entry:
// each a light drawn by the light selection and packed as this frame places
// and lights it.
kernel void restir_pt_light_tiles(constant serenity::contracts::FrameConstants& frame [[buffer(binding::constants)]],
                                  primitive_acceleration_structure structure [[buffer(binding::structure)]],
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
