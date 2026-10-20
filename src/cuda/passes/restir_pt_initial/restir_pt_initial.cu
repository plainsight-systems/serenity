// ReSTIR PT's initial sampling's program (restir_pt_initial.h): steps 1 to 4
// of passes/restir_pt/restir_pt.cuh, one thread a traced pixel
// (restir_pt.cuh, Traced: every pixel, or the frame's field). The path tree
// along the camera's ray, resampled to one path into the pixel's path
// reservoir; the pixel's primary hit for the merge; the light the camera
// sees directly into the frame's light buffer. Metal's restir_pt_initial
// (metal/passes/restir_pt_initial/restir_pt_initial.metal), as a ray
// generation program reading the frame's launch parameters
// (frame/traced_module.cuh).

#include <optix.h>

#include "core/contracts/camera.h"
#include "core/contracts/frame_constants.h"
#include "core/contracts/light_tile.h"
#include "core/contracts/mnee_request.h"
#include "core/contracts/path_reservoir.h"
#include "core/contracts/path_reuse_settings.h"
#include "core/contracts/transform.h"
#include "cuda/frame/traced_module.cuh"
#include "cuda/integrator/mnee.cuh"
#include "cuda/passes/restir_pt/restir_pt.cuh"

namespace serenity {
namespace shaders {

// One pixel's tree (steps 1 to 4), with MNEE: its requests written on the
// frame's work list, whose light the MNEE pass adds to the light buffer
// after.
__device__ inline void initial_pixel() {
  const serenity::contracts::FrameConstants& frame = launch.frame;
  const serenity::contracts::PathReuseSettings& settings = launch.path_settings;
  const uint3 index = optixGetLaunchIndex();
  const uint2 grid_place = make_uint2(index.x, index.y);
  const restir_pt::Traced traced = restir_pt::traced(frame, settings);
  const uint2 pixel = traced.pixel(grid_place);
  if (!traced.inside(pixel)) {
    return;
  }
  const uint i = traced.slot(pixel);
  const restir_pt::Scene scene{&launch.structures, launch.scene,
                               launch.transforms, launch.glows,
                               launch.light_power};
  PathNumbers numbers;
  const CameraRay ray = restir_pt::primary_ray(
      launch.camera, pixel, frame.frame_index,
      make_uint2(frame.width, frame.height), frame_jitter(frame), numbers);
  // confidence_cap is 0 exactly in a graph without restir_pt_temporal,
  // where neither array is given (contracts/path_reuse_settings.h).
  const bool temporal = settings.confidence_cap > 0.0f;
  restir_pt::TreeGather tree = restir_pt::start_tree(
      settings, numbers, scene.shapes(),
      restir_pt::block_tile(launch.light_tiles, pixel, frame.frame_index),
      mnee::Queue{launch.mnee_requests, launch.mnee_header}, i);
  (void)path::walk_path(scene, path::walk_from(ray), numbers, tree);
  launch.path_reservoirs[i] = restir_pt::finish_tree(tree);
  launch.primaries[i] = tree.primary;
  if (temporal && tree.primary.distance > 0.0f) {
    launch.previous_positions[i] = to_packed(restir_pt::previous_position(
        tree, launch.transforms, launch.last_transforms));
  }
  launch.light[i] = make_float4(tree.seen, 0.0f);
}

}  // namespace shaders
}  // namespace serenity

extern "C" __global__ void __raygen__restir_pt_initial() {
  serenity::shaders::initial_pixel();
}
