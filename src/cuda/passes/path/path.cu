// The path pass's programs (path.h): one path per pixel per frame, through
// a point in the pixel drawn anew each frame, folded into the accumulated
// image, whose mean is the frame's radiance: the passes after it read the
// image itself. Metal's path_trace (metal/passes/path/path.metal), as a ray
// generation program reading the frame's launch parameters
// (frame/traced_module.cuh).

#include <optix.h>

#include "core/contracts/camera.h"
#include "core/contracts/frame_constants.h"
#include "cuda/camera/thin_lens.cuh"
#include "cuda/film/accumulate.cuh"
#include "cuda/frame/traced_module.cuh"
#include "cuda/integrator/path.cuh"
#include "cuda/sampler/sampler.cuh"
#include "cuda/scene/scene_block.cuh"

namespace serenity {
namespace shaders {

// One pixel's path, by the path tracer (integrator/path.cuh), folded in:
// the program's body, in the namespace whose names it uses.
__device__ inline void path_pixel() {
  const uint3 index = optixGetLaunchIndex();
  const uint2 pixel = make_uint2(index.x, index.y);
  const serenity::contracts::FrameConstants& frame = launch.frame;
  const serenity::contracts::CameraData& camera = launch.camera;

  // The scene, through its block, and as this frame places and lights it
  // (scene/scene_block.cuh).
  const path::Scene scene{&launch.structures, launch.scene, launch.transforms,
                          launch.glows};

  // The camera's ray as pixel_ray() draws it (camera/thin_lens.cuh), written
  // out here as Metal's pass writes it (its kernel measured faster so): the
  // same numbers in the same order. A path graph's frames never take the
  // camera's jitter (core/frame/schedule.h, jitters), so its point is
  // always the pixel's own.
  PathNumbers numbers = path_numbers(pixel, frame.frame_index);
  const float2 within = next_numbers2(numbers);  // the point in the pixel
  // The point on the lens, drawn only through one, so a pinhole's paths
  // draw what they drew before lenses; a pinhole reads none.
  const float2 lens =
      has_lens(camera) ? next_numbers2(numbers) : make_float2(0.0f);
  const CameraRay ray = camera_ray(camera, make_float2(pixel) + within, lens,
                                   make_uint2(frame.width, frame.height));
  const float3 sample = path::radiance(scene, ray, numbers);

  const bool failed = !finite(sample);
  float4* row = reinterpret_cast<float4*>(launch.accumulated.rows +
                                          std::size_t{pixel.y} *
                                              launch.accumulated.pitch);
  row[pixel.x] =
      accumulate(row[pixel.x], sample, frame.accumulated_frames == 0u);
  count_non_finite(failed, launch.non_finite);
}

}  // namespace shaders
}  // namespace serenity

extern "C" __global__ void __raygen__path() { serenity::shaders::path_pixel(); }
