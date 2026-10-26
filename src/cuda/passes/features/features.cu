// The features pass's program (features.h): one thread a pixel, the
// camera's ray the light passes trace there, followed through glass and
// mirrors to the first surface that is neither; the four feature images of
// contract 17 (core/contracts/features.h, which states every rule this file
// keeps). Metal's features kernel (metal/passes/features/features.metal),
// as a ray generation program reading the frame's launch parameters
// (frame/traced_module.cuh). The images are written as Metal writes its
// RGBA16Float textures, each float rounded to the nearest half, ties to
// even; the depth as a float.

#include <cuda_fp16.h>
#include <optix.h>

#include "core/contracts/bsdf.h"
#include "core/contracts/camera.h"
#include "core/contracts/features.h"
#include "core/contracts/frame_constants.h"
#include "core/contracts/medium.h"
#include "core/contracts/surface_interaction.h"
#include "core/contracts/transform.h"
#include "cuda/acceleration/trace.cuh"
#include "cuda/camera/thin_lens.cuh"
#include "cuda/frame/traced_module.cuh"
#include "cuda/integrator/surface.cuh"
#include "cuda/light_selection/uniform_light.cuh"
#include "cuda/materials/dielectric.cuh"
#include "cuda/math/transform.cuh"
#include "cuda/sampler/sampler.cuh"
#include "cuda/scene/scene_block.cuh"

namespace serenity {
namespace shaders {
namespace {

namespace contracts = serenity::contracts;

// A pixel of an RGBA16Float image: each channel rounded to the nearest
// half, ties to even, Metal's conversion on a texture write.
__device__ inline ushort4 half4_bits(float4 v) {
  return make_ushort4(__half_as_ushort(__float2half_rn(v.x)),
                      __half_as_ushort(__float2half_rn(v.y)),
                      __half_as_ushort(__float2half_rn(v.z)),
                      __half_as_ushort(__float2half_rn(v.w)));
}

// The scene as the path tracer views it (integrator/path.cuh).
using Scene = SceneView<UniformLight>;

// What a feature ray settles on (contract 17): the albedo, weighed by what
// the delta surfaces before it pass on, and the normal.
struct Found {
  float3 albedo;
  float3 normal;
};

// A surface the ray reached: where, and what it is.
struct Reached {
  bool found;  // false: the sky
  float3 point;
  contracts::SurfaceInteraction surface;
  bool light;            // a light's shape: emits, scatters nothing
  contracts::Bsdf bsdf;  // resolved unless `light` or not found
};

// The camera's rays and their mirrors' see the fireflies: every shape
// (contract 19, everything).
__device__ inline Reached reach(Scene scene, Ray along) {
  const Hit hit = trace(scene.structure, scene.shapes(), along,
                        contracts::RayMask::everything);
  if (!hit.found) {
    return Reached{false, make_float3(0.0f), contracts::SurfaceInteraction{},
                   false, contracts::Bsdf{}};
  }
  const float3 point = hit.at.point;
  const contracts::SurfaceInteraction surface = surface_interaction(
      scene.shapes(), hit.primitive, hit.at, along.direction);
  if (light_at(scene.lights(), hit.primitive).is_light) {
    return Reached{true, point, surface, true, contracts::Bsdf{}};
  }
  return Reached{true, point, surface, false,
                 resolve_bsdf(scene.materials(), scene.textures(), surface)};
}

// A conductor smooth enough to be a mirror to the features (contract 17).
__device__ inline bool mirror(contracts::Bsdf bsdf) {
  return bsdf.kind == contracts::BsdfKind::conductor &&
         bsdf.alpha <= contracts::feature_mirror_alpha;
}

__device__ inline bool delta(contracts::Bsdf bsdf) {
  return bsdf.kind == contracts::BsdfKind::dielectric || mirror(bsdf);
}

// A surface that is not delta, or a light, as the features see it: its
// albedo (a light's 1) and shading normal.
__device__ inline Found settled(Reached at) {
  if (at.light) {
    return Found{make_float3(1.0f), to_float3(at.surface.shading_normal)};
  }
  // Lambert and the coat's base: their color; a rough conductor: f0. No
  // default: the kinds a Bsdf resolves to are all here, each returning,
  // so a kind left out traps after (device/layout.cuh, unhandled_kind).
  const float3 normal = to_float3(at.bsdf.normal);
  switch (at.bsdf.kind) {
    case contracts::BsdfKind::lambert:
    case contracts::BsdfKind::coated:
    case contracts::BsdfKind::conductor:
      return Found{saturate(to_float3(at.bsdf.color)), normal};
    case contracts::BsdfKind::none:
    case contracts::BsdfKind::dielectric:
      return Found{make_float3(1.0f), normal};
  }
  unhandled_kind();
  return Found{make_float3(1.0f), normal};
}

// The medium a ray is in after crossing `surface` (as the path tracer's step
// 6 decides it, integrator/path.cuh).
__device__ inline uint medium_beyond(contracts::SurfaceInteraction surface) {
  return (surface.flags & contracts::arrived_from_outside) != 0u
             ? surface.interior
             : contracts::no_medium;
}

// Where `along`, leaving `from` in `medium`, settles: through each delta
// surface by its heavier branch (refraction unless Fresnel's F > 1/2 or the
// ray is totally reflected; a mirror's reflection), its absorption and a
// mirror's f0 weighing the albedo, at most feature_delta_hops of them; the
// sky, or the hops run out, as albedo 1 and no normal.
__device__ inline Found follow(Scene scene, Ray along, uint medium,
                               float3 from) {
  float3 weight = make_float3(1.0f);
  for (uint hop = 0; hop <= contracts::feature_delta_hops; ++hop) {
    const Reached at = reach(scene, along);
    if (!at.found) {
      return Found{weight, make_float3(0.0f)};
    }
    weight *= transmittance(scene.media(), medium, distance(from, at.point));
    if (at.light || !delta(at.bsdf)) {
      const Found found = settled(at);
      return Found{weight * found.albedo, found.normal};
    }
    float3 next = make_float3(0.0f);
    if (mirror(at.bsdf)) {
      weight *= saturate(to_float3(at.bsdf.color));
      next = reflect(along.direction, to_float3(at.bsdf.normal));
    } else {
      const Boundary boundary = dielectric_boundary(
          at.bsdf.ior, along.direction, to_float3(at.bsdf.normal));
      const bool reflects =
          boundary.total_internal || boundary.reflectance > 0.5f;
      if (!reflects) {
        medium = medium_beyond(at.surface);
      }
      next = reflects ? boundary.reflected : boundary.refracted;
    }
    from = at.point;
    along = leave(at.surface, next);
  }
  return Found{weight, make_float3(0.0f)};
}

// What the camera's ray along `direction` settles on, from the primary
// surface `at` it reached (contract 17): `at` itself unless it is delta; a
// mirror's reflection, weighed by its f0; glass's path split in two, each
// half followed on and blended by Fresnel (Open Image Denoise's rule).
__device__ inline Found replaced(Scene scene, Reached at, float3 direction) {
  if (at.light || !delta(at.bsdf)) {
    return settled(at);
  }
  if (mirror(at.bsdf)) {
    const Found seen = follow(
        scene, leave(at.surface, reflect(direction, to_float3(at.bsdf.normal))),
        contracts::no_medium, at.point);
    return Found{saturate(to_float3(at.bsdf.color)) * seen.albedo, seen.normal};
  }
  const Boundary boundary =
      dielectric_boundary(at.bsdf.ior, direction, to_float3(at.bsdf.normal));
  const Found reflected = follow(scene, leave(at.surface, boundary.reflected),
                                 contracts::no_medium, at.point);
  const Found transmitted =
      boundary.total_internal
          ? reflected
          : follow(scene, leave(at.surface, boundary.refracted),
                   medium_beyond(at.surface), at.point);
  const float f = boundary.reflectance;
  return Found{f * reflected.albedo + (1.0f - f) * transmitted.albedo,
               f * reflected.normal + (1.0f - f) * transmitted.normal};
}

// One pixel's features.
__device__ inline void features_pixel() {
  const contracts::FrameConstants& frame = launch.frame;
  const contracts::CameraData& camera = launch.camera;
  const uint3 index = optixGetLaunchIndex();
  const uint2 pixel = make_uint2(index.x, index.y);
  if (pixel.x >= frame.width || pixel.y >= frame.height) {
    return;
  }
  const uint2 size = make_uint2(frame.width, frame.height);
  const uint at_pixel = pixel.y * frame.width + pixel.x;
  const Scene scene{&launch.structures, launch.scene, launch.transforms,
                    launch.glows};
  // The light passes' ray at this pixel and frame: their numbers, drawn in
  // their order (passes/path/path.cu).
  PathNumbers numbers = path_numbers(pixel, frame.frame_index);
  const CameraRay ray =
      pixel_ray(camera, pixel, size, frame_jitter(frame), numbers);
  const Ray along = make_ray(ray.origin, ray.direction, 0.0f, unbounded);
  const serenity::contracts::FeatureConstants& constants =
      launch.feature_constants;
  const bool previous = constants.previous != 0u;

  const Reached at = reach(scene, along);
  if (!at.found) {
    // The sky: albedo 1, no normal, no depth; its motion the camera's
    // turn alone, of the direction at infinity.
    const float2 moved =
        previous
            ? direction_pixel(constants.previous_camera, ray.direction, size) -
                  direction_pixel(camera, ray.direction, size)
            : make_float2(0.0f);
    launch.albedo[at_pixel] = half4_bits(make_float4(1.0f));
    launch.normal[at_pixel] = half4_bits(make_float4(0.0f));
    launch.depth[at_pixel] = 0.0f;
    launch.motion[at_pixel] =
        half4_bits(make_float4(moved.x, moved.y, 0.0f, 0.0f));
    return;
  }

  // The primary surface's depth and motion: its point now, and where its
  // shape held it last frame (its object-space point placed by last
  // frame's transform, contract 10), each through its frame's camera.
  const float view_z = camera_depth(camera, at.point);
  float4 moved = make_float4(0.0f);
  if (previous) {
    const float3 before =
        transform_to_world(launch.feature_last_transforms[at.surface.primitive],
                           to_float3(at.surface.object_position));
    const float view_z_before = camera_depth(constants.previous_camera, before);
    // A point behind last frame's camera has no place in its image:
    // dx, dy 0 and dz its depth then less now's, below -view_z, which a
    // surface in front of both cameras never has, so a depth test
    // refuses it (contract 17).
    const float2 offset =
        view_z_before > 0.0f
            ? camera_pixel(constants.previous_camera, before, size) -
                  camera_pixel(camera, at.point, size)
            : make_float2(0.0f);
    moved = make_float4(offset.x, offset.y, view_z_before - view_z, 0.0f);
  }

  const Found found = replaced(scene, at, ray.direction);
  launch.albedo[at_pixel] = half4_bits(make_float4(found.albedo, 1.0f));
  launch.normal[at_pixel] = half4_bits(make_float4(found.normal, 0.0f));
  launch.depth[at_pixel] = view_z;
  launch.motion[at_pixel] = half4_bits(moved);
}

}  // namespace
}  // namespace shaders
}  // namespace serenity

extern "C" __global__ void __raygen__features() {
  serenity::shaders::features_pixel();
}
