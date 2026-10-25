// The features pass (features.h): one thread a pixel, the camera's ray the
// raster pass drew there (visibility.h), its primary hit rebuilt from the
// visibility buffer, followed through glass and mirrors to the first
// surface that is neither; the four feature images of contract 17
// (core/contracts/features.h, which states every rule this file keeps).

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "core/contracts/camera.h"
#include "core/contracts/features.h"
#include "core/contracts/frame_constants.h"
#include "core/contracts/medium.h"
#include "core/contracts/surface_interaction.h"
#include "core/contracts/transform.h"
#include "metal/acceleration/trace.metal.h"
#include "metal/camera/thin_lens.metal.h"
#include "metal/integrator/surface.metal.h"
#include "metal/light_selection/uniform_light.metal.h"
#include "metal/materials/dielectric.metal.h"
#include "metal/math/transform.metal.h"
#include "metal/passes/bindings.h"
#include "metal/passes/features/visibility.h"
#include "metal/sampler/sampler.metal.h"
#include "metal/scene/scene_block.metal.h"

using namespace metal;
using namespace serenity::shaders;
namespace binding = serenity::bindings::features;
namespace contracts = serenity::contracts;

namespace {

// The scene as the path tracer views it (metal/integrator/path.metal.h).
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

// The surface `hit` along `along` is, as the features see it.
inline Reached reached(Scene scene, Hit hit, metal::raytracing::ray along) {
  if (!hit.found) {
    return Reached{false, float3(0.0f), contracts::SurfaceInteraction{}, false,
                   contracts::Bsdf{}};
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

// The camera's rays and their mirrors' see the fireflies: every shape
// (contract 19, everything).
inline Reached reach(Scene scene, metal::raytracing::ray along) {
  return reached(scene,
                 trace(scene.structure, scene.shapes(), along,
                       contracts::RayMask::everything),
                 along);
}

// A conductor smooth enough to be a mirror to the features (contract 17).
inline bool mirror(contracts::Bsdf bsdf) {
  return bsdf.kind == contracts::BsdfKind::conductor &&
         bsdf.alpha <= contracts::feature_mirror_alpha;
}

inline bool delta(contracts::Bsdf bsdf) {
  return bsdf.kind == contracts::BsdfKind::dielectric || mirror(bsdf);
}

// A surface that is not delta, or a light, as the features see it: its
// albedo (a light's 1) and shading normal.
inline Found settled(Reached at) {
  if (at.light) {
    return Found{float3(1.0f), to_float3(at.surface.shading_normal)};
  }
  // Lambert and the coat's base: their color; a rough conductor: f0. No
  // default: the kinds a Bsdf resolves to are all here.
  float3 albedo = float3(1.0f);
  switch (at.bsdf.kind) {
    case contracts::BsdfKind::lambert:
    case contracts::BsdfKind::coated:
    case contracts::BsdfKind::conductor:
      albedo = metal::saturate(to_float3(at.bsdf.color));
      break;
    case contracts::BsdfKind::none:
    case contracts::BsdfKind::dielectric:
      break;
  }
  return Found{albedo, to_float3(at.bsdf.normal)};
}

// The medium a ray is in after crossing `surface` (as the path tracer's step
// 6 decides it, metal/integrator/path.metal.h).
inline uint medium_beyond(contracts::SurfaceInteraction surface) {
  return (surface.flags & contracts::arrived_from_outside) != 0u
             ? surface.interior
             : contracts::no_medium;
}

// Where `along`, leaving `from` in `medium`, settles: through each delta
// surface by its heavier branch (refraction unless Fresnel's F > 1/2 or the
// ray is totally reflected; a mirror's reflection), its absorption and a
// mirror's f0 weighing the albedo, at most feature_delta_hops of them; the
// sky, or the hops run out, as albedo 1 and no normal.
inline Found follow(Scene scene, metal::raytracing::ray along, uint medium,
                    float3 from) {
  float3 weight = float3(1.0f);
  for (uint hop = 0; hop <= contracts::feature_delta_hops; ++hop) {
    const Reached at = reach(scene, along);
    if (!at.found) {
      return Found{weight, float3(0.0f)};
    }
    weight *=
        transmittance(scene.media(), medium, metal::distance(from, at.point));
    if (at.light || !delta(at.bsdf)) {
      const Found found = settled(at);
      return Found{weight * found.albedo, found.normal};
    }
    float3 next = float3(0.0f);
    if (mirror(at.bsdf)) {
      weight *= metal::saturate(to_float3(at.bsdf.color));
      next = metal::reflect(along.direction, to_float3(at.bsdf.normal));
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
  return Found{weight, float3(0.0f)};
}

// What the camera's ray along `direction` settles on, from the primary
// surface `at` it reached (contract 17): `at` itself unless it is delta; a
// mirror's reflection, weighed by its f0; glass's path split in two, each
// half followed on and blended by Fresnel (Open Image Denoise's rule).
inline Found replaced(Scene scene, Reached at, float3 direction) {
  if (at.light || !delta(at.bsdf)) {
    return settled(at);
  }
  if (mirror(at.bsdf)) {
    const Found seen = follow(
        scene,
        leave(at.surface, metal::reflect(direction, to_float3(at.bsdf.normal))),
        contracts::no_medium, at.point);
    return Found{metal::saturate(to_float3(at.bsdf.color)) * seen.albedo,
                 seen.normal};
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

}  // namespace

kernel void features(
    constant contracts::FrameConstants& frame [[buffer(binding::constants)]],
    constant contracts::CameraData& camera [[buffer(binding::camera)]],
    serenity::shaders::SceneStructure structure [[buffer(binding::structure)]],
    constant serenity::gpu::SceneBlock& block [[buffer(binding::scene)]],
    constant float* sphere_glows [[buffer(binding::glows)]],
    constant contracts::Transform* transforms [[buffer(binding::transforms)]],
    constant contracts::FeatureConstants& constants
    [[buffer(binding::feature_constants)]],
    constant contracts::Transform* last_transforms
    [[buffer(binding::last_transforms)]],
    texture2d<float, access::write> albedo [[texture(binding::albedo)]],
    texture2d<float, access::write> normal [[texture(binding::normal)]],
    texture2d<float, access::write> depth [[texture(binding::depth)]],
    texture2d<float, access::write> motion [[texture(binding::motion)]],
    texture2d<uint, access::read> visibility [[texture(binding::visibility)]],
    constant serenity::gpu::MeshBlock& meshes [[buffer(binding::mesh_block)]],
    uint2 pixel [[thread_position_in_grid]]) {
  if (pixel.x >= frame.width || pixel.y >= frame.height) {
    return;
  }
  const uint2 size = uint2(frame.width, frame.height);
  const Scene scene{structure, &block, transforms, sphere_glows};
  // The ray the raster pass drew at this pixel: through its center plus
  // the frame's jitter where the frame takes it (at the light size, the
  // light passes' ray), else its center (contract 17), a pinhole's.
  const PixelJitter jitter = frame_jitter(frame);
  const CameraRay ray = camera_ray(
      camera, float2(pixel) + 0.5f + (jitter.on ? jitter.offset : float2(0.0f)),
      float2(0.0f), size);
  const metal::raytracing::ray along(ray.origin, ray.direction, 0.0f,
                                     unbounded);
  const bool previous = constants.previous != 0u;

  // Its primary hit, the triangle the rasterizer found nearest, rebuilt as
  // trace() builds one (acceleration/trace.metal.h, hit_of()).
  const uint2 drawn = visibility.read(pixel).xy;
  const Reached at =
      drawn.x == serenity::bindings::visibility::sky_shape
          ? reached(scene, Hit{false, 0.0f, 0u, MeshPoint{}, 0u}, along)
          : reached(scene,
                    hit_of(meshes, scene.shapes(), along, drawn.x, drawn.y),
                    along);
  if (!at.found) {
    // The sky: albedo 1, no normal, no depth; its motion the camera's
    // turn alone, of the direction at infinity.
    const float2 moved =
        previous
            ? direction_pixel(constants.previous_camera, ray.direction, size) -
                  direction_pixel(camera, ray.direction, size)
            : float2(0.0f);
    albedo.write(float4(1.0f), pixel);
    normal.write(float4(0.0f), pixel);
    depth.write(float4(0.0f), pixel);
    motion.write(float4(moved, 0.0f, 0.0f), pixel);
    return;
  }

  // The primary surface's depth and motion: its point now, and where its
  // shape held it last frame (its object-space point placed by last
  // frame's transform, contract 10), each through its frame's camera.
  const float view_z = camera_depth(camera, at.point);
  float4 moved = float4(0.0f);
  if (previous) {
    const float3 before =
        transform_to_world(last_transforms[at.surface.primitive],
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
            : float2(0.0f);
    moved = float4(offset, view_z_before - view_z, 0.0f);
  }

  const Found found = replaced(scene, at, ray.direction);
  albedo.write(float4(found.albedo, 1.0f), pixel);
  normal.write(float4(found.normal, 0.0f), pixel);
  depth.write(float4(view_z), pixel);
  motion.write(moved, pixel);
}
