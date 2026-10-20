#pragma once

// Axis: Camera (shader half of camera/thin_lens.h).
//
// The ray from the camera through a point of a W x H image, by the formula
// in contracts/camera.h: from a point on the lens toward where the pinhole
// ray meets the plane of focus.
//
// Translated line for line from Metal's metal/camera/thin_lens.metal.h
// (docs/research/2026-10-19-cuda-backend.md, "Phase 3's design"): its
// prose is Metal's, and a cost or a measurement it gives is the M3 Max's.

#include "core/contracts/camera.h"
#include "core/contracts/frame_constants.h"
#include "cuda/device/layout.cuh"
#include "cuda/math/vector.cuh"
#include "cuda/math/warp.cuh"
#include "cuda/sampler/sampler.cuh"

namespace serenity {
namespace shaders {

struct CameraRay {
  float3 origin;
  float3 direction;  // unit length
};

// Whether the camera has a lens, rather than a pinhole: the one test every
// caller makes (the path pass draws a lens point only through one).
__device__ inline bool has_lens(const serenity::contracts::CameraData& camera) {
  return camera.lens_radius > 0.0f;
}

// `position` is in pixels from the top-left corner of an image of `size`
// pixels: pixel (x, y)'s center is (x + 0.5, y + 0.5). `lens` is a point in
// the unit square, mapped to the lens's disk (metal/camera/warp.metal.h,
// concentric_disk); unread for a pinhole, whose ray is the pinhole's
// exactly.
__device__ inline CameraRay camera_ray(
    const serenity::contracts::CameraData& camera, float2 position, float2 lens,
    uint2 size) {
  const float sx = 2.0f * position.x / float(size.x) - 1.0f;
  const float sy = 1.0f - 2.0f * position.y / float(size.y);
  const float3 forward = to_float3(camera.forward);
  const float3 right = to_float3(camera.right);
  const float3 up = to_float3(camera.up);
  const float3 d = forward + sx * right + sy * up;
  const float3 origin = to_float3(camera.origin);
  if (!has_lens(camera)) {
    return CameraRay{origin, normalize(d)};
  }
  const float3 focus = origin + (camera.focus_distance / dot(d, forward)) * d;
  const float2 on_disk = concentric_disk(lens) * camera.lens_radius;
  const float3 start =
      origin + on_disk.x * normalize(right) + on_disk.y * normalize(up);
  return CameraRay{start, normalize(focus - start)};
}

// Where in its pixel a frame's camera rays pass (contracts/frame_constants.h):
// the frame's jitter, the same offset from every pixel's center, where the
// frame takes it; else each pixel's own point, drawn from its numbers.
struct PixelJitter {
  bool on;
  float2 offset;  // pixels, from the pixel's center
};

__device__ inline PixelJitter frame_jitter(
    const serenity::contracts::FrameConstants& frame) {
  return PixelJitter{frame.jittered != 0u,
                     make_float2(frame.jitter_x, frame.jitter_y)};
}

// The camera's ray through pixel `pixel` of an image of `size`, drawn from
// `numbers`, a pixel's path numbers at their start
// (metal/camera/sampler.metal.h, path_numbers): the point in the pixel, the
// frame's jitter where `jitter` is on (drawing nothing for it) and else drawn,
// then the point on the lens, drawn only through one, so a pinhole's paths draw
// what they drew before lenses and an unjittered frame's what it drew before
// the jitter. The one ray every light pass, every shift's destination and the
// features take a pixel's path from (ES.3: they must agree to the bit);
// `numbers` is left after what it drew, for the path to go on drawing. The path
// pass writes the same draw out instead, for its kernel's speed
// (passes/path/path.cu).
__device__ inline CameraRay pixel_ray(
    const serenity::contracts::CameraData& camera, uint2 pixel, uint2 size,
    PixelJitter jitter, PathNumbers& numbers) {
  // The point in the pixel. Which of the two is the frame's, the same for
  // every thread of a dispatch, so the branch costs no divergence (GPU.4).
  const float2 within =
      jitter.on ? make_float2(0.5f) + jitter.offset : next_numbers2(numbers);
  const float2 lens =
      has_lens(camera) ? next_numbers2(numbers) : make_float2(0.0f);
  return camera_ray(camera, make_float2(pixel) + within, lens, size);
}

// The inverse, for ReSTIR PT's temporal search and the features' motion
// (metal/passes/restir_pt/restir_pt.metal.h, previous_pixel;
// metal/passes/features/features.metal).
//
// camera_depth: how far in front of the camera `point` is, along forward:
// dot(point - origin, forward). camera_pixel: where the pinhole ray through
// `point` crosses the image of `size` pixels, in pixels from its top-left
// corner, by contracts/camera.h's formula solved for (px, py); the lens
// ignored, as a point's place in the image is its pinhole ray's, the lens
// blurring it about that place. Meaningful only where camera_depth > 0.
__device__ inline float camera_depth(
    const serenity::contracts::CameraData& camera, float3 point) {
  return dot(point - to_float3(camera.origin), to_float3(camera.forward));
}

// direction_pixel: where the pinhole ray along direction `v` (any length,
// pointing in front of the camera) crosses the image: camera_pixel's
// formula on a direction, so a point at infinity, the sky, is placed by
// the camera's orientation alone (the features' motion, core/contracts/
// features.h). camera_pixel is direction_pixel of the point less the
// origin.
__device__ inline float2 direction_pixel(
    const serenity::contracts::CameraData& camera, float3 v, uint2 size) {
  // d = v / depth has dot(d, forward) = 1, as the ray formula's d, whose
  // right and up parts are sx right and sy up; right, up and forward are
  // square to one another (contracts/camera.h).
  const float3 d = v / dot(v, to_float3(camera.forward));
  const float3 right = to_float3(camera.right);
  const float3 up = to_float3(camera.up);
  const float sx = dot(d, right) / dot(right, right);
  const float sy = dot(d, up) / dot(up, up);
  return make_float2((sx + 1.0f) * 0.5f * float(size.x),
                     (1.0f - sy) * 0.5f * float(size.y));
}

__device__ inline float2 camera_pixel(
    const serenity::contracts::CameraData& camera, float3 point, uint2 size) {
  return direction_pixel(camera, point - to_float3(camera.origin), size);
}

}  // namespace shaders
}  // namespace serenity
