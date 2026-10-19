// The checkerboard's fill (checkerboard_fill.h, which states every rule
// this file keeps): one thread a slot of the packed grid (contract 20), the
// slot's traced pixel copied into the hand-off and its other pixel filled,
// from the samples the frame before traced where its surface was, else
// from the traced pixels beside it.

#include <metal_stdlib>

#include "core/contracts/checkerboard.h"
#include "core/contracts/frame_constants.h"
#include "metal/passes/bindings.h"

using namespace metal;
namespace binding = serenity::bindings::checkerboard_fill;
namespace contracts = serenity::contracts;

namespace {

// How near another sample's depth must be to a pixel's for both to be one
// surface (ES.45): within depth_tolerance of the larger, 2%, well under
// the gap between a firefly and the surface behind it; and beyond that by
// what the surface's depth changes over jitter_reach pixels, since each
// frame's sample is at its own jittered point in its pixel and lens, so
// two frames' samples of one pixel are up to a pixel apart on each axis,
// and at the far table, seen at a grazing angle and out of focus, a pixel
// spans more than 2% of its depth (SVGF's test, scaled by the depth's
// screen-space slope). The slope is the shallower one-sided difference to
// the pixel's neighbors on each axis, so a silhouette, where one side is
// another surface, does not widen it.
constant constexpr float depth_tolerance = 0.02f;
constant constexpr float jitter_reach = 1.0f;

// What the fill reads: this frame's light, depth and motion, and the frame
// before's history, its light with its depth in a.
struct Images {
  texture2d<float, access::read> accumulated;
  texture2d<float, access::read> previous_light;
  texture2d<float, access::read> depth;
  texture2d<float, access::read> motion;
  uint2 size;
};

bool inside(int2 p, uint2 size) {
  return all(p >= int2(0)) && all(p < int2(size));
}

// Whether a sample at view depth `other` is of the surface a pixel sees at
// `depth`, whose depth changes `slope` a pixel: both the sky (depth 0), or
// both surfaces within depth_tolerance of the larger and jitter_reach
// times the slope.
bool same_surface(float depth, float other, float slope) {
  if (!(depth > 0.0f) || !(other > 0.0f)) {
    return depth == 0.0f && other == 0.0f;
  }
  return abs(depth - other) <=
         depth_tolerance * max(depth, other) + jitter_reach * slope;
}

// How much the depth `depth` at `pixel` changes a pixel (above): on each
// axis the smaller of the differences to the two neighbors, the larger of
// the two axes; a neighbor outside the image or with no surface (the sky)
// gives none on its side. 0 for the sky.
float depth_slope(Images images, uint2 pixel, float depth) {
  if (!(depth > 0.0f)) {
    return 0.0f;
  }
  constexpr int2 axes[2] = {int2(1, 0), int2(0, 1)};
  float slope = 0.0f;
  for (uint a = 0u; a < 2u; ++a) {
    float shallower = INFINITY;
    for (int side = -1; side <= 1; side += 2) {
      const int2 p = int2(pixel) + side * axes[a];
      if (!inside(p, images.size)) {
        continue;
      }
      const float d = images.depth.read(uint2(p)).r;
      if (d > 0.0f) {
        shallower = min(shallower, abs(d - depth));
      }
    }
    if (isfinite(shallower)) {
      slope = max(slope, shallower);
    }
  }
  return slope;
}

// The light the frame before traced where the pixel's surface was, the
// point `at` (in pixels, a pixel's center at + 0.5), at view depth
// `expected`: of the four pixels around `at` (the bilinear footprint), the
// two frame `frame` - 1 traced, each kept where its depth then is
// `expected`'s surface's, weighed by its bilinear weight and the kept ones'
// weights renormalized. Into `light`; false where neither is kept.
bool reprojected(Images images, float2 at, float expected, float slope,
                 uint frame, thread float3& light) {
  const float2 f = at - 0.5f;  // texel centers on whole numbers
  const float2 corner = floor(f);
  const float2 w = f - corner;
  float3 sum = float3(0.0f);
  float total = 0.0f;
  for (uint n = 0u; n < 4u; ++n) {
    const int2 offset = int2(int(n & 1u), int(n >> 1u));
    const int2 texel = int2(corner) + offset;
    if (!inside(texel, images.size) ||
        !contracts::checkerboard_traced(uint(texel.x), uint(texel.y),
                                        frame - 1u)) {
      continue;
    }
    const float4 then =
        images.previous_light.read(uint2(texel));  // rgb the light, a the depth
    if (!same_surface(expected, then.a, slope)) {
      continue;
    }
    const float weight =
        (offset.x != 0 ? w.x : 1.0f - w.x) * (offset.y != 0 ? w.y : 1.0f - w.y);
    sum += weight * then.rgb;
    total += weight;
  }
  if (!(total > 0.0f)) {
    return false;
  }
  light = sum / total;
  return true;
}

// The untraced pixel `pixel`'s light in frame `frame` (checkerboard_fill.h).
float3 filled(Images images, uint2 pixel, uint frame, bool history) {
  const float depth = images.depth.read(pixel).r;
  const float slope = depth_slope(images, pixel, depth);
  if (history) {
    // Where the surface was: the pixel's center moved by its motion
    // (contract 17: old = new + motion, in pixels), at the depth the
    // motion predicts; a point behind last frame's camera (a predicted
    // depth not above 0 for a surface) has no place there.
    const float4 motion = images.motion.read(pixel);
    const float expected = depth == 0.0f ? 0.0f : depth + motion.z;
    float3 light;
    if ((depth == 0.0f || expected > 0.0f) &&
        reprojected(images, float2(pixel) + 0.5f + motion.xy, expected, slope,
                    frame, light)) {
      return light;
    }
  }
  // The traced pixels beside it, this frame's: those of its surface, else
  // all of them.
  constexpr int2 beside[4] = {int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1)};
  float3 same = float3(0.0f);
  float3 any = float3(0.0f);
  uint same_count = 0u;
  uint any_count = 0u;
  for (uint n = 0u; n < 4u; ++n) {
    const int2 p = int2(pixel) + beside[n];
    if (!inside(p, images.size)) {
      continue;
    }
    const uint2 q = uint2(p);
    const float3 light = images.accumulated.read(q).rgb;
    any += light;
    ++any_count;
    if (same_surface(depth, images.depth.read(q).r, slope)) {
      same += light;
      ++same_count;
    }
  }
  if (same_count != 0u) {
    return same / float(same_count);
  }
  return any_count != 0u ? any / float(any_count) : float3(0.0f);
}

// A pixel's light into the hand-off, and with its depth into the history
// the next frame's fill reads (metal/frame/checkerboard_images.h).
void hand_off(texture2d<float, access::write> light,
              texture2d<float, access::write> history, Images images,
              uint2 pixel, float4 value) {
  light.write(value, pixel);
  history.write(float4(value.rgb, images.depth.read(pixel).r), pixel);
}

}  // namespace

kernel void checkerboard_fill(
    constant contracts::FrameConstants& frame [[buffer(binding::constants)]],
    constant contracts::CheckerboardConstants& own
    [[buffer(binding::fill_constants)]],
    texture2d<float, access::read> accumulated
    [[texture(binding::accumulated)]],
    texture2d<float, access::read> previous_light
    [[texture(binding::previous_light)]],
    texture2d<float, access::read> depth [[texture(binding::depth)]],
    texture2d<float, access::read> motion [[texture(binding::motion)]],
    texture2d<float, access::write> light [[texture(binding::light)]],
    texture2d<float, access::write> history [[texture(binding::history)]],
    uint2 slot [[thread_position_in_grid]]) {
  const uint2 size = uint2(frame.width, frame.height);
  if (slot.y >= size.y || slot.x >= contracts::checkerboard_width(size.x)) {
    return;
  }
  const Images images{accumulated, previous_light, depth, motion, size};
  const uint traced_x =
      contracts::checkerboard_x(slot.x, slot.y, frame.frame_index);
  const uint other_x = traced_x ^ 1u;  // the slot's other pixel
  if (traced_x < size.x) {
    const uint2 pixel = uint2(traced_x, slot.y);
    hand_off(light, history, images, pixel, accumulated.read(pixel));
  }
  if (other_x < size.x) {
    const uint2 pixel = uint2(other_x, slot.y);
    hand_off(light, history, images, pixel,
             float4(filled(images, pixel, frame.frame_index, own.history != 0u),
                    0.0f));
  }
}
