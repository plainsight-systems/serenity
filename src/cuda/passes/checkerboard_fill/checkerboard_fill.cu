// The checkerboard's fill (checkerboard_fill.h): Metal's checkerboard_fill
// (metal/passes/checkerboard_fill/checkerboard_fill.metal, whose comments
// give the test and the alternatives measured), a kernel, one thread a slot
// of the packed grid (contract 20), taking the frame's launch parameters by
// address (frame/frame_launch.h). Every read is exact, at a pixel: the
// pitched images' rows (cuda/device/images.h) and the features' half floats
// widened, where Metal reads its textures without a sampler.

#include <cuda_fp16.h>

#include "core/contracts/checkerboard.h"
#include "core/contracts/frame_constants.h"
#include "cuda/frame/frame_launch.h"
#include "cuda/math/vector.cuh"

namespace serenity {
namespace shaders {
namespace {

namespace contracts = serenity::contracts;

// How near another sample's depth must be to a pixel's for both to be one
// surface (ES.45): Metal's (checkerboard_fill.metal says why each).
inline constexpr float depth_tolerance = 0.02f;
inline constexpr float jitter_reach = 1.0f;

// What the fill reads: this frame's light, depth and motion, and the frame
// before's history, its light with its depth in w.
struct Images {
  serenity::cuda::PitchedRows accumulated;
  serenity::cuda::PitchedRows previous_light;
  const float* depth;
  const ushort4* motion;
  uint2 size;
};

__device__ inline float4 read_rows(serenity::cuda::PitchedRows image,
                                   uint2 pixel) {
  return reinterpret_cast<const float4*>(image.rows + std::uint64_t{pixel.y} *
                                                          image.pitch)[pixel.x];
}

__device__ inline void write_rows(serenity::cuda::PitchedRows image,
                                  uint2 pixel, float4 value) {
  reinterpret_cast<float4*>(image.rows + std::uint64_t{pixel.y} *
                                             image.pitch)[pixel.x] = value;
}

__device__ inline float read_depth(Images images, uint2 pixel) {
  return images.depth[pixel.y * images.size.x + pixel.x];
}

__device__ inline float4 read_motion(Images images, uint2 pixel) {
  const ushort4 bits = images.motion[pixel.y * images.size.x + pixel.x];
  return make_float4(__half2float(__ushort_as_half(bits.x)),
                     __half2float(__ushort_as_half(bits.y)),
                     __half2float(__ushort_as_half(bits.z)),
                     __half2float(__ushort_as_half(bits.w)));
}

__device__ inline bool inside(int x, int y, uint2 size) {
  return x >= 0 && y >= 0 && x < int(size.x) && y < int(size.y);
}

// Whether a sample at view depth `other` is of the surface a pixel sees at
// `depth`, whose depth changes `slope` a pixel: both the sky (depth 0), or
// both surfaces within depth_tolerance of the larger and jitter_reach
// times the slope.
__device__ inline bool same_surface(float depth, float other, float slope) {
  if (!(depth > 0.0f) || !(other > 0.0f)) {
    return depth == 0.0f && other == 0.0f;
  }
  return abs(depth - other) <=
         depth_tolerance * max(depth, other) + jitter_reach * slope;
}

// How much the depth `depth` at `pixel` changes a pixel: on each axis the
// smaller of the differences to the two neighbors, the larger of the two
// axes; a neighbor outside the image or with no surface (the sky) gives
// none on its side. 0 for the sky.
__device__ inline float depth_slope(Images images, uint2 pixel, float depth) {
  if (!(depth > 0.0f)) {
    return 0.0f;
  }
  float slope = 0.0f;
  for (uint a = 0u; a < 2u; ++a) {
    float shallower = INFINITY;
    for (int side = -1; side <= 1; side += 2) {
      const int x = int(pixel.x) + (a == 0u ? side : 0);
      const int y = int(pixel.y) + (a == 1u ? side : 0);
      if (!inside(x, y, images.size)) {
        continue;
      }
      const float d = read_depth(images, make_uint2(uint(x), uint(y)));
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
// `expected`: of the four pixels around `at`, the two frame `frame` - 1
// traced, each kept where its depth then is `expected`'s surface's, weighed
// by its bilinear weight and the kept ones' weights renormalized. Into
// `light`; false where neither is kept.
__device__ inline bool reprojected(Images images, float2 at, float expected,
                                   float slope, uint frame, float3& light) {
  const float2 f = at - 0.5f;  // texel centers on whole numbers
  const float2 corner = floor(f);
  const float2 w = f - corner;
  float3 sum = make_float3(0.0f);
  float total = 0.0f;
  for (uint n = 0u; n < 4u; ++n) {
    const int ox = int(n & 1u);
    const int oy = int(n >> 1u);
    const int x = int(corner.x) + ox;
    const int y = int(corner.y) + oy;
    if (!inside(x, y, images.size) ||
        !contracts::checkerboard_traced(uint(x), uint(y), frame - 1u)) {
      continue;
    }
    // xyz the light, w the depth
    const float4 then =
        read_rows(images.previous_light, make_uint2(uint(x), uint(y)));
    if (!same_surface(expected, then.w, slope)) {
      continue;
    }
    const float weight =
        (ox != 0 ? w.x : 1.0f - w.x) * (oy != 0 ? w.y : 1.0f - w.y);
    sum += weight * xyz(then);
    total += weight;
  }
  if (!(total > 0.0f)) {
    return false;
  }
  light = sum / total;
  return true;
}

// The untraced pixel `pixel`'s light in frame `frame` (checkerboard_fill.h).
__device__ inline float3 filled(Images images, uint2 pixel, uint frame,
                                bool history) {
  const float depth = read_depth(images, pixel);
  const float slope = depth_slope(images, pixel, depth);
  if (history) {
    // Where the surface was: the pixel's center moved by its motion
    // (contract 17: old = new + motion, in pixels), at the depth the
    // motion predicts; a point behind last frame's camera (a predicted
    // depth not above 0 for a surface) has no place there.
    const float4 motion = read_motion(images, pixel);
    const float expected = depth == 0.0f ? 0.0f : depth + motion.z;
    float3 light;
    if ((depth == 0.0f || expected > 0.0f) &&
        reprojected(images,
                    make_float2(pixel) + 0.5f + make_float2(motion.x, motion.y),
                    expected, slope, frame, light)) {
      return light;
    }
  }
  // The traced pixels beside it, this frame's: those of its surface, else
  // all of them.
  constexpr int beside_x[4] = {-1, 1, 0, 0};
  constexpr int beside_y[4] = {0, 0, -1, 1};
  float3 same = make_float3(0.0f);
  float3 any = make_float3(0.0f);
  uint same_count = 0u;
  uint any_count = 0u;
  for (uint n = 0u; n < 4u; ++n) {
    const int x = int(pixel.x) + beside_x[n];
    const int y = int(pixel.y) + beside_y[n];
    if (!inside(x, y, images.size)) {
      continue;
    }
    const uint2 q = make_uint2(uint(x), uint(y));
    const float3 light = xyz(read_rows(images.accumulated, q));
    any += light;
    ++any_count;
    if (same_surface(depth, read_depth(images, q), slope)) {
      same += light;
      ++same_count;
    }
  }
  if (same_count != 0u) {
    return same / float(same_count);
  }
  return any_count != 0u ? any / float(any_count) : make_float3(0.0f);
}

// A pixel's light into the image handed on, and with its depth into the
// history the next frame's fill reads (cuda/frame/checkerboard_images.h).
__device__ inline void hand_off(const serenity::cuda::FrameLaunch& launch,
                                Images images, uint2 pixel, float4 value) {
  write_rows(launch.filled, pixel, value);
  write_rows(launch.history_light, pixel,
             make_float4(xyz(value), read_depth(images, pixel)));
}

__device__ inline void fill_slot(const serenity::cuda::FrameLaunch& launch) {
  const contracts::FrameConstants& frame = launch.frame;
  const uint2 slot = make_uint2(blockIdx.x * blockDim.x + threadIdx.x,
                                blockIdx.y * blockDim.y + threadIdx.y);
  const uint2 size = make_uint2(frame.width, frame.height);
  if (slot.y >= size.y || slot.x >= contracts::checkerboard_width(size.x)) {
    return;
  }
  const Images images{launch.accumulated, launch.previous_light, launch.depth,
                      launch.motion, size};
  const uint traced_x =
      contracts::checkerboard_x(slot.x, slot.y, frame.frame_index);
  const uint other_x = traced_x ^ 1u;  // the slot's other pixel
  if (traced_x < size.x) {
    const uint2 pixel = make_uint2(traced_x, slot.y);
    hand_off(launch, images, pixel, read_rows(launch.accumulated, pixel));
  }
  if (other_x < size.x) {
    const uint2 pixel = make_uint2(other_x, slot.y);
    hand_off(launch, images, pixel,
             make_float4(filled(images, pixel, frame.frame_index,
                                launch.checkerboard_constants.history != 0u),
                         0.0f));
  }
}

}  // namespace
}  // namespace shaders
}  // namespace serenity

extern "C" __global__ void checkerboard_fill(
    const serenity::cuda::FrameLaunch* launch) {
  serenity::shaders::fill_slot(*launch);
}
