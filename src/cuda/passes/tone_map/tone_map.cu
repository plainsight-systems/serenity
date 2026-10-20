// The tone-map pass's kernels (tone_map.h): the core's steps 1 to 6
// (core/passes/tone_map.h), with its constants, in four kernels, Metal's
// (metal/passes/tone_map/tone_map.metal) line for line but for how an image
// is read and written.
//
// Every image is sampled at its levels' texel centers, (i + 0.5) / size,
// through a texture object that filters as Metal's one sampler does
// (cuda/device/images.h): bilinear, clamped to the edge, normalized
// coordinates. A bloom level is written, and read exactly by the step that
// adds to it, through its surface object, as four half floats rounded to
// nearest even, Metal's conversion to RGBA16Float; the target is written as
// 8-bit unorm, rounded to nearest, Metal's conversion to RGBA8Unorm.

#include <cuda_fp16.h>

#include <cstdint>

#include "core/passes/tone_map.h"
#include "cuda/math/srgb.cuh"
#include "cuda/math/vector.cuh"

namespace serenity {
namespace shaders {
namespace {

using serenity::passes::ToneMap;
using Texture = cudaTextureObject_t;
using Surface = cudaSurfaceObject_t;

// Step 5's span above the roll-off's start: 1 - neutral_start.
inline constexpr float neutral_span = 1.0f - serenity::passes::neutral_start;

// Step 1: a read of L, exposed and clamped to the ceiling, per channel.
__device__ float3 expose(float3 radiance, float scale) {
  return min(radiance * scale, make_float3(serenity::passes::bloom_ceiling));
}

__device__ float3 sample(Texture image, float2 uv) {
  return rgb(tex2D<float4>(image, uv.x, uv.y));
}

// Step 2's filter about `uv` in a source whose texels are `texel` apart:
// five 2 x 2 boxes, from 13 bilinear reads of `read`: a 3 x 3 grid two
// texels apart, named by compass point from the center, and four reads one
// texel diagonally off it.
template <typename Read>
__device__ float3 down13(Read read, float2 uv, float2 texel) {
  const auto s = [&](float dx, float dy) {
    return read(uv + make_float2(dx, dy) * texel);
  };
  const float3 north_west = s(-2, -2);
  const float3 north = s(0, -2);
  const float3 north_east = s(2, -2);
  const float3 west = s(-2, 0);
  const float3 center = s(0, 0);
  const float3 east = s(2, 0);
  const float3 south_west = s(-2, 2);
  const float3 south = s(0, 2);
  const float3 south_east = s(2, 2);
  const float3 inner_north_west = s(-1, -1);
  const float3 inner_north_east = s(1, -1);
  const float3 inner_south_west = s(-1, 1);
  const float3 inner_south_east = s(1, 1);
  const float3 middle = 0.25f * (inner_north_west + inner_north_east +
                                 inner_south_west + inner_south_east);
  const float3 corners = 0.25f * ((north_west + north + west + center) +
                                  (north + north_east + center + east) +
                                  (west + center + south_west + south) +
                                  (center + east + south + south_east));
  return serenity::passes::down_middle * middle +
         serenity::passes::down_corner * corners;
}

// Step 3's 3 x 3 tent of `level`, of `size` texels, about `uv`, a texel of
// `level` apart.
__device__ float3 tent(Texture level, uint2 size, float2 uv) {
  const float2 texel = 1.0f / make_float2(size);
  const auto s = [&](float dx, float dy) {
    return sample(level, uv + make_float2(dx, dy) * texel);
  };
  return (s(-1, -1) + 2.0f * s(0, -1) + s(1, -1) + 2.0f * s(-1, 0) +
          4.0f * s(0, 0) + 2.0f * s(1, 0) + s(-1, 1) + 2.0f * s(0, 1) +
          s(1, 1)) /
         16.0f;
}

// Step 5: Khronos' PBR Neutral.
__device__ float3 neutral(float3 color) {
  const float x = min(color.x, min(color.y, color.z));
  color -= x < serenity::passes::neutral_toe
               ? x - serenity::passes::neutral_toe_curve * x * x
               : serenity::passes::neutral_f90;
  const float peak = max(color.x, max(color.y, color.z));
  if (peak < serenity::passes::neutral_start) {
    return color;
  }
  const float rolled =
      1.0f - neutral_span * neutral_span /
                 (peak + neutral_span - serenity::passes::neutral_start);
  color *= rolled / peak;
  const float g =
      1.0f -
      1.0f / (serenity::passes::neutral_desaturation * (peak - rolled) + 1.0f);
  return mix(color, make_float3(rolled), g);
}

__device__ float2 center(uint2 pixel, uint2 size) {
  return (make_float2(pixel) + 0.5f) / make_float2(size);
}

// The thread's pixel of a per-pixel launch (cuda/device/module.h).
__device__ uint2 launch_pixel() {
  return make_uint2(blockIdx.x * blockDim.x + threadIdx.x,
                    blockIdx.y * blockDim.y + threadIdx.y);
}

// A bloom level's pixel, as four half floats, through its surface.
__device__ void write_level(Surface level, uint2 pixel, float3 value) {
  const ushort4 bits = make_ushort4(__half_as_ushort(__float2half_rn(value.x)),
                                    __half_as_ushort(__float2half_rn(value.y)),
                                    __half_as_ushort(__float2half_rn(value.z)),
                                    __half_as_ushort(__float2half_rn(1.0f)));
  surf2Dwrite(bits, level, int(pixel.x * sizeof(ushort4)), int(pixel.y));
}

__device__ float3 read_level(Surface level, uint2 pixel) {
  const ushort4 bits =
      surf2Dread<ushort4>(level, int(pixel.x * sizeof(ushort4)), int(pixel.y));
  return make_float3(__half2float(__ushort_as_half(bits.x)),
                     __half2float(__ushort_as_half(bits.y)),
                     __half2float(__ushort_as_half(bits.z)));
}

__device__ unsigned char unorm8(float v) {
  return static_cast<unsigned char>(__float2uint_rn(saturate(v) * 255.0f));
}

// Steps 1 and 2 for B_0: the exposed radiance, filtered to B_0's size and
// divided by the level count.
__device__ void tone_map_down_first_body(ToneMap settings, Texture radiance,
                                         uint2 radiance_size, Surface level,
                                         uint2 size) {
  const uint2 pixel = launch_pixel();
  if (pixel.x >= size.x || pixel.y >= size.y) {
    return;
  }
  const float scale = exp2(settings.exposure);
  const float2 texel = 1.0f / make_float2(radiance_size);
  const auto read = [&](float2 uv) {
    return expose(sample(radiance, uv), scale);
  };
  const float3 filtered = down13(read, center(pixel, size), texel);
  write_level(level, pixel, filtered / float(serenity::passes::bloom_levels));
}

// Step 2 for B_1 .. B_5: the level before, filtered to this one's size.
__device__ void tone_map_down_body(Texture source, uint2 source_size,
                                   Surface level, uint2 size) {
  const uint2 pixel = launch_pixel();
  if (pixel.x >= size.x || pixel.y >= size.y) {
    return;
  }
  const float2 texel = 1.0f / make_float2(source_size);
  const auto read = [&](float2 uv) { return sample(source, uv); };
  write_level(level, pixel, down13(read, center(pixel, size), texel));
}

// Step 3 for one level: B_k += tent(B_(k+1)), in place.
__device__ void tone_map_up_body(Texture below, uint2 below_size, Surface level,
                                 uint2 size) {
  const uint2 pixel = launch_pixel();
  if (pixel.x >= size.x || pixel.y >= size.y) {
    return;
  }
  const float3 sum =
      read_level(level, pixel) + tent(below, below_size, center(pixel, size));
  write_level(level, pixel, sum);
}

// Steps 1 and 4 to 6, at the frame's size, into the target: the radiance
// read exactly, at the pixel, from the accumulated image's rows.
__device__ void tone_map_finish_body(ToneMap settings,
                                     const std::uint8_t* radiance,
                                     std::uint64_t pitch, Texture bloom,
                                     uint2 bloom_size, uchar4* target,
                                     uint2 size) {
  const uint2 pixel = launch_pixel();
  if (pixel.x >= size.x || pixel.y >= size.y) {
    return;
  }
  const float4* row = reinterpret_cast<const float4*>(
      radiance + std::uint64_t{pixel.y} * pitch);
  // Step 1.
  const float3 exposed = expose(rgb(row[pixel.x]), exp2(settings.exposure));
  // Step 4.
  const float3 glare = tent(bloom, bloom_size, center(pixel, size));
  const float3 composite =
      (1.0f - settings.bloom) * exposed + settings.bloom * glare;
  // Steps 5 and 6.
  const float3 shown = transfer_srgb(saturate(neutral(composite)));
  target[pixel.y * size.x + pixel.x] =
      make_uchar4(unorm8(shown.x), unorm8(shown.y), unorm8(shown.z), 255);
}

}  // namespace
}  // namespace shaders
}  // namespace serenity

// The kernels, each one call to its body above (cuda/device/module.h
// launches them by these names).
extern "C" __global__ void tone_map_down_first(
    serenity::passes::ToneMap settings, cudaTextureObject_t radiance,
    uint2 radiance_size, cudaSurfaceObject_t level, uint2 size) {
  serenity::shaders::tone_map_down_first_body(settings, radiance, radiance_size,
                                              level, size);
}

extern "C" __global__ void tone_map_down(cudaTextureObject_t source,
                                         uint2 source_size,
                                         cudaSurfaceObject_t level,
                                         uint2 size) {
  serenity::shaders::tone_map_down_body(source, source_size, level, size);
}

extern "C" __global__ void tone_map_up(cudaTextureObject_t below,
                                       uint2 below_size,
                                       cudaSurfaceObject_t level, uint2 size) {
  serenity::shaders::tone_map_up_body(below, below_size, level, size);
}

extern "C" __global__ void tone_map_finish(serenity::passes::ToneMap settings,
                                           const std::uint8_t* radiance,
                                           std::uint64_t pitch,
                                           cudaTextureObject_t bloom,
                                           uint2 bloom_size, uchar4* target,
                                           uint2 size) {
  serenity::shaders::tone_map_finish_body(settings, radiance, pitch, bloom,
                                          bloom_size, target, size);
}
