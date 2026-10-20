// The depth of field's six kernels (depth_of_field.h): Metal's
// (metal/passes/depth_of_field/depth_of_field.metal, whose header,
// metal/passes/depth_of_field/depth_of_field.h, states every rule this file
// keeps, Jimenez's steps in order) line for line but for how an image is
// read and written: the presort to half resolution, the tiles' largest
// near radius and its dilation, the scatter-as-gather over the 49-tap disc,
// the median, and the composite at the output's size.
//
// Every read Metal makes without a sampler is exact, at a pixel: the
// radiance image's rows by address and pitch (cuda/device/images.h,
// PitchedImage), the depth's floats, rows tightly packed, and the half-float
// images through their surface objects, widened (cuda/device/images.h,
// LevelImage). A half-float image is written through its surface, each
// channel rounded to nearest even, Metal's conversion to RGBA16Float. The
// composite's two bilinear reads, Metal's `sample` (normalized coordinates,
// clamped to the edge), are the filtered images' texture objects, which
// filter as Metal's sampler does (cuda/device/images.h).
//
// Metal's vector functions these kernels call are math/vector.cuh's, under
// the shading language's names (round's halfway cases away from zero, as
// Metal's, not rint's to even), and its pi, the same float as Metal's.

#include <cuda_fp16.h>

#include <cstdint>

#include "core/contracts/depth_of_field.h"
#include "cuda/math/vector.cuh"

namespace serenity {
namespace shaders {
namespace {

namespace contracts = serenity::contracts;
using Texture = cudaTextureObject_t;
using Surface = cudaSurfaceObject_t;

// --- Reading and writing the images (above) ---------------------------------

// The thread's pixel of a per-pixel launch (cuda/device/module.h).
__device__ inline uint2 launch_pixel() {
  return make_uint2(blockIdx.x * blockDim.x + threadIdx.x,
                    blockIdx.y * blockDim.y + threadIdx.y);
}

__device__ inline float4 read_radiance(const std::uint8_t* radiance,
                                       std::uint64_t pitch, uint2 pixel) {
  return reinterpret_cast<const float4*>(radiance + std::uint64_t{pixel.y} *
                                                        pitch)[pixel.x];
}

__device__ inline void write_radiance(std::uint8_t* radiance,
                                      std::uint64_t pitch, uint2 pixel,
                                      float4 value) {
  reinterpret_cast<float4*>(radiance +
                            std::uint64_t{pixel.y} * pitch)[pixel.x] = value;
}

// The full-size depth at `pixel` of an image `width` pixels wide.
__device__ inline float read_depth(const float* depth, uint width,
                                   uint2 pixel) {
  return depth[pixel.y * width + pixel.x];
}

// A half-float image's pixel, all four channels, through its surface.
__device__ inline float4 read_half4(Surface image, uint2 pixel) {
  const ushort4 bits =
      surf2Dread<ushort4>(image, int(pixel.x * sizeof(ushort4)), int(pixel.y));
  return make_float4(__half2float(__ushort_as_half(bits.x)),
                     __half2float(__ushort_as_half(bits.y)),
                     __half2float(__ushort_as_half(bits.z)),
                     __half2float(__ushort_as_half(bits.w)));
}

__device__ inline void write_half4(Surface image, float4 value, uint2 pixel) {
  const ushort4 bits = make_ushort4(__half_as_ushort(__float2half_rn(value.x)),
                                    __half_as_ushort(__float2half_rn(value.y)),
                                    __half_as_ushort(__float2half_rn(value.z)),
                                    __half_as_ushort(__float2half_rn(value.w)));
  surf2Dwrite(bits, image, int(pixel.x * sizeof(ushort4)), int(pixel.y));
}

// --- Metal's helpers --------------------------------------------------------

// The signed circle of confusion of view depth `z`, in output pixels
// (contract 21): negative in front of the plane of focus; the sky, z = 0,
// at infinity; clamped to the largest radius.
__device__ inline float circle_of_confusion(
    const contracts::DepthOfFieldConstants& k, float z) {
  const float c =
      z > 0.0f ? k.coc_scale * (z - k.focus_distance) / z : k.coc_scale;
  return clamp(c, -k.max_radius, k.max_radius);
}

// A pixel of a `size` grid, clamped into it.
__device__ inline uint2 clamped(int2 p, uint2 size) {
  return make_uint2(clamp(p, make_int2(0), make_int2(size) - 1));
}

// Jimenez's spread comparison: how much of a disc of `radius` covers a
// point `distance` from its center, a pixel's soft edge.
__device__ inline float spreads(float radius, float distance) {
  return saturate(radius - distance + 0.5f);
}

// Jimenez's sample alpha, without its pi: the share of a pixel's light a
// disc of `radius` gives each pixel it covers; a radius under half a pixel
// is half a pixel's, the pixel itself.
__device__ inline float spread_weight(float radius) {
  return 1.0f / max(radius * radius, 0.25f);
}

// 1. The presort: one thread a half-resolution pixel, its 2 x 2 block of
// the output's (clamped at an odd edge): their mean, and the circle of
// confusion of the nearest surface among them, in half pixels; the sky is
// the farthest of all.
__device__ void dof_presort_body(contracts::DepthOfFieldConstants k,
                                 const std::uint8_t* radiance,
                                 std::uint64_t pitch, const float* depth,
                                 Surface presort) {
  const uint2 p = launch_pixel();
  if (p.x >= k.half_width || p.y >= k.half_height) {
    return;
  }
  const uint2 size = make_uint2(k.width, k.height);
  float3 color = make_float3(0.0f);
  float nearest = 0.0f;  // 0: the sky, until a surface is seen
  for (uint q = 0; q < 4u; ++q) {
    const uint2 x = clamped(
        make_int2(contracts::dof_scale * p + make_uint2(q & 1u, q >> 1)), size);
    color += rgb(read_radiance(radiance, pitch, x));
    const float z = read_depth(depth, k.width, x);
    if (z > 0.0f && (nearest == 0.0f || z < nearest)) {
      nearest = z;
    }
  }
  write_half4(presort,
              make_float4(0.25f * color, circle_of_confusion(k, nearest) /
                                             float(contracts::dof_scale)),
              p);
}

// 2. The tiles: one thread a tile, the largest near-field radius among its
// half pixels, in half pixels (0 where none is in front of the focus).
__device__ void dof_tiles_body(contracts::DepthOfFieldConstants k, float* tiles,
                               Surface presort) {
  const uint2 t = launch_pixel();
  if (t.x >= k.tiles_x || t.y >= k.tiles_y) {
    return;
  }
  float largest = 0.0f;
  const uint2 first = t * contracts::dof_tile;
  const uint2 last =
      min(first + contracts::dof_tile, make_uint2(k.half_width, k.half_height));
  for (uint y = first.y; y < last.y; ++y) {
    for (uint x = first.x; x < last.x; ++x) {
      largest = max(largest, -read_half4(presort, make_uint2(x, y)).w);
    }
  }
  tiles[t.y * k.tiles_x + t.x] = largest;
}

// 3. The dilation: one thread a tile, the largest near radius of the tiles
// within k.dilation of it that reach it (depth_of_field.h, step 3).
__device__ void dof_dilate_body(contracts::DepthOfFieldConstants k,
                                const float* tiles, float* dilated) {
  const uint2 t = launch_pixel();
  if (t.x >= k.tiles_x || t.y >= k.tiles_y) {
    return;
  }
  const int reach = int(k.dilation);
  float largest = 0.0f;
  for (int dy = -reach; dy <= reach; ++dy) {
    for (int dx = -reach; dx <= reach; ++dx) {
      const int2 n = make_int2(t) + make_int2(dx, dy);
      if (any(n < make_int2(0)) || n.x >= int(k.tiles_x) ||
          n.y >= int(k.tiles_y)) {
        continue;
      }
      const float radius = tiles[uint(n.y) * k.tiles_x + uint(n.x)];
      const int away = max(abs(dx), abs(dy));
      if (radius >= float(max(away - 1, 0) * int(contracts::dof_tile))) {
        largest = max(largest, radius);
      }
    }
  }
  dilated[t.y * k.tiles_x + t.x] = largest;
}

// 4. The gather: one thread a half pixel, scatter as gather over the
// 49-tap disc into its far and near fields (depth_of_field.h, step 4).
__device__ void dof_gather_body(contracts::DepthOfFieldConstants k,
                                const float* dilated, Surface presort,
                                Surface far, Surface near) {
  const uint2 p = launch_pixel();
  if (p.x >= k.half_width || p.y >= k.half_height) {
    return;
  }
  const uint2 size = make_uint2(k.half_width, k.half_height);
  const float4 center = read_half4(presort, p);
  const float own = center.w;  // signed, half pixels
  const uint2 tile = p / contracts::dof_tile;
  const float radius = max(abs(own), dilated[tile.y * k.tiles_x + tile.x]);
  if (radius < 0.5f) {
    // No blur reaches this pixel: its own light, nothing in front.
    write_half4(far, make_float4(rgb(center), 0.0f), p);
    write_half4(near, make_float4(0.0f), p);
    return;
  }
  // The center: in its own field, by its own spread.
  const float own_weight = spread_weight(abs(own));
  float3 far_sum = make_float3(0.0f);
  float far_weight = 0.0f;
  float3 near_sum = make_float3(0.0f);
  float near_weight = 0.0f;
  if (own >= 0.0f) {
    far_sum = own_weight * rgb(center);
    far_weight = own_weight;
  } else {
    near_sum = own_weight * rgb(center);
    near_weight = own_weight;
  }
  // A tap behind the plane of focus reaches the center with at most the
  // center's own radius where it lies behind it (depth_of_field.h).
  const float behind_limit = max(own, 0.0f);
  for (uint ring = 1u; ring <= contracts::dof_rings; ++ring) {
    const uint taps = contracts::dof_ring_taps * ring;
    const float distance = radius * float(ring) / float(contracts::dof_rings);
    for (uint i = 0u; i < taps; ++i) {
      const float angle =
          2.0f * pi * (float(i) + 0.5f * float(ring & 1u)) / float(taps);
      const float2 offset = distance * make_float2(cos(angle), sin(angle));
      const float4 tap = read_half4(
          presort, clamped(make_int2(round(make_float2(p) + offset)), size));
      const float c = tap.w;
      const float weight = spread_weight(abs(c));
      if (c >= 0.0f) {
        const float reach = c > own ? min(c, behind_limit) : c;
        const float w = spreads(reach, distance) * weight;
        far_sum += w * rgb(tap);
        far_weight += w;
      } else {
        const float w = spreads(-c, distance) * weight;
        near_sum += w * rgb(tap);
        near_weight += w;
      }
    }
  }
  // The near field's coverage: each tap stands for pi radius^2 / taps
  // pixels, each spreading 1 / (pi r^2) of its light here (Jimenez's
  // normalization), times his factor.
  const float coverage = saturate(contracts::dof_alpha_gain * near_weight *
                                  radius * radius / float(contracts::dof_taps));
  const float3 far_color =
      far_weight > 0.0f ? far_sum / far_weight : rgb(center);
  const float3 near_color =
      near_weight > 0.0f ? near_sum / near_weight : make_float3(0.0f);
  write_half4(far, make_float4(far_color, 0.0f), p);
  write_half4(near, make_float4(near_color, coverage), p);
}

// The median of nine, by the exchange network of Paeth's (Graphics Gems),
// 19 exchanges: elementwise, so on four channels at once.
__device__ inline void exchange(float4& a, float4& b) {
  const float4 low = min(a, b);
  b = max(a, b);
  a = low;
}

__device__ inline float4 median9(float4 (&v)[9]) {
  exchange(v[1], v[2]);
  exchange(v[4], v[5]);
  exchange(v[7], v[8]);
  exchange(v[0], v[1]);
  exchange(v[3], v[4]);
  exchange(v[6], v[7]);
  exchange(v[1], v[2]);
  exchange(v[4], v[5]);
  exchange(v[7], v[8]);
  exchange(v[0], v[3]);
  exchange(v[5], v[8]);
  exchange(v[4], v[7]);
  exchange(v[3], v[6]);
  exchange(v[1], v[4]);
  exchange(v[2], v[5]);
  exchange(v[4], v[7]);
  exchange(v[4], v[2]);
  exchange(v[6], v[4]);
  exchange(v[4], v[2]);
  return v[4];
}

// 5. The median: one thread a half pixel, the 3 x 3 median of each field,
// channel by channel (clamped at the edge).
__device__ void dof_median_body(contracts::DepthOfFieldConstants k, Surface far,
                                Surface near, Surface far_filtered,
                                Surface near_filtered) {
  const uint2 p = launch_pixel();
  if (p.x >= k.half_width || p.y >= k.half_height) {
    return;
  }
  const uint2 size = make_uint2(k.half_width, k.half_height);
  float4 f[9];
  float4 n[9];
  for (uint i = 0; i < 9u; ++i) {
    const uint2 x = clamped(
        make_int2(p) + make_int2(int(i % 3u) - 1, int(i / 3u) - 1), size);
    f[i] = read_half4(far, x);
    n[i] = read_half4(near, x);
  }
  write_half4(far_filtered, median9(f), p);
  write_half4(near_filtered, median9(n), p);
}

// 6. The composite: one thread an output pixel, in place (depth_of_field.h,
// step 6): the network's own light where the pixel's circle is under a
// pixel, the far field where it is over two, the near field over both.
__device__ void dof_composite_body(contracts::DepthOfFieldConstants k,
                                   std::uint8_t* radiance, std::uint64_t pitch,
                                   const float* depth, Texture far_filtered,
                                   Texture near_filtered) {
  const uint2 x = launch_pixel();
  if (x.x >= k.width || x.y >= k.height) {
    return;
  }
  // The output pixel's center in the half grid's normalized coordinates;
  // the filtered images' textures are Metal's bilinear sampler (above).
  const float2 at = (make_float2(x) + 0.5f) /
                    (float(contracts::dof_scale) *
                     make_float2(make_uint2(k.half_width, k.half_height)));
  const float4 sharp = read_radiance(radiance, pitch, x);
  const float c = abs(circle_of_confusion(k, read_depth(depth, k.width, x)));
  const float3 far_color = rgb(tex2D<float4>(far_filtered, at.x, at.y));
  const float4 near = tex2D<float4>(near_filtered, at.x, at.y);
  const float3 base = mix(rgb(sharp), far_color, smoothstep(1.0f, 2.0f, c));
  write_radiance(radiance, pitch, x,
                 make_float4(mix(base, rgb(near), near.w), sharp.w));
}

}  // namespace
}  // namespace shaders
}  // namespace serenity

// The kernels, each one call to its body above (cuda/device/module.h
// launches them by these names): the constants by value, then Metal's
// bindings in the order its kernels declare them.
extern "C" __global__ void dof_presort(
    serenity::contracts::DepthOfFieldConstants k, const std::uint8_t* radiance,
    std::uint64_t pitch, const float* depth, cudaSurfaceObject_t presort) {
  serenity::shaders::dof_presort_body(k, radiance, pitch, depth, presort);
}

extern "C" __global__ void dof_tiles(
    serenity::contracts::DepthOfFieldConstants k, float* tiles,
    cudaSurfaceObject_t presort) {
  serenity::shaders::dof_tiles_body(k, tiles, presort);
}

extern "C" __global__ void dof_dilate(
    serenity::contracts::DepthOfFieldConstants k, const float* tiles,
    float* dilated) {
  serenity::shaders::dof_dilate_body(k, tiles, dilated);
}

extern "C" __global__ void dof_gather(
    serenity::contracts::DepthOfFieldConstants k, const float* dilated,
    cudaSurfaceObject_t presort, cudaSurfaceObject_t far,
    cudaSurfaceObject_t near) {
  serenity::shaders::dof_gather_body(k, dilated, presort, far, near);
}

extern "C" __global__ void dof_median(
    serenity::contracts::DepthOfFieldConstants k, cudaSurfaceObject_t far,
    cudaSurfaceObject_t near, cudaSurfaceObject_t far_filtered,
    cudaSurfaceObject_t near_filtered) {
  serenity::shaders::dof_median_body(k, far, near, far_filtered, near_filtered);
}

extern "C" __global__ void dof_composite(
    serenity::contracts::DepthOfFieldConstants k, std::uint8_t* radiance,
    std::uint64_t pitch, const float* depth, cudaTextureObject_t far_filtered,
    cudaTextureObject_t near_filtered) {
  serenity::shaders::dof_composite_body(k, radiance, pitch, depth, far_filtered,
                                        near_filtered);
}
