// The denoise pass's kernels (denoise.h states the pass, its launches and
// their costs; core/contracts/denoise.h the network's constants): Metal's
// (metal/passes/denoise/denoise.metal, whose comments give each kernel's
// step of training/model.py's Denoiser.frame, the precision and why each
// piece of threadgroup memory is there) translated line for line:
//
//   denoise_depth_histogram<L>, denoise_depth_select<L>   the input's depth
//       normalization's median (training/data.py, input_frame), exactly;
//   denoise_input       steps 1 and 2's input: the 30 channels a pixel;
//   conv3x3_n<NT>       step 2's 3 x 3 convolutions, with ReLU;
//   denoise_maxpool     the U-Net's 2 x 2 max pool (its nearest upsampling
//                       is conv3x3's, as a decoder reads the level below);
//   denoise_output      the head (1 x 1), step 3's kernels, step 4's blend
//                       and step 5's remodulation, one output pixel a thread.
//
// What is CUDA's, not Metal's. A block of threads is contract 18's, four
// warps of 32 (a warp Metal's simdgroup, its index threadIdx.x / 32, its
// lane threadIdx.x % 32); threadgroup memory is shared memory, its atomics
// shared-memory atomics, threadgroup_barrier __syncthreads() and
// simdgroup_barrier __syncwarp(). The simdgroup matrix products are the
// tensor cores' m16n8k8 (cuda/device/matrix.cuh): a Metal row's four 8 x 8
// pixel blocks two 16-pixel tiles, K one block of 8 input channels, N one
// block of 8 output channels, half in, float sums; where Metal stores its
// accumulators through threadgroup memory, the fragments are stored into
// the same layout by the fragment's formula. Half floats are carried as
// their bits (ushort, ushort4 for Metal's half4) and converted by
// cuda_fp16.h's intrinsics, rounding to nearest even as Metal's half(float)
// does. The images Metal reads through textures are device memory read
// exactly by address: the light's hand-off and the radiance written,
// pitched rows of four floats (cuda/device/images.h); the features at the
// light's size and the full-size albedo, contract 17's formats, rows
// tightly packed (cuda/frame/feature_images.h). Each kernel's parameters
// are Metal's bindings in order, every texture it reads among them (Metal's
// argument table keeps a binding from one dispatch to the next, so a kernel
// there reads some an earlier one set).
//
// Buffers of activations hold a level's pixels with a one-pixel border of
// zeros on every side, channels innermost, rows `pitch` pixels apart, as
// Metal's (cuda/frame/denoise_images.h).

#include <cuda_fp16.h>

#include <cstdint>

#include "core/contracts/denoise.h"
#include "cuda/device/layout.cuh"
#include "cuda/device/matrix.cuh"
#include "cuda/math/vector.cuh"

namespace serenity {
namespace shaders {
namespace {

namespace contracts = serenity::contracts;
using contracts::DenoiseConstants;
using contracts::MedianState;

// --- Half floats, as bits -------------------------------------------------

__device__ inline float4 widen(ushort4 h) {
  return make_float4(
      __half2float(__ushort_as_half(h.x)), __half2float(__ushort_as_half(h.y)),
      __half2float(__ushort_as_half(h.z)), __half2float(__ushort_as_half(h.w)));
}

__device__ inline ushort4 narrow(float4 v) {
  return make_ushort4(__half_as_ushort(__float2half_rn(v.x)),
                      __half_as_ushort(__float2half_rn(v.y)),
                      __half_as_ushort(__float2half_rn(v.z)),
                      __half_as_ushort(__float2half_rn(v.w)));
}

// Metal's max(float4, float), each component; its own name, since a max
// declared here would hide math/vector.cuh's for every call below.
__device__ inline float4 max_each(float4 a, float b) {
  return make_float4(max(a.x, b), max(a.y, b), max(a.z, b), max(a.w, b));
}

// A pixel of a pitched image of four floats (cuda/device/images.h).
__device__ inline float4 read_rows(const std::uint8_t* rows,
                                   std::uint64_t pitch, uint2 pixel) {
  return reinterpret_cast<const float4*>(rows + std::uint64_t{pixel.y} *
                                                    pitch)[pixel.x];
}

__device__ inline void write_rows(std::uint8_t* rows, std::uint64_t pitch,
                                  uint2 pixel, float4 value) {
  reinterpret_cast<float4*>(rows + std::uint64_t{pixel.y} * pitch)[pixel.x] =
      value;
}

// The thread's pixel of a per-pixel launch (cuda/device/module.h).
__device__ inline uint2 launch_pixel() {
  return make_uint2(blockIdx.x * blockDim.x + threadIdx.x,
                    blockIdx.y * blockDim.y + threadIdx.y);
}

// Output pixel denoise_scale p + (q & 1, q >> 1): subpixel q of input pixel
// p (contract 18).
__device__ inline uint2 subpixel(uint2 p, uint q) {
  return make_uint2(contracts::denoise_scale * p.x + (q & 1u),
                    contracts::denoise_scale * p.y + (q >> 1));
}

// The kernels' spacings, copied at compile time from the core's into a
// constant of device memory, since device code may index a host constant
// only in a constant expression (cuda/reuse/pairing.cuh says the same).
static constexpr __device__ uint
    device_spacing_pixels[contracts::denoise_spacings] = {
        contracts::spacing_pixels[0], contracts::spacing_pixels[1],
        contracts::spacing_pixels[2]};
static_assert(contracts::denoise_spacings == 3u, "the copy lists every one");

// OIDN's PU transfer (training/loss.py, pu): radiance >= 0 to about [0, 1].
// logf: math/vector.cuh has no log, and CUDA's global one is not looked up
// by accident.
__device__ inline float pu(float y) {
  y = max(y, 0.0f);
  float v;
  if (y <= contracts::pu_y0) {
    v = contracts::pu_a * y;
  } else if (y <= contracts::pu_y1) {
    v = contracts::pu_b * pow(y, contracts::pu_c) + contracts::pu_d;
  } else {
    v = contracts::pu_e * logf(y + contracts::pu_f) + contracts::pu_g;
  }
  return v * contracts::pu_norm_scale;
}

__device__ inline float3 albedo_factor(float3 albedo) {
  return contracts::albedo_floor +
         (1.0f - contracts::albedo_floor) * clamp(albedo, 0.0f, 1.0f);
}

struct Moved {
  float3 color;
  float valid;
};

// Step 1: last frame's output at output pixel `x` moved by `mv` (input
// pixels, old = new + mv): read bilinearly at x + 1/2 + 2 mv, clamped to
// the padded image's edge (grid_sample's border padding); valid when that
// point lies inside twice the light's extent (its edges included).
__device__ inline Moved moved(const DenoiseConstants& k, const float4* history,
                              uint2 x, float2 mv) {
  if (k.history == 0u) {
    return Moved{make_float3(0.0f), 0.0f};
  }
  const float sx = float(x.x) + 0.5f + float(contracts::denoise_scale) * mv.x;
  const float sy = float(x.y) + 0.5f + float(contracts::denoise_scale) * mv.y;
  const float w = float(contracts::denoise_scale * k.light_width);
  const float h = float(contracts::denoise_scale * k.light_height);
  if (!(sx >= 0.0f && sx <= w && sy >= 0.0f && sy <= h)) {
    return Moved{make_float3(0.0f), 0.0f};
  }
  const uint width = contracts::denoise_scale * k.padded_width;
  const uint height = contracts::denoise_scale * k.padded_height;
  const float u = clamp(sx - 0.5f, 0.0f, float(width - 1u));
  const float v = clamp(sy - 0.5f, 0.0f, float(height - 1u));
  const uint x0 = uint(u);
  const uint y0 = uint(v);
  const uint x1 = min(x0 + 1u, width - 1u);
  const uint y1 = min(y0 + 1u, height - 1u);
  const float fx = u - float(x0);
  const float fy = v - float(y0);
  const float3 top =
      mix(rgb(history[y0 * width + x0]), rgb(history[y0 * width + x1]), fx);
  const float3 bottom =
      mix(rgb(history[y1 * width + x0]), rgb(history[y1 * width + x1]), fx);
  return Moved{mix(top, bottom, fy), 1.0f};
}

__device__ inline bool inside_light(const DenoiseConstants& k, int2 p) {
  return p.x >= 0 && p.y >= 0 && uint(p.x) < k.light_width &&
         uint(p.y) < k.light_height;
}

// Tap t of the kernel at `spacing` around input pixel p: p + spacing (i, j),
// t = (j + 1) * 3 + (i + 1).
__device__ inline int2 tap_at(uint2 p, uint spacing, uint t) {
  return make_int2(p) +
         int(spacing) * make_int2(int(t % 3u) - 1, int(t / 3u) - 1);
}

// The depth median's radix selection (below): level 0 selects by a depth's
// bits from top_shift up, levels 1 and 2 by level_bits more each, each
// middle rank in its own half of the bins.
inline constexpr uint top_shift = 20u;
inline constexpr uint level_bits = 10u;
inline constexpr uint level_bins = 1u << level_bits;
inline constexpr uint level_mask = level_bins - 1u;
static_assert(2u * level_bins == contracts::median_bins,
              "levels 1 and 2 hold both ranks' bins");

// ---------------------------------------------------------------------------
// The depth's median over the light's surface pixels (depth > 0), as
// np.median: level 0 counts the float's bits 30..20 (the sign is 0), level 1
// bits 19..10 and level 2 bits 9..0, the last two only of depths whose
// higher bits are each middle rank's so far. Each block counts into its own
// bins and adds them to the device's once (GDSA.5); integers, so the result
// is the same whatever the order (GDSA.2).
// ---------------------------------------------------------------------------

template <uint Level>
__device__ void denoise_depth_histogram(const DenoiseConstants& k,
                                        uint* histogram,
                                        const MedianState& state,
                                        const float* depth) {
  const uint2 pixel = launch_pixel();
  const uint index = threadIdx.y * blockDim.x + threadIdx.x;
  const uint threads = blockDim.x * blockDim.y;
  __shared__ uint bins[contracts::median_bins];
  for (uint b = index; b < contracts::median_bins; b += threads) {
    bins[b] = 0u;
  }
  __syncthreads();
  if (pixel.x < k.light_width && pixel.y < k.light_height) {
    const float z = depth[pixel.y * k.light_width + pixel.x];
    if (z > 0.0f) {
      const uint bits = as_uint(z);
      if (Level == 0u) {
        atomicAdd(&bins[bits >> top_shift], 1u);
      } else {
        const uint shift = Level == 1u ? top_shift : top_shift - level_bits;
        const uint bin = (bits >> (shift - level_bits)) & level_mask;
        for (uint r = 0; r < 2u; ++r) {
          if (state.count != 0u && (bits >> shift) == state.prefix[r]) {
            atomicAdd(&bins[r * level_bins + bin], 1u);
          }
        }
      }
    }
  }
  __syncthreads();
  for (uint b = index; b < contracts::median_bins; b += threads) {
    const uint n = bins[b];
    if (n != 0u) {
      atomicAdd(&histogram[b], n);
    }
  }
}

// One thread walks the bins (2048 at most, a few microseconds); then the
// block clears them for the next level or frame.
template <uint Level>
__device__ void denoise_depth_select(uint* histogram, MedianState& state) {
  const uint index = threadIdx.x;
  const uint threads = blockDim.x;
  if (index == 0u) {
    if (Level == 0u) {
      uint n = 0u;
      for (uint b = 0; b < contracts::median_bins; ++b) {
        n += histogram[b];
      }
      state.count = n;
      state.rank[0] = n == 0u ? 0u : (n - 1u) / 2u;
      state.rank[1] = n / 2u;
    }
    if (state.count == 0u) {
      state.median = 1.0f;
    } else {
      for (uint r = 0; r < 2u; ++r) {
        const uint first = Level == 0u ? 0u : r * level_bins;
        const uint bins = Level == 0u ? contracts::median_bins : level_bins;
        uint below = 0u;
        uint b = 0u;
        for (; b + 1u < bins && below + histogram[first + b] <= state.rank[r];
             ++b) {
          below += histogram[first + b];
        }
        state.rank[r] -= below;
        state.prefix[r] = Level == 0u ? b : (state.prefix[r] << level_bits) | b;
      }
      if (Level == 2u) {
        state.median =
            (as_float(state.prefix[0]) + as_float(state.prefix[1])) * 0.5f;
      }
    }
  }
  __syncthreads();
  for (uint b = index; b < contracts::median_bins; b += threads) {
    histogram[b] = 0u;
  }
}

// ---------------------------------------------------------------------------
// The network's input, one thread an input pixel of the padded grid: zero
// outside the light's extent (training/evaluate.py's zero padding); inside
// it, training/data.py's input_frame and model.py's step 1 and 2 inputs.
// Also the demodulated light, which the output's kernels average.
// ---------------------------------------------------------------------------

__device__ void denoise_input(const DenoiseConstants& k,
                              const MedianState& state, const float4* history,
                              ushort4* input, float4* demod,
                              const std::uint8_t* light,
                              std::uint64_t light_pitch, const ushort4* albedo,
                              const ushort4* normal, const float* depth,
                              const ushort4* motion) {
  const uint2 p = launch_pixel();
  if (p.x >= k.padded_width || p.y >= k.padded_height) {
    return;
  }
  const uint pitch = k.padded_width + 2u;
  // A pixel's channels as half4s, a run of 64 bytes (GPU.2).
  constexpr uint quads = contracts::input_channels / 4u;
  ushort4* out = input + ((p.y + 1u) * pitch + p.x + 1u) * quads;
  float values[contracts::input_channels];
  for (uint c = 0; c < contracts::input_channels; ++c) {
    values[c] = 0.0f;
  }
  float3 d = make_float3(0.0f);
  if (inside_light(k, make_int2(p))) {
    // the features' pixel, rows of the light's width
    const uint at = p.y * k.light_width + p.x;
    const float3 a = clamp(rgb(widen(albedo[at])), 0.0f, 1.0f);
    d = rgb(read_rows(light, light_pitch, p)) / albedo_factor(a);
    const float3 n = xyz(widen(normal[at]));
    const float z = depth[at];
    const float4 m4 = widen(motion[at]);
    const float2 mv = make_float2(m4.x, m4.y);
    values[contracts::input_radiance + 0u] = pu(d.x);
    values[contracts::input_radiance + 1u] = pu(d.y);
    values[contracts::input_radiance + 2u] = pu(d.z);
    values[contracts::input_albedo + 0u] = a.x;
    values[contracts::input_albedo + 1u] = a.y;
    values[contracts::input_albedo + 2u] = a.z;
    values[contracts::input_normal + 0u] = dot(n, to_float3(k.right));
    values[contracts::input_normal + 1u] = dot(n, to_float3(k.up));
    values[contracts::input_normal + 2u] = dot(n, to_float3(k.backward));
    values[contracts::input_depth] =
        z > 0.0f ? logf(1.0f + z / state.median) : 0.0f;
    values[contracts::input_motion + 0u] = mv.x;
    values[contracts::input_motion + 1u] = mv.y;
    // pixel_unshuffle: subpixel q = dy * 2 + dx of each history channel c
    // (contract 18), and its validity.
    constexpr uint subpixels = contracts::denoise_subpixels;
    for (uint q = 0; q < subpixels; ++q) {
      const uint2 x = subpixel(p, q);
      const Moved m = moved(k, history, x, mv);
      values[contracts::input_history + 0u * subpixels + q] = pu(m.color.x);
      values[contracts::input_history + 1u * subpixels + q] = pu(m.color.y);
      values[contracts::input_history + 2u * subpixels + q] = pu(m.color.z);
      values[contracts::input_validity + q] = m.valid;
    }
    // The frame's camera jitter, in input pixels, at every pixel of the
    // light (contract 18), 0 in the padding as every input is.
    values[contracts::input_jitter + 0u] = k.jitter_x;
    values[contracts::input_jitter + 1u] = k.jitter_y;
  }
  for (uint c = 0; c < quads; ++c) {
    out[c] = narrow(make_float4(values[4u * c], values[4u * c + 1u],
                                values[4u * c + 2u], values[4u * c + 3u]));
  }
  demod[p.y * k.padded_width + p.x] = make_float4(d, 0.0f);
}

// ---------------------------------------------------------------------------
// A 3 x 3 convolution, zero-padded by 1, with its bias and, unless the
// params say not, a ReLU: an implicit matrix product on the tensor cores,
// pixels by input channels (9 taps x in_blocks blocks of 8) times input
// channels by output channels. A block of threads is 4 warps, 4 rows of 32
// pixels; each warp computes its row's 32 pixels, two 16-pixel tiles, for
// NT blocks of 8 output channels (2 x NT accumulators, NT at most 4, as
// Metal's), a grid's z one such chunk of the outputs.
//
// For each block of 8 input channels the block of threads first copies,
// with every thread, the block's 6 x 34 pixels of input (its 4 x 32 and the
// border the taps reach) and its 9 x 8 x NT x 8 weights into shared memory,
// coalesced (GPU.2); every warp's 9 taps then load their fragments from
// there by ldmatrix, so each input value leaves device memory once a block,
// not 9 times a warp (GDSA.16). Reads past the level's border are zeros.
// Results go through shared memory, a block of channels at a time, to be
// biased, rectified and narrowed to half.
// ---------------------------------------------------------------------------

// The tile (contract 18): conv_rows rows of conv_pixels, one a warp, with
// the border its taps reach; blocks of `block` channels (contract 18's
// channel_block), K and N of an m16n8k8 product, two half4s a pixel, 16
// bytes, so every row ldmatrix reads is 16-byte aligned in the tiles
// below.
inline constexpr uint conv_rows = contracts::conv_rows;
inline constexpr uint conv_pixels = contracts::conv_pixels;
inline constexpr uint halo_rows = conv_rows + 2u;
inline constexpr uint halo_pixels = conv_pixels + 2u;
inline constexpr uint conv_threads = contracts::denoise_threads;
inline constexpr uint conv_taps = 9u;
inline constexpr uint block = contracts::channel_block;
inline constexpr uint block_quads = block / 4u;
static_assert(block == 8u, "a block is an m16n8k8 product's K and N");
// A row's conv_pixels pixels as pixel_tiles tiles of an m16n8k8 product's
// 16 rows.
inline constexpr uint tile_pixels = 16u;
inline constexpr uint pixel_tiles = conv_pixels / tile_pixels;
static_assert(pixel_tiles * tile_pixels == conv_pixels, "a row is whole tiles");

__device__ inline uint warp_index() {
  return threadIdx.x / contracts::simd_width;
}
__device__ inline uint warp_lane() {
  return threadIdx.x % contracts::simd_width;
}

template <uint NT>
__device__ void conv3x3(const contracts::ConvParams& p, const ushort4* input,
                        const ushort4* weights, const float* bias,
                        ushort4* output, const ushort4* coarse) {
  constexpr uint channels = NT * block;
  constexpr uint quads = channels / 4u;
  __shared__ __align__(16) ushort4 halo4[halo_rows * halo_pixels * block_quads];
  __shared__ __align__(16) ushort4 taps4[conv_taps * block * quads];
  __shared__ float stage[conv_rows * conv_pixels * block];
  const uint sg = warp_index();
  const uint lane = warp_lane();
  const uint tid = sg * contracts::simd_width + lane;
  const uint y0 = blockIdx.y * conv_rows;
  const uint x0 = blockIdx.x * conv_pixels;
  const uint y = y0 + sg;
  const bool active = y < p.height;  // every thread still copies and waits
  const uint first_out = blockIdx.z * channels;
  // the weights' row: every output channel
  const uint out_width = p.out_blocks * block;

  matrix::Accumulator acc[pixel_tiles][NT];
  for (uint m = 0; m < pixel_tiles; ++m) {
    for (uint n = 0; n < NT; ++n) {
      acc[m][n] = matrix::zero_accumulator();
    }
  }
  const uint in_stride = p.in_stride / 4u;  // in half4s, as the reads below
  const uint up_stride = p.up_stride / 4u;
  for (uint kb = 0; kb < p.in_blocks; ++kb) {
    // The block's input, bordered rows y0 .. y0 + conv_rows + 1 and
    // columns x0 .. x0 + conv_pixels + 1 (input pixels y0 - 1 .. and
    // x0 - 1 ..), block_quads half4s a pixel.
    for (uint i = tid; i < halo_rows * halo_pixels * block_quads;
         i += conv_threads) {
      const uint pixel = i / block_quads;
      const uint quad = i % block_quads;
      const uint row = y0 + pixel / halo_pixels;
      const uint column = x0 + pixel % halo_pixels;
      ushort4 v = make_ushort4(0, 0, 0, 0);  // half 0.0's bits
      if (kb < p.up_blocks) {
        // Upsampled: input pixel (column - 1, row - 1) reads the level
        // below's at half its coordinates; the border stays zero.
        if (row >= 1u && row <= p.height && column >= 1u && column <= p.width) {
          v = coarse[(((row - 1u) / 2u + 1u) * p.up_pitch + (column - 1u) / 2u +
                      1u) *
                         up_stride +
                     kb * block_quads + quad];
        }
      } else if (row <= p.height + 1u && column <= p.width + 1u) {
        v = input[(row * p.in_pitch + column) * in_stride +
                  (kb - p.up_blocks) * block_quads + quad];
      }
      halo4[pixel * block_quads + quad] = v;
    }
    // Its weights: rows (tap, input channel), this chunk's outputs.
    const uint weight_quads = out_width / 4u;
    for (uint i = tid; i < conv_taps * block * quads; i += conv_threads) {
      const uint row = i / quads;
      const uint q = i % quads;
      const uint tap = row / block;
      const uint k = row % block;
      taps4[row * quads + q] =
          weights[((tap * p.in_blocks + kb) * block + k) * weight_quads +
                  first_out / 4u + q];
    }
    __syncthreads();
    if (active) {
      for (uint tap = 0; tap < conv_taps; ++tap) {
        const uint dy = tap / 3u;
        const uint dx = tap % 3u;
        // A: tile m's 16 pixels, lane L naming pixel L % 16's 8 channels;
        // B: the tap's 8 input channels, lane L naming channel L % 8's row
        // of this chunk's outputs, at output block n.
        matrix::FragmentA a[pixel_tiles];
        matrix::FragmentB b[NT];
        for (uint m = 0; m < pixel_tiles; ++m) {
          a[m] =
              matrix::load_a(&halo4[((sg + dy) * halo_pixels + m * tile_pixels +
                                     dx + lane % tile_pixels) *
                                    block_quads]);
        }
        for (uint n = 0; n < NT; ++n) {
          b[n] = matrix::load_b(
              &taps4[(tap * block + lane % block) * quads + n * block_quads]);
        }
        for (uint m = 0; m < pixel_tiles; ++m) {
          for (uint n = 0; n < NT; ++n) {
            matrix::multiply_accumulate(acc[m][n], a[m], b[n]);
          }
        }
      }
    }
    __syncthreads();
  }
  if (!active) {
    return;
  }
  float* mine = stage + sg * conv_pixels * block;
  const uint x = x0 + lane;
  ushort4* out = output + (((y + 1u) * p.out_pitch + x + 1u) * p.out_stride +
                           p.out_offset + first_out) /
                              4u;
  // A block of channels at a time through the staging, [pixel][channel]
  // as Metal's simdgroup_store lays it out, each lane then writing its own
  // pixel's block, block_quads half4s, inside the level only.
  for (uint n = 0; n < NT; ++n) {
    for (uint m = 0; m < pixel_tiles; ++m) {
      for (uint i = 0; i < 4u; ++i) {
        mine[(m * tile_pixels + matrix::fragment_row(lane, i)) * block +
             matrix::fragment_column(lane, i)] = acc[m][n].c[i];
      }
    }
    __syncwarp();
    if (x < p.width) {
      const float* values = mine + lane * block;
      const float* b = bias + first_out + n * block;
      for (uint c = 0; c < block_quads; ++c) {
        const uint at = 4u * c;
        float4 v = make_float4(values[at], values[at + 1u], values[at + 2u],
                               values[at + 3u]) +
                   make_float4(b[at], b[at + 1u], b[at + 2u], b[at + 3u]);
        if (p.relu != 0u) {
          v = max_each(v, 0.0f);
        }
        out[n * block_quads + c] = narrow(v);
      }
    }
    __syncwarp();
  }
}

// ---------------------------------------------------------------------------
// 2 x 2 max pool, four channels a thread, between bordered buffers (the
// destination's size in the params). One dimension of threads, channels
// fastest, so a warp's writes are one run of memory (GPU.2). The launch
// rounds up to whole blocks; a thread past the last is past the last row.
// ---------------------------------------------------------------------------

// A half's maximum, exact: one of the two halves (__hmax), as Metal's max.
__device__ inline ushort4 max_halves(ushort4 a, ushort4 b) {
  const auto m = [](ushort x, ushort y) {
    return __half_as_ushort(__hmax(__ushort_as_half(x), __ushort_as_half(y)));
  };
  return make_ushort4(m(a.x, b.x), m(a.y, b.y), m(a.z, b.z), m(a.w, b.w));
}

__device__ void denoise_maxpool(const contracts::ResampleParams& p,
                                const ushort4* input, ushort4* output) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint quads = p.channels / 4u;
  const uint q = i % quads;
  const uint pixel = i / quads;
  const uint x = pixel % p.width;
  const uint y = pixel / p.width;
  if (y >= p.height) {
    return;
  }
  const uint in_stride = p.in_stride / 4u;
  const uint c = p.in_offset / 4u + q;
  const auto read = [&](uint sx, uint sy) {
    return input[((sy + 1u) * p.in_pitch + sx + 1u) * in_stride + c];
  };
  const ushort4 v = max_halves(
      max_halves(read(2u * x, 2u * y), read(2u * x + 1u, 2u * y)),
      max_halves(read(2u * x, 2u * y + 1u), read(2u * x + 1u, 2u * y + 1u)));
  output[((y + 1u) * p.out_pitch + x + 1u) * (p.out_stride / 4u) +
         p.out_offset / 4u + q] = v;
}

// ---------------------------------------------------------------------------
// The output, over twice the padded grid (training/evaluate.py runs the
// head and kernels over the padding too, and the history it keeps is that
// size). A block is four warps and a row of 32 input pixels. First the
// head, a 1 x 1 convolution, as a matrix product on the tensor cores: the
// 32 pixels' last features (half, read straight from their buffer, as
// Metal's simdgroup_load reads them: ldmatrix reads only shared memory, so
// the fragments are loaded from device memory by the fragment's formula,
// cuda/device/matrix.cuh) times the head's weights (half, laid out
// [feature][output], 124 padded to head_columns), float sums, each warp 32
// of the outputs; the scores stored transposed in shared memory, an
// output's 32 pixels side by side, so the lanes reading them hit different
// banks, and never in device memory (GDSA.16). Then a thread an output
// pixel X, its subpixel q its warp, p = X / 2 its input pixel: per spacing,
// the softmax of its 9 taps' scores, a tap outside the light's extent
// masked, over the demodulated light; the softmax of the 3 spacing scores
// mixing the three; the blend with the moved history by sigmoid(score 30) x
// valid; the blend kept as the next frame's history and, inside the output,
// multiplied by the full-size albedo's factor into the radiance image.
// ---------------------------------------------------------------------------

inline constexpr uint head_columns = contracts::head_columns;

__device__ void denoise_output(
    const DenoiseConstants& k, const ushort* features, const ushort* head,
    const float* head_bias, const float4* demod, const float4* history,
    float4* next_history, const ushort4* motion, const ushort4* full_albedo,
    std::uint8_t* radiance, std::uint64_t radiance_pitch) {
  __shared__ float scores_of[head_columns * conv_pixels];  // [output][pixel]
  const uint sg = warp_index();
  const uint lane = warp_lane();
  const uint y = blockIdx.y;
  const uint x0 = blockIdx.x * conv_pixels;
  const uint width = contracts::denoise_scale * k.padded_width;

  // The head: the features of pixels x0 .. x0 + 31 of row y, bordered
  // (past the grid's right edge read and never used), times its weights.
  {
    const ushort* row =
        features + ((y + 1u) * (k.padded_width + 2u) + x0 + 1u) * k.features;
    // pixel_tiles tiles of 16 pixels by output_blocks blocks of `block`
    // of the warp's simd_width outputs.
    constexpr uint output_blocks = contracts::simd_width / block;
    const uint first_output = sg * contracts::simd_width;
    matrix::Accumulator acc[pixel_tiles][output_blocks];
    for (uint m = 0; m < pixel_tiles; ++m) {
      for (uint n = 0; n < output_blocks; ++n) {
        acc[m][n] = matrix::zero_accumulator();
      }
    }
    for (uint kb = 0; kb < k.features / block; ++kb) {
      matrix::FragmentB b[output_blocks];
      for (uint n = 0; n < output_blocks; ++n) {
        b[n] = matrix::load_b_device(
            head + kb * block * head_columns + first_output + n * block,
            head_columns, lane);
      }
      for (uint m = 0; m < pixel_tiles; ++m) {
        const matrix::FragmentA a = matrix::load_a_device(
            row + m * tile_pixels * k.features + kb * block, k.features, lane);
        for (uint n = 0; n < output_blocks; ++n) {
          matrix::multiply_accumulate(acc[m][n], a, b[n]);
        }
      }
    }
    // Transposed, as Metal's simdgroup_store(..., transpose = true).
    for (uint m = 0; m < pixel_tiles; ++m) {
      for (uint n = 0; n < output_blocks; ++n) {
        for (uint i = 0; i < 4u; ++i) {
          scores_of[(first_output + n * block +
                     matrix::fragment_column(lane, i)) *
                        conv_pixels +
                    m * tile_pixels + matrix::fragment_row(lane, i)] =
              acc[m][n].c[i];
        }
      }
    }
  }
  __syncthreads();

  const uint q = sg;
  const uint2 p = make_uint2(x0 + lane, y);
  if (p.x >= k.padded_width) {
    return;
  }
  const uint2 x = subpixel(p, q);
  float scores[contracts::denoise_scores];
  for (uint s = 0; s < contracts::denoise_scores; ++s) {
    const uint o = s * contracts::denoise_subpixels + q;
    scores[s] = scores_of[o * conv_pixels + lane] + head_bias[o];
  }

  // The kernels: score si * 9 + t for tap t at spacing si (tap_at).
  float3 filtered[contracts::denoise_spacings];
  for (uint si = 0; si < contracts::denoise_spacings; ++si) {
    const uint spacing = device_spacing_pixels[si];
    float logits[contracts::denoise_taps];
    float largest = -INFINITY;
    for (uint t = 0; t < contracts::denoise_taps; ++t) {
      const int2 at = tap_at(p, spacing, t);
      logits[t] = inside_light(k, at) ? scores[si * contracts::denoise_taps + t]
                                      : contracts::masked_score;
      largest = max(largest, logits[t]);
    }
    float total = 0.0f;
    float3 sum = make_float3(0.0f);
    for (uint t = 0; t < contracts::denoise_taps; ++t) {
      const int2 at = tap_at(p, spacing, t);
      const float e = exp(logits[t] - largest);
      total += e;
      if (inside_light(k, at)) {
        sum += e * rgb(demod[uint(at.y) * k.padded_width + uint(at.x)]);
      }
    }
    filtered[si] = sum / total;
  }
  const uint w0 = contracts::denoise_spacings * contracts::denoise_taps;
  const float largest = max(scores[w0], max(scores[w0 + 1u], scores[w0 + 2u]));
  float3 mixed = make_float3(0.0f);
  float total = 0.0f;
  for (uint si = 0; si < contracts::denoise_spacings; ++si) {
    const float e = exp(scores[w0 + si] - largest);
    total += e;
    mixed += e * filtered[si];
  }
  mixed /= total;

  float2 mv = make_float2(0.0f);
  if (inside_light(k, make_int2(p))) {
    const float4 m4 = widen(motion[p.y * k.light_width + p.x]);
    mv = make_float2(m4.x, m4.y);
  }
  const Moved m = moved(k, history, x, mv);
  const float alpha =
      m.valid / (1.0f + exp(-scores[contracts::denoise_scores - 1u]));
  const float3 out = alpha * m.color + (1.0f - alpha) * mixed;
  next_history[x.y * width + x.x] = make_float4(out, 0.0f);
  if (x.x < k.output_width && x.y < k.output_height) {
    // the full-size albedo's rows are the output's width (denoise.h)
    const float3 albedo = rgb(widen(full_albedo[x.y * k.output_width + x.x]));
    write_rows(radiance, radiance_pitch, x,
               make_float4(out * albedo_factor(albedo), 1.0f));
  }
}

}  // namespace
}  // namespace shaders
}  // namespace serenity

// The kernels, each one call to its body above (cuda/device/module.h
// launches them by these names; denoise.cpp names their parameters). The
// convolutions and the output are launched in contract 18's blocks alone,
// which __launch_bounds__ states, so the compiler sizes their registers
// for 128 threads.

namespace contracts = serenity::contracts;

extern "C" __global__ void denoise_depth_histogram0(
    contracts::DenoiseConstants k, unsigned* histogram,
    const contracts::MedianState* state, const float* depth) {
  serenity::shaders::denoise_depth_histogram<0>(k, histogram, *state, depth);
}
extern "C" __global__ void denoise_depth_histogram1(
    contracts::DenoiseConstants k, unsigned* histogram,
    const contracts::MedianState* state, const float* depth) {
  serenity::shaders::denoise_depth_histogram<1>(k, histogram, *state, depth);
}
extern "C" __global__ void denoise_depth_histogram2(
    contracts::DenoiseConstants k, unsigned* histogram,
    const contracts::MedianState* state, const float* depth) {
  serenity::shaders::denoise_depth_histogram<2>(k, histogram, *state, depth);
}

extern "C" __global__ void denoise_depth_select0(
    unsigned* histogram, contracts::MedianState* state) {
  serenity::shaders::denoise_depth_select<0>(histogram, *state);
}
extern "C" __global__ void denoise_depth_select1(
    unsigned* histogram, contracts::MedianState* state) {
  serenity::shaders::denoise_depth_select<1>(histogram, *state);
}
extern "C" __global__ void denoise_depth_select2(
    unsigned* histogram, contracts::MedianState* state) {
  serenity::shaders::denoise_depth_select<2>(histogram, *state);
}

extern "C" __global__ void denoise_input(
    contracts::DenoiseConstants k, const contracts::MedianState* state,
    const float4* history, ushort4* input, float4* demod,
    const std::uint8_t* light, std::uint64_t light_pitch, const ushort4* albedo,
    const ushort4* normal, const float* depth, const ushort4* motion) {
  serenity::shaders::denoise_input(k, *state, history, input, demod, light,
                                   light_pitch, albedo, normal, depth, motion);
}

extern "C" __global__ void __launch_bounds__(contracts::denoise_threads)
    conv3x3_n1(contracts::ConvParams p, const ushort4* input,
               const ushort4* weights, const float* bias, ushort4* output,
               const ushort4* coarse) {
  serenity::shaders::conv3x3<1>(p, input, weights, bias, output, coarse);
}
extern "C" __global__ void __launch_bounds__(contracts::denoise_threads)
    conv3x3_n2(contracts::ConvParams p, const ushort4* input,
               const ushort4* weights, const float* bias, ushort4* output,
               const ushort4* coarse) {
  serenity::shaders::conv3x3<2>(p, input, weights, bias, output, coarse);
}
extern "C" __global__ void __launch_bounds__(contracts::denoise_threads)
    conv3x3_n3(contracts::ConvParams p, const ushort4* input,
               const ushort4* weights, const float* bias, ushort4* output,
               const ushort4* coarse) {
  serenity::shaders::conv3x3<3>(p, input, weights, bias, output, coarse);
}
extern "C" __global__ void __launch_bounds__(contracts::denoise_threads)
    conv3x3_n4(contracts::ConvParams p, const ushort4* input,
               const ushort4* weights, const float* bias, ushort4* output,
               const ushort4* coarse) {
  serenity::shaders::conv3x3<4>(p, input, weights, bias, output, coarse);
}

extern "C" __global__ void denoise_maxpool(contracts::ResampleParams p,
                                           const ushort4* input,
                                           ushort4* output) {
  serenity::shaders::denoise_maxpool(p, input, output);
}

extern "C" __global__ void __launch_bounds__(contracts::denoise_threads)
    denoise_output(contracts::DenoiseConstants k,
                   const unsigned short* features, const unsigned short* head,
                   const float* head_bias, const float4* demod,
                   const float4* history, float4* next_history,
                   const ushort4* motion, const ushort4* full_albedo,
                   std::uint8_t* radiance, std::uint64_t radiance_pitch) {
  serenity::shaders::denoise_output(k, features, head, head_bias, demod,
                                    history, next_history, motion, full_albedo,
                                    radiance, radiance_pitch);
}
