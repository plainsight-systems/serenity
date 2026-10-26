// The denoise pass's kernels (passes/denoise/denoise.h states the pass, its
// dispatches and their costs; core/contracts/denoise.h the network's
// constants). training/model.py, Denoiser.frame, step by step; each kernel
// names the step of that docstring it carries out:
//
//   denoise_depth_histogram<L>, denoise_depth_select<L>   the input's depth
//       normalization's median (training/data.py, input_frame), exactly;
//   denoise_input       steps 1 and 2's input: the 30 channels a pixel,
//                       planar, for the network's tensor;
//   denoise_output      the head (1 x 1), step 3's kernels, step 4's blend
//                       and step 5's remodulation, one output pixel a thread.
//
// The U-Net between them is the network's Metal package, on Apple's own
// kernels (denoise.h).
//
// Precision. The network's input and features are stored as half floats,
// as its tensors are; the head's sums are accumulated in float on
// simdgroup matrices. The light, its demodulation (up to 50 x the
// radiance, past half's range), the history, the head's scores and
// everything after them are float. Every sum here runs in a fixed order, a
// simdgroup's matrix product or a thread's loop, never an atomic (GDSA.2).
//
// Threadgroup memory only where it reorders access (GPU.5): the head's
// scores, transposed for the lanes. Staging the output kernel's taps was
// measured and gained nothing.
//
// The input and features are planar (metal/frame/denoise_images.h): a
// channel's pixels row after row, then the next channel's.

#include <metal_atomic>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

#include "core/contracts/denoise.h"
#include "metal/device/layout.metal.h"
#include "metal/passes/bindings.h"

using namespace metal;
using serenity::shaders::to_float3;
namespace binding = serenity::bindings::denoise;
namespace contracts = serenity::contracts;

namespace {

// OIDN's PU transfer (training/loss.py, pu): radiance >= 0 to about [0, 1].
inline float pu(float y) {
  y = max(y, 0.0f);
  float v;
  if (y <= contracts::pu_y0) {
    v = contracts::pu_a * y;
  } else if (y <= contracts::pu_y1) {
    v = contracts::pu_b * pow(y, contracts::pu_c) + contracts::pu_d;
  } else {
    v = contracts::pu_e * log(y + contracts::pu_f) + contracts::pu_g;
  }
  return v * contracts::pu_norm_scale;
}

inline float3 albedo_factor(float3 albedo) {
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
inline Moved moved(constant contracts::DenoiseConstants& k,
                   device const float4* history, uint2 x, float2 mv) {
  if (k.history == 0u) {
    return Moved{float3(0.0f), 0.0f};
  }
  const float sx = float(x.x) + 0.5f + float(contracts::denoise_scale) * mv.x;
  const float sy = float(x.y) + 0.5f + float(contracts::denoise_scale) * mv.y;
  const float w = float(contracts::denoise_scale * k.light_width);
  const float h = float(contracts::denoise_scale * k.light_height);
  if (!(sx >= 0.0f && sx <= w && sy >= 0.0f && sy <= h)) {
    return Moved{float3(0.0f), 0.0f};
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
      mix(history[y0 * width + x0].rgb, history[y0 * width + x1].rgb, fx);
  const float3 bottom =
      mix(history[y1 * width + x0].rgb, history[y1 * width + x1].rgb, fx);
  return Moved{mix(top, bottom, fy), 1.0f};
}

inline bool inside_light(constant contracts::DenoiseConstants& k, int2 p) {
  return p.x >= 0 && p.y >= 0 && uint(p.x) < k.light_width &&
         uint(p.y) < k.light_height;
}

// Tap t of the kernel at `spacing` around input pixel p: p + spacing (i, j),
// t = (j + 1) * 3 + (i + 1).
inline int2 tap_at(uint2 p, uint spacing, uint t) {
  return int2(p) + int(spacing) * int2(int(t % 3u) - 1, int(t / 3u) - 1);
}

// The depth median's radix selection (below): level 0 selects by a depth's
// bits from top_shift up, levels 1 and 2 by level_bits more each, each
// middle rank in its own half of the bins.
constant constexpr uint top_shift = 20u;
constant constexpr uint level_bits = 10u;
constant constexpr uint level_bins = 1u << level_bits;
constant constexpr uint level_mask = level_bins - 1u;
static_assert(2u * level_bins == contracts::median_bins,
              "levels 1 and 2 hold both ranks' bins");

}  // namespace

// ---------------------------------------------------------------------------
// The depth's median over the light's surface pixels (depth > 0), as
// np.median: level 0 counts the float's bits 30..20 (the sign is 0), level 1
// bits 19..10 and level 2 bits 9..0, the last two only of depths whose
// higher bits are each middle rank's so far. Each threadgroup counts into
// its own bins and adds them to the device's once (GDSA.5); integers, so
// the result is the same whatever the order (GDSA.2).
// ---------------------------------------------------------------------------

template <uint Level>
kernel void denoise_depth_histogram(
    constant contracts::DenoiseConstants& k [[buffer(binding::constants)]],
    device atomic_uint* histogram [[buffer(binding::histogram)]],
    device const contracts::MedianState& state [[buffer(binding::median)]],
    texture2d<float, access::read> depth [[texture(binding::depth)]],
    uint2 pixel [[thread_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]],
    uint2 size [[threads_per_threadgroup]]) {
  const uint index = local.y * size.x + local.x;
  const uint threads = size.x * size.y;
  threadgroup atomic_uint bins[contracts::median_bins];
  for (uint b = index; b < contracts::median_bins; b += threads) {
    atomic_store_explicit(&bins[b], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (pixel.x < k.light_width && pixel.y < k.light_height) {
    const float z = depth.read(pixel).r;
    if (z > 0.0f) {
      const uint bits = as_type<uint>(z);
      if (Level == 0u) {
        atomic_fetch_add_explicit(&bins[bits >> top_shift], 1u,
                                  memory_order_relaxed);
      } else {
        const uint shift = Level == 1u ? top_shift : top_shift - level_bits;
        const uint bin = (bits >> (shift - level_bits)) & level_mask;
        for (uint r = 0; r < 2u; ++r) {
          if (state.count != 0u && (bits >> shift) == state.prefix[r]) {
            atomic_fetch_add_explicit(&bins[r * level_bins + bin], 1u,
                                      memory_order_relaxed);
          }
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint b = index; b < contracts::median_bins; b += threads) {
    const uint n = atomic_load_explicit(&bins[b], memory_order_relaxed);
    if (n != 0u) {
      atomic_fetch_add_explicit(&histogram[b], n, memory_order_relaxed);
    }
  }
}

// One thread walks the bins (2048 at most, a few microseconds); then the
// threadgroup clears them for the next level or frame.
template <uint Level>
kernel void denoise_depth_select(device uint* histogram
                                 [[buffer(binding::histogram)]],
                                 device contracts::MedianState& state
                                 [[buffer(binding::median)]],
                                 uint index [[thread_index_in_threadgroup]],
                                 uint threads [[threads_per_threadgroup]]) {
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
        state.median = (as_type<float>(state.prefix[0]) +
                        as_type<float>(state.prefix[1])) *
                       0.5f;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
  for (uint b = index; b < contracts::median_bins; b += threads) {
    histogram[b] = 0u;
  }
}

template [[host_name("denoise_depth_histogram0")]] kernel void
denoise_depth_histogram<0>(constant contracts::DenoiseConstants&,
                           device atomic_uint*,
                           device const contracts::MedianState&,
                           texture2d<float, access::read>, uint2, uint2, uint2);
template [[host_name("denoise_depth_histogram1")]] kernel void
denoise_depth_histogram<1>(constant contracts::DenoiseConstants&,
                           device atomic_uint*,
                           device const contracts::MedianState&,
                           texture2d<float, access::read>, uint2, uint2, uint2);
template [[host_name("denoise_depth_histogram2")]] kernel void
denoise_depth_histogram<2>(constant contracts::DenoiseConstants&,
                           device atomic_uint*,
                           device const contracts::MedianState&,
                           texture2d<float, access::read>, uint2, uint2, uint2);
template [[host_name("denoise_depth_select0")]] kernel void
denoise_depth_select<0>(device uint*, device contracts::MedianState&, uint,
                        uint);
template [[host_name("denoise_depth_select1")]] kernel void
denoise_depth_select<1>(device uint*, device contracts::MedianState&, uint,
                        uint);
template [[host_name("denoise_depth_select2")]] kernel void
denoise_depth_select<2>(device uint*, device contracts::MedianState&, uint,
                        uint);

// ---------------------------------------------------------------------------
// The network's input, one thread an input pixel of the padded grid: zero
// outside the light's extent (training/evaluate.py's zero padding); inside
// it, training/data.py's input_frame and model.py's step 1 and 2 inputs.
// Also the demodulated light, which the output's kernels average.
// ---------------------------------------------------------------------------

kernel void denoise_input(
    constant contracts::DenoiseConstants& k [[buffer(binding::constants)]],
    device const contracts::MedianState& state [[buffer(binding::median)]],
    device const float4* history [[buffer(binding::history)]],
    device half* input [[buffer(binding::output)]],
    device float4* demod [[buffer(binding::demod)]],
    texture2d<float, access::read> light [[texture(binding::light)]],
    texture2d<float, access::read> albedo [[texture(binding::albedo)]],
    texture2d<float, access::read> normal [[texture(binding::normal)]],
    texture2d<float, access::read> depth [[texture(binding::depth)]],
    texture2d<float, access::read> motion [[texture(binding::motion)]],
    uint2 p [[thread_position_in_grid]]) {
  if (p.x >= k.padded_width || p.y >= k.padded_height) {
    return;
  }
  float values[contracts::network_inputs];
  for (uint c = 0; c < contracts::network_inputs; ++c) {
    values[c] = 0.0f;
  }
  float3 d = float3(0.0f);
  if (inside_light(k, int2(p))) {
    const float3 a = clamp(albedo.read(p).rgb, 0.0f, 1.0f);
    d = light.read(p).rgb / albedo_factor(a);
    const float3 n = normal.read(p).xyz;
    const float z = depth.read(p).r;
    const float2 mv = motion.read(p).xy;
    values[contracts::input_radiance + 0u] = pu(d.r);
    values[contracts::input_radiance + 1u] = pu(d.g);
    values[contracts::input_radiance + 2u] = pu(d.b);
    values[contracts::input_albedo + 0u] = a.r;
    values[contracts::input_albedo + 1u] = a.g;
    values[contracts::input_albedo + 2u] = a.b;
    values[contracts::input_normal + 0u] = dot(n, to_float3(k.right));
    values[contracts::input_normal + 1u] = dot(n, to_float3(k.up));
    values[contracts::input_normal + 2u] = dot(n, to_float3(k.backward));
    values[contracts::input_depth] =
        z > 0.0f ? log(1.0f + z / state.median) : 0.0f;
    values[contracts::input_motion + 0u] = mv.x;
    values[contracts::input_motion + 1u] = mv.y;
    // pixel_unshuffle: subpixel q = dy * 2 + dx of each history channel c
    // (contract 18), and its validity.
    constexpr uint subpixels = contracts::denoise_subpixels;
    for (uint q = 0; q < subpixels; ++q) {
      const uint2 x = contracts::denoise_scale * p + uint2(q & 1u, q >> 1);
      const Moved m = moved(k, history, x, mv);
      values[contracts::input_history + 0u * subpixels + q] = pu(m.color.r);
      values[contracts::input_history + 1u * subpixels + q] = pu(m.color.g);
      values[contracts::input_history + 2u * subpixels + q] = pu(m.color.b);
      values[contracts::input_validity + q] = m.valid;
    }
    // The frame's camera jitter, in input pixels, at every pixel of the
    // light (contract 18), 0 in the padding as every input is.
    values[contracts::input_jitter + 0u] = k.jitter_x;
    values[contracts::input_jitter + 1u] = k.jitter_y;
  }
  // Planar, as the network's input tensor is (metal/frame/
  // denoise_images.h): channel c's plane, row p.y, pixel p.x, so a
  // simdgroup's 32 pixels of a row write 64 adjacent bytes a channel
  // (GPU.2).
  const uint pitch = contracts::tensor_pitch(k.padded_width);
  const uint plane = pitch * k.padded_height;
  const uint at = p.y * pitch + p.x;
  for (uint c = 0; c < contracts::network_inputs; ++c) {
    input[c * plane + at] = half(values[c]);
  }
  demod[p.y * k.padded_width + p.x] = float4(d, 0.0f);
}

// The head's tile (contract 18): a row of conv_pixels input pixels a
// threadgroup, in blocks of `block` (contract 18's channel_block), a
// simdgroup matrix's side.
constant constexpr uint conv_pixels = contracts::conv_pixels;
constant constexpr uint block = contracts::channel_block;
static_assert(block == 8u, "a block is a simdgroup matrix's 8 channels");

// ---------------------------------------------------------------------------
// The output, over twice the padded grid (training/evaluate.py runs the
// head and kernels over the padding too, and the history it keeps is that
// size). A threadgroup is four simdgroups and a row of 32 input pixels.
// First the head, a 1 x 1 convolution, as a matrix product on simdgroup
// matrices: the 32 pixels' last features (half, read straight from their
// buffer) times the head's weights (half, laid out [feature][output], 124
// padded to head_columns), float sums, each simdgroup 32 of the outputs;
// the scores stored transposed in threadgroup memory, an output's 32
// pixels side by side, so the lanes reading them hit different banks, and
// never in device memory (GDSA.16). Half weights, as the convolutions':
// float weights made the pass 9 ms slower at a 1728 x 1117 light and its
// output no closer to PyTorch's (docs/research/2026-10-15-denoiser-live.md).
// Then a thread an output pixel X, its subpixel q its simdgroup, p = X / 2 its
// input pixel: per spacing, the softmax of its 9 taps' scores, a tap outside
// the light's extent masked, over the demodulated light; the softmax of the 3
// spacing scores mixing the three; the blend with the moved history by
// sigmoid(score 30) x valid; the blend kept as the next frame's history and,
// inside the output, multiplied by the full-size albedo's factor into the
// radiance image.
// ---------------------------------------------------------------------------

constant constexpr uint head_columns = contracts::head_columns;

kernel void denoise_output(
    constant contracts::DenoiseConstants& k [[buffer(binding::constants)]],
    device const half* features [[buffer(binding::input)]],
    device const half* head [[buffer(binding::weights)]],
    device const float* head_bias [[buffer(binding::bias)]],
    device const float4* demod [[buffer(binding::demod)]],
    device const float4* history [[buffer(binding::history)]],
    device float4* next_history [[buffer(binding::next_history)]],
    texture2d<float, access::read> motion [[texture(binding::motion)]],
    texture2d<float, access::read> full_albedo
    [[texture(binding::full_albedo)]],
    texture2d<float, access::write> radiance [[texture(binding::radiance)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scores_of[head_columns * conv_pixels];  // [output][pixel]
  const uint y = group.y;
  const uint x0 = group.x * conv_pixels;
  const uint width = contracts::denoise_scale * k.padded_width;

  // The head: the features of pixels x0 .. x0 + 31 of row y, bordered
  // (past the grid's right edge read and never used), times its weights.
  {
    // planar (denoise_images.h): channel c of pixel x at c * plane + y *
    // pitch + x, the pitch a multiple of conv_pixels, so a row's 32 pixels
    // stay in its row; past the grid's right edge read and never used
    const uint pitch = contracts::tensor_pitch(k.padded_width);
    const uint plane = pitch * k.padded_height;
    device const half* row = features + y * pitch + x0;
    // pixel_blocks blocks of `block` pixels by output_blocks blocks of
    // `block` of the simdgroup's simd_width outputs.
    constexpr uint pixel_blocks = conv_pixels / block;
    constexpr uint output_blocks = contracts::simd_width / block;
    const uint first_output = sg * contracts::simd_width;
    simdgroup_float8x8 acc[pixel_blocks][output_blocks];
    for (uint m = 0; m < pixel_blocks; ++m) {
      for (uint n = 0; n < output_blocks; ++n) {
        acc[m][n] = simdgroup_float8x8(0.0f);
      }
    }
    for (uint kb = 0; kb < k.features / block; ++kb) {
      simdgroup_half8x8 b[output_blocks];
      for (uint n = 0; n < output_blocks; ++n) {
        simdgroup_load(
            b[n], head + kb * block * head_columns + first_output + n * block,
            head_columns);
      }
      for (uint m = 0; m < pixel_blocks; ++m) {
        simdgroup_half8x8 a;
        // 8 pixels by 8 channels: the planes' 8 rows of 8 pixels,
        // transposed
        simdgroup_load(a, row + (kb * block) * plane + m * block, plane,
                       ulong2(0), true);
        for (uint n = 0; n < output_blocks; ++n) {
          simdgroup_multiply_accumulate(acc[m][n], a, b[n], acc[m][n]);
        }
      }
    }
    for (uint m = 0; m < pixel_blocks; ++m) {
      for (uint n = 0; n < output_blocks; ++n) {
        simdgroup_store(
            acc[m][n],
            scores_of + (first_output + n * block) * conv_pixels + m * block,
            conv_pixels, ulong2(0), true);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint q = sg;
  const uint2 p = uint2(x0 + lane, y);
  if (p.x >= k.padded_width) {
    return;
  }
  const uint2 x = contracts::denoise_scale * p + uint2(q & 1u, q >> 1);
  float scores[contracts::denoise_scores];
  for (uint s = 0; s < contracts::denoise_scores; ++s) {
    const uint o = s * contracts::denoise_subpixels + q;
    scores[s] = scores_of[o * conv_pixels + lane] + head_bias[o];
  }

  // The kernels: score si * 9 + t for tap t at spacing si (tap_at).
  float3 filtered[contracts::denoise_spacings];
  for (uint si = 0; si < contracts::denoise_spacings; ++si) {
    const uint spacing = contracts::spacing_pixels[si];
    float logits[contracts::denoise_taps];
    float largest = -INFINITY;
    for (uint t = 0; t < contracts::denoise_taps; ++t) {
      const int2 at = tap_at(p, spacing, t);
      logits[t] = inside_light(k, at) ? scores[si * contracts::denoise_taps + t]
                                      : contracts::masked_score;
      largest = max(largest, logits[t]);
    }
    float total = 0.0f;
    float3 sum = float3(0.0f);
    for (uint t = 0; t < contracts::denoise_taps; ++t) {
      const int2 at = tap_at(p, spacing, t);
      const float e = exp(logits[t] - largest);
      total += e;
      if (inside_light(k, at)) {
        sum += e * demod[uint(at.y) * k.padded_width + uint(at.x)].rgb;
      }
    }
    filtered[si] = sum / total;
  }
  const uint w0 = contracts::denoise_spacings * contracts::denoise_taps;
  const float largest = max(scores[w0], max(scores[w0 + 1u], scores[w0 + 2u]));
  float3 mixed = float3(0.0f);
  float total = 0.0f;
  for (uint si = 0; si < contracts::denoise_spacings; ++si) {
    const float e = exp(scores[w0 + si] - largest);
    total += e;
    mixed += e * filtered[si];
  }
  mixed /= total;

  const float2 mv = inside_light(k, int2(p)) ? motion.read(p).xy : float2(0.0f);
  const Moved m = moved(k, history, x, mv);
  const float alpha =
      m.valid / (1.0f + exp(-scores[contracts::denoise_scores - 1u]));
  const float3 out = alpha * m.color + (1.0f - alpha) * mixed;
  next_history[x.y * width + x.x] = float4(out, 0.0f);
  if (x.x < k.output_width && x.y < k.output_height) {
    radiance.write(float4(out * albedo_factor(full_albedo.read(x).rgb), 1.0f),
                   x);
  }
}
