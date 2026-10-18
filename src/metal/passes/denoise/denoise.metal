// The denoise pass's kernels (passes/denoise/denoise.h states the pass, its
// dispatches and their costs; core/contracts/denoise.h the network's
// constants). training/model.py, Denoiser.frame, step by step; each kernel
// names the step of that docstring it carries out:
//
//   denoise_depth_histogram<L>, denoise_depth_select<L>   the input's depth
//       normalization's median (training/data.py, input_frame), exactly;
//   denoise_input       steps 1 and 2's input: the 28 channels a pixel;
//   conv3x3_n<NT>       step 2's 3 x 3 convolutions, with ReLU;
//   denoise_maxpool     the U-Net's 2 x 2 max pool (its nearest upsampling
//                       is conv3x3's, as a decoder reads the level below);
//   denoise_output      the head (1 x 1), step 3's kernels, step 4's blend
//                       and step 5's remodulation, one output pixel a thread.
//
// Precision. Activations and weights are stored as half floats and every
// sum is accumulated in float: the convolutions on simdgroup matrices, half
// inputs and float accumulators. The light, its demodulation (up to 50 x
// the radiance, past half's range), the history, the head's scores and
// everything after them are float. The match against PyTorch's float32 is
// measured (docs/research/2026-10-15-denoiser-live.md). Every sum runs in a
// fixed order, a simdgroup's matrix product or a thread's loop, never an
// atomic: the same inputs give the same image, run after run (GDSA.2).
//
// Threadgroup memory only where it is reused or reorders access (GPU.5): a
// convolution's input halo and weights, read by every tap of every
// simdgroup, and the head's scores, transposed for the lanes. Staging the
// output kernel's taps too was measured and gained nothing, and so was
// aliasing the convolution's halo and results to raise occupancy (GPU.3).
//
// Buffers of activations hold a level's pixels with a one-pixel border of
// zeros on every side, channels innermost, rows `pitch` pixels apart: the
// convolutions' zero padding is read from memory, so the matrices load
// without a test per pixel.

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
    return contracts::albedo_floor + (1.0f - contracts::albedo_floor) * clamp(albedo, 0.0f, 1.0f);
}

struct Moved {
    float3 color;
    float valid;
};

// Step 1: last frame's output at output pixel `x` moved by `mv` (input
// pixels, old = new + mv): read bilinearly at x + 1/2 + 2 mv, clamped to
// the padded image's edge (grid_sample's border padding); valid when that
// point lies inside twice the light's extent (its edges included).
inline Moved moved(constant contracts::DenoiseConstants& k, device const float4* history, uint2 x, float2 mv) {
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
    const float3 top = mix(history[y0 * width + x0].rgb, history[y0 * width + x1].rgb, fx);
    const float3 bottom = mix(history[y1 * width + x0].rgb, history[y1 * width + x1].rgb, fx);
    return Moved{mix(top, bottom, fy), 1.0f};
}

inline bool inside_light(constant contracts::DenoiseConstants& k, int2 p) {
    return p.x >= 0 && p.y >= 0 && uint(p.x) < k.light_width && uint(p.y) < k.light_height;
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
static_assert(2u * level_bins == contracts::median_bins, "levels 1 and 2 hold both ranks' bins");

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
kernel void denoise_depth_histogram(constant contracts::DenoiseConstants& k [[buffer(binding::constants)]],
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
                atomic_fetch_add_explicit(&bins[bits >> top_shift], 1u, memory_order_relaxed);
            } else {
                const uint shift = Level == 1u ? top_shift : top_shift - level_bits;
                const uint bin = (bits >> (shift - level_bits)) & level_mask;
                for (uint r = 0; r < 2u; ++r) {
                    if (state.count != 0u && (bits >> shift) == state.prefix[r]) {
                        atomic_fetch_add_explicit(&bins[r * level_bins + bin], 1u, memory_order_relaxed);
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
kernel void denoise_depth_select(device uint* histogram [[buffer(binding::histogram)]],
                                 device contracts::MedianState& state [[buffer(binding::median)]],
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
                for (; b + 1u < bins && below + histogram[first + b] <= state.rank[r]; ++b) {
                    below += histogram[first + b];
                }
                state.rank[r] -= below;
                state.prefix[r] = Level == 0u ? b : (state.prefix[r] << level_bits) | b;
            }
            if (Level == 2u) {
                state.median = (as_type<float>(state.prefix[0]) + as_type<float>(state.prefix[1])) * 0.5f;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint b = index; b < contracts::median_bins; b += threads) {
        histogram[b] = 0u;
    }
}

template [[host_name("denoise_depth_histogram0")]] kernel void denoise_depth_histogram<0>(
    constant contracts::DenoiseConstants&, device atomic_uint*, device const contracts::MedianState&,
    texture2d<float, access::read>, uint2, uint2, uint2);
template [[host_name("denoise_depth_histogram1")]] kernel void denoise_depth_histogram<1>(
    constant contracts::DenoiseConstants&, device atomic_uint*, device const contracts::MedianState&,
    texture2d<float, access::read>, uint2, uint2, uint2);
template [[host_name("denoise_depth_histogram2")]] kernel void denoise_depth_histogram<2>(
    constant contracts::DenoiseConstants&, device atomic_uint*, device const contracts::MedianState&,
    texture2d<float, access::read>, uint2, uint2, uint2);
template [[host_name("denoise_depth_select0")]] kernel void denoise_depth_select<0>(
    device uint*, device contracts::MedianState&, uint, uint);
template [[host_name("denoise_depth_select1")]] kernel void denoise_depth_select<1>(
    device uint*, device contracts::MedianState&, uint, uint);
template [[host_name("denoise_depth_select2")]] kernel void denoise_depth_select<2>(
    device uint*, device contracts::MedianState&, uint, uint);

// ---------------------------------------------------------------------------
// The network's input, one thread an input pixel of the padded grid: zero
// outside the light's extent (training/evaluate.py's zero padding); inside
// it, training/data.py's input_frame and model.py's step 1 and 2 inputs.
// Also the demodulated light, which the output's kernels average.
// ---------------------------------------------------------------------------

kernel void denoise_input(constant contracts::DenoiseConstants& k [[buffer(binding::constants)]],
                          device const contracts::MedianState& state [[buffer(binding::median)]],
                          device const float4* history [[buffer(binding::history)]],
                          device half4* input [[buffer(binding::output)]],
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
    const uint pitch = k.padded_width + 2u;
    // A pixel's channels as half4s, a run of 64 bytes (GPU.2).
    constexpr uint quads = contracts::input_channels / 4u;
    device half4* out = input + ((p.y + 1u) * pitch + p.x + 1u) * quads;
    float values[contracts::input_channels];
    for (uint c = 0; c < contracts::input_channels; ++c) {
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
        values[contracts::input_depth] = z > 0.0f ? log(1.0f + z / state.median) : 0.0f;
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
    for (uint c = 0; c < quads; ++c) {
        out[c] = half4(float4(values[4u * c], values[4u * c + 1u], values[4u * c + 2u], values[4u * c + 3u]));
    }
    demod[p.y * k.padded_width + p.x] = float4(d, 0.0f);
}

// ---------------------------------------------------------------------------
// A 3 x 3 convolution, zero-padded by 1, with its bias and, unless the
// params say not, a ReLU: an implicit matrix product on simdgroup
// matrices, pixels by input channels (9 taps x in_blocks blocks of 8) times
// input channels by output channels. A threadgroup is 4 simdgroups, 4 rows
// of 32 pixels; each simdgroup computes its row's 32 pixels for NT blocks
// of 8 output channels (4 x NT accumulators, NT at most 4, past which the
// accumulators spill: measured, docs/research/2026-10-15-denoiser-live.md),
// a grid's z one such chunk of the outputs.
//
// For each block of 8 input channels the threadgroup first copies, with
// every thread, the block's 6 x 34 pixels of input (its 4 x 32 and the
// border the taps reach) and its 9 x 8 x NT x 8 weights into threadgroup
// memory, coalesced (GPU.2); every simdgroup's 9 taps then load their
// matrices from there, so each input value leaves device memory once a
// threadgroup, not 9 times a simdgroup (GDSA.16). Reads past the level's
// border are zeros. Results go through threadgroup memory, a block at a
// time, to be biased, rectified and narrowed to half.
// ---------------------------------------------------------------------------

// The tile (contract 18): conv_rows rows of conv_pixels, one a simdgroup,
// with the border its taps reach; blocks of `block` channels (contract
// 18's channel_block), a simdgroup matrix's side, two half4s a pixel.
constant constexpr uint conv_rows = contracts::conv_rows;
constant constexpr uint conv_pixels = contracts::conv_pixels;
constant constexpr uint halo_rows = conv_rows + 2u;
constant constexpr uint halo_pixels = conv_pixels + 2u;
constant constexpr uint conv_threads = contracts::denoise_threads;
constant constexpr uint conv_taps = 9u;
constant constexpr uint block = contracts::channel_block;
constant constexpr uint block_quads = block / 4u;
static_assert(block == 8u, "a block is a simdgroup matrix's 8 channels");

// Device buffers of half floats are read and written here a half4 at a
// time, typed so (ES.48: no cast of device pointers); their channel offsets
// and strides, counted in halves, are multiples of 4. The threadgroup tiles
// are written a half4 at a time and read by simdgroup_load, which takes
// halves: that one reinterpretation, of the tiles, is the kernel's.
template <uint NT>
kernel void conv3x3(constant contracts::ConvParams& p [[buffer(binding::params)]],
                    device const half4* input [[buffer(binding::input)]],
                    device const half4* weights [[buffer(binding::weights)]],
                    device const float* bias [[buffer(binding::bias)]],
                    device half4* output [[buffer(binding::output)]],
                    device const half4* coarse [[buffer(binding::coarse)]],
                    uint3 group [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint channels = NT * block;
    threadgroup half halo[halo_rows * halo_pixels * block];
    threadgroup half taps[conv_taps * block * channels];
    threadgroup float stage[conv_rows * conv_pixels * block];
    threadgroup half4* const halo4 = reinterpret_cast<threadgroup half4*>(halo);
    threadgroup half4* const taps4 = reinterpret_cast<threadgroup half4*>(taps);
    const uint tid = sg * contracts::simd_width + lane;
    const uint y0 = group.y * conv_rows;
    const uint x0 = group.x * conv_pixels;
    const uint y = y0 + sg;
    const bool active = y < p.height;  // every thread still copies and waits
    const uint first_out = group.z * channels;
    const uint out_width = p.out_blocks * block;  // the weights' row: every output channel

    // A row's conv_pixels pixels as pixel_blocks matrices of `block` pixels.
    constexpr uint pixel_blocks = conv_pixels / block;
    simdgroup_float8x8 acc[pixel_blocks][NT];
    for (uint m = 0; m < pixel_blocks; ++m) {
        for (uint n = 0; n < NT; ++n) {
            acc[m][n] = simdgroup_float8x8(0.0f);
        }
    }
    const uint in_stride = p.in_stride / 4u;  // in half4s, as the reads below
    const uint up_stride = p.up_stride / 4u;
    for (uint kb = 0; kb < p.in_blocks; ++kb) {
        // The block's input, bordered rows y0 .. y0 + conv_rows + 1 and
        // columns x0 .. x0 + conv_pixels + 1 (input pixels y0 - 1 .. and
        // x0 - 1 ..), block_quads half4s a pixel.
        for (uint i = tid; i < halo_rows * halo_pixels * block_quads; i += conv_threads) {
            const uint pixel = i / block_quads;
            const uint quad = i % block_quads;
            const uint row = y0 + pixel / halo_pixels;
            const uint column = x0 + pixel % halo_pixels;
            half4 v = half4(0.0h);
            if (kb < p.up_blocks) {
                // Upsampled: input pixel (column - 1, row - 1) reads the level
                // below's at half its coordinates; the border stays zero.
                if (row >= 1u && row <= p.height && column >= 1u && column <= p.width) {
                    v = coarse[(((row - 1u) / 2u + 1u) * p.up_pitch + (column - 1u) / 2u + 1u) * up_stride +
                               kb * block_quads + quad];
                }
            } else if (row <= p.height + 1u && column <= p.width + 1u) {
                v = input[(row * p.in_pitch + column) * in_stride + (kb - p.up_blocks) * block_quads + quad];
            }
            halo4[pixel * block_quads + quad] = v;
        }
        // Its weights: rows (tap, input channel), this chunk's outputs.
        constexpr uint quads = channels / 4u;
        const uint weight_quads = out_width / 4u;
        for (uint i = tid; i < conv_taps * block * quads; i += conv_threads) {
            const uint row = i / quads;
            const uint q = i % quads;
            const uint tap = row / block;
            const uint k = row % block;
            taps4[row * quads + q] =
                weights[((tap * p.in_blocks + kb) * block + k) * weight_quads + first_out / 4u + q];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (uint tap = 0; tap < conv_taps; ++tap) {
                const uint dy = tap / 3u;
                const uint dx = tap % 3u;
                simdgroup_half8x8 a[pixel_blocks];
                simdgroup_half8x8 b[NT];
                for (uint m = 0; m < pixel_blocks; ++m) {
                    simdgroup_load(a[m], halo + ((sg + dy) * halo_pixels + m * block + dx) * block, block);
                }
                for (uint n = 0; n < NT; ++n) {
                    simdgroup_load(b[n], taps + tap * block * channels + n * block, channels);
                }
                for (uint m = 0; m < pixel_blocks; ++m) {
                    for (uint n = 0; n < NT; ++n) {
                        simdgroup_multiply_accumulate(acc[m][n], a[m], b[n], acc[m][n]);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!active) {
        return;
    }
    threadgroup float* mine = stage + sg * conv_pixels * block;
    const uint x = x0 + lane;
    device half4* out = output + (((y + 1u) * p.out_pitch + x + 1u) * p.out_stride + p.out_offset + first_out) / 4u;
    // A block of channels at a time through the staging, each lane then
    // writing its own pixel's block, block_quads half4s, inside the level
    // only.
    for (uint n = 0; n < NT; ++n) {
        for (uint m = 0; m < pixel_blocks; ++m) {
            simdgroup_store(acc[m][n], mine + m * block * block, block);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (x < p.width) {
            threadgroup const float* values = mine + lane * block;
            device const float* b = bias + first_out + n * block;
            for (uint c = 0; c < block_quads; ++c) {
                const uint at = 4u * c;
                float4 v = float4(values[at], values[at + 1u], values[at + 2u], values[at + 3u]) +
                           float4(b[at], b[at + 1u], b[at + 2u], b[at + 3u]);
                if (p.relu != 0u) {
                    v = max(v, 0.0f);
                }
                out[n * block_quads + c] = half4(v);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template [[host_name("conv3x3_n1")]] kernel void conv3x3<1>(constant contracts::ConvParams&, device const half4*,
                                                             device const half4*, device const float*, device half4*,
                                                             device const half4*, uint3, uint, uint);
template [[host_name("conv3x3_n2")]] kernel void conv3x3<2>(constant contracts::ConvParams&, device const half4*,
                                                             device const half4*, device const float*, device half4*,
                                                             device const half4*, uint3, uint, uint);
template [[host_name("conv3x3_n3")]] kernel void conv3x3<3>(constant contracts::ConvParams&, device const half4*,
                                                             device const half4*, device const float*, device half4*,
                                                             device const half4*, uint3, uint, uint);
template [[host_name("conv3x3_n4")]] kernel void conv3x3<4>(constant contracts::ConvParams&, device const half4*,
                                                             device const half4*, device const float*, device half4*,
                                                             device const half4*, uint3, uint, uint);

// ---------------------------------------------------------------------------
// 2 x 2 max pool, four channels a thread, between bordered buffers (the
// destination's size in the params). One
// dimension of threads, channels fastest, so a SIMD group's writes are one
// run of memory (GPU.2).
// ---------------------------------------------------------------------------

// Its buffers typed half4, as conv3x3's: every stride and offset a multiple
// of 4 halves.
kernel void denoise_maxpool(constant contracts::ResampleParams& p [[buffer(binding::params)]],
                            device const half4* input [[buffer(binding::input)]],
                            device half4* output [[buffer(binding::output)]],
                            uint i [[thread_position_in_grid]]) {
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
    const auto read = [&](uint sx, uint sy) { return input[((sy + 1u) * p.in_pitch + sx + 1u) * in_stride + c]; };
    const half4 v = max(max(read(2u * x, 2u * y), read(2u * x + 1u, 2u * y)),
                        max(read(2u * x, 2u * y + 1u), read(2u * x + 1u, 2u * y + 1u)));
    output[((y + 1u) * p.out_pitch + x + 1u) * (p.out_stride / 4u) + p.out_offset / 4u + q] = v;
}

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
// output no closer to PyTorch's (docs/research/2026-10-15-denoiser-live.md). Then a thread an output pixel X, its
// subpixel q its simdgroup, p = X / 2 its input pixel: per spacing, the
// softmax of its 9 taps' scores, a tap outside the light's extent masked,
// over the demodulated light; the softmax of the 3 spacing scores mixing
// the three; the blend with the moved history by sigmoid(score 30) x
// valid; the blend kept as the next frame's history and, inside the
// output, multiplied by the full-size albedo's factor into the radiance
// image.
// ---------------------------------------------------------------------------

constant constexpr uint head_columns = contracts::head_columns;

kernel void denoise_output(constant contracts::DenoiseConstants& k [[buffer(binding::constants)]],
                           device const half* features [[buffer(binding::input)]],
                           device const half* head [[buffer(binding::weights)]],
                           device const float* head_bias [[buffer(binding::bias)]],
                           device const float4* demod [[buffer(binding::demod)]],
                           device const float4* history [[buffer(binding::history)]],
                           device float4* next_history [[buffer(binding::next_history)]],
                           texture2d<float, access::read> motion [[texture(binding::motion)]],
                           texture2d<float, access::read> full_albedo [[texture(binding::full_albedo)]],
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
        device const half* row = features + ((y + 1u) * (k.padded_width + 2u) + x0 + 1u) * k.features;
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
                simdgroup_load(b[n], head + kb * block * head_columns + first_output + n * block, head_columns);
            }
            for (uint m = 0; m < pixel_blocks; ++m) {
                simdgroup_half8x8 a;
                simdgroup_load(a, row + m * block * k.features + kb * block, k.features);
                for (uint n = 0; n < output_blocks; ++n) {
                    simdgroup_multiply_accumulate(acc[m][n], a, b[n], acc[m][n]);
                }
            }
        }
        for (uint m = 0; m < pixel_blocks; ++m) {
            for (uint n = 0; n < output_blocks; ++n) {
                simdgroup_store(acc[m][n], scores_of + (first_output + n * block) * conv_pixels + m * block,
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
            logits[t] = inside_light(k, at) ? scores[si * contracts::denoise_taps + t] : contracts::masked_score;
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
    const float alpha = m.valid / (1.0f + exp(-scores[contracts::denoise_scores - 1u]));
    const float3 out = alpha * m.color + (1.0f - alpha) * mixed;
    next_history[x.y * width + x.x] = float4(out, 0.0f);
    if (x.x < k.output_width && x.y < k.output_height) {
        radiance.write(float4(out * albedo_factor(full_albedo.read(x).rgb), 1.0f), x);
    }
}
