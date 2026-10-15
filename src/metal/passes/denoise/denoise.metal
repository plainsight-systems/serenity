// The denoise pass's kernels (passes/denoise/denoise.h states the pass, its
// dispatches and their costs; core/contracts/denoise.h the network's
// constants). training/model.py, Denoiser.frame, step by step; each kernel
// names the step of that docstring it carries out:
//
//   denoise_depth_histogram<L>, denoise_depth_select<L>   the input's depth
//       normalization's median (training/data.py, input_frame), exactly;
//   denoise_input       steps 1 and 2's input: the 28 channels a pixel;
//   conv3x3_n<NT>       step 2's 3 x 3 convolutions, with ReLU;
//   denoise_maxpool, denoise_upsample   the U-Net's 2 x 2 max pool and
//                       nearest upsampling into the concatenation;
//   denoise_output      the head (1 x 1), step 3's kernels, step 4's blend
//                       and step 5's remodulation, one output pixel a thread.
//
// Precision. Activations and weights are stored as half floats and every
// sum is accumulated in float: the convolutions on simdgroup matrices, half
// inputs and float accumulators. The light, its demodulation (up to 50 x
// the radiance, past half's range), the history, the head's scores and
// everything after them are float. The match against PyTorch's float32 is
// measured (docs/research/2026-10-15-denoiser-live.md).
//
// Buffers of activations hold a level's pixels with a one-pixel border of
// zeros on every side, channels innermost, rows `pitch` pixels apart: the
// convolutions' zero padding is read from memory, so the matrices load
// without a test per pixel.

#include <metal_atomic>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

#include "core/contracts/denoise.h"
#include "metal/passes/bindings.h"

using namespace metal;
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
                atomic_fetch_add_explicit(&bins[bits >> 20], 1u, memory_order_relaxed);
            } else {
                const uint shift = Level == 1u ? 20u : 10u;
                const uint bin = (bits >> (shift - 10u)) & 1023u;
                for (uint r = 0; r < 2u; ++r) {
                    if (state.count != 0u && (bits >> shift) == state.prefix[r]) {
                        atomic_fetch_add_explicit(&bins[r * 1024u + bin], 1u, memory_order_relaxed);
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
                const uint first = Level == 0u ? 0u : r * 1024u;
                const uint bins = Level == 0u ? contracts::median_bins : 1024u;
                uint below = 0u;
                uint b = 0u;
                for (; b + 1u < bins && below + histogram[first + b] <= state.rank[r]; ++b) {
                    below += histogram[first + b];
                }
                state.rank[r] -= below;
                state.prefix[r] = Level == 0u ? b : (state.prefix[r] << 10) | b;
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
    const uint pitch = k.padded_width + 2u;
    device half* out = input + ((p.y + 1u) * pitch + p.x + 1u) * contracts::input_channels;
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
        values[0] = pu(d.r);
        values[1] = pu(d.g);
        values[2] = pu(d.b);
        values[3] = a.r;
        values[4] = a.g;
        values[5] = a.b;
        values[6] = dot(n, float3(k.right.x, k.right.y, k.right.z));
        values[7] = dot(n, float3(k.up.x, k.up.y, k.up.z));
        values[8] = dot(n, float3(k.backward.x, k.backward.y, k.backward.z));
        values[9] = z > 0.0f ? log(1.0f + z / state.median) : 0.0f;
        values[10] = mv.x;
        values[11] = mv.y;
        // pixel_unshuffle: channel 12 + c * 4 + dy * 2 + dx, and 24 + dy * 2 + dx.
        for (uint q = 0; q < contracts::denoise_subpixels; ++q) {
            const uint2 x = contracts::denoise_scale * p + uint2(q & 1u, q >> 1);
            const Moved m = moved(k, history, x, mv);
            values[12 + q] = pu(m.color.r);
            values[16 + q] = pu(m.color.g);
            values[20 + q] = pu(m.color.b);
            values[24 + q] = m.valid;
        }
    }
    for (uint c = 0; c < contracts::input_channels; ++c) {
        out[c] = half(values[c]);
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

constant constexpr uint conv_rows = 4u;
constant constexpr uint conv_pixels = 32u;
constant constexpr uint halo_rows = conv_rows + 2u;
constant constexpr uint halo_pixels = conv_pixels + 2u;
constant constexpr uint conv_threads = conv_rows * 32u;

template <uint NT>
kernel void conv3x3(constant contracts::ConvParams& p [[buffer(binding::params)]],
                    device const half* input [[buffer(binding::input)]],
                    device const half* weights [[buffer(binding::weights)]],
                    device const float* bias [[buffer(binding::bias)]],
                    device half* output [[buffer(binding::output)]],
                    uint3 group [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint channels = NT * 8u;
    threadgroup half halo[halo_rows * halo_pixels * 8u];
    threadgroup half taps[9u * 8u * channels];
    threadgroup float stage[conv_rows * conv_pixels * 8u];
    const uint tid = sg * 32u + lane;
    const uint y0 = group.y * conv_rows;
    const uint x0 = group.x * conv_pixels;
    const uint y = y0 + sg;
    const bool active = y < p.height;  // every thread still copies and waits
    const uint first_out = group.z * channels;
    const uint out_width = p.out_blocks * 8u;  // the weights' row: every output channel

    simdgroup_float8x8 acc[4][NT];
    for (uint m = 0; m < 4u; ++m) {
        for (uint n = 0; n < NT; ++n) {
            acc[m][n] = simdgroup_float8x8(0.0f);
        }
    }
    for (uint kb = 0; kb < p.in_blocks; ++kb) {
        // The block's input, bordered rows y0 .. y0 + 5 and columns x0 ..
        // x0 + 33 (input pixels y0 - 1 .. and x0 - 1 ..), two half4 a pixel.
        for (uint i = tid; i < halo_rows * halo_pixels * 2u; i += conv_threads) {
            const uint pixel = i / 2u;
            const uint row = y0 + pixel / halo_pixels;
            const uint column = x0 + pixel % halo_pixels;
            half4 v = half4(0.0h);
            if (row <= p.height + 1u && column <= p.width + 1u) {
                v = *reinterpret_cast<device const half4*>(input + (row * p.in_pitch + column) * p.in_stride +
                                                           kb * 8u + (i % 2u) * 4u);
            }
            *reinterpret_cast<threadgroup half4*>(halo + pixel * 8u + (i % 2u) * 4u) = v;
        }
        // Its weights: rows (tap, input channel), this chunk's outputs.
        constexpr uint quads = channels / 4u;
        for (uint i = tid; i < 72u * quads; i += conv_threads) {
            const uint row = i / quads;
            const uint q = i % quads;
            const uint tap = row / 8u;
            const uint k = row % 8u;
            *reinterpret_cast<threadgroup half4*>(taps + row * channels + q * 4u) =
                *reinterpret_cast<device const half4*>(weights + ((tap * p.in_blocks + kb) * 8u + k) * out_width +
                                                       first_out + q * 4u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (uint tap = 0; tap < 9u; ++tap) {
                const uint dy = tap / 3u;
                const uint dx = tap % 3u;
                simdgroup_half8x8 a[4];
                simdgroup_half8x8 b[NT];
                for (uint m = 0; m < 4u; ++m) {
                    simdgroup_load(a[m], halo + ((sg + dy) * halo_pixels + m * 8u + dx) * 8u, 8u);
                }
                for (uint n = 0; n < NT; ++n) {
                    simdgroup_load(b[n], taps + tap * 8u * channels + n * 8u, channels);
                }
                for (uint m = 0; m < 4u; ++m) {
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
    threadgroup float* mine = stage + sg * conv_pixels * 8u;
    const uint x = x0 + lane;
    device half* out = output + ((y + 1u) * p.out_pitch + x + 1u) * p.out_stride + p.out_offset + first_out;
    // A block of 8 channels at a time through the staging, each lane then
    // writing its pixel's 8, inside the level only.
    for (uint n = 0; n < NT; ++n) {
        for (uint m = 0; m < 4u; ++m) {
            simdgroup_store(acc[m][n], mine + m * 8u * 8u, 8u);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (x < p.width) {
            threadgroup const float* values = mine + lane * 8u;
            for (uint c = 0; c < 8u; ++c) {
                float v = values[c] + bias[first_out + n * 8u + c];
                if (p.relu != 0u) {
                    v = max(v, 0.0f);
                }
                out[n * 8u + c] = half(v);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define SERENITY_CONV3X3(NT)                                                                                        \
    template [[host_name("conv3x3_n" #NT)]] kernel void conv3x3<NT>(                                                \
        constant contracts::ConvParams&, device const half*, device const half*, device const float*, device half*, \
        uint3, uint, uint);
SERENITY_CONV3X3(1)
SERENITY_CONV3X3(2)
SERENITY_CONV3X3(3)
SERENITY_CONV3X3(4)
#undef SERENITY_CONV3X3

// ---------------------------------------------------------------------------
// 2 x 2 max pool and nearest upsampling, four channels a thread, each
// between bordered buffers (the destination's size in the params). One
// dimension of threads, channels fastest, so a SIMD group's writes are one
// run of memory (GPU.2).
// ---------------------------------------------------------------------------

kernel void denoise_maxpool(constant contracts::ResampleParams& p [[buffer(binding::params)]],
                            device const half* input [[buffer(binding::input)]],
                            device half* output [[buffer(binding::output)]],
                            uint i [[thread_position_in_grid]]) {
    const uint quads = p.channels / 4u;
    const uint q = i % quads;
    const uint pixel = i / quads;
    const uint x = pixel % p.width;
    const uint y = pixel / p.width;
    if (y >= p.height) {
        return;
    }
    const uint c = p.in_offset + q * 4u;
    const auto read = [&](uint sx, uint sy) {
        return *reinterpret_cast<device const half4*>(input + ((sy + 1u) * p.in_pitch + sx + 1u) * p.in_stride + c);
    };
    const half4 v = max(max(read(2u * x, 2u * y), read(2u * x + 1u, 2u * y)),
                        max(read(2u * x, 2u * y + 1u), read(2u * x + 1u, 2u * y + 1u)));
    *reinterpret_cast<device half4*>(output + ((y + 1u) * p.out_pitch + x + 1u) * p.out_stride + p.out_offset +
                                     q * 4u) = v;
}

kernel void denoise_upsample(constant contracts::ResampleParams& p [[buffer(binding::params)]],
                             device const half* input [[buffer(binding::input)]],
                             device half* output [[buffer(binding::output)]],
                             uint i [[thread_position_in_grid]]) {
    const uint quads = p.channels / 4u;
    const uint q = i % quads;
    const uint pixel = i / quads;
    const uint x = pixel % p.width;
    const uint y = pixel / p.width;
    if (y >= p.height) {
        return;
    }
    const half4 v = *reinterpret_cast<device const half4*>(
        input + ((y / 2u + 1u) * p.in_pitch + x / 2u + 1u) * p.in_stride + p.in_offset + q * 4u);
    *reinterpret_cast<device half4*>(output + ((y + 1u) * p.out_pitch + x + 1u) * p.out_stride + p.out_offset +
                                     q * 4u) = v;
}

// ---------------------------------------------------------------------------
// The output, over twice the padded grid (training/evaluate.py runs the
// head and kernels over the padding too, and the history it keeps is that
// size). A threadgroup is two simdgroups, one row of 16 input pixels and
// their 4 x 16 output pixels (so its staging stays within 12 KB). First the
// head, a 1 x 1 convolution, as a matrix product on simdgroup matrices in
// float: the 16 pixels' last features (staged in threadgroup memory as
// floats) times the head's weights (laid out [feature][output], 124 padded
// to head_columns), each simdgroup 64 of the outputs; the scores stay in threadgroup memory, never in device memory
// (GDSA.16). Then a thread an output pixel X, p = X / 2 its input pixel:
// per spacing, the softmax of its 9 taps' scores, a tap outside the
// light's extent masked, over the demodulated light; the softmax of the 3
// spacing scores mixing the three; the blend with the moved history by
// sigmoid(score 30) x valid; the blend kept as the next frame's history
// and, inside the output, multiplied by the full-size albedo's factor into
// the radiance image.
// ---------------------------------------------------------------------------

constant constexpr uint head_columns = 128u;  // contracts::head_outputs, padded to the simdgroups' 2 x 64
constant constexpr uint output_pixels = 16u;  // input pixels a threadgroup
constant constexpr uint score_stride = head_columns + 1u;  // a row of scores: odd, so rows fall in other banks
constant constexpr int reach = 4;  // the widest spacing: the demodulated light a threadgroup's taps read
constant constexpr uint reach_rows = 2u * uint(reach) + 1u;
constant constexpr uint reach_pixels = output_pixels + 2u * uint(reach);
constant constexpr uint max_features = 64u;   // the head's widest input the staging holds (denoise.cpp checks)

kernel void denoise_output(constant contracts::DenoiseConstants& k [[buffer(binding::constants)]],
                           device const half* features [[buffer(binding::input)]],
                           device const float* head [[buffer(binding::weights)]],
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
    threadgroup float staged[output_pixels * max_features];
    threadgroup float scores_of[output_pixels * score_stride];
    threadgroup float4 light[reach_rows * reach_pixels];
    const uint tid = sg * 32u + lane;
    const uint y = group.y;
    const uint x0 = group.x * output_pixels;
    const uint c_count = k.features;

    // The head's inputs, past the grid's right edge zeros.
    // The demodulated light every tap of the row reads, zero outside the
    // light's extent (where every tap is masked).
    for (uint i = tid; i < reach_rows * reach_pixels; i += 64u) {
        const int2 at = int2(int(x0) - reach + int(i % reach_pixels), int(y) - reach + int(i / reach_pixels));
        light[i] = inside_light(k, at) ? demod[uint(at.y) * k.padded_width + uint(at.x)] : float4(0.0f);
    }
    for (uint i = tid; i < output_pixels * c_count; i += 64u) {
        const uint px = i / c_count;
        const uint c = i % c_count;
        staged[i] = x0 + px < k.padded_width
                        ? float(features[((y + 1u) * (k.padded_width + 2u) + x0 + px + 1u) * c_count + c])
                        : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        simdgroup_float8x8 acc[2][8];
        for (uint m = 0; m < 2u; ++m) {
            for (uint n = 0; n < 8u; ++n) {
                acc[m][n] = simdgroup_float8x8(0.0f);
            }
        }
        for (uint kb = 0; kb < c_count / 8u; ++kb) {
            simdgroup_float8x8 a[2];
            for (uint m = 0; m < 2u; ++m) {
                simdgroup_load(a[m], staged + m * 8u * c_count + kb * 8u, c_count);
            }
            for (uint n = 0; n < 8u; ++n) {
                simdgroup_float8x8 b;
                simdgroup_load(b, head + kb * 8u * head_columns + sg * 64u + n * 8u, head_columns);
                for (uint m = 0; m < 2u; ++m) {
                    simdgroup_multiply_accumulate(acc[m][n], a[m], b, acc[m][n]);
                }
            }
        }
        for (uint m = 0; m < 2u; ++m) {
            for (uint n = 0; n < 8u; ++n) {
                simdgroup_store(acc[m][n], scores_of + m * 8u * score_stride + sg * 64u + n * 8u, score_stride);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // A thread an output pixel: subpixel q, input pixel x0 + i.
    const uint q = tid / output_pixels;
    const uint i = tid % output_pixels;
    const uint2 p = uint2(x0 + i, y);
    if (p.x >= k.padded_width) {
        return;
    }
    const uint2 x = contracts::denoise_scale * p + uint2(q & 1u, q >> 1);
    const uint width = contracts::denoise_scale * k.padded_width;
    float scores[contracts::denoise_scores];
    for (uint s = 0; s < contracts::denoise_scores; ++s) {
        const uint o = s * contracts::denoise_subpixels + q;
        scores[s] = scores_of[i * score_stride + o] + head_bias[o];
    }

    // The kernels: taps at p + s (i, j), k = spacing * 9 + (j + 1) * 3 + (i + 1).
    constexpr int spacing[contracts::denoise_spacings] = {1, 2, 4};
    float3 filtered[contracts::denoise_spacings];
    for (uint si = 0; si < contracts::denoise_spacings; ++si) {
        float logits[contracts::denoise_taps];
        float largest = -INFINITY;
        for (uint t = 0; t < contracts::denoise_taps; ++t) {
            const int2 at = int2(p) + spacing[si] * int2(int(t % 3u) - 1, int(t / 3u) - 1);
            logits[t] = inside_light(k, at) ? scores[si * contracts::denoise_taps + t] : contracts::masked_score;
            largest = max(largest, logits[t]);
        }
        float total = 0.0f;
        float3 sum = float3(0.0f);
        for (uint t = 0; t < contracts::denoise_taps; ++t) {
            const int2 at = int2(p) + spacing[si] * int2(int(t % 3u) - 1, int(t / 3u) - 1);
            const float e = exp(logits[t] - largest);
            total += e;
            const int2 local = at - int2(int(x0) - reach, int(y) - reach);
            sum += e * light[uint(local.y) * reach_pixels + uint(local.x)].rgb;
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
