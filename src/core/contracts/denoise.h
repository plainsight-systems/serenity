#pragma once

// Contract 18: the denoiser's network as it runs (Milestone 3, Phase 4;
// docs/research/2026-10-15-denoiser-live.md). Owned by the Pass family;
// read by the core's loader (core/denoise/network.h), which checks an
// exported network against these numbers, and by the Metal backend's
// denoise pass (metal/passes/denoise/denoise.h), host and shaders alike.
//
// The network is training/model.py's, exactly as trained: these are its
// constants, each the value training/export.py writes into layout.json,
// so a network exported with other ones is refused at load rather than
// run with the wrong pre- or post-processing (E.2, I.5).
//
//   - Upscaling by denoise_scale each way: the light passes render at the
//     output size halved, rounded up (core/frame/schedule.h, light_size).
//   - The U-Net's input is zero-padded to a multiple of unet_alignment on
//     each side, as training/evaluate.py pads it; taps and history reads
//     never land in the padding (the extent, below).
//   - network_inputs channels at each input pixel (training/export.py,
//     "inputs"): pu(demodulated radiance) 0..2, albedo clamped to [0, 1]
//     3..5, the normal in view space 6..8, log(1 + z / median z) 9, motion
//     10..11, pixel_unshuffle(pu(moved history)) 12..23, pixel_unshuffle(
//     history validity) 24..27; stored in channel_block-padded half floats
//     (input_channels), the 4 past network_inputs zero.
//   - scores per output pixel: taps (3 spacings x 9) 0..26, one weight per
//     spacing 27..29, the history blend 30; the head's channel k * 4 + dy * 2
//     + dx is score k of output pixel (2y + dy, 2x + dx).
//   - Demodulation by NRD's factor, albedo_floor + (1 - albedo_floor) a.
//   - OIDN's PU transfer (training/loss.py), its constants below.
//
// Layout rules as for every shared contract (contracts/frame_constants.h).

#include "core/contracts/float3.h"
#include "core/contracts/shared_layout.h"

namespace serenity {
namespace contracts {

SERENITY_CONSTANT uint32_t denoise_scale = 2u;
SERENITY_CONSTANT uint32_t unet_alignment = 16u;
SERENITY_CONSTANT uint32_t unet_levels = 4u;  // max pools below the input's level
SERENITY_CONSTANT uint32_t network_inputs = 28u;
SERENITY_CONSTANT uint32_t channel_block = 8u;  // a simdgroup matrix's side: every stored width a multiple
SERENITY_CONSTANT uint32_t input_channels = 32u;  // network_inputs, padded to channel_block
SERENITY_CONSTANT uint32_t denoise_taps = 9u;
SERENITY_CONSTANT uint32_t denoise_spacings = 3u;  // 1, 2 and 4
SERENITY_CONSTANT uint32_t denoise_scores = 31u;   // 27 taps, 3 spacing weights, 1 history blend
SERENITY_CONSTANT uint32_t denoise_subpixels = 4u;
SERENITY_CONSTANT uint32_t head_outputs = 124u;  // denoise_scores x denoise_subpixels
SERENITY_CONSTANT float masked_score = -1.0e4f;
SERENITY_CONSTANT float albedo_floor = 0.02f;

// OIDN's PU2 fit, as training/loss.py has it; pu_norm_scale is 1 / pu(65504).
SERENITY_CONSTANT float pu_a = 1.41283765e03f;
SERENITY_CONSTANT float pu_b = 1.64593172e00f;
SERENITY_CONSTANT float pu_c = 4.31384981e-01f;
SERENITY_CONSTANT float pu_d = -2.94139609e-03f;
SERENITY_CONSTANT float pu_e = 1.92653254e-01f;
SERENITY_CONSTANT float pu_f = 6.26026094e-03f;
SERENITY_CONSTANT float pu_g = 9.98620152e-01f;
SERENITY_CONSTANT float pu_y0 = 1.57945760e-06f;
SERENITY_CONSTANT float pu_y1 = 3.22087631e-02f;
SERENITY_CONSTANT float pu_norm_scale = 0.318967155f;

// What the denoise pass's kernels read beyond the frame's constants, one
// per frame in flight (metal/passes/denoise/denoise.h). Sizes in pixels:
// the light image's (the extent, h x w), the U-Net's padded grid (a
// multiple of unet_alignment, at least the extent), and the output
// written (at most twice the extent).
struct DenoiseConstants {
    Float3 right;  // the camera's unit right axis
    uint32_t light_width;
    Float3 up;  // its unit up axis
    uint32_t light_height;
    Float3 backward;  // toward the viewer: -forward
    uint32_t history;  // 1 when last frame's output is this frame's history, else 0
    uint32_t padded_width;
    uint32_t padded_height;
    uint32_t output_width;
    uint32_t output_height;
    uint32_t features;  // the U-Net's last width: the head's inputs a pixel
    uint32_t padding[3];
};

static_assert(sizeof(DenoiseConstants) == 80, "DenoiseConstants must be the same 80 bytes on the host and in shaders");

// One 3 x 3 convolution's dispatch (metal/passes/denoise/denoise.metal,
// conv3x3): its level's size, where it reads and writes in its buffers
// (half floats, channels innermost, rows `pitch` pixels apart with a
// one-pixel zero border), and its widths in blocks of channel_block. A
// decoder's reads its first up_blocks blocks from the level below,
// upsampled by nearest neighbor as it reads them (its own pitch and
// stride), and the rest from `input`, the skip: the concatenation
// [upsampled | skip] is never written.
struct ConvParams {
    uint32_t width;
    uint32_t height;
    uint32_t in_pitch;     // pixels a row of the input, borders included
    uint32_t in_stride;    // channels a pixel of the input
    uint32_t in_blocks;    // input channels read, / channel_block: the upsampled ones and the skip's
    uint32_t out_pitch;
    uint32_t out_stride;   // channels a pixel of the output buffer
    uint32_t out_offset;   // the first output channel written
    uint32_t out_blocks;   // output channels, / channel_block
    uint32_t chunk_blocks; // output blocks one threadgroup computes
    uint32_t relu;         // 1: ReLU after the bias
    uint32_t up_blocks;    // of in_blocks, read upsampled from the level below; 0 but in a decoder
    uint32_t up_pitch;     // that level's pitch and stride
    uint32_t up_stride;
    uint32_t padding[2];
};

static_assert(sizeof(ConvParams) == 64, "ConvParams must be the same 64 bytes on the host and in shaders");

// One max pool's or nearest upsample's dispatch: the destination's size,
// and each buffer's pitch, stride and first channel; `channels` copied.
struct ResampleParams {
    uint32_t width;  // of the destination
    uint32_t height;
    uint32_t channels;
    uint32_t in_pitch;
    uint32_t in_stride;
    uint32_t in_offset;
    uint32_t out_pitch;
    uint32_t out_stride;
    uint32_t out_offset;
    uint32_t padding[3];
};

static_assert(sizeof(ResampleParams) == 48, "ResampleParams must be the same 48 bytes on the host and in shaders");

// The depth median's working state (training/data.py normalizes depth by
// the median over the frame's surface pixels): found exactly, as numpy's
// np.median finds it, by three levels of radix selection over the float's
// bits, 11, 10 and 10 of them (metal/passes/denoise/denoise.metal). For
// both middle ranks, (n - 1) / 2 and n / 2, whose mean is the median of an
// even count.
SERENITY_CONSTANT uint32_t median_bins = 2048u;  // per level: 2^11, or two ranks' 2^10
struct MedianState {
    uint32_t count;      // surface pixels: depth > 0
    uint32_t rank[2];    // each middle rank, within the bins selected so far
    uint32_t prefix[2];  // each one's bits selected so far
    float median;        // 1 where there is no surface, as training/data.py
    uint32_t padding[2];
};

static_assert(sizeof(MedianState) == 32, "MedianState must be the same 32 bytes on the host and in shaders");

#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<DenoiseConstants>, "DenoiseConstants is written to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<ConvParams>, "ConvParams is written to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<ResampleParams>, "ResampleParams is written to the GPU as bytes");
#endif

}  // namespace contracts
}  // namespace serenity
