#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/contracts/denoise.h"
#include "core/denoise/network.h"
#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/history.h"
#include "metal/device/device.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: Frame graph (the denoiser's memory: contract 18).
//
// What the denoise pass (metal/passes/denoise/denoise.h) reads and writes
// beyond the frame's textures, made for the light's size and remade when it
// changes, before the frame's submission begins, draining the submission
// first, as the images between passes are (frame_images.h; GPU.9, MEM.9):
//
//   - each U-Net level's activations, half floats, channels innermost, a
//     one-pixel zero border around each level (contracts::ConvParams): the
//     network's input (contracts::input_channels); the encoders' outputs,
//     each a skip; the max pools', the bottleneck's two, the decoders' and
//     the last convolution's. No concatenation and no upsampled copy: a
//     decoder convolution reads the level below, upsampled, and the skip,
//     each where it is (conv3x3). The padded grid is the light's size rounded up to
//     contracts::unet_alignment, as training/evaluate.py pads;
//   - the demodulated light, a float4 an input pixel of the padded grid;
//   - the history, last frame's output and this frame's, a float4 an output
//     pixel of twice the padded grid, two of them by the frame's parity;
//   - the depth median's bins and state (contracts::MedianState);
//   - every dispatch's ConvParams and ResampleParams, written once per size.
//
// Shared storage, zeroed when made: the borders must be zero and nothing
// writes them, and a private buffer's first contents are not defined.
// Nothing is aliased: some 0.9 GB at a 1728 x 1117 light for the wide
// network, on a machine with far more (the alternatives, in the note).
//
// The history. The output feeds the next frame, so which frames it holds
// is the core's reuse history (core/frame/history.h, ReuseHistory), as
// ReSTIR's reservoirs: prepare() joins the frame to it, and the frame reads
// last frame's output only when the history holds a frame before it. So a
// history's first frame, a resize and a frame out of order start the
// denoiser over, as they start ReSTIR's reuse over. A HistoryError is a
// MetalError naming the denoiser.
//
// Throws MetalError if the light's size is not one Metal makes an image of,
// or the device cannot make a buffer.
class DenoiseImages {
public:
    DenoiseImages(const Device& device, Submission& submission, const denoise::Widths& widths);

    DenoiseImages(const DenoiseImages&) = delete;
    DenoiseImages& operator=(const DenoiseImages&) = delete;
    DenoiseImages(DenoiseImages&&) = delete;
    DenoiseImages& operator=(DenoiseImages&&) = delete;
    ~DenoiseImages() = default;

    // Ready for frame `inputs`, its light of `light`: the buffers made or
    // remade, the frame joined to the history. Whether the frame reads last
    // frame's output.
    bool prepare(const frame::FrameInputs& inputs, frame::Extent light);

    // The conv3x3 dispatches, in order: enc 0..3, bottleneck a and b,
    // dec 0..3, dec_final. The pools, levels 1..4.
    static constexpr std::size_t convolutions = 11;
    static constexpr std::size_t resamplings = 4;

    struct Conv {
        contracts::ConvParams params;
        MTL::GPUAddress params_address = 0;
        MTL::GPUAddress input = 0;
        MTL::GPUAddress output = 0;
        MTL::GPUAddress coarse = 0;  // a decoder's level below; else its input again, unread
    };
    struct Resample {
        contracts::ResampleParams params;
        MTL::GPUAddress params_address = 0;
        MTL::GPUAddress input = 0;
        MTL::GPUAddress output = 0;
    };

    // Each valid after the first prepare().
    const Conv& conv(std::size_t i) const { return convs_.at(i); }
    const Resample& pool(std::size_t i) const { return pools_.at(i); }
    frame::Extent light() const noexcept { return light_; }
    frame::Extent padded() const noexcept { return padded_; }
    MTL::GPUAddress input() const noexcept;
    MTL::GPUAddress features() const noexcept;  // the last convolution's output
    MTL::GPUAddress demod() const noexcept;
    MTL::GPUAddress histogram() const noexcept;
    MTL::GPUAddress median() const noexcept;
    // Last frame's output and this frame's, by the prepared frame's parity.
    MTL::GPUAddress history() const noexcept;
    MTL::GPUAddress next_history() const noexcept;

private:
    struct Buffer {
        NS::SharedPtr<MTL::Buffer> buffer;
        Resident resident;  // released after the GPU is done with it (submission.h, Lifetime)
    };
    // A level's activations, bordered (above): its size, pitch and stride.
    struct Activations {
        Buffer memory;
        frame::Extent size;
        std::uint32_t pitch = 0;
        std::uint32_t stride = 0;
    };

    void make(Buffer& buffer, std::size_t bytes, const char* what);
    void make(Activations& a, frame::Extent size, std::uint32_t stride, const char* what);
    void layout();

    NS::SharedPtr<MTL::Device> device_;
    Submission& submission_;
    denoise::Widths widths_;
    frame::ReuseHistory history_;
    frame::Extent light_;
    frame::Extent padded_;
    std::uint64_t index_ = 0;  // the prepared frame's: its parity picks the history

    Activations input_;
    std::array<Activations, 4> skip_;   // levels 0..3, the encoders' outputs
    std::array<Activations, 4> pooled_;  // levels 1..4
    Activations bottleneck_a_;
    Activations bottleneck_b_;
    std::array<Activations, 4> decoded_;  // levels 0..3, dec 3..0's outputs
    Activations final_;
    Buffer demod_;
    std::array<Buffer, 2> history_buffers_;
    Buffer histogram_;
    Buffer median_;
    Buffer params_;
    std::array<Conv, convolutions> convs_{};
    std::array<Resample, resamplings> pools_{};
};

// The output channel blocks one conv3x3 threadgroup computes for a
// convolution of `out_blocks` blocks: the largest of 4 .. 1 that divides
// it, so every pipeline's accumulators fill and none spills
// (passes/denoise/denoise.metal).
std::uint32_t chunk_blocks(std::uint32_t out_blocks);

}  // namespace serenity::metal
