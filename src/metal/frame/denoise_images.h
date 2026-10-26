#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <array>
#include <cstddef>
#include <cstdint>

#include "core/contracts/denoise.h"
#include "core/denoise/network.h"
#include "core/denoise/unet.h"
#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/history.h"
#include "metal/device/device.h"
#include "metal/device/resident.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: Frame graph (the denoiser's memory: contract 18).
//
// What the denoise pass (metal/passes/denoise/denoise.h) reads and writes
// beyond the frame's textures, made for the light's size and remade when it
// changes, before the frame's submission begins, draining the submission
// first, as the images between passes are (frame_images.h; GPU.9, MEM.9):
//
//   - the network's input and its features, half floats, planar (a
//     channel's pixels of the padded grid row after row, then the next
//     channel's), as Metal's tensors lay them out, the innermost stride 1
//     (docs/research/2026-10-23-apple-silicon-design.md, section 6, change
//     3): the 30 inputs the input kernel writes and the network reads, the
//     last convolution's features the network writes and the output
//     kernel's head reads; a tensor over each, (W, H, C, 1) innermost
//     first, which Metal's machine-learning encoder binds. The U-Net's
//     own activations are the network's, in its intermediates heap
//     (passes/denoise/denoise.h);
//   - the demodulated light, a float4 an input pixel of the padded grid;
//   - the history, last frame's output and this frame's, a float4 an output
//     pixel of twice the padded grid, two of them by the frame's parity;
//   - the depth median's bins and state (contracts::MedianState);
//   - every dispatch's ConvParams and ResampleParams, written once per size;
//   - the hand-off of the light's radiance (metal/frame/renderer.h, "Two
//     halves"): one RGBA32Float image at the light size per frame slot,
//     which the checkerboard's fill writes whole from the light's
//     accumulated image (passes/checkerboard_fill/checkerboard_fill.h) and
//     the denoise pass reads, so frame N + 1's light never writes the
//     hand-off frame N's denoiser reads (CP.3; GPU.7, MEM.4: one per frame in
//     flight). Private, the GPU's alone; 62 MB at 1728 x 1117.
//
// The buffers are in shared storage, zeroed when made: the history is read
// before a frame writes it (where the reuse history holds none, unused but
// read), and a private buffer's first contents are not defined. The hand-off
// images are written whole before they are read. Nothing is aliased: the two
// histories, 2 x 7.7 M float4s at a 1728 x 1117 light, most of it.
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
  DenoiseImages(const Device& device, Submission& submission,
                const denoise::Widths& widths);

  DenoiseImages(const DenoiseImages&) = delete;
  DenoiseImages& operator=(const DenoiseImages&) = delete;
  DenoiseImages(DenoiseImages&&) = delete;
  DenoiseImages& operator=(DenoiseImages&&) = delete;
  ~DenoiseImages() = default;

  // Ready for frame `inputs`, its light of `light`: the buffers made or
  // remade, the frame joined to the history. Whether the frame reads last
  // frame's output.
  bool prepare(const frame::FrameInputs& inputs, frame::Extent light);

  // Each valid after the first prepare().
  frame::Extent light() const noexcept { return light_; }
  frame::Extent padded() const noexcept { return padded_; }
  MTL::GPUAddress input() const noexcept { return input_.address(); }
  MTL::GPUAddress features() const noexcept { return features_.address(); }
  // The tensors over them (above).
  MTL::Tensor* input_tensor() const noexcept { return input_tensor_.get(); }
  MTL::Tensor* features_tensor() const noexcept {
    return features_tensor_.get();
  }
  MTL::GPUAddress demod() const noexcept;
  MTL::GPUAddress histogram() const noexcept;
  MTL::GPUAddress median() const noexcept;
  // Last frame's output and this frame's, by the prepared frame's parity.
  MTL::GPUAddress history() const noexcept;
  MTL::GPUAddress next_history() const noexcept;
  // Frame slot `slot`'s hand-off of the light's radiance (above). Throws
  // MetalError for a slot that is not one (I.6).
  MTL::Texture* light_radiance(std::uint32_t slot) const;

 private:
  using Buffer = ResidentBuffer;  // metal/device/resident.h

  void make(Buffer& buffer, std::size_t bytes, std::string_view what);
  // A planar tensor of `channels` over `buffer`, at the padded grid, made
  // resident; throws MetalError if Metal makes none.
  NS::SharedPtr<MTL::Tensor> tensor_over(const Buffer& buffer,
                                         std::uint32_t channels,
                                         Resident& resident);
  // At a new size, the light's `light`: every buffer, tensor and image.
  void allocate(frame::Extent light);

  NS::SharedPtr<MTL::Device> device_;
  Submission& submission_;
  denoise::Widths widths_;
  frame::ReuseHistory history_;
  frame::Extent light_;
  frame::Extent padded_;
  // the prepared frame's: its parity picks the history
  std::uint64_t index_ = 0;

  Buffer input_;
  Buffer features_;
  NS::SharedPtr<MTL::Tensor> input_tensor_;
  NS::SharedPtr<MTL::Tensor> features_tensor_;
  Resident input_tensor_resident_;
  Resident features_tensor_resident_;
  Buffer demod_;
  std::array<Buffer, 2> history_buffers_;
  Buffer histogram_;
  Buffer median_;
  // the hand-off, by slot
  std::array<ResidentTexture, frames_in_flight> light_radiance_;
};

}  // namespace serenity::metal
