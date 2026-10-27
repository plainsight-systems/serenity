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
//   (the network's input and features are the network's, in memory the
//   Neural Engine shares: passes/denoise/neural_engine.h)
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
  DenoiseImages(const Device& device, Submission& submission);

  DenoiseImages(const DenoiseImages&) = delete;
  DenoiseImages& operator=(const DenoiseImages&) = delete;
  DenoiseImages(DenoiseImages&&) = delete;
  DenoiseImages& operator=(DenoiseImages&&) = delete;
  ~DenoiseImages() = default;

  // Ready for frame `inputs`, its light of `light` padded to `grid` (the
  // network's rung for it: passes/denoise/rungs.h, smallest_rung()): the
  // buffers made or remade, the frame joined to the history. Whether the
  // frame reads last frame's output.
  bool prepare(const frame::FrameInputs& inputs, frame::Extent light,
               frame::Extent grid);

  // Each valid after the first prepare().
  frame::Extent light() const noexcept { return light_; }
  frame::Extent padded() const noexcept { return padded_; }
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
  // At a new size, the light's `light`: every buffer, tensor and image.
  void allocate(frame::Extent light);

  NS::SharedPtr<MTL::Device> device_;
  Submission& submission_;
  frame::ReuseHistory history_;
  frame::Extent light_;
  frame::Extent padded_;
  // the prepared frame's: its parity picks the history
  std::uint64_t index_ = 0;

  Buffer demod_;
  std::array<Buffer, 2> history_buffers_;
  Buffer histogram_;
  Buffer median_;
  // the hand-off, by slot
  std::array<ResidentTexture, frames_in_flight> light_radiance_;
};

}  // namespace serenity::metal
