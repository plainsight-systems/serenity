#pragma once

#include <cuda.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "core/contracts/denoise.h"
#include "core/denoise/network.h"
#include "core/denoise/unet.h"
#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/history.h"
#include "cuda/device/buffer.h"
#include "cuda/device/device.h"
#include "cuda/device/frame_sequence.h"
#include "cuda/device/images.h"
#include "cuda/device/submission.h"

namespace serenity::cuda {

// Axis: Frame graph (the denoiser's memory: contract 18).
//
// What the denoise pass (cuda/passes/denoise/denoise.h) reads and writes
// beyond the frame's images, Metal's DenoiseImages (metal/frame/
// denoise_images.h, which says what each holds and why none is aliased)
// in device memory, at the core's sizes: which activations there are,
// their sizes and channels, the padded grid and which each dispatch reads
// and writes are the core's network at the light's size (core/denoise/
// unet.h); their layout, a pitch with a one-pixel zero border and slack
// past the end, is Metal's, the same here:
//
//   - each U-Net level's activations, half floats, channels innermost;
//   - the demodulated light, four floats an input pixel of the padded grid;
//   - the history, last frame's output and this frame's, four floats an
//     output pixel of twice the padded grid, two of them by the frame's
//     parity;
//   - the depth median's bins and state (contracts::MedianState);
//   - the hand-off of the light's radiance: one image of four floats a
//     pixel at the light's size per frame slot (cuda::frames_in_flight),
//     pitched (cuda/device/images.h), which the checkerboard's fill writes
//     whole and the denoise pass reads, so frame N + 1's light never writes
//     the hand-off frame N's denoiser reads (CP.3; GPU.7: one per frame in
//     flight).
//
// Every dispatch's ConvParams and ResampleParams are kept here, on the
// host, filled once per size; the pass hands each to its kernel by value,
// as a launch argument, where Metal binds them from a buffer (F.16: 64 and
// 48 bytes, copied by the driver at the launch), so no buffer holds them.
//
// The buffers are zeroed when made: the borders must be zero and nothing
// writes them, and the median's state starts at no count, each zeroed and
// waited for (cuda/device/buffer.h, zero_now), once per size, outside the
// frame loop (GPU.8: never a device-wide wait in it). The hand-off images are
// written whole before they are read. Made when the light's size is first known
// and remade when it changes, the frames in flight finished first, before the
// frame's submission begins (MEM.9, GPU.9: nothing allocated per frame).
//
// The history. Which frames it holds is the core's reuse history
// (frame::ReuseHistory), as Metal's: prepare() joins the frame to it, on a
// copy kept once the buffers exist (E.4), and the frame reads last frame's
// output only when the history holds a frame before it. A HistoryError is a
// CudaError naming the denoiser.
//
// Throws CudaError if the light's size is empty, the device cannot make a
// buffer or an image, or the history refuses a frame.
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

  // The conv3x3 dispatches and the pools, in the core's order
  // (core/denoise/unet.h).
  static constexpr std::size_t convolutions = denoise::unet_convolutions;
  static constexpr std::size_t resamplings = denoise::unet_pools;

  // A dispatch's parameters and where it reads and writes (C.2: plain
  // data, filled together once per size).
  struct Conv {
    contracts::ConvParams params{};
    CUdeviceptr input = 0;
    CUdeviceptr output = 0;
    // a decoder's level below; else its input again, unread
    CUdeviceptr coarse = 0;
  };
  struct Resample {
    contracts::ResampleParams params{};
    CUdeviceptr input = 0;
    CUdeviceptr output = 0;
  };

  // Each valid after the first prepare(); the addresses 0 before it.
  const Conv& conv(std::size_t i) const { return convs_.at(i); }
  const Resample& pool(std::size_t i) const { return pools_.at(i); }
  frame::Extent light() const noexcept { return light_; }
  frame::Extent padded() const noexcept { return padded_; }
  CUdeviceptr input() const noexcept;
  CUdeviceptr features() const noexcept;  // the last convolution's output
  CUdeviceptr demod() const noexcept { return demod_.address(); }
  CUdeviceptr histogram() const noexcept { return histogram_.address(); }
  CUdeviceptr median() const noexcept { return median_.address(); }
  // Last frame's output and this frame's, by the prepared frame's parity.
  CUdeviceptr history() const noexcept;
  CUdeviceptr next_history() const noexcept;
  // Frame slot `slot`'s hand-off of the light's radiance (above). Throws
  // CudaError for a slot that is not one, or before the first prepare()
  // (I.6).
  const PitchedImage& light_radiance(std::uint32_t slot) const;

 private:
  // A level's activations, bordered (above): its size, pitch and stride.
  struct Activations {
    DeviceBuffer memory;
    frame::Extent size;
    std::uint32_t pitch = 0;
    std::uint32_t stride = 0;
  };

  // `bytes` of device memory, zeroed (above).
  static DeviceBuffer zeroed(std::size_t bytes);
  static void make(Activations& a, frame::Extent size, std::uint32_t stride);
  // At a new size, the light's `light`: the core's network at that size
  // (core/denoise/unet.h), every buffer and image made (allocate), then
  // every dispatch's parameters filled (plan).
  void allocate(const denoise::UNet& net, frame::Extent light);
  void plan(const denoise::UNet& net);

  const Device& device_;
  Submission& submission_;
  denoise::Widths widths_;
  frame::ReuseHistory history_;
  frame::Extent light_;
  frame::Extent padded_;
  // the prepared frame's: its parity picks the history
  std::uint64_t index_ = 0;

  // by the core's index (core/denoise/unet.h)
  std::array<Activations, denoise::unet_activations> activations_;
  DeviceBuffer demod_;
  std::array<DeviceBuffer, 2> history_buffers_;
  DeviceBuffer histogram_;
  DeviceBuffer median_;
  // the hand-off, by slot
  std::array<std::unique_ptr<PitchedImage>, frames_in_flight> light_radiance_;
  std::array<Conv, convolutions> convs_{};
  std::array<Resample, resamplings> pools_{};
};

// The output channel blocks one conv3x3 block of threads computes for a
// convolution of `out_blocks` blocks: the largest of 4 .. 1 that divides
// it, so every kernel's accumulators fill and none spills
// (cuda/passes/denoise/denoise.cu), Metal's rule.
std::uint32_t chunk_blocks(std::uint32_t out_blocks);

}  // namespace serenity::cuda
