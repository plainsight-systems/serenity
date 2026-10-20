#pragma once

#include <cuda.h>

#include <array>

#include "core/contracts/denoise.h"
#include "core/denoise/network.h"
#include "core/frame/extent.h"
#include "cuda/device/buffer.h"
#include "cuda/device/images.h"
#include "cuda/device/module.h"
#include "cuda/frame/denoise_images.h"

namespace serenity::cuda {

// What a frame gives the denoise pass, everything Metal's record() takes
// from FrameResources (metal/frame/frame_resources.h), by name (I.24: one
// struct, so no two of its many addresses can pass in each other's
// places; C.2: plain data, each field the frame's own). An address of 0 or
// a null pointer is an input missing, which record() refuses.
struct DenoiseInputs {
  // The frame's constants (contract 18), by value: each launch takes them
  // as an argument (F.16: 80 bytes, copied by the driver).
  contracts::DenoiseConstants constants{};
  // The denoiser's buffers, prepared for the frame (denoise_images.h).
  const DenoiseImages* images = nullptr;
  // The frame slot's hand-off of the light's radiance
  // (DenoiseImages::light_radiance), read exactly, at the light's size.
  const PitchedImage* light_radiance = nullptr;
  // The features at the light's size (cuda/frame/feature_images.h,
  // contract 17's formats, rows tightly packed): albedo, normal and motion
  // four half floats a pixel, depth one float.
  CUdeviceptr albedo = 0;
  CUdeviceptr normal = 0;
  CUdeviceptr depth = 0;
  CUdeviceptr motion = 0;
  // The full-size albedo, four half floats a pixel, rows tightly packed,
  // `full_size` pixels: the output's size.
  CUdeviceptr full_albedo = 0;
  frame::Extent full_size;
  // The radiance image written, four floats a pixel, at the output's size.
  const PitchedImage* radiance = nullptr;
};

// Axis: Pass (the denoiser: contract 18, core/contracts/denoise.h).
//
// Milestone 3's trained network on the frame, Metal's DenoisePass
// (metal/passes/denoise/denoise.h, which states the network, how it
// matches PyTorch and its cost) translated line for line: its kernels
// (denoise.cu) Metal's (denoise.metal), its 23 dispatches 23 launches on
// the frame's stream in Metal's order:
//
//    1. the depth's median over the light's surface pixels, exactly, in
//       three histogram and select launches each (6);
//    2. the network's 30 input channels a pixel and the demodulated light
//       (1);
//    3. the U-Net: 11 convolutions, 3 x 3 with ReLU, on the tensor cores
//       (cuda/device/matrix.cuh: half in, float sums), and 4 max pools
//       (15);
//    4. the output: the 1 x 1 head, on the tensor cores too, the kernels'
//       softmaxes, the history blend, the remodulation (1).
//
// Each launch reads what the one before wrote: the stream runs them in
// order, so none needs a barrier (Metal's 23; GPU.8). A block of threads
// is contract 18's, four warps of 32, a warp a row of 32 pixels; Metal's
// threadgroup memory is shared memory, its simdgroup matrices the tensor
// cores' m16n8k8 products, a Metal row's four 8 x 8 pixel blocks two
// 16-pixel tiles. Every sum runs in a fixed order, never an atomic but
// the median's integer counts: the same inputs give the same image, run
// after run (GDSA.2).
//
// The weights are the network's (core/denoise/network.h), converted once at
// construction into Metal's layouts and uploaded once (MEM.9): each 3 x 3
// convolution's to half floats laid out [tap][input channel][output
// channel], the input's 30 channels padded to 32 with zeros; the head's to
// half floats laid out [feature][output], its 124 outputs padded to
// contracts::head_columns; biases in float. Each dispatch's parameters are
// DenoiseImages', passed by value.
//
// Cost: Metal's (its header), less what CUDA's launches cost on the host,
// a few microseconds each. Measured with the demo graph's frame time.
//
// Throws CudaError if the module cannot be loaded or the weights cannot be
// uploaded; at record(), if an input is missing or a launch fails.
class DenoisePass {
 public:
  explicit DenoisePass(const denoise::Network& network);

  // Launches the pass on `stream`. Throws CudaError if `inputs` lacks one
  // of its inputs, its full-size albedo or radiance image is not of the
  // constants' output size, or a launch fails.
  void record(CUstream stream, const DenoiseInputs& inputs) const;

 private:
  // The convolution kernel that computes `chunk` output blocks a block of
  // threads.
  CUfunction convolution(std::uint32_t chunk) const;

  // The four steps above, in order; record() checks `inputs` first.
  void record_median(CUstream stream, const DenoiseInputs& inputs) const;
  void record_input(CUstream stream, const DenoiseInputs& inputs) const;
  void record_unet(CUstream stream, const DenoiseInputs& inputs) const;
  void record_output(CUstream stream, const DenoiseInputs& inputs) const;

  Module module_;
  std::array<CUfunction, 3> histogram_{};
  std::array<CUfunction, 3> select_{};
  CUfunction input_ = nullptr;
  // conv3x3_n1 .. n4
  std::array<CUfunction, 4> conv_{};
  CUfunction maxpool_ = nullptr;
  CUfunction output_ = nullptr;
  // Each convolution's weights and bias in the order they run
  // (core/denoise/unet.h), then the head's (above).
  std::array<DeviceBuffer, DenoiseImages::convolutions> weights_;
  std::array<DeviceBuffer, DenoiseImages::convolutions> biases_;
  DeviceBuffer head_;
  DeviceBuffer head_bias_;
};

}  // namespace serenity::cuda
