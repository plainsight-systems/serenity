#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <array>
#include <memory>

#include "core/denoise/network.h"
#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/static_arrays.h"
#include "metal/device/submission.h"
#include "metal/frame/denoise_images.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (the denoiser: contract 18, core/contracts/denoise.h).
//
// Milestone 3's trained network, run on the frame inside the frame's one
// compute encoder (docs/research/2026-10-15-denoiser-live.md says why this
// path, and how it matches PyTorch): training/model.py's Denoiser.frame,
// with training/data.py's input and training/evaluate.py's padding, on
// Metal compute kernels of our own (passes/denoise/denoise.metal):
//
//    1. the depth's median over the light's surface pixels, exactly, in
//       three histogram and select dispatches each (6);
//    2. the network's 30 input channels a pixel, last frame's output moved
//       by the motion vectors and the frame's camera jitter among them, and
//       the demodulated light (1);
//    3. the U-Net: 11 convolutions, 3 x 3 with ReLU, on simdgroup matrices
//       (half in, float sums), each decoder's reading the level below,
//       upsampled, and its skip where they are; and 4 max pools (15);
//    4. the output: the 1 x 1 head, the three softmax kernels over the
//       demodulated light, the spacings' softmax, the history blend, the
//       remodulation by the full-size albedo (1).
//
// 23 dispatches, each after a barrier on the one before it (GPU.8: each
// reads what the one before wrote), the first after the renderer's barrier
// before the pass (metal/frame/renderer.cpp). The weights are the network's,
// converted once at construction: each 3 x 3 convolution's to half floats
// laid out [tap][input channel][output channel], the input's 30 channels
// padded to 32 with zeros; the head's to half floats laid out [feature]
// [output], its 124 outputs padded to head_columns; biases in float
// (StaticArrays).
//
// What a frame gives it (FrameResources): the light pass's accumulated
// image and the features at the light size, the full-size albedo, the
// radiance image it writes at the output size, its constants and its
// buffers (metal/frame/denoise_images.h). Throws MetalError, at record(),
// if any is missing.
//
// Cost, at a light of P pixels (the padded grid's): per input pixel the
// U-Net's multiply-adds (training/model.py, macs_per_input_pixel: about
// 70,000 for the wide network, 37,000 for the small), less the head's,
// which the output kernel computes per output pixel, 31 dot products of
// the last width each; reads and writes of the activations, half floats,
// about 2 x 9 x the widths' sum bytes a pixel through the cache. Measured
// in the note.
class DenoisePass {
 public:
  DenoisePass(const Device& device, const Library& library,
              Submission& submission, const denoise::Network& network);

  // Records the pass. Throws MetalError if `resources` lacks one of the
  // inputs above.
  void record(MTL4::ComputeCommandEncoder* encoder,
              const FrameResources& resources) const;

 private:
  // The convolution kernel that computes `chunk` output blocks a
  // threadgroup; owned by this pass.
  MTL::ComputePipelineState* pipeline(std::uint32_t chunk) const;

  // The four steps above, in order; record() checks `resources` first.
  void record_median(MTL4::ComputeCommandEncoder* encoder,
                     const FrameResources& resources) const;
  void record_input(MTL4::ComputeCommandEncoder* encoder,
                    const FrameResources& resources) const;
  void record_unet(MTL4::ComputeCommandEncoder* encoder,
                   const FrameResources& resources) const;
  void record_output(MTL4::ComputeCommandEncoder* encoder,
                     const FrameResources& resources) const;

  std::array<NS::SharedPtr<MTL::ComputePipelineState>, 3> histogram_;
  std::array<NS::SharedPtr<MTL::ComputePipelineState>, 3> select_;
  NS::SharedPtr<MTL::ComputePipelineState> input_;
  std::array<NS::SharedPtr<MTL::ComputePipelineState>, 4>
      conv_;  // conv3x3_n1 .. n4
  NS::SharedPtr<MTL::ComputePipelineState> maxpool_;
  NS::SharedPtr<MTL::ComputePipelineState> output_;
  // Each convolution's weights and bias, then the head's (above).
  std::unique_ptr<StaticArrays> weights_;
};

}  // namespace serenity::metal
