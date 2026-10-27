#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <array>
#include <memory>

#include "core/denoise/network.h"
#include "core/frame/extent.h"
#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/resident.h"
#include "metal/device/static_arrays.h"
#include "metal/device/submission.h"
#include "metal/frame/denoise_images.h"
#include "metal/frame/frame_resources.h"
#include "metal/passes/denoise/neural_engine.h"

namespace serenity::metal {

// Axis: Pass (the denoiser: contract 18, core/contracts/denoise.h).
//
// Milestone 3's trained network, run on the frame (docs/research/
// 2026-10-15-denoiser-live.md says how it matches PyTorch):
// training/model.py's Denoiser.frame, with training/data.py's input and
// training/evaluate.py's padding. Its U-Net on the Neural Engine, beside the
// GPU (neural_engine.h; docs/research/2026-10-26-neural-engine-denoiser.md);
// the rest on compute kernels of our own (passes/denoise/denoise.metal):
//
//    1. the depth's median over the light's surface pixels, exactly, in
//       three histogram and select dispatches each (6);
//    2. the network's 30 input channels a pixel, planar, last frame's output
//       moved by the motion vectors and the frame's camera jitter among
//       them, and the demodulated light (1);
//    3. the U-Net through its last convolution (training/model.py,
//       UNet.features) on the Neural Engine, between the halves of the
//       frame's image part a Submission hand-off splits (submission.h, "A
//       hand-off"): steps 1 and 2 end the part before it, step 4 begins the
//       part after, with a barrier that makes the Neural Engine's writes
//       visible to the GPU's reads;
//    4. the output: the 1 x 1 head, the three softmax kernels over the
//       demodulated light, the spacings' softmax, the history blend, the
//       remodulation by the full-size albedo (1).
//
// The head's weights, the network's own, converted once at construction to
// half floats laid out [feature][output], its 124 outputs padded to
// head_columns; its biases in float (StaticArrays).
//
// What a frame gives it (FrameResources): the light pass's accumulated
// image and the features at the light size, the full-size albedo, the
// radiance image it writes at the output size, its constants and its
// buffers and tensors (metal/frame/denoise_images.h). Throws MetalError,
// at record(), if any is missing.
//
// The light is padded to the network's rung for it (rungs.h, smallest_rung()),
// as to the U-Net's alignment, zeros past its edge; at the demo's lights (960 x
// 540, 1728 x 1117) the rung is the aligned grid.
//
// Cost, at a light of P pixels (the rung's): the U-Net's multiply-adds
// (training/model.py, macs_per_input_pixel: about 70,000 for the wide
// network), on the Neural Engine, off the GPU; the head's per output pixel,
// 31 dot products of the last width each; the input's and the features'
// 16-bit planes written and read once.
class DenoisePass {
 public:
  // The network's Core ML model, `unet.mlmodelc` in the directory `network`
  // was read from (training/export_coreml.py), loaded (neural_engine.h).
  // Throws MetalError if there is none, or Core ML loads none.
  DenoisePass(const Device& device, const Library& library,
              Submission& submission, const denoise::Network& network);

  // The grid a light of `light` runs at: the network's rung for it. Throws
  // MetalError for a light past the largest.
  frame::Extent grid(frame::Extent light) const;

  // Ready for the grid `grid` (DenoiseImages's padded grid): the network's
  // memory at it and its first prediction (neural_engine.h). Throws
  // MetalError as NeuralEngineNetwork::prepare() does.
  void prepare(frame::Extent grid);

  // Records the pass: steps 1 and 2 into `encoder`, which it ends; the
  // submission's hand-off and the network between its halves; step 4
  // into a compute encoder of the part after, with the renderer's argument
  // table, which it leaves in `encoder`, open, for the passes after it.
  // Throws MetalError if `resources` lacks one of the inputs above, or the
  // pass was not prepared for its size, or as Submission::hand_off() does.
  void record(MTL4::ComputeCommandEncoder*& encoder,
              const FrameResources& resources) const;

 private:
  // Steps 1, 2 and 4 above, in order; record() checks `resources` first.
  void record_median(MTL4::ComputeCommandEncoder* encoder,
                     const FrameResources& resources) const;
  void record_input(MTL4::ComputeCommandEncoder* encoder,
                    const FrameResources& resources) const;
  void record_output(MTL4::ComputeCommandEncoder* encoder,
                     const FrameResources& resources) const;

  Submission* submission_;  // never null: the constructor's, outliving it
  std::array<NS::SharedPtr<MTL::ComputePipelineState>, 3> histogram_;
  std::array<NS::SharedPtr<MTL::ComputePipelineState>, 3> select_;
  NS::SharedPtr<MTL::ComputePipelineState> input_;
  NS::SharedPtr<MTL::ComputePipelineState> output_;
  // The head's weights and bias (above).
  std::unique_ptr<StaticArrays> weights_;
  // The U-Net on the Neural Engine, its input and features (above).
  std::unique_ptr<NeuralEngineNetwork> network_;
};

}  // namespace serenity::metal
