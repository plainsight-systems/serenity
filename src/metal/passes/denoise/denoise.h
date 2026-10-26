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

namespace serenity::metal {

// Axis: Pass (the denoiser: contract 18, core/contracts/denoise.h).
//
// Milestone 3's trained network, run on the frame (docs/research/
// 2026-10-15-denoiser-live.md says how it matches PyTorch):
// training/model.py's Denoiser.frame, with training/data.py's input and
// training/evaluate.py's padding. Its U-Net on Apple's own kernels, as Open
// Image Denoise runs its U-Net on Metal (through MPSGraph), here through
// Metal 4's machine-learning path (docs/research/
// 2026-10-23-apple-silicon-design.md, section 6, change 3); the rest on
// compute kernels of our own (passes/denoise/denoise.metal):
//
//    1. the depth's median over the light's surface pixels, exactly, in
//       three histogram and select dispatches each (6);
//    2. the network's 30 input channels a pixel, planar, last frame's output
//       moved by the motion vectors and the frame's camera jitter among
//       them, and the demodulated light (1);
//    3. the U-Net through its last convolution (training/model.py,
//       UNet.features), the network's Metal package
//       (training/export_metal.py: an ML program, 16-bit tensors and
//       weights, which Xcode's metal-package-builder turns into the
//       package), its pipeline compiled for the light's padded size: one
//       dispatch of Metal's machine-learning encoder, between a barrier
//       from the compute stage before it and one to the compute stage after
//       it (the encoder ends the frame's compute encoder and opens another);
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
// Cost, at a light of P pixels (the padded grid's): the U-Net's multiply-
// adds (training/model.py, macs_per_input_pixel: about 70,000 for the
// wide network), in Apple's kernels; the head's per output pixel, 31 dot
// products of the last width each; the input's and the features' 16-bit
// planes written and read once.
class DenoisePass {
 public:
  // The network's Metal package, `unet.mtlpackage` in the directory
  // `network` was read from (training/export_metal.py), loaded and its
  // network function found. Throws MetalError if there is none, or Metal
  // makes no compiler, table or library of it.
  DenoisePass(const Device& device, const Library& library,
              Submission& submission, const denoise::Network& network);

  // Ready for a light whose padded grid is `padded` (DenoiseImages): the
  // network's pipeline compiled for it, and its intermediates heap, made
  // resident; nothing at the size it was last ready for. Throws MetalError
  // if Metal compiles no pipeline or makes no heap.
  void prepare(frame::Extent padded);

  // Records the pass into `commands`: steps 1 and 2 into `encoder`, which
  // it ends; step 3 into a machine-learning encoder; step 4 into a new
  // compute encoder, with the renderer's argument table, which it leaves
  // in `encoder`, open, for the passes after it. Throws MetalError if
  // `resources` lacks one of the inputs above, or the pass was not
  // prepared for its size.
  void record(MTL4::CommandBuffer* commands,
              MTL4::ComputeCommandEncoder*& encoder,
              const FrameResources& resources) const;

 private:
  // Steps 1, 2 and 4 above, in order; record() checks `resources` first.
  void record_median(MTL4::ComputeCommandEncoder* encoder,
                     const FrameResources& resources) const;
  void record_input(MTL4::ComputeCommandEncoder* encoder,
                    const FrameResources& resources) const;
  void record_output(MTL4::ComputeCommandEncoder* encoder,
                     const FrameResources& resources) const;

  NS::SharedPtr<MTL::Device> device_;
  Submission* submission_;  // never null: the constructor's, outliving it
  std::array<NS::SharedPtr<MTL::ComputePipelineState>, 3> histogram_;
  std::array<NS::SharedPtr<MTL::ComputePipelineState>, 3> select_;
  NS::SharedPtr<MTL::ComputePipelineState> input_;
  NS::SharedPtr<MTL::ComputePipelineState> output_;
  // The head's weights and bias (above).
  std::unique_ptr<StaticArrays> weights_;
  // The network: its package's function, the compiler that compiles it, a
  // pipeline and an intermediates heap for the size prepared, each
  // resident, and the argument table its encoder binds the tensors in.
  NS::SharedPtr<MTL4::LibraryFunctionDescriptor> network_;
  NS::SharedPtr<MTL4::Compiler> compiler_;
  NS::SharedPtr<MTL4::MachineLearningPipelineState> pipeline_;
  NS::SharedPtr<MTL::Heap> heap_;
  Resident pipeline_resident_;
  Resident heap_resident_;
  NS::SharedPtr<MTL4::ArgumentTable> network_arguments_;
  frame::Extent prepared_;
};

}  // namespace serenity::metal
