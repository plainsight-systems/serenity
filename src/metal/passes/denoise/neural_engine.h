#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include "core/frame/extent.h"
#include "metal/device/device.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: GPU backend (the denoiser's network off the GPU).
//
// The denoiser's U-Net (training/model.py, UNet.features) on the M3 Max's
// Neural Engine, through Core ML, beside the GPU: the GPU traces frame
// N + 1 while the Neural Engine runs frame N's network
// (docs/research/2026-10-26-neural-engine-denoiser.md, section 3). Core ML
// reaches the Neural Engine and Metal does not, so this is the one place
// the backend names Core ML, Core Video and IOSurface, in Objective-C++
// (neural_engine.mm), as display_link.mm is Core Animation's.
//
// The model. `unet.mlmodelc` beside the network's weights
// (training/export_coreml.py): an ML program in 16 bits, compiled, its
// shapes enumerated, since the Neural Engine runs only shapes it knew when
// it compiled the model (a flexible one ran 5 times slower and partly off
// it). Its shapes are the rungs a light is padded to (rungs.h,
// smallest_rung()), read from the model, so they are written once, by the
// export. Loaded for the CPU and the Neural Engine
// (MLComputeUnitsCPUAndNeuralEngine); the export refuses a model any of
// whose operations Core ML's compute plan puts elsewhere. The first load on
// a machine compiles it for the Neural Engine, some 12 s, which Core ML
// caches; a rung's first prediction 0.2 to 0.35 s, which prepare() takes.
//
// The memory, one copy, no copying: the network's input (contract 18's 30
// planes) and its features (the last width's planes), each in a pixel
// buffer over an IOSurface, half floats, a plane's rows one after another
// (width x channels * height), which Core ML reads and writes in place
// (MLMultiArray over the pixel buffer; the features through the
// prediction's output backing); Metal reads and writes the same pages
// through a buffer over the surface's memory, so the input kernel and the
// head (denoise.metal) address them as before. A row's pitch must be
// contract 18's tensor_pitch(): refused at prepare() otherwise (the
// export's rungs are multiples of 32 halves, a row 64-byte aligned, so
// Core Video pads none). Measured: the features Metal reads are Core ML's,
// bit for bit, and a prediction takes the time it takes on Core ML's own
// arrays (6.57 and 23.37 ms at the demo's rungs).
//
// The frame. run() hands the network to the Neural Engine between the
// halves of the frame's image part that a Submission hand-off splits
// (submission.h, "A hand-off"): Metal's shared-event listener, on a serial
// queue of its own, wakes when the GPU sets `ready`, runs the prediction
// (blocking that queue's thread for the network's time), reports, and sets
// `done`. The hand-off's queue order keeps the one copy safe: the input
// kernel of frame N + 1 is recorded after the part that waits for frame N's
// `done`, so the prediction has read the input before it is rewritten; and
// frame N + 1's prediction starts after its own input kernel, whose part
// begins with a barrier after frame N's head, so frame N's head has read the
// features before they are rewritten.
//
// Failure is the submission's (E.2): a prediction that fails, or that
// Core ML answers in memory not ours, reports the failure and still sets
// `done`, so the GPU never waits on it forever; the submission that
// carried it throws, naming Core ML's error, when it is settled. The
// worker is noexcept (E.12): an exception there, a failed allocation
// building its message or an Objective-C exception from Core ML, ends the
// program rather than leave the GPU waiting. Loading, a missing model,
// shapes that are not enumerated, inputs or outputs of another name,
// count or type, and a rung not the model's throw MetalError.
//
// Cost a frame, on the CPU: the worker registered, which copies its block
// to the heap with what it holds (two shared pointers, the events) and
// allocates by the API's shape, as Submission's feedback handler does
// (submission.h, "Cost of a frame"); one listener wake, one prediction
// call, one event set; Core ML allocates its result's provider per call
// (the API's shape: our output backing is filled, the provider around it
// is new). These go through the C heap, which the allocation test
// (tests/gpu/allocation_test.cpp) does not count.
class NeuralEngineNetwork {
 public:
  // The model in `directory` (above), its input of `input_channels`
  // channels and its features of `feature_channels`.
  NeuralEngineNetwork(const Device& device, Submission& submission,
                      const std::filesystem::path& directory,
                      std::uint32_t input_channels,
                      std::uint32_t feature_channels);
  ~NeuralEngineNetwork();

  NeuralEngineNetwork(const NeuralEngineNetwork&) = delete;
  NeuralEngineNetwork& operator=(const NeuralEngineNetwork&) = delete;
  NeuralEngineNetwork(NeuralEngineNetwork&&) = delete;
  NeuralEngineNetwork& operator=(NeuralEngineNetwork&&) = delete;

  // The model's shapes, width x height.
  const std::vector<frame::Extent>& rungs() const noexcept;

  // Ready for `rung`, its planes' rows `row_bytes` apart (the pitch the
  // kernels address: contract 18's tensor_pitch()): the last rung's memory
  // let go, then its pixel buffers and Metal's buffers over them, one
  // prediction run, and the buffers made resident. Nothing at the rung and
  // pitch it was ready for. Frames in flight must have drained
  // (DenoiseImages has, at a new size). Throws MetalError if `rung` is not
  // one, `row_bytes` is not the pitch it was ready for at the same rung,
  // Core Video's rows are not `row_bytes` apart, or a buffer or the first
  // prediction cannot be made as above; then, but for the pitch refused
  // at the same rung, it is ready for none (E.4: nothing half made kept).
  void prepare(frame::Extent rung, std::size_t row_bytes);
  // The rung it is ready for; none (0 x 0) before prepare() or after one
  // that failed.
  frame::Extent prepared() const noexcept { return prepared_; }

  // The input and the features, planar half floats, contract 18's pitch;
  // 0 when not prepared.
  MTL::GPUAddress input() const noexcept;
  MTL::GPUAddress features() const noexcept;

  // The network on the Neural Engine between `hand_off`'s halves (above).
  // Throws MetalError if not prepared.
  void run(const HandOff& hand_off) const;

 private:
  struct State;  // Core ML's and Core Video's objects (neural_engine.mm)

  NS::SharedPtr<MTL::Device> device_;
  Submission* submission_;  // never null: the constructor's, outliving it
  std::uint32_t input_channels_;
  std::uint32_t features_channels_;
  std::shared_ptr<State> state_;  // shared with the listener's blocks
  std::vector<frame::Extent> rungs_;
  frame::Extent prepared_;
  std::size_t row_bytes_ = 0;  // the prepared rung's
  NS::SharedPtr<MTL::Buffer> input_;
  NS::SharedPtr<MTL::Buffer> features_;
  Resident input_resident_;
  Resident features_resident_;
};

}  // namespace serenity::metal
