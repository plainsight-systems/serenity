#include "metal/passes/denoise/denoise.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "core/contracts/denoise.h"
#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/passes/bindings.h"
#include "metal/passes/denoise/rungs.h"

namespace serenity::metal {

namespace {

// The depth median's dispatches: a histogram's threadgroups square, a
// select's one threadgroup (denoise.metal reads both sizes from its
// threads).
constexpr NS::UInteger median_side = 16;
constexpr NS::UInteger select_threads = 256;

// The head's weights and bias among the static arrays.
constexpr std::size_t head_index = 0;

// Threadgroups of `per_group` covering `threads`.
constexpr NS::UInteger groups(std::uint32_t threads, NS::UInteger per_group) {
  return (threads + per_group - 1) / per_group;
}

}  // namespace

DenoisePass::DenoisePass(const Device& device, const Library& library,
                         Submission& submission,
                         const denoise::Network& network)
    : submission_(&submission),
      histogram_{library.compute_pipeline("denoise_depth_histogram0"),
                 library.compute_pipeline("denoise_depth_histogram1"),
                 library.compute_pipeline("denoise_depth_histogram2")},
      select_{library.compute_pipeline("denoise_depth_select0"),
              library.compute_pipeline("denoise_depth_select1"),
              library.compute_pipeline("denoise_depth_select2")},
      input_(library.compute_pipeline("denoise_input",
                                      {.group_threads = apple_group_threads})),
      output_(library.compute_pipeline("denoise_output")) {
  const auto pool = scoped_pool();
  // The head as the output kernel's matrix product reads it: [feature]
  // [output], the 124 outputs padded to head_columns with zeros, half
  // floats.
  const denoise::Convolution& h = network.head;
  std::vector<Half> head(std::size_t{h.inputs} * contracts::head_columns,
                         Half{0});
  for (std::uint32_t o = 0; o < h.outputs; ++o) {
    for (std::uint32_t c = 0; c < h.inputs; ++c) {
      head[std::size_t{c} * contracts::head_columns + o] =
          static_cast<Half>(h.weight[std::size_t{o} * h.inputs + c]);
    }
  }
  const std::array<std::span<const std::byte>, 2> arrays{
      std::as_bytes(std::span(head)), std::as_bytes(std::span(h.bias))};
  weights_ = std::make_unique<StaticArrays>(device, submission, arrays);

  network_ = std::make_unique<NeuralEngineNetwork>(
      device, submission, std::filesystem::path(network.source),
      contracts::network_inputs, network.widths.dec.back());
}

frame::Extent DenoisePass::grid(frame::Extent light) const {
  const std::optional<frame::Extent> rung =
      smallest_rung(network_->rungs(), light);
  if (!rung) {
    throw MetalError("DenoisePass: a light of " + frame::to_string(light) +
                     " is past the network's largest shape");
  }
  return *rung;
}

void DenoisePass::prepare(frame::Extent grid) {
  // The planes' rows as the input kernel and the head address them
  // (denoise.metal: contract 18's pitch, half floats).
  network_->prepare(
      grid, std::size_t{contracts::tensor_pitch(grid.width)} * sizeof(Half));
}

void DenoisePass::record(MTL4::ComputeCommandEncoder*& encoder,
                         const FrameResources& resources) const {
  const DenoiseImages* images = resources.denoise;
  if (encoder == nullptr || images == nullptr ||
      resources.denoise_constants == 0 || resources.light_radiance == nullptr ||
      resources.albedo == nullptr || resources.normal == nullptr ||
      resources.depth == nullptr || resources.motion == nullptr ||
      resources.full_albedo == nullptr || resources.radiance == nullptr) {
    throw MetalError(
        "DenoisePass: no encoder, or the frame lacks one of its buffers, "
        "constants, light, features, full-size albedo or radiance image");
  }
  if (network_->prepared() != images->padded()) {
    throw MetalError("DenoisePass: not prepared for the light's size");
  }
  record_median(encoder, resources);
  record_input(encoder, resources);
  encoder->endEncoding();
  // 3. The network on the Neural Engine, between the part before and the
  // part after (submission.h, "A hand-off").
  const HandOff hand_off = submission_->hand_off();
  network_->run(hand_off);
  encoder = hand_off.after->computeCommandEncoder();
  if (encoder == nullptr) {
    throw MetalError(
        "DenoisePass: the part after the network made no compute "
        "encoder");
  }
  encoder->setArgumentTable(resources.arguments);
  // The Neural Engine's features, and the input kernel's demodulated light,
  // visible to the head (GPU.8).
  encoder->barrierAfterQueueStages(MTL::StageDispatch, MTL::StageDispatch,
                                   MTL4::VisibilityOptionDevice);
  record_output(encoder, resources);
}

void DenoisePass::record_median(MTL4::ComputeCommandEncoder* encoder,
                                const FrameResources& resources) const {
  namespace binding = bindings::denoise;
  MTL4::ArgumentTable* arguments = resources.arguments;
  const DenoiseImages& images = *resources.denoise;
  const frame::Extent light = images.light();
  arguments->setAddress(resources.denoise_constants, binding::constants);
  arguments->setAddress(images.histogram(), binding::histogram);
  arguments->setAddress(images.median(), binding::median);
  arguments->setTexture(resources.depth->gpuResourceID(), binding::depth);
  for (std::size_t level = 0; level < histogram_.size(); ++level) {
    encoder->setComputePipelineState(histogram_.at(level).get());
    encoder->dispatchThreadgroups(
        MTL::Size{groups(light.width, median_side),
                  groups(light.height, median_side), 1},
        MTL::Size{median_side, median_side, 1});
    barrier_after_dispatch(encoder);
    encoder->setComputePipelineState(select_.at(level).get());
    encoder->dispatchThreadgroups(MTL::Size{1, 1, 1},
                                  MTL::Size{select_threads, 1, 1});
    barrier_after_dispatch(encoder);
  }
}

// 2. The network's input and the demodulated light.
void DenoisePass::record_input(MTL4::ComputeCommandEncoder* encoder,
                               const FrameResources& resources) const {
  namespace binding = bindings::denoise;
  MTL4::ArgumentTable* arguments = resources.arguments;
  const DenoiseImages& images = *resources.denoise;
  arguments->setAddress(images.history(), binding::history);
  arguments->setAddress(network_->input(), binding::output);
  arguments->setAddress(images.demod(), binding::demod);
  arguments->setTexture(resources.light_radiance->gpuResourceID(),
                        binding::light);
  arguments->setTexture(resources.albedo->gpuResourceID(), binding::albedo);
  arguments->setTexture(resources.normal->gpuResourceID(), binding::normal);
  arguments->setTexture(resources.motion->gpuResourceID(), binding::motion);
  encoder->setComputePipelineState(input_.get());
  dispatch_per_pixel(encoder, input_.get(), images.padded());
}

// 4. The output: a threadgroup a row of conv_pixels input pixels, a thread
// an output pixel of twice the padded grid.
void DenoisePass::record_output(MTL4::ComputeCommandEncoder* encoder,
                                const FrameResources& resources) const {
  namespace binding = bindings::denoise;
  MTL4::ArgumentTable* arguments = resources.arguments;
  const DenoiseImages& images = *resources.denoise;
  const frame::Extent grid = images.padded();
  arguments->setAddress(network_->features(), binding::input);
  arguments->setAddress(weights_->address(head_index), binding::weights);
  arguments->setAddress(weights_->address(head_index + 1), binding::bias);
  arguments->setAddress(images.demod(), binding::demod);
  arguments->setAddress(images.history(), binding::history);
  arguments->setAddress(images.next_history(), binding::next_history);
  arguments->setTexture(resources.full_albedo->gpuResourceID(),
                        binding::full_albedo);
  arguments->setTexture(resources.radiance->gpuResourceID(), binding::radiance);
  encoder->setComputePipelineState(output_.get());
  encoder->dispatchThreadgroups(
      MTL::Size{groups(grid.width, contracts::conv_pixels), grid.height, 1},
      MTL::Size{contracts::denoise_threads, 1, 1});
}

}  // namespace serenity::metal
