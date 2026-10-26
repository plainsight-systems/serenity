#include "metal/passes/denoise/denoise.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "core/contracts/denoise.h"
#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/passes/bindings.h"

namespace serenity::metal {

namespace {

// The network package's file, in the network's directory
// (training/export_metal.py), and its function, the network.
constexpr const char* network_package = "unet.mtlpackage";
constexpr const char* network_function = "main";

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

NS::SharedPtr<NS::String> string_of(const std::string& text) {
  return NS::TransferPtr(
      NS::String::alloc()->init(text.c_str(), NS::UTF8StringEncoding));
}

}  // namespace

DenoisePass::DenoisePass(const Device& device, const Library& library,
                         Submission& submission,
                         const denoise::Network& network)
    : device_(NS::RetainPtr(device.handle())),
      submission_(submission),
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

  // The network's package (training/export_metal.py), as Apple's workflow
  // loads one: the package as a library, its network function, a compiler
  // to compile it for each size (prepare()).
  const std::filesystem::path package =
      std::filesystem::path(network.source) / network_package;
  if (!std::filesystem::exists(package)) {
    throw MetalError("DenoisePass: no " + package.string() +
                     " beside the network's weights; make it with "
                     "training/export_metal.py");
  }
  NS::Error* error = nullptr;
  const auto url = NS::URL::fileURLWithPath(string_of(package.string()).get());
  auto library_of = NS::TransferPtr(device_->newLibrary(url, &error));
  if (!library_of) {
    throw MetalError("DenoisePass: Metal loads no library from " +
                     package.string() + ": " + describe(error));
  }
  network_ = NS::TransferPtr(MTL4::LibraryFunctionDescriptor::alloc()->init());
  network_->setName(string_of(network_function).get());
  network_->setLibrary(library_of.get());
  auto compiler = NS::TransferPtr(MTL4::CompilerDescriptor::alloc()->init());
  compiler_ = NS::TransferPtr(device_->newCompiler(compiler.get(), &error));
  auto table = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
  table->setMaxBufferBindCount(bindings::network::features + 1);
  network_arguments_ =
      NS::TransferPtr(device_->newArgumentTable(table.get(), &error));
  if (!compiler_ || !network_arguments_) {
    throw MetalError("DenoisePass: Metal made no compiler or argument table: " +
                     describe(error));
  }
}

void DenoisePass::prepare(frame::Extent padded) {
  if (padded == prepared_) {
    return;
  }
  // Frames in flight may still run the old pipeline (DenoiseImages, which
  // prepares first, has drained them at a new size).
  const auto pool = scoped_pool();
  auto descriptor =
      NS::TransferPtr(MTL4::MachineLearningPipelineDescriptor::alloc()->init());
  descriptor->setMachineLearningFunctionDescriptor(network_.get());
  // The input's (W, H, C, 1), innermost first (denoise_images.h).
  const std::array<NS::Integer, 4> dims{padded.width, padded.height,
                                        contracts::network_inputs, 1};
  descriptor->setInputDimensions(
      NS::TransferPtr(MTL::TensorExtents::alloc()->init(4, dims.data())).get(),
      bindings::network::input);
  NS::Error* error = nullptr;
  auto pipeline = NS::TransferPtr(
      compiler_->newMachineLearningPipelineState(descriptor.get(), &error));
  if (!pipeline) {
    throw MetalError("DenoisePass: Metal compiled no network for " +
                     frame::to_string(padded) + ": " + describe(error));
  }
  auto heap = NS::TransferPtr(MTL::HeapDescriptor::alloc()->init());
  heap->setType(MTL::HeapTypePlacement);
  heap->setStorageMode(MTL::StorageModePrivate);
  heap->setSize(std::max<NS::UInteger>(pipeline->intermediatesHeapSize(), 1));
  auto made = NS::TransferPtr(device_->newHeap(heap.get()));
  if (!made) {
    throw MetalError("DenoisePass: the device made no intermediates heap");
  }
  pipeline_resident_.reset();
  heap_resident_.reset();
  pipeline_ = pipeline;
  heap_ = made;
  pipeline_resident_ = submission_.keep_resident(pipeline_.get());
  heap_resident_ = submission_.keep_resident(heap_.get());
  prepared_ = padded;
}

void DenoisePass::record(MTL4::CommandBuffer* commands,
                         MTL4::ComputeCommandEncoder*& encoder,
                         const FrameResources& resources) const {
  const DenoiseImages* images = resources.denoise;
  if (commands == nullptr || encoder == nullptr || images == nullptr ||
      images->input() == 0 || images->input_tensor() == nullptr ||
      resources.denoise_constants == 0 || resources.light_radiance == nullptr ||
      resources.albedo == nullptr || resources.normal == nullptr ||
      resources.depth == nullptr || resources.motion == nullptr ||
      resources.full_albedo == nullptr || resources.radiance == nullptr) {
    throw MetalError(
        "DenoisePass: no command buffer or encoder, or the frame lacks one of "
        "its buffers, tensors, constants, light, features, full-size albedo "
        "or radiance image");
  }
  if (!pipeline_ || images->padded() != prepared_) {
    throw MetalError("DenoisePass: not prepared for the light's size");
  }
  record_median(encoder, resources);
  record_input(encoder, resources);
  encoder->endEncoding();
  // 3. The network, after the input kernel's writes and before the output
  // kernel's reads (GPU.8; Metal's machine-learning stage).
  MTL4::MachineLearningCommandEncoder* network =
      commands->machineLearningCommandEncoder();
  if (network == nullptr) {
    throw MetalError("DenoisePass: no machine-learning encoder");
  }
  network->barrierAfterQueueStages(MTL::StageDispatch,
                                   MTL::StageMachineLearning,
                                   MTL4::VisibilityOptionDevice);
  network_arguments_->setResource(images->input_tensor()->gpuResourceID(),
                                  bindings::network::input);
  network_arguments_->setResource(images->features_tensor()->gpuResourceID(),
                                  bindings::network::features);
  network->setPipelineState(pipeline_.get());
  network->setArgumentTable(network_arguments_.get());
  network->dispatchNetwork(heap_.get());
  network->endEncoding();
  encoder = commands->computeCommandEncoder();
  if (encoder == nullptr) {
    throw MetalError("DenoisePass: the command buffer made no compute encoder");
  }
  encoder->setArgumentTable(resources.arguments);
  encoder->barrierAfterQueueStages(MTL::StageMachineLearning,
                                   MTL::StageDispatch,
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
  arguments->setAddress(images.input(), binding::output);
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
  arguments->setAddress(images.features(), binding::input);
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
