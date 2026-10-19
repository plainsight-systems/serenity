#include "metal/passes/denoise/denoise.h"

#include <array>
#include <cstddef>
#include <cstdint>
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

// A 3 x 3 kernel's taps.
constexpr std::uint32_t conv_taps = 9;

constexpr std::uint32_t padded(std::uint32_t channels) {
  return (channels + contracts::channel_block - 1u) / contracts::channel_block *
         contracts::channel_block;
}

// A 3 x 3 convolution's weights, PyTorch's [out][in][ky][kx], as conv3x3
// reads them: [tap = ky * 3 + kx][in, padded][out], half floats, padding 0.
std::vector<Half> laid_out(const denoise::Convolution& c) {
  const std::uint32_t in = padded(c.inputs);
  const std::uint32_t out =
      c.outputs;  // a multiple of channel_block (network.h)
  std::vector<Half> w(std::size_t{conv_taps} * in * out, Half{0});
  for (std::uint32_t o = 0; o < c.outputs; ++o) {
    for (std::uint32_t i = 0; i < c.inputs; ++i) {
      for (std::uint32_t tap = 0; tap < conv_taps; ++tap) {
        const float v =
            c.weight[(std::size_t{o} * c.inputs + i) * conv_taps + tap];
        w[(std::size_t{tap} * in + i) * out + o] = static_cast<Half>(v);
      }
    }
  }
  return w;
}

// The depth median's dispatches: a histogram's threadgroups square, a
// select's one threadgroup (denoise.metal reads both sizes from its
// threads); the max pool's threadgroups, each thread four channels.
constexpr NS::UInteger median_side = 16;
constexpr NS::UInteger select_threads = 256;
constexpr NS::UInteger pool_threads = 256;
constexpr std::uint32_t pool_channels = 4;  // a half4 a thread

// Where each convolution's weights and bias are among the static arrays,
// convolution i's at 2i and 2i + 1, and the head's after the last
// convolution's (the constructor lays them out so).
constexpr std::size_t weights_index(std::size_t i) { return 2 * i; }
constexpr std::size_t bias_index(std::size_t i) { return 2 * i + 1; }
constexpr std::size_t head_index =
    2 * DenoiseImages::convolutions;  // and its bias after it

// Threadgroups of `per_group` covering `threads`.
constexpr NS::UInteger groups(std::uint32_t threads, NS::UInteger per_group) {
  return (threads + per_group - 1) / per_group;
}

}  // namespace

DenoisePass::DenoisePass(const Device& device, const Library& library,
                         Submission& submission,
                         const denoise::Network& network)
    : histogram_{library.compute_pipeline("denoise_depth_histogram0"),
                 library.compute_pipeline("denoise_depth_histogram1"),
                 library.compute_pipeline("denoise_depth_histogram2")},
      select_{library.compute_pipeline("denoise_depth_select0"),
              library.compute_pipeline("denoise_depth_select1"),
              library.compute_pipeline("denoise_depth_select2")},
      input_(library.compute_pipeline("denoise_input")),
      conv_{library.compute_pipeline("conv3x3_n1"),
            library.compute_pipeline("conv3x3_n2"),
            library.compute_pipeline("conv3x3_n3"),
            library.compute_pipeline("conv3x3_n4")},
      maxpool_(library.compute_pipeline("denoise_maxpool")),
      output_(library.compute_pipeline("denoise_output")) {
  // The convolutions in DenoiseImages' order (denoise_images.h): the
  // encoders, the bottleneck's two, the decoders, the last.
  std::array<const denoise::Convolution*, DenoiseImages::convolutions> convs{};
  for (std::size_t l = 0; l < DenoiseImages::levels; ++l) {
    convs.at(l) = &network.enc.at(l);
    convs.at(DenoiseImages::first_decoder + l) = &network.dec.at(l);
  }
  convs[DenoiseImages::first_bottleneck] = &network.bottleneck_a;
  convs[DenoiseImages::first_bottleneck + 1] = &network.bottleneck_b;
  convs[DenoiseImages::last_convolution] = &network.dec_final;
  std::vector<std::vector<Half>> weights;
  weights.reserve(convs.size());
  for (const denoise::Convolution* c : convs) {
    weights.push_back(laid_out(*c));
  }
  // In weights_index(), bias_index() and head_index()'s order (above).
  std::vector<std::span<const std::byte>> arrays;
  for (std::size_t i = 0; i < convs.size(); ++i) {
    arrays.push_back(std::as_bytes(std::span(weights[i])));
    arrays.push_back(std::as_bytes(std::span(convs.at(i)->bias)));
  }
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
  arrays.push_back(std::as_bytes(std::span(head)));
  arrays.push_back(std::as_bytes(std::span(network.head.bias)));
  weights_ = std::make_unique<StaticArrays>(device, submission, arrays);
}

MTL::ComputePipelineState* DenoisePass::pipeline(std::uint32_t chunk) const {
  if (chunk < 1 || chunk > conv_.size()) {
    throw MetalError("DenoisePass: no convolution kernel computes " +
                     std::to_string(chunk) + " blocks");
  }
  return conv_.at(chunk - 1).get();
}

void DenoisePass::record(MTL4::ComputeCommandEncoder* encoder,
                         const FrameResources& resources) const {
  const DenoiseImages* images = resources.denoise;
  if (images == nullptr || images->input() == 0 ||
      resources.denoise_constants == 0 || resources.light_radiance == nullptr ||
      resources.albedo == nullptr || resources.normal == nullptr ||
      resources.depth == nullptr || resources.motion == nullptr ||
      resources.full_albedo == nullptr || resources.radiance == nullptr) {
    throw MetalError(
        "DenoisePass: the frame lacks one of its buffers, constants, light, "
        "features, full-size "
        "albedo or radiance image");
  }
  record_median(encoder, resources);
  record_input(encoder, resources);
  record_unet(encoder, resources);
  record_output(encoder, resources);
}

// 1. The depth's median: three levels of histogram and select, each after
// a barrier on the one before. The first histogram's barrier is the
// renderer's, before the pass (metal/frame/renderer.cpp): the depth it
// reads was written before it.
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

// 3. The U-Net: encoders and pools, the bottleneck, the decoders (each
// upsampling as it reads), the last convolution (denoise_images.h's
// order), each after a barrier on the one before.
void DenoisePass::record_unet(MTL4::ComputeCommandEncoder* encoder,
                              const FrameResources& resources) const {
  namespace binding = bindings::denoise;
  MTL4::ArgumentTable* arguments = resources.arguments;
  const DenoiseImages& images = *resources.denoise;
  const auto convolve = [&](std::size_t i) {
    const DenoiseImages::Conv& c = images.conv(i);
    barrier_after_dispatch(encoder);
    arguments->setAddress(c.params_address, binding::params);
    arguments->setAddress(c.input, binding::input);
    arguments->setAddress(weights_->address(weights_index(i)),
                          binding::weights);
    arguments->setAddress(weights_->address(bias_index(i)), binding::bias);
    arguments->setAddress(c.output, binding::output);
    arguments->setAddress(c.coarse, binding::coarse);
    encoder->setComputePipelineState(pipeline(c.params.chunk_blocks));
    encoder->dispatchThreadgroups(
        MTL::Size{groups(c.params.width, contracts::conv_pixels),
                  groups(c.params.height, contracts::conv_rows),
                  c.params.out_blocks / c.params.chunk_blocks},
        MTL::Size{contracts::denoise_threads, 1, 1});
  };
  const auto max_pool = [&](const DenoiseImages::Resample& r) {
    barrier_after_dispatch(encoder);
    arguments->setAddress(r.params_address, binding::params);
    arguments->setAddress(r.input, binding::input);
    arguments->setAddress(r.output, binding::output);
    encoder->setComputePipelineState(maxpool_.get());
    encoder->dispatchThreads(
        MTL::Size{NS::UInteger{r.params.width} * r.params.height *
                      (r.params.channels / pool_channels),
                  1, 1},
        MTL::Size{pool_threads, 1, 1});
  };
  for (std::size_t l = 0; l < DenoiseImages::levels; ++l) {
    convolve(l);
    max_pool(images.pool(l));
  }
  for (std::size_t i = DenoiseImages::first_bottleneck;
       i < DenoiseImages::convolutions; ++i) {
    convolve(i);
  }
}

// 4. The output: a threadgroup a row of conv_pixels input pixels, a thread
// an output pixel of twice the padded grid.
void DenoisePass::record_output(MTL4::ComputeCommandEncoder* encoder,
                                const FrameResources& resources) const {
  namespace binding = bindings::denoise;
  MTL4::ArgumentTable* arguments = resources.arguments;
  const DenoiseImages& images = *resources.denoise;
  const frame::Extent grid = images.padded();
  barrier_after_dispatch(encoder);
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
