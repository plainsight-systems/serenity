#include "metal/passes/denoise/denoise.h"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "core/contracts/denoise.h"
#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/passes/bindings.h"

namespace serenity::metal {

namespace {

// A half float, the convolutions' weights: _Float16, a compiler extension
// (P.2), until the toolchain's C++ has std::float16_t (C++23), as
// metal/frame/feature_images.cpp has it.
using Half = _Float16;
static_assert(sizeof(Half) == 2);

constexpr std::uint32_t padded(std::uint32_t channels) {
    return (channels + contracts::channel_block - 1u) / contracts::channel_block * contracts::channel_block;
}

// A 3 x 3 convolution's weights, PyTorch's [out][in][ky][kx], as conv3x3
// reads them: [tap = ky * 3 + kx][in, padded][out], half floats, padding 0.
std::vector<Half> laid_out(const denoise::Convolution& c) {
    const std::uint32_t in = padded(c.inputs);
    const std::uint32_t out = c.outputs;  // a multiple of channel_block (network.h)
    std::vector<Half> w(std::size_t{9} * in * out, Half{0});
    for (std::uint32_t o = 0; o < c.outputs; ++o) {
        for (std::uint32_t i = 0; i < c.inputs; ++i) {
            for (std::uint32_t tap = 0; tap < 9; ++tap) {
                const float v = c.weight[(std::size_t{o} * c.inputs + i) * 9 + tap];
                w[(std::size_t{tap} * in + i) * out + o] = static_cast<Half>(v);
            }
        }
    }
    return w;
}

// The output kernel's (denoise.metal): the head's outputs padded to its
// two simdgroups of 64, and the widest last feature width it stages.
constexpr std::uint32_t head_columns = 128;
constexpr std::uint32_t max_head_features = 64;
static_assert(contracts::head_outputs <= head_columns);

void barrier(MTL4::ComputeCommandEncoder* encoder) {
    encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
}

}  // namespace

DenoisePass::DenoisePass(const Device& device, const Library& library, Submission& submission,
                         const denoise::Network& network)
    : histogram_{library.compute_pipeline("denoise_depth_histogram0"),
                 library.compute_pipeline("denoise_depth_histogram1"),
                 library.compute_pipeline("denoise_depth_histogram2")},
      select_{library.compute_pipeline("denoise_depth_select0"), library.compute_pipeline("denoise_depth_select1"),
              library.compute_pipeline("denoise_depth_select2")},
      input_(library.compute_pipeline("denoise_input")),
      conv_{library.compute_pipeline("conv3x3_n1"), library.compute_pipeline("conv3x3_n2"),
            library.compute_pipeline("conv3x3_n3"), library.compute_pipeline("conv3x3_n4")},
      maxpool_(library.compute_pipeline("denoise_maxpool")),
      upsample_(library.compute_pipeline("denoise_upsample")),
      output_(library.compute_pipeline("denoise_output")) {
    // The convolutions in DenoiseImages' order (denoise_images.h).
    const std::array<const denoise::Convolution*, DenoiseImages::convolutions> convs = {
        &network.enc[0], &network.enc[1], &network.enc[2], &network.enc[3], &network.bottleneck_a,
        &network.bottleneck_b, &network.dec[0], &network.dec[1], &network.dec[2], &network.dec[3],
        &network.dec_final};
    std::vector<std::vector<Half>> weights;
    weights.reserve(convs.size());
    for (const denoise::Convolution* c : convs) {
        weights.push_back(laid_out(*c));
    }
    std::vector<std::span<const std::byte>> arrays;
    for (std::size_t i = 0; i < convs.size(); ++i) {
        arrays.push_back(std::as_bytes(std::span(weights[i])));
        arrays.push_back(std::as_bytes(std::span(convs[i]->bias)));
    }
    // The head as the output kernel's matrix product reads it: [feature]
    // [output], the 124 outputs padded to 128 with zeros, floats.
    const denoise::Convolution& h = network.head;
    if (h.inputs > max_head_features) {
        throw MetalError("DenoisePass: the network's last width, " + std::to_string(h.inputs) +
                         ", is past the " + std::to_string(max_head_features) + " the output kernel stages");
    }
    std::vector<float> head(std::size_t{h.inputs} * head_columns, 0.0f);
    for (std::uint32_t o = 0; o < h.outputs; ++o) {
        for (std::uint32_t c = 0; c < h.inputs; ++c) {
            head[std::size_t{c} * head_columns + o] = h.weight[std::size_t{o} * h.inputs + c];
        }
    }
    arrays.push_back(std::as_bytes(std::span(head)));
    arrays.push_back(std::as_bytes(std::span(network.head.bias)));
    weights_ = std::make_unique<StaticArrays>(device, submission, arrays);
}

NS::SharedPtr<MTL::ComputePipelineState> DenoisePass::pipeline(std::uint32_t chunk) const {
    if (chunk < 1 || chunk > conv_.size()) {
        throw MetalError("DenoisePass: no convolution kernel computes " + std::to_string(chunk) + " blocks");
    }
    return conv_[chunk - 1];
}

void DenoisePass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    const DenoiseImages* images = resources.denoise;
    if (images == nullptr || images->input() == 0 || resources.denoise_constants == 0 ||
        resources.light_radiance == nullptr || resources.albedo == nullptr || resources.normal == nullptr ||
        resources.depth == nullptr || resources.motion == nullptr || resources.full_albedo == nullptr ||
        resources.radiance == nullptr) {
        throw MetalError("DenoisePass: the frame lacks one of its buffers, constants, light, features, full-size "
                         "albedo or radiance image");
    }
    namespace binding = bindings::denoise;
    MTL4::ArgumentTable* arguments = resources.arguments;
    const frame::Extent light = images->light();
    const frame::Extent grid = images->padded();

    // 1. The depth's median: three levels of histogram and select.
    arguments->setAddress(resources.denoise_constants, binding::constants);
    arguments->setAddress(images->histogram(), binding::histogram);
    arguments->setAddress(images->median(), binding::median);
    arguments->setTexture(resources.depth->gpuResourceID(), binding::depth);
    for (std::size_t level = 0; level < histogram_.size(); ++level) {
        encoder->setComputePipelineState(histogram_[level].get());
        encoder->dispatchThreadgroups(MTL::Size{(light.width + 15) / 16, (light.height + 15) / 16, 1},
                                      MTL::Size{16, 16, 1});
        barrier(encoder);
        encoder->setComputePipelineState(select_[level].get());
        encoder->dispatchThreadgroups(MTL::Size{1, 1, 1}, MTL::Size{256, 1, 1});
        barrier(encoder);
    }

    // 2. The network's input and the demodulated light.
    arguments->setAddress(images->history(), binding::history);
    arguments->setAddress(images->input(), binding::output);
    arguments->setAddress(images->demod(), binding::demod);
    arguments->setTexture(resources.light_radiance->gpuResourceID(), binding::light);
    arguments->setTexture(resources.albedo->gpuResourceID(), binding::albedo);
    arguments->setTexture(resources.normal->gpuResourceID(), binding::normal);
    arguments->setTexture(resources.motion->gpuResourceID(), binding::motion);
    encoder->setComputePipelineState(input_.get());
    dispatch_per_pixel(encoder, input_.get(), grid);

    // 3. The U-Net: encoders and pools, the bottleneck, upsamplings and
    // decoders, the last convolution (denoise_images.h's order).
    const auto convolve = [&](std::size_t i) {
        const DenoiseImages::Conv& c = images->conv(i);
        barrier(encoder);
        arguments->setAddress(c.params_address, binding::params);
        arguments->setAddress(c.input, binding::input);
        arguments->setAddress(weights_->address(2 * i), binding::weights);
        arguments->setAddress(weights_->address(2 * i + 1), binding::bias);
        arguments->setAddress(c.output, binding::output);
        const NS::SharedPtr<MTL::ComputePipelineState> conv = pipeline(c.params.chunk_blocks);
        encoder->setComputePipelineState(conv.get());
        encoder->dispatchThreadgroups(MTL::Size{(c.params.width + 31) / 32, (c.params.height + 3) / 4,
                                                c.params.out_blocks / c.params.chunk_blocks},
                                      MTL::Size{128, 1, 1});
    };
    const auto resample = [&](const DenoiseImages::Resample& r, const MTL::ComputePipelineState* kernel) {
        barrier(encoder);
        arguments->setAddress(r.params_address, binding::params);
        arguments->setAddress(r.input, binding::input);
        arguments->setAddress(r.output, binding::output);
        encoder->setComputePipelineState(kernel);
        encoder->dispatchThreads(
            MTL::Size{NS::UInteger{r.params.width} * r.params.height * (r.params.channels / 4), 1, 1},
            MTL::Size{256, 1, 1});
    };
    for (std::size_t l = 0; l < 4; ++l) {
        convolve(l);
        resample(images->pool(l), maxpool_.get());
    }
    convolve(4);
    convolve(5);
    for (std::size_t i = 0; i < 4; ++i) {
        resample(images->upsample(i), upsample_.get());
        convolve(6 + i);
    }
    convolve(10);

    // 4. The output: a threadgroup a row of 16 input pixels, a thread an
    // output pixel of twice the padded grid.
    barrier(encoder);
    arguments->setAddress(images->features(), binding::input);
    arguments->setAddress(weights_->address(2 * DenoiseImages::convolutions), binding::weights);
    arguments->setAddress(weights_->address(2 * DenoiseImages::convolutions + 1), binding::bias);
    arguments->setAddress(images->demod(), binding::demod);
    arguments->setAddress(images->history(), binding::history);
    arguments->setAddress(images->next_history(), binding::next_history);
    arguments->setTexture(resources.full_albedo->gpuResourceID(), binding::full_albedo);
    arguments->setTexture(resources.radiance->gpuResourceID(), binding::radiance);
    encoder->setComputePipelineState(output_.get());
    encoder->dispatchThreadgroups(MTL::Size{(grid.width + 15) / 16, grid.height, 1}, MTL::Size{64, 1, 1});
}

}  // namespace serenity::metal
