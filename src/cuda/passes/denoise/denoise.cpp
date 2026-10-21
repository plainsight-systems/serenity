#include "cuda/passes/denoise/denoise.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/denoise/unet.h"
#include "cuda/device/error.h"
#include "serenity/cuda_modules/denoise.h"

namespace serenity::cuda {

namespace {

// A half float on the host: _Float16, a compiler extension (P.2), as
// FeatureImages widens its own (cuda/frame/feature_images.cpp); its
// conversion from float rounds to nearest even, as Metal's host's.
using Half = _Float16;

// A 3 x 3 kernel's taps.
constexpr std::uint32_t conv_taps = 9;

constexpr std::uint32_t padded(std::uint32_t channels) {
  return (channels + contracts::channel_block - 1u) / contracts::channel_block *
         contracts::channel_block;
}

// A 3 x 3 convolution's weights, PyTorch's [out][in][ky][kx], as conv3x3
// reads them: [tap = ky * 3 + kx][in, padded][out], half floats, padding 0.
// Metal's layout.
std::vector<Half> laid_out(const denoise::Convolution& c) {
  const std::uint32_t in = padded(c.inputs);
  // a multiple of channel_block (network.h)
  const std::uint32_t out = c.outputs;
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

// `values` in device memory of their own, uploaded once.
template <typename T>
DeviceBuffer uploaded(std::span<const T> values) {
  const std::span<const std::byte> bytes = std::as_bytes(values);
  DeviceBuffer buffer(bytes.size());
  upload_now(buffer.address(), bytes, "DenoisePass");
  return buffer;
}

// The depth median's launches: a histogram's blocks square, a select's one
// block of 256 threads, the size its block scan is built for (denoise.cu,
// select_block, which refuses another); the max pool's blocks, each thread
// four channels. Metal's.
constexpr unsigned median_side = 16;
constexpr unsigned select_threads = 256;
constexpr unsigned pool_threads = 256;
constexpr std::uint32_t pool_channels = 4;  // a half4 a thread

// Blocks of `per_block` covering `threads`.
constexpr unsigned groups(std::uint32_t threads, unsigned per_block) {
  return (threads + per_block - 1) / per_block;
}

}  // namespace

DenoisePass::DenoisePass(const denoise::Network& network)
    : module_(cuda_modules::denoise),
      histogram_{module_.kernel("denoise_depth_histogram0"),
                 module_.kernel("denoise_depth_histogram1"),
                 module_.kernel("denoise_depth_histogram2")},
      select_{module_.kernel("denoise_depth_select0"),
              module_.kernel("denoise_depth_select1"),
              module_.kernel("denoise_depth_select2")},
      input_(module_.kernel("denoise_input")),
      conv_{module_.kernel("conv3x3_n1"), module_.kernel("conv3x3_n2"),
            module_.kernel("conv3x3_n3"), module_.kernel("conv3x3_n4")},
      maxpool_(module_.kernel("denoise_maxpool")),
      output_(module_.kernel("denoise_output")) {
  // The convolutions in the order they run (core/denoise/unet.h): the
  // encoders, the bottleneck's two, the decoders, the last.
  const std::array<const denoise::Convolution*, DenoiseImages::convolutions>
      convs = denoise::in_order(network);
  for (std::size_t i = 0; i < convs.size(); ++i) {
    const std::vector<Half> w = laid_out(*convs.at(i));
    weights_.at(i) = uploaded(std::span<const Half>(w));
    biases_.at(i) = uploaded(std::span<const float>(convs.at(i)->bias));
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
  head_ = uploaded(std::span<const Half>(head));
  head_bias_ = uploaded(std::span<const float>(h.bias));
}

CUfunction DenoisePass::convolution(std::uint32_t chunk) const {
  if (chunk < 1 || chunk > conv_.size()) {
    throw CudaError("DenoisePass: no convolution kernel computes " +
                    std::to_string(chunk) + " blocks");
  }
  return conv_.at(chunk - 1);
}

void DenoisePass::record(CUstream stream, const DenoiseInputs& inputs) const {
  const DenoiseImages* images = inputs.images;
  if (images == nullptr || images->input() == 0 ||
      inputs.light_radiance == nullptr || inputs.albedo == 0 ||
      inputs.normal == 0 || inputs.depth == 0 || inputs.motion == 0 ||
      inputs.full_albedo == 0 || inputs.radiance == nullptr) {
    throw CudaError(
        "DenoisePass: the frame lacks one of its buffers, light, features, "
        "full-size albedo or radiance image");
  }
  const frame::Extent output{inputs.constants.output_width,
                             inputs.constants.output_height};
  if (!(inputs.full_size == output) || !(inputs.radiance->size() == output) ||
      !(inputs.light_radiance->size() == images->light())) {
    throw CudaError(
        "DenoisePass: the full-size albedo or the radiance image is not of "
        "the output's size, or the light's hand-off not of the light's");
  }
  record_median(stream, inputs);
  record_input(stream, inputs);
  record_unet(stream, inputs);
  record_output(stream, inputs);
}

// 1. The depth's median: three levels of histogram and select, each after
// the one before on the stream.
void DenoisePass::record_median(CUstream stream,
                                const DenoiseInputs& inputs) const {
  const DenoiseImages& images = *inputs.images;
  const frame::Extent light = images.light();
  contracts::DenoiseConstants k = inputs.constants;
  CUdeviceptr histogram = images.histogram();
  CUdeviceptr median = images.median();
  CUdeviceptr depth = inputs.depth;
  for (std::size_t level = 0; level < histogram_.size(); ++level) {
    auto counted = std::to_array<void*>({&k, &histogram, &median, &depth});
    launch(stream, histogram_.at(level),
           LaunchShape{.blocks_x = groups(light.width, median_side),
                       .blocks_y = groups(light.height, median_side),
                       .threads_x = median_side,
                       .threads_y = median_side},
           counted);
    auto selected = std::to_array<void*>({&histogram, &median});
    launch(stream, select_.at(level), LaunchShape{.threads_x = select_threads},
           selected);
  }
}

// 2. The network's input and the demodulated light.
void DenoisePass::record_input(CUstream stream,
                               const DenoiseInputs& inputs) const {
  const DenoiseImages& images = *inputs.images;
  contracts::DenoiseConstants k = inputs.constants;
  CUdeviceptr median = images.median();
  CUdeviceptr history = images.history();
  CUdeviceptr output = images.input();
  CUdeviceptr demod = images.demod();
  CUdeviceptr light = inputs.light_radiance->address();
  std::uint64_t light_pitch = inputs.light_radiance->pitch();
  CUdeviceptr albedo = inputs.albedo;
  CUdeviceptr normal = inputs.normal;
  CUdeviceptr depth = inputs.depth;
  CUdeviceptr motion = inputs.motion;
  auto arguments =
      std::to_array<void*>({&k, &median, &history, &output, &demod, &light,
                            &light_pitch, &albedo, &normal, &depth, &motion});
  launch_per_pixel(stream, input_, images.padded(), arguments);
}

// 3. The U-Net: encoders and pools, the bottleneck, the decoders (each
// upsampling as it reads), the last convolution (denoise_images.h's
// order), each after the one before on the stream.
void DenoisePass::record_unet(CUstream stream,
                              const DenoiseInputs& inputs) const {
  const DenoiseImages& images = *inputs.images;
  const auto convolve = [&](std::size_t i) {
    const DenoiseImages::Conv& c = images.conv(i);
    contracts::ConvParams params = c.params;
    CUdeviceptr input = c.input;
    CUdeviceptr weights = weights_.at(i).address();
    CUdeviceptr bias = biases_.at(i).address();
    CUdeviceptr output = c.output;
    CUdeviceptr coarse = c.coarse;
    auto arguments = std::to_array<void*>(
        {&params, &input, &weights, &bias, &output, &coarse});
    launch(stream, convolution(params.chunk_blocks),
           LaunchShape{.blocks_x = groups(params.width, contracts::conv_pixels),
                       .blocks_y = groups(params.height, contracts::conv_rows),
                       .blocks_z = params.out_blocks / params.chunk_blocks,
                       .threads_x = contracts::denoise_threads},
           arguments);
  };
  const auto max_pool = [&](const DenoiseImages::Resample& r) {
    contracts::ResampleParams params = r.params;
    CUdeviceptr input = r.input;
    CUdeviceptr output = r.output;
    auto arguments = std::to_array<void*>({&params, &input, &output});
    launch(stream, maxpool_,
           linear_shape(std::uint64_t{params.width} * params.height *
                            (params.channels / pool_channels),
                        pool_threads),
           arguments);
  };
  for (std::size_t l = 0; l < denoise::unet_levels; ++l) {
    convolve(l);
    max_pool(images.pool(l));
  }
  for (std::size_t i = denoise::first_bottleneck;
       i < DenoiseImages::convolutions; ++i) {
    convolve(i);
  }
}

// 4. The output: a block a row of conv_pixels input pixels, a thread an
// output pixel of twice the padded grid.
void DenoisePass::record_output(CUstream stream,
                                const DenoiseInputs& inputs) const {
  const DenoiseImages& images = *inputs.images;
  const frame::Extent grid = images.padded();
  contracts::DenoiseConstants k = inputs.constants;
  CUdeviceptr features = images.features();
  CUdeviceptr head = head_.address();
  CUdeviceptr head_bias = head_bias_.address();
  CUdeviceptr demod = images.demod();
  CUdeviceptr history = images.history();
  CUdeviceptr next_history = images.next_history();
  CUdeviceptr motion = inputs.motion;
  CUdeviceptr full_albedo = inputs.full_albedo;
  CUdeviceptr radiance = inputs.radiance->address();
  std::uint64_t radiance_pitch = inputs.radiance->pitch();
  auto arguments = std::to_array<void*>(
      {&k, &features, &head, &head_bias, &demod, &history, &next_history,
       &motion, &full_albedo, &radiance, &radiance_pitch});
  launch(stream, output_,
         LaunchShape{.blocks_x = groups(grid.width, contracts::conv_pixels),
                     .blocks_y = grid.height,
                     .threads_x = contracts::denoise_threads},
         arguments);
}

}  // namespace serenity::cuda
