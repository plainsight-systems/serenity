#pragma once

// CUDA's runner of a denoiser fixture (gpu/support/denoise_fixture.h)
// through its denoise pass (cuda/passes/denoise/denoise.h), as the renderer
// runs it, Metal's (tests/gpu/support/denoise_runner.h) under the same name
// and interface: the renderer's own images at the light size and the
// full-size albedo, uploaded in the formats the renderer's are (contract
// 17's: albedo, normal and motion four half floats a pixel, depth one
// float, rows tightly packed, cuda/frame/feature_images.h; the light's
// hand-off RGBA32Float, pitched), the pass recorded frame after frame with
// its own history (cuda/frame/denoise_images.h, joined to the history as
// frames 0, 1, ...), and each frame's radiance image read back.

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "core/contracts/denoise.h"
#include "core/contracts/linear_image.h"
#include "core/denoise/network.h"
#include "core/frame/frame_inputs.h"
#include "core/output/pfm.h"
#include "cuda/device/buffer.h"
#include "cuda/device/device.h"
#include "cuda/device/error.h"
#include "cuda/device/images.h"
#include "cuda/device/submission.h"
#include "cuda/frame/denoise_images.h"
#include "cuda/passes/denoise/denoise.h"
#include "gpu/support/denoise_fixture.h"

namespace serenity::tests {

namespace detail {

// `image` of `size`, checked; its pixels.
inline std::size_t pixels_of(const contracts::LinearImage& image,
                             frame::Extent size) {
  if (!(image.extent == size)) {
    throw std::runtime_error(
        "denoise fixture: an image of another size than the pass reads");
  }
  return std::size_t{size.width} * size.height;
}

// `image`'s RGB into a new device buffer of four half floats a pixel (the
// alpha 0), or of one float a pixel, its red: contract 17's formats.
enum class Format : std::uint8_t { half4, float1 };
inline cuda::DeviceBuffer uploaded(const contracts::LinearImage& image,
                                   frame::Extent size, Format format) {
  const std::size_t pixels = pixels_of(image, size);
  if (format == Format::float1) {
    std::vector<float> texels(pixels);
    for (std::size_t p = 0; p < pixels; ++p) {
      texels[p] = image.rgb[p * 3];
    }
    cuda::DeviceBuffer buffer(texels.size() * sizeof(float));
    cuda::upload_now(buffer.address(), std::as_bytes(std::span(texels)),
                     "denoise fixture");
    return buffer;
  }
  std::vector<_Float16> texels(pixels * 4, _Float16{0});
  for (std::size_t p = 0; p < pixels; ++p) {
    for (std::size_t c = 0; c < 3; ++c) {
      texels[p * 4 + c] = static_cast<_Float16>(image.rgb[p * 3 + c]);
    }
  }
  cuda::DeviceBuffer buffer(texels.size() * sizeof(_Float16));
  cuda::upload_now(buffer.address(), std::as_bytes(std::span(texels)),
                   "denoise fixture");
  return buffer;
}

// `image`'s RGB into `target`, RGBA32Float pitched, the alpha 0.
inline void upload(const cuda::PitchedImage& target,
                   const contracts::LinearImage& image) {
  const frame::Extent size = target.size();
  const std::size_t pixels = pixels_of(image, size);
  std::vector<float> texels(pixels * 4, 0.0f);
  for (std::size_t p = 0; p < pixels; ++p) {
    for (std::size_t c = 0; c < 3; ++c) {
      texels[p * 4 + c] = image.rgb[p * 3 + c];
    }
  }
  CUDA_MEMCPY2D copy{};
  copy.srcMemoryType = CU_MEMORYTYPE_HOST;
  copy.srcHost = texels.data();
  copy.srcPitch = std::size_t{size.width} * 4 * sizeof(float);
  copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  copy.dstDevice = target.address();
  copy.dstPitch = target.pitch();
  copy.WidthInBytes = copy.srcPitch;
  copy.Height = size.height;
  cuda::check(cuMemcpy2D(&copy), "denoise fixture: cuMemcpy2D");
  // A copy on the legacy stream: waited for before the pass's stream reads
  // it (cuda/device/buffer.h, upload_now).
  cuda::check(cuCtxSynchronize(), "denoise fixture: cuCtxSynchronize");
}

// `image`'s RGB, read back.
inline contracts::LinearImage download(const cuda::PitchedImage& image) {
  const frame::Extent size = image.size();
  std::vector<float> texels(std::size_t{size.width} * size.height * 4);
  image.read(texels);
  contracts::LinearImage out;
  out.extent = size;
  out.rgb.resize(std::size_t{size.width} * size.height * 3);
  for (std::size_t p = 0; p < std::size_t{size.width} * size.height; ++p) {
    for (std::size_t c = 0; c < 3; ++c) {
      out.rgb[p * 3 + c] = texels[p * 4 + c];
    }
  }
  return out;
}

// Frame `k`'s constants: the camera's axes, the sizes, whether its
// history is kept, and its jitter.
inline contracts::DenoiseConstants constants_of(
    const DenoiseFixture& fixture, std::size_t k,
    const denoise::Network& network, const cuda::DenoiseImages& images,
    frame::Extent output, bool history) {
  contracts::DenoiseConstants c{};
  c.right = fixture.right;
  c.up = fixture.up;
  c.backward = fixture.backward;
  c.light_width = fixture.light.width;
  c.light_height = fixture.light.height;
  c.history = history ? 1u : 0u;
  c.padded_width = images.padded().width;
  c.padded_height = images.padded().height;
  c.output_width = output.width;
  c.output_height = output.height;
  c.features = network.widths.dec.back();
  c.jitter_x = fixture.jitters.at(k)[0];
  c.jitter_y = fixture.jitters.at(k)[1];
  return c;
}

}  // namespace detail

// Runs `fixture` through the denoise pass with its network, changed first
// by `change` if given (a test's mutation); each frame's output.
inline std::vector<contracts::LinearImage> run_denoise_fixture(
    const DenoiseFixture& fixture,
    const std::function<void(denoise::Network&)>& change = {}) {
  using detail::Format;
  denoise::Network network = denoise::load_network(fixture.dir / "network");
  if (change) {
    change(network);
  }
  cuda::Device device;
  cuda::Submission submission(device);
  const cuda::DenoisePass pass(network);
  cuda::DenoiseImages images(device, submission, network.widths);
  const frame::Extent light = fixture.light;
  const frame::Extent output{contracts::denoise_scale * light.width,
                             contracts::denoise_scale * light.height};
  const cuda::PitchedImage light_radiance(device, submission, light);
  const cuda::PitchedImage radiance(device, submission, output);

  std::vector<contracts::LinearImage> outputs;
  for (std::size_t k = 0; k < fixture.frames; ++k) {
    // Frame k of a history from frame 0 (reused_since 0, the default).
    frame::FrameInputs inputs{};
    inputs.index = k;
    const bool history = images.prepare(inputs, light);
    detail::upload(light_radiance,
                   output::read_pfm(fixture.file(k, "radiance")));
    const auto read = [&](std::string_view name, frame::Extent size,
                          Format format) {
      return detail::uploaded(output::read_pfm(fixture.file(k, name)), size,
                              format);
    };
    const cuda::DeviceBuffer albedo = read("albedo", light, Format::half4);
    const cuda::DeviceBuffer normal = read("normal", light, Format::half4);
    const cuda::DeviceBuffer depth = read("depth", light, Format::float1);
    const cuda::DeviceBuffer motion = read("motion", light, Format::half4);
    const cuda::DeviceBuffer full_albedo =
        read("full_albedo", output, Format::half4);
    const cuda::FrameSlot begun = submission.begin();
    pass.record(begun.stream,
                cuda::DenoiseInputs{
                    .constants = detail::constants_of(fixture, k, network,
                                                      images, output, history),
                    .images = &images,
                    .light_radiance = &light_radiance,
                    .albedo = albedo.address(),
                    .normal = normal.address(),
                    .depth = depth.address(),
                    .motion = motion.address(),
                    .full_albedo = full_albedo.address(),
                    .full_size = output,
                    .radiance = &radiance,
                });
    submission.commit();
    submission.wait_until_complete(begun.sequence);
    outputs.push_back(detail::download(radiance));
  }
  submission.finish();
  return outputs;
}

}  // namespace serenity::tests
