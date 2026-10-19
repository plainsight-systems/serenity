#pragma once

// A denoiser fixture (training/denoise_check.py, "fixture") run through the
// denoise pass on the GPU, as the renderer runs it: the renderer's own
// images at the light size and the full-size albedo, uploaded into textures
// of the formats the renderer's are (metal/frame/feature_images.h; the
// accumulated image RGBA32Float), the pass recorded frame after frame with
// its own history (metal/frame/denoise_images.h, joined to the history as
// frames 0, 1, ...), and each frame's radiance image read back. For the
// match against PyTorch: tests/gpu/denoise_test.cpp on the committed
// fixture, and tests/gpu/denoise_check.cpp on a real network's frames.

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "core/contracts/denoise.h"
#include "core/contracts/linear_image.h"
#include "core/denoise/json.h"
#include "core/denoise/network.h"
#include "core/frame/frame_inputs.h"
#include "core/output/pfm.h"
#include "gpu/support/shared_resources.h"
#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/submission.h"
#include "metal/device/support.h"
#include "metal/frame/denoise_images.h"
#include "metal/frame/frame_resources.h"
#include "metal/passes/bindings.h"
#include "metal/passes/denoise/denoise.h"
#include "serenity/metallib/shaders.h"

namespace serenity::tests {

struct DenoiseFixture {
  std::filesystem::path dir;
  frame::Extent light;
  std::size_t frames = 0;
  contracts::Float3 right{};
  contracts::Float3 up{};
  contracts::Float3 backward{};
  // Each frame's camera jitter (x, y), in light pixels (contract 18).
  std::vector<std::array<float, 2>> jitters;

  std::filesystem::path file(std::size_t k, std::string_view name) const {
    return dir /
           ("frame-" + std::to_string(k) + "-" + std::string(name) + ".pfm");
  }
};

inline DenoiseFixture read_denoise_fixture(const std::filesystem::path& dir) {
  std::ifstream file{dir / "fixture.json", std::ios::binary};
  if (!file) {
    throw std::runtime_error("cannot read " + (dir / "fixture.json").string());
  }
  const std::string text{std::istreambuf_iterator<char>(file),
                         std::istreambuf_iterator<char>()};
  const denoise::Json doc = denoise::parse_json(text);
  const auto count = [](const denoise::Json& node) {
    return static_cast<std::uint32_t>(node.number("a count"));
  };
  const auto vector = [](const denoise::Json& node) {
    const denoise::JsonArray& a = node.array("an axis");
    return contracts::Float3{static_cast<float>(a.at(0).number("x")),
                             static_cast<float>(a.at(1).number("y")),
                             static_cast<float>(a.at(2).number("z"))};
  };
  DenoiseFixture f;
  f.dir = dir;
  const denoise::JsonArray& light = doc.at("light").array("light");
  f.light = {count(light.at(0)), count(light.at(1))};
  f.frames = count(doc.at("frames"));
  f.right = vector(doc.at("axes").at("right"));
  f.up = vector(doc.at("axes").at("up"));
  f.backward = vector(doc.at("axes").at("backward"));
  const denoise::JsonArray& jitters = doc.at("jitter").array("jitter");
  if (jitters.size() != f.frames) {
    throw std::runtime_error(
        "denoise fixture: a jitter a frame, and fixture.json has another "
        "count");
  }
  for (const denoise::Json& j : jitters) {
    const denoise::JsonArray& xy = j.array("a jitter");
    f.jitters.push_back({static_cast<float>(xy.at(0).number("x")),
                         static_cast<float>(xy.at(1).number("y"))});
  }
  return f;
}

namespace detail {

using metal::Half;

// `image`'s RGB into `texture` (RGBA32Float, RGBA16Float or R32Float).
inline void upload(MTL::Texture* texture, const contracts::LinearImage& image) {
  const frame::Extent size = image.extent;
  if (texture->width() != size.width || texture->height() != size.height) {
    throw std::runtime_error(
        "denoise fixture: an image of another size than its texture");
  }
  const std::size_t pixels = std::size_t{size.width} * size.height;
  const MTL::Region region = MTL::Region::Make2D(0, 0, size.width, size.height);
  switch (texture->pixelFormat()) {
    case MTL::PixelFormatRGBA32Float: {
      std::vector<float> texels(pixels * 4, 0.0f);
      for (std::size_t p = 0; p < pixels; ++p) {
        std::memcpy(&texels[p * 4], &image.rgb[p * 3], 3 * sizeof(float));
      }
      texture->replaceRegion(region, 0, texels.data(),
                             size.width * 4 * sizeof(float));
      return;
    }
    case MTL::PixelFormatRGBA16Float: {
      std::vector<Half> texels(pixels * 4, Half{0});
      for (std::size_t p = 0; p < pixels; ++p) {
        for (std::size_t c = 0; c < 3; ++c) {
          texels[p * 4 + c] = static_cast<Half>(image.rgb[p * 3 + c]);
        }
      }
      texture->replaceRegion(region, 0, texels.data(),
                             size.width * 4 * sizeof(Half));
      return;
    }
    case MTL::PixelFormatR32Float: {
      std::vector<float> texels(pixels);
      for (std::size_t p = 0; p < pixels; ++p) {
        texels[p] = image.rgb[p * 3];
      }
      texture->replaceRegion(region, 0, texels.data(),
                             size.width * sizeof(float));
      return;
    }
    default:
      throw std::runtime_error(
          "denoise fixture: a texture format it does not upload");
  }
}

inline contracts::LinearImage download(MTL::Texture* texture) {
  const frame::Extent size{static_cast<std::uint32_t>(texture->width()),
                           static_cast<std::uint32_t>(texture->height())};
  std::vector<float> texels(std::size_t{size.width} * size.height * 4);
  texture->getBytes(texels.data(), size.width * 4 * sizeof(float),
                    MTL::Region::Make2D(0, 0, size.width, size.height), 0);
  return contracts::from_rgba(size, texels);
}

// The pass's images, of the formats the renderer's are, at the light size
// and the output's, resident while this lives; and its constants.
struct FixtureImages {
  NS::SharedPtr<MTL::Texture> radiance_in;
  NS::SharedPtr<MTL::Texture> albedo;
  NS::SharedPtr<MTL::Texture> normal;
  NS::SharedPtr<MTL::Texture> depth;
  NS::SharedPtr<MTL::Texture> motion;
  NS::SharedPtr<MTL::Texture> full_albedo;
  NS::SharedPtr<MTL::Texture> radiance_out;
  NS::SharedPtr<MTL::Buffer> constants;
  std::vector<metal::Resident> resident;
};

inline FixtureImages make_images(const metal::Device& device,
                                 metal::Submission& submission,
                                 frame::Extent light, frame::Extent output) {
  constexpr MTL::TextureUsage usage =
      MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite;
  FixtureImages m;
  m.radiance_in =
      shared_texture(device, MTL::PixelFormatRGBA32Float, light, usage);
  m.albedo = shared_texture(device, MTL::PixelFormatRGBA16Float, light, usage);
  m.normal = shared_texture(device, MTL::PixelFormatRGBA16Float, light, usage);
  m.depth = shared_texture(device, MTL::PixelFormatR32Float, light, usage);
  m.motion = shared_texture(device, MTL::PixelFormatRGBA16Float, light, usage);
  m.full_albedo =
      shared_texture(device, MTL::PixelFormatRGBA16Float, output, usage);
  m.radiance_out =
      shared_texture(device, MTL::PixelFormatRGBA32Float, output, usage);
  m.constants = shared_buffer(device, submission, {},
                              sizeof(contracts::DenoiseConstants));
  // Textures and buffers are allocations (metal-cpp's MTLResource.hpp).
  const std::array<MTL::Allocation*, 7> textures{
      m.radiance_in.get(), m.albedo.get(), m.normal.get(),
      m.depth.get(),       m.motion.get(), m.full_albedo.get(),
      m.radiance_out.get()};
  for (MTL::Allocation* texture : textures) {
    m.resident.push_back(submission.keep_resident(texture));
  }
  return m;
}

// Frame k's images, from the fixture's files.
inline void upload_frame(const DenoiseFixture& fixture, std::size_t k,
                         const FixtureImages& m) {
  upload(m.radiance_in.get(), output::read_pfm(fixture.file(k, "radiance")));
  upload(m.albedo.get(), output::read_pfm(fixture.file(k, "albedo")));
  upload(m.normal.get(), output::read_pfm(fixture.file(k, "normal")));
  upload(m.depth.get(), output::read_pfm(fixture.file(k, "depth")));
  upload(m.motion.get(), output::read_pfm(fixture.file(k, "motion")));
  upload(m.full_albedo.get(), output::read_pfm(fixture.file(k, "full_albedo")));
}

// Frame `k`'s constants: the camera's axes, the sizes, whether its
// history is kept, and its jitter.
inline contracts::DenoiseConstants constants_of(
    const DenoiseFixture& fixture, std::size_t k,
    const denoise::Network& network, const metal::DenoiseImages& images,
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

// Records the pass over `m` into a command buffer of its own, commits it
// and waits.
inline void record_frame(metal::Submission& submission,
                         MTL4::ArgumentTable* table,
                         const metal::DenoisePass& pass, const FixtureImages& m,
                         const metal::DenoiseImages& images,
                         frame::Extent output) {
  const auto pool = metal::scoped_pool();
  const metal::FrameSlot begun = submission.begin();
  MTL4::ComputeCommandEncoder* encoder =
      begun.commands->computeCommandEncoder();
  encoder->setArgumentTable(table);
  metal::FrameResources resources{};
  resources.arguments = table;
  resources.size = output;
  resources.radiance = m.radiance_out.get();
  resources.albedo = m.albedo.get();
  resources.normal = m.normal.get();
  resources.depth = m.depth.get();
  resources.motion = m.motion.get();
  resources.light_radiance = m.radiance_in.get();
  resources.full_albedo = m.full_albedo.get();
  resources.denoise_constants = m.constants->gpuAddress();
  resources.denoise = &images;
  pass.record(encoder, resources);
  encoder->endEncoding();
  submission.commit();
  (void)submission.wait_until_complete(begun.sequence);
}

}  // namespace detail

// Runs `fixture` through the denoise pass with its network, changed first
// by `change` if given (a test's mutation); each frame's output.
inline std::vector<contracts::LinearImage> run_denoise_fixture(
    const DenoiseFixture& fixture,
    const std::function<void(denoise::Network&)>& change = {}) {
  denoise::Network network = denoise::load_network(fixture.dir / "network");
  if (change) {
    change(network);
  }
  metal::Device device;
  metal::Submission submission(device);
  metal::Library library(device, metallib::shaders);
  metal::DenoisePass pass(device, library, submission, network);
  metal::DenoiseImages images(device, submission, network.widths);
  const frame::Extent light = fixture.light;
  const frame::Extent output{contracts::denoise_scale * light.width,
                             contracts::denoise_scale * light.height};
  const detail::FixtureImages made =
      detail::make_images(device, submission, light, output);

  auto descriptor =
      NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
  // Every binding of the pass's kernels (passes/bindings.h, denoise).
  descriptor->setMaxBufferBindCount(bindings::denoise::coarse + 1);
  descriptor->setMaxTextureBindCount(bindings::denoise::radiance + 1);
  NS::Error* error = nullptr;
  auto table = NS::TransferPtr(
      device.handle()->newArgumentTable(descriptor.get(), &error));
  if (!table) {
    throw std::runtime_error("denoise fixture: no argument table");
  }

  std::vector<contracts::LinearImage> outputs;
  for (std::size_t k = 0; k < fixture.frames; ++k) {
    detail::upload_frame(fixture, k, made);
    // Frame k of a history from frame 0 (reused_since 0, the default).
    frame::FrameInputs inputs{};
    inputs.index = k;
    const bool history = images.prepare(inputs, light);
    const contracts::DenoiseConstants c =
        detail::constants_of(fixture, k, network, images, output, history);
    std::memcpy(made.constants->contents(), &c, sizeof(c));
    detail::record_frame(submission, table.get(), pass, made, images, output);
    outputs.push_back(detail::download(made.radiance_out.get()));
  }
  submission.finish();
  return outputs;
}

}  // namespace serenity::tests
