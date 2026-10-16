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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/contracts/denoise.h"
#include "core/contracts/linear_image.h"
#include "core/denoise/json.h"
#include "core/denoise/network.h"
#include "core/frame/frame_inputs.h"
#include "core/output/pfm.h"
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

    std::filesystem::path file(std::size_t k, const char* name) const {
        return dir / ("frame-" + std::to_string(k) + "-" + name + ".pfm");
    }
};

inline DenoiseFixture read_denoise_fixture(const std::filesystem::path& dir) {
    std::ifstream file{dir / "fixture.json", std::ios::binary};
    if (!file) {
        throw std::runtime_error("cannot read " + (dir / "fixture.json").string());
    }
    const std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    const denoise::Json doc = denoise::parse_json(text);
    const auto count = [](const denoise::Json& node) { return static_cast<std::uint32_t>(node.number("a count")); };
    const auto vector = [](const denoise::Json& node) {
        const denoise::JsonArray& a = node.array("an axis");
        return contracts::Float3{static_cast<float>(a.at(0).number("x")), static_cast<float>(a.at(1).number("y")),
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
    return f;
}

namespace detail {

using metal::Half;

inline NS::SharedPtr<MTL::Texture> shared_texture(MTL::Device* device, MTL::PixelFormat format, frame::Extent size) {
    auto descriptor = NS::TransferPtr(MTL::TextureDescriptor::alloc()->init());
    descriptor->setTextureType(MTL::TextureType2D);
    descriptor->setPixelFormat(format);
    descriptor->setWidth(size.width);
    descriptor->setHeight(size.height);
    descriptor->setStorageMode(MTL::StorageModeShared);
    descriptor->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    auto texture = NS::TransferPtr(device->newTexture(descriptor.get()));
    if (!texture) {
        throw std::runtime_error("denoise fixture: the device made no texture");
    }
    return texture;
}

// `image`'s RGB into `texture` (RGBA32Float, RGBA16Float or R32Float).
inline void upload(MTL::Texture* texture, const contracts::LinearImage& image) {
    const frame::Extent size = image.extent;
    if (texture->width() != size.width || texture->height() != size.height) {
        throw std::runtime_error("denoise fixture: an image of another size than its texture");
    }
    const std::size_t pixels = std::size_t{size.width} * size.height;
    const MTL::Region region = MTL::Region::Make2D(0, 0, size.width, size.height);
    switch (texture->pixelFormat()) {
    case MTL::PixelFormatRGBA32Float: {
        std::vector<float> texels(pixels * 4, 0.0f);
        for (std::size_t p = 0; p < pixels; ++p) {
            std::memcpy(&texels[p * 4], &image.rgb[p * 3], 3 * sizeof(float));
        }
        texture->replaceRegion(region, 0, texels.data(), size.width * 4 * sizeof(float));
        return;
    }
    case MTL::PixelFormatRGBA16Float: {
        std::vector<Half> texels(pixels * 4, Half{0});
        for (std::size_t p = 0; p < pixels; ++p) {
            for (std::size_t c = 0; c < 3; ++c) {
                texels[p * 4 + c] = static_cast<Half>(image.rgb[p * 3 + c]);
            }
        }
        texture->replaceRegion(region, 0, texels.data(), size.width * 4 * sizeof(Half));
        return;
    }
    case MTL::PixelFormatR32Float: {
        std::vector<float> texels(pixels);
        for (std::size_t p = 0; p < pixels; ++p) {
            texels[p] = image.rgb[p * 3];
        }
        texture->replaceRegion(region, 0, texels.data(), size.width * sizeof(float));
        return;
    }
    default:
        throw std::runtime_error("denoise fixture: a texture format it does not upload");
    }
}

inline contracts::LinearImage download(MTL::Texture* texture) {
    const frame::Extent size{static_cast<std::uint32_t>(texture->width()),
                             static_cast<std::uint32_t>(texture->height())};
    std::vector<float> texels(std::size_t{size.width} * size.height * 4);
    texture->getBytes(texels.data(), size.width * 4 * sizeof(float), MTL::Region::Make2D(0, 0, size.width, size.height),
                      0);
    return contracts::from_rgba(size, texels);
}

}  // namespace detail

// Runs `fixture` through the denoise pass with its network, changed first
// by `change` if given (a test's mutation); each frame's output.
inline std::vector<contracts::LinearImage> run_denoise_fixture(
    const DenoiseFixture& fixture, const std::function<void(denoise::Network&)>& change = {}) {
    denoise::Network network = denoise::load_network(fixture.dir / "network");
    if (change) {
        change(network);
    }
    metal::Device device;
    metal::Submission submission(device);
    metal::Library library(device, metallib::shaders);
    metal::DenoisePass pass(device, library, submission, network);
    metal::DenoiseImages images(device, submission, network.widths);
    MTL::Device* mtl = device.handle();
    const frame::Extent light = fixture.light;
    const frame::Extent output{contracts::denoise_scale * light.width, contracts::denoise_scale * light.height};

    using detail::shared_texture;
    const auto radiance_in = shared_texture(mtl, MTL::PixelFormatRGBA32Float, light);
    const auto albedo = shared_texture(mtl, MTL::PixelFormatRGBA16Float, light);
    const auto normal = shared_texture(mtl, MTL::PixelFormatRGBA16Float, light);
    const auto depth = shared_texture(mtl, MTL::PixelFormatR32Float, light);
    const auto motion = shared_texture(mtl, MTL::PixelFormatRGBA16Float, light);
    const auto full_albedo = shared_texture(mtl, MTL::PixelFormatRGBA16Float, output);
    const auto radiance_out = shared_texture(mtl, MTL::PixelFormatRGBA32Float, output);
    auto constants = NS::TransferPtr(mtl->newBuffer(sizeof(contracts::DenoiseConstants), MTL::ResourceStorageModeShared));
    std::vector<metal::Resident> resident;
    for (MTL::Allocation* a : {static_cast<MTL::Allocation*>(radiance_in.get()), static_cast<MTL::Allocation*>(albedo.get()),
                               static_cast<MTL::Allocation*>(normal.get()), static_cast<MTL::Allocation*>(depth.get()),
                               static_cast<MTL::Allocation*>(motion.get()), static_cast<MTL::Allocation*>(full_albedo.get()),
                               static_cast<MTL::Allocation*>(radiance_out.get()),
                               static_cast<MTL::Allocation*>(constants.get())}) {
        resident.push_back(submission.keep_resident(a));
    }
    auto descriptor = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
    // Every binding of the pass's kernels (passes/bindings.h, denoise).
    descriptor->setMaxBufferBindCount(bindings::denoise::coarse + 1);
    descriptor->setMaxTextureBindCount(bindings::denoise::radiance + 1);
    NS::Error* error = nullptr;
    auto table = NS::TransferPtr(mtl->newArgumentTable(descriptor.get(), &error));
    if (!table) {
        throw std::runtime_error("denoise fixture: no argument table");
    }

    std::vector<contracts::LinearImage> outputs;
    for (std::size_t k = 0; k < fixture.frames; ++k) {
        detail::upload(radiance_in.get(), output::read_pfm(fixture.file(k, "radiance")));
        detail::upload(albedo.get(), output::read_pfm(fixture.file(k, "albedo")));
        detail::upload(normal.get(), output::read_pfm(fixture.file(k, "normal")));
        detail::upload(depth.get(), output::read_pfm(fixture.file(k, "depth")));
        detail::upload(motion.get(), output::read_pfm(fixture.file(k, "motion")));
        detail::upload(full_albedo.get(), output::read_pfm(fixture.file(k, "full_albedo")));

        frame::FrameInputs inputs{};
        inputs.index = k;
        inputs.reused_since = 0;
        const bool history = images.prepare(inputs, light);
        contracts::DenoiseConstants c{};
        c.right = fixture.right;
        c.up = fixture.up;
        c.backward = fixture.backward;
        c.light_width = light.width;
        c.light_height = light.height;
        c.history = history ? 1u : 0u;
        c.padded_width = images.padded().width;
        c.padded_height = images.padded().height;
        c.output_width = output.width;
        c.output_height = output.height;
        c.features = network.widths.dec[3];
        std::memcpy(constants->contents(), &c, sizeof(c));

        const auto pool = metal::scoped_pool();
        const metal::FrameSlot begun = submission.begin();
        MTL4::ComputeCommandEncoder* encoder = begun.commands->computeCommandEncoder();
        encoder->setArgumentTable(table.get());
        metal::FrameResources resources{};
        resources.arguments = table.get();
        resources.size = output;
        resources.radiance = radiance_out.get();
        resources.albedo = albedo.get();
        resources.normal = normal.get();
        resources.depth = depth.get();
        resources.motion = motion.get();
        resources.light_radiance = radiance_in.get();
        resources.full_albedo = full_albedo.get();
        resources.denoise_constants = constants->gpuAddress();
        resources.denoise = &images;
        pass.record(encoder, resources);
        encoder->endEncoding();
        submission.commit();
        (void)submission.wait_until_complete(begun.sequence);
        outputs.push_back(detail::download(radiance_out.get()));
    }
    submission.finish();
    return outputs;
}

}  // namespace serenity::tests
