// The tone-map pass (metal/passes/tone_map/tone_map.h), on the GPU, against
// the core's steps 1 to 6 (core/frame/tone_map.h), worked out here on the
// CPU in doubles from the header's words: exposure and its ceiling, the
// pyramid's sizes and its 13-tap filter, the tent up, the composite, PBR
// Neutral and sRGB. The pass is given radiance written here, so every image
// is one chosen to show a property: a uniform field, a lone light, a light
// past the half-float range, frames as small as a pixel. And the pass in a
// renderer: the path tracer's graph, frames in flight, a resize.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "core/frame/graph_file.h"
#include "core/frame/tone_map.h"
#include "core/scene/scene.h"
#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/frame_images.h"
#include "metal/frame/renderer.h"
#include "metal/passes/tone_map/tone_map.h"
#include "serenity/metallib/shaders.h"

using namespace serenity;

namespace {

using Color = std::array<double, 3>;

// An image of linear colors, rows from the top.
struct Field {
    std::uint32_t width = 0, height = 0;
    std::vector<Color> texels;

    Color& at(std::uint32_t x, std::uint32_t y) { return texels[std::size_t{y} * width + x]; }
    const Color& at(std::uint32_t x, std::uint32_t y) const { return texels[std::size_t{y} * width + x]; }
};

Field field(std::uint32_t width, std::uint32_t height, Color fill) {
    return {width, height, std::vector<Color>(std::size_t{width} * height, fill)};
}

// ---- The core's steps, on the CPU ---------------------------------------

// Bilinear at normalized (u, v), edges clamped.
Color bilinear(const Field& f, double u, double v) {
    const double x = u * f.width - 0.5, y = v * f.height - 0.5;
    const double fx = std::floor(x), fy = std::floor(y);
    const double tx = x - fx, ty = y - fy;
    const auto texel = [&](long i, long j) {
        return f.at(std::uint32_t(std::clamp<long>(i, 0, f.width - 1)),
                    std::uint32_t(std::clamp<long>(j, 0, f.height - 1)));
    };
    const long i = long(fx), j = long(fy);
    Color out{};
    for (int c = 0; c < 3; ++c) {
        const double top = texel(i, j)[c] * (1 - tx) + texel(i + 1, j)[c] * tx;
        const double bottom = texel(i, j + 1)[c] * (1 - tx) + texel(i + 1, j + 1)[c] * tx;
        out[c] = top * (1 - ty) + bottom * ty;
    }
    return out;
}

// Step 2: five 2 x 2 boxes about (u, v), from 13 bilinear reads of
// `source` through `read`.
template <typename Read>
Color down13(Read read, double u, double v, const Field& source) {
    const double tx = 1.0 / source.width, ty = 1.0 / source.height;
    const auto s = [&](int dx, int dy) { return read(u + dx * tx, v + dy * ty); };
    const Color a = s(-2, -2), b = s(0, -2), c = s(2, -2), d = s(-2, 0), e = s(0, 0), f = s(2, 0);
    const Color g = s(-2, 2), h = s(0, 2), i = s(2, 2), j = s(-1, -1), k = s(1, -1), l = s(-1, 1), m = s(1, 1);
    Color out{};
    for (int n = 0; n < 3; ++n) {
        const double middle = 0.25 * (j[n] + k[n] + l[n] + m[n]);
        const double corners = 0.25 * ((a[n] + b[n] + d[n] + e[n]) + (b[n] + c[n] + e[n] + f[n]) +
                                       (d[n] + e[n] + g[n] + h[n]) + (e[n] + f[n] + h[n] + i[n]));
        out[n] = frame::down_middle * middle + frame::down_corner * corners;
    }
    return out;
}

// Step 3's tent about (u, v), a texel of `level` apart.
Color tent(const Field& level, double u, double v) {
    const double tx = 1.0 / level.width, ty = 1.0 / level.height;
    Color out{};
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const double w = (dx == 0 ? 2.0 : 1.0) * (dy == 0 ? 2.0 : 1.0) / 16.0;
            const Color s = bilinear(level, u + dx * tx, v + dy * ty);
            for (int c = 0; c < 3; ++c) {
                out[c] += w * s[c];
            }
        }
    }
    return out;
}

// Step 5.
Color neutral(Color c) {
    const double x = std::min({c[0], c[1], c[2]});
    const double offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    for (double& v : c) {
        v -= offset;
    }
    const double peak = std::max({c[0], c[1], c[2]});
    const double start = frame::neutral_start;
    if (peak < start) {
        return c;
    }
    const double d = 1.0 - start;
    const double rolled = 1.0 - d * d / (peak + d - start);
    const double g = 1.0 - 1.0 / (double(frame::neutral_desaturation) * (peak - rolled) + 1.0);
    for (double& v : c) {
        v = (v * rolled / peak) * (1.0 - g) + rolled * g;
    }
    return c;
}

int srgb8(double v) {
    v = std::clamp(v, 0.0, 1.0);
    const double e = v <= 0.0031308 ? 12.92 * v : 1.055 * std::pow(v, 1.0 / 2.4) - 0.055;
    return int(std::lround(e * 255.0));
}

using Bytes = std::vector<std::array<int, 3>>;

// Steps 1 to 6 over `radiance`.
Bytes reference(const Field& radiance, frame::ToneMap settings) {
    // Step 1: E, and each read of it clamped.
    const auto clamped = [](Color c) {
        for (double& v : c) {
            v = std::min(v, double(frame::bloom_ceiling));
        }
        return c;
    };
    Field exposed = radiance;
    for (Color& t : exposed.texels) {
        for (double& v : t) {
            v = std::exp2(double(settings.exposure)) * v;
        }
    }
    // Step 2.
    std::vector<Field> pyramid;
    const Field* source = &exposed;
    for (unsigned k = 0; k < frame::bloom_levels; ++k) {
        Field level = field(std::max(1u, (source->width + 1) / 2), std::max(1u, (source->height + 1) / 2), {});
        const auto read = [&](double u, double v) {
            const Color c = bilinear(*source, u, v);
            return k == 0 ? clamped(c) : c;
        };
        for (std::uint32_t y = 0; y < level.height; ++y) {
            for (std::uint32_t x = 0; x < level.width; ++x) {
                level.at(x, y) = down13(read, (x + 0.5) / level.width, (y + 0.5) / level.height, *source);
            }
        }
        pyramid.push_back(std::move(level));
        source = &pyramid.back();
    }
    // Step 3.
    for (int k = int(frame::bloom_levels) - 2; k >= 0; --k) {
        Field& level = pyramid[std::size_t(k)];
        const Field& below = pyramid[std::size_t(k) + 1];
        for (std::uint32_t y = 0; y < level.height; ++y) {
            for (std::uint32_t x = 0; x < level.width; ++x) {
                const Color t = tent(below, (x + 0.5) / level.width, (y + 0.5) / level.height);
                for (int c = 0; c < 3; ++c) {
                    level.at(x, y)[c] += t[c];
                }
            }
        }
    }
    // Steps 4 to 6.
    Bytes out;
    for (std::uint32_t y = 0; y < radiance.height; ++y) {
        for (std::uint32_t x = 0; x < radiance.width; ++x) {
            const Color glare = tent(pyramid[0], (x + 0.5) / radiance.width, (y + 0.5) / radiance.height);
            Color c{};
            for (int n = 0; n < 3; ++n) {
                c[n] = (1.0 - settings.bloom) * clamped(exposed.at(x, y))[n] +
                       settings.bloom * glare[n] / frame::bloom_levels;
            }
            const Color shown = neutral(c);
            out.push_back({srgb8(shown[0]), srgb8(shown[1]), srgb8(shown[2])});
        }
    }
    return out;
}

// ---- The pass, on the GPU -----------------------------------------------

// The tone-map pass alone over `radiance`, through the renderer's own pass
// type, into an 8-bit target.
Bytes tone_map(const Field& radiance, frame::ToneMap settings) {
    metal::Device device;
    metal::Submission submission(device);
    metal::Library library(device, metallib::shaders);
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    MTL::Device* mtl = device.handle();
    const frame::Extent size{radiance.width, radiance.height};

    auto descriptor = NS::TransferPtr(MTL::TextureDescriptor::alloc()->init());
    descriptor->setPixelFormat(MTL::PixelFormatRGBA32Float);
    descriptor->setWidth(size.width);
    descriptor->setHeight(size.height);
    descriptor->setStorageMode(MTL::StorageModeShared);
    descriptor->setUsage(MTL::TextureUsageShaderRead);
    auto image = NS::TransferPtr(mtl->newTexture(descriptor.get()));
    REQUIRE(image);
    std::vector<float> texels;
    for (const Color& t : radiance.texels) {
        texels.insert(texels.end(), {float(t[0]), float(t[1]), float(t[2]), 1.0f});
    }
    image->replaceRegion(MTL::Region(0, 0, size.width, size.height), 0, texels.data(), size.width * 16);
    submission.make_resident(image.get());

    metal::FrameImages pyramid(device, submission, false, true);
    pyramid.prepare(size);
    metal::Offscreen target(device, submission, size);
    metal::ToneMapPass pass(device, library, submission, settings);

    auto tables = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
    tables->setMaxBufferBindCount(1);
    tables->setMaxTextureBindCount(3);
    NS::Error* error = nullptr;
    auto table = NS::TransferPtr(mtl->newArgumentTable(tables.get(), &error));
    REQUIRE(table);

    metal::FrameResources resources;
    resources.arguments = table.get();
    resources.target = target.texture();
    resources.size = size;
    resources.radiance = image.get();
    for (std::uint32_t k = 0; k < frame::bloom_levels; ++k) {
        resources.bloom[k] = pyramid.bloom(k);
    }
    const metal::FrameSlot frame = submission.begin();
    MTL4::ComputeCommandEncoder* encoder = frame.commands->computeCommandEncoder();
    encoder->setArgumentTable(table.get());
    pass.record(encoder, resources);
    encoder->endEncoding();
    submission.commit();
    (void)submission.wait_until_complete(frame.sequence);

    std::vector<std::uint8_t> rgba(std::size_t{size.width} * size.height * 4);
    target.read_rgba(rgba);
    Bytes out;
    for (std::size_t i = 0; i < rgba.size(); i += 4) {
        out.push_back({rgba[i], rgba[i + 1], rgba[i + 2]});
    }
    return out;
}

// The largest difference between two images, in 8-bit levels.
int farthest(const Bytes& a, const Bytes& b) {
    REQUIRE(a.size() == b.size());
    int most = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (int c = 0; c < 3; ++c) {
            most = std::max(most, std::abs(a[i][c] - b[i][c]));
        }
    }
    return most;
}

// A scene in small: a dim gradient, a few fireflies of hundreds, and a
// brass-colored patch near the roll-off's start.
Field scene_like(std::uint32_t width, std::uint32_t height) {
    Field f = field(width, height, {});
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const double u = double(x) / width, v = double(y) / height;
            f.at(x, y) = {0.02 + 0.1 * u, 0.03 + 0.05 * v, 0.06 + 0.02 * u * v};
        }
    }
    for (std::uint32_t y = height / 3; y < height / 2; ++y) {
        for (std::uint32_t x = width / 2; x < 3 * width / 4; ++x) {
            f.at(x, y) = {0.9, 0.62, 0.25};
        }
    }
    f.at(width / 5, height / 4) = {300.0, 240.0, 60.0};
    f.at(4 * width / 5, 3 * height / 4) = {75.0, 60.0, 15.0};
    f.at(width - 1, 0) = {120.0, 96.0, 24.0};  // at an edge
    return f;
}

constexpr frame::ToneMap settings(float exposure, float bloom) {
    return frame::ToneMap{exposure, bloom, {0.0f, 0.0f}};
}

}  // namespace

TEST_CASE("the tone map computes the core's steps, at even and odd sizes") {
    // Within 2 levels of 255: the pyramid is stored as half floats and the
    // hardware's bilinear weights are fixed point, where the reference has
    // doubles.
    for (const auto [width, height] : {std::array<std::uint32_t, 2>{64, 32}, {61, 37}, {200, 113}}) {
        const Field radiance = scene_like(width, height);
        for (const frame::ToneMap s : {settings(0.0f, 0.04f), settings(1.5f, 0.3f), settings(-2.0f, 0.0f)}) {
            INFO(width << " x " << height << ", exposure " << s.exposure << ", bloom " << s.bloom);
            CHECK(farthest(tone_map(radiance, s), reference(radiance, s)) <= 2);
        }
    }
    // The check can see the bloom: with it and without, the images differ
    // far more than the tolerance.
    const Field radiance = scene_like(64, 32);
    CHECK(farthest(reference(radiance, settings(0.0f, 0.3f)), reference(radiance, settings(0.0f, 0.0f))) > 20);
}

TEST_CASE("a uniform image stays uniform: bloom moves light and makes none") {
    for (const auto [width, height] : {std::array<std::uint32_t, 2>{50, 30}, {33, 65}}) {
        const Color c{0.3, 0.2, 0.1};
        const Bytes out = tone_map(field(width, height, c), settings(0.0f, 0.5f));
        const Color shown = neutral(c);
        const std::array<int, 3> expected{srgb8(shown[0]), srgb8(shown[1]), srgb8(shown[2])};
        INFO(width << " x " << height);
        CHECK(farthest(out, Bytes(out.size(), expected)) <= 1);
        CHECK(farthest(out, Bytes(out.size(), out.front())) == 0);
    }
}

TEST_CASE("frames as small as a pixel have every level") {
    for (const auto [width, height] :
         {std::array<std::uint32_t, 2>{1, 1}, {2, 1}, {1, 3}, {5, 3}, {33, 1}}) {
        INFO(width << " x " << height);
        const Field radiance = scene_like(width, height);
        CHECK(farthest(tone_map(radiance, settings(0.0f, 0.2f)), reference(radiance, settings(0.0f, 0.2f))) <= 2);
    }
}

TEST_CASE("light past the half-float range glares as if at the ceiling, and nothing turns to NaN") {
    Field radiance = field(48, 48, {0.01, 0.01, 0.01});
    radiance.at(24, 24) = {1e30, 1e30, 1e30};
    radiance.at(10, 30) = {3e38, 0.0, 0.0};
    for (const frame::ToneMap s : {settings(10.0f, 0.04f), settings(0.0f, 0.5f)}) {
        const Bytes out = tone_map(radiance, s);
        INFO("exposure " << s.exposure << ", bloom " << s.bloom);
        CHECK(farthest(out, reference(radiance, s)) <= 2);
        CHECK(out[24 * 48 + 24] == std::array<int, 3>{255, 255, 255});
    }
}

TEST_CASE("a lone firefly glares: a halo falling off with distance, its color kept, its core white") {
    Field radiance = field(129, 129, {0.0, 0.0, 0.0});
    radiance.at(64, 64) = {100.0, 80.0, 20.0};

    // Without bloom, the light stays in its pixel.
    const Bytes sharp = tone_map(radiance, settings(0.0f, 0.0f));
    int lit = 0;
    for (const auto& p : sharp) {
        lit += p != std::array<int, 3>{0, 0, 0};
    }
    CHECK(lit == 1);

    const Bytes glare = tone_map(radiance, settings(0.0f, 0.04f));
    const auto at = [&](int x, int y) { return glare[std::size_t(y) * 129 + std::size_t(x)]; };
    // White-hot at its core.
    for (int c = 0; c < 3; ++c) {
        CHECK(at(64, 64)[c] >= 240);
    }
    // Falling off, never rising, along a row, and reaching far: B_5's texels
    // are 64 frame pixels apart.
    for (int d = 1; d < 60; ++d) {
        INFO("distance " << d);
        CHECK(at(64 + d, 64)[0] <= at(64 + d - 1, 64)[0]);
    }
    CHECK(at(64 + 8, 64)[0] > 0);
    CHECK(at(64 + 24, 64)[0] > 0);
    // The glare keeps the firefly's yellow.
    const auto near = at(64 + 4, 64);
    CHECK(near[0] > near[1]);
    CHECK(near[1] > near[2]);
    // Round: the same at the same distance in each direction.
    CHECK(at(64 + 6, 64) == at(64, 64 + 6));
    CHECK(at(64 - 6, 64) == at(64, 64 - 6));
}

TEST_CASE("the path tracer's graph runs through the tone map, frames in flight, across a resize") {
    // Each frame, rendered back to back with the frame before still in
    // flight, is the frame rendered alone: the radiance image they share is
    // never written while the frame before still reads it.
    const scene::SceneDescription scene = scene::load(SERENITY_SCENES_DIR "/brass_sphere_flight.toml");
    const frame::Schedule graph =
        frame::parse_schedule("passes = [\"preview\", \"tone_map\"]\n[tone_map]\nexposure = 0.5\nbloom = 0.1\n", "t");
    const frame::Extent size{480, 270};
    const auto inputs = [&](std::uint64_t i) {
        // One instant per image: the scene moves (core/frame/history.h).
        return frame::FrameInputs{
            .time = frame::Seconds(0.1 * i), .index = i, .accumulated_since = i, .camera = scene.camera};
    };
    const auto read = [&](metal::Offscreen& target) {
        std::vector<std::uint8_t> rgba(std::size_t{size.width} * size.height * 4);
        target.read_rgba(rgba);
        return rgba;
    };

    std::vector<std::vector<std::uint8_t>> alone;
    for (std::uint64_t i = 0; i < 4; ++i) {
        metal::Device device;
        metal::Submission submission(device);
        metal::Offscreen target(device, submission, size);
        metal::Renderer renderer(device, submission, graph, &scene);
        (void)submission.wait_until_complete(metal::render_to_offscreen(submission, target, renderer, inputs(i)));
        alone.push_back(read(target));
    }

    metal::Device device;
    metal::Submission submission(device);
    metal::Renderer renderer(device, submission, graph, &scene);
    std::vector<std::unique_ptr<metal::Offscreen>> targets;
    std::vector<std::uint64_t> sequences;
    for (std::uint64_t i = 0; i < 4; ++i) {
        targets.push_back(std::make_unique<metal::Offscreen>(device, submission, size));
        sequences.push_back(metal::render_to_offscreen(submission, *targets.back(), renderer, inputs(i)));
    }
    (void)submission.wait_until_complete(sequences.back());
    for (std::uint64_t i = 0; i < 4; ++i) {
        INFO("frame " << i);
        CHECK(read(*targets[i]) == alone[i]);
    }

    // A resize remakes the images, and the next frame is whole.
    metal::Offscreen small(device, submission, {97, 61});
    (void)submission.wait_until_complete(metal::render_to_offscreen(submission, small, renderer, inputs(0)));
    std::vector<std::uint8_t> rgba(97 * 61 * 4);
    small.read_rgba(rgba);
    CHECK(std::any_of(rgba.begin(), rgba.end(), [](std::uint8_t b) { return b > 0; }));

    // And the path tracer's own graph, in graphs/, runs.
    const frame::Schedule path = frame::load_schedule(SERENITY_GRAPHS_DIR "/path.toml");
    metal::Renderer tracer(device, submission, path, &scene);
    metal::Offscreen shown(device, submission, size);
    std::uint64_t last = 0;
    for (std::uint64_t i = 0; i < 3; ++i) {
        last = metal::render_to_offscreen(submission, shown, tracer, inputs(i));
    }
    (void)submission.wait_until_complete(last);
    CHECK(renderer.non_finite_samples() == 0);
    CHECK(tracer.non_finite_samples() == 0);
}
