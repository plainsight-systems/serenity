// The tone-map pass (metal/passes/tone_map/tone_map.h), on the GPU, against
// the core's steps 1 to 6 (core/passes/tone_map.h), worked out here on the
// CPU in doubles from the header's words: exposure and its ceiling, the
// pyramid's sizes and its 13-tap filter, the tent up, the composite, PBR
// Neutral and sRGB. The constants are written out here as the header's words
// give them, not read from it, so a wrong constant there does not match
// itself here. The pass is given radiance written here, so every image is
// one chosen to show a property: a uniform field, a lone light, light past
// the half-float range, alone and over a broad field, frames as small as a
// pixel. The pass among others, in a renderer, is the frame graph's
// (frame_images_test.cpp).

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

#include <doctest/doctest.h>

#include "core/passes/tone_map.h"
#include "gpu/support/probe_runner.h"
#include "gpu/support/rendering.h"
#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/frame_images.h"
#include "metal/frame/frame_resources.h"
#include "metal/passes/tone_map/tone_map.h"
#include "serenity/metallib/shaders.h"
#include "support/references.h"

using namespace serenity;

namespace {

// ---- The core's constants, from its header's words ------------------------

constexpr unsigned levels = 6;             // the pyramid's levels
constexpr double ceiling = 65504.0;        // step 1: the largest half float
constexpr double middle_weight = 0.5;      // step 2: the middle box's weight,
constexpr double corner_weight = 0.125;    // and each corner box's
constexpr double neutral_start = 0.76;     // step 5: 0.8 - F90, F90 = 0.04
constexpr double neutral_desaturation = 0.15;
constexpr double f90 = 0.04;

// A half float, the pyramid's storage: _Float16, a compiler extension
// (P.2), until the toolchain's C++ has std::float16_t (C++23).
using Half = _Float16;

using Color = std::array<double, 3>;

// An image of linear colors, rows from the top.
struct Field {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<Color> texels;

    Color& at(std::uint32_t x, std::uint32_t y) { return texels[std::size_t{y} * width + x]; }
    const Color& at(std::uint32_t x, std::uint32_t y) const { return texels[std::size_t{y} * width + x]; }
};

Field field(std::uint32_t width, std::uint32_t height, Color fill) {
    return {width, height, std::vector<Color>(std::size_t{width} * height, fill)};
}

// The size of the level below one of `width` x `height`: half, rounded up,
// at least 1 (step 2).
frame::Extent half_size(frame::Extent size) {
    return {std::max(1u, (size.width + 1) / 2), std::max(1u, (size.height + 1) / 2)};
}

// ---- The core's steps, on the CPU ---------------------------------------

// `v` stored as a half float, as the pyramid stores it: rounded to nearest,
// where the M3 Max rounds toward zero (core/passes/tone_map.h, step 3), a
// part in two thousand at most, within the comparisons' tolerance.
double half(double v) {
    return static_cast<double>(static_cast<Half>(v));
}

// Bilinear at normalized (u, v), edges clamped.
Color bilinear(const Field& f, double u, double v) {
    const double x = u * f.width - 0.5;
    const double y = v * f.height - 0.5;
    const double fx = std::floor(x);
    const double fy = std::floor(y);
    const double tx = x - fx;
    const double ty = y - fy;
    const auto texel = [&](long i, long j) {
        return f.at(static_cast<std::uint32_t>(std::clamp<long>(i, 0, static_cast<long>(f.width) - 1)),
                    static_cast<std::uint32_t>(std::clamp<long>(j, 0, static_cast<long>(f.height) - 1)));
    };
    const auto i = static_cast<long>(fx);
    const auto j = static_cast<long>(fy);
    Color out{};
    for (std::size_t c = 0; c < 3; ++c) {
        const double top = texel(i, j)[c] * (1 - tx) + texel(i + 1, j)[c] * tx;
        const double bottom = texel(i, j + 1)[c] * (1 - tx) + texel(i + 1, j + 1)[c] * tx;
        out[c] = top * (1 - ty) + bottom * ty;
    }
    return out;
}

// Step 2: five 2 x 2 boxes about (u, v), from 13 bilinear reads of `source`
// through `read`.
template <std::invocable<double, double> Read>
Color down13(Read read, double u, double v, const Field& source) {
    const double tx = 1.0 / source.width;
    const double ty = 1.0 / source.height;
    const auto s = [&](int dx, int dy) { return read(u + dx * tx, v + dy * ty); };
    // The 3 x 3 grid two texels apart, and the four between its middle and
    // its corners.
    const Color a = s(-2, -2);
    const Color b = s(0, -2);
    const Color c = s(2, -2);
    const Color d = s(-2, 0);
    const Color e = s(0, 0);
    const Color f = s(2, 0);
    const Color g = s(-2, 2);
    const Color h = s(0, 2);
    const Color i = s(2, 2);
    const Color j = s(-1, -1);
    const Color k = s(1, -1);
    const Color l = s(-1, 1);
    const Color m = s(1, 1);
    Color out{};
    for (std::size_t n = 0; n < 3; ++n) {
        const double middle = 0.25 * (j[n] + k[n] + l[n] + m[n]);
        const double corners = 0.25 * ((a[n] + b[n] + d[n] + e[n]) + (b[n] + c[n] + e[n] + f[n]) +
                                       (d[n] + e[n] + g[n] + h[n]) + (e[n] + f[n] + h[n] + i[n]));
        out[n] = middle_weight * middle + corner_weight * corners;
    }
    return out;
}

// Step 3's tent about (u, v), a texel of `level` apart.
Color tent(const Field& level, double u, double v) {
    const double tx = 1.0 / level.width;
    const double ty = 1.0 / level.height;
    Color out{};
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const double w = (dx == 0 ? 2.0 : 1.0) * (dy == 0 ? 2.0 : 1.0) / 16.0;
            const Color s = bilinear(level, u + dx * tx, v + dy * ty);
            for (std::size_t c = 0; c < 3; ++c) {
                out[c] += w * s[c];
            }
        }
    }
    return out;
}

// Step 5: PBR Neutral.
Color neutral(Color c) {
    const double x = std::min({c[0], c[1], c[2]});
    const double offset = x < 2.0 * f90 ? x - x * x / (4.0 * f90) : f90;
    for (double& v : c) {
        v -= offset;
    }
    const double peak = std::max({c[0], c[1], c[2]});
    if (peak < neutral_start) {
        return c;
    }
    const double d = 1.0 - neutral_start;
    const double rolled = 1.0 - d * d / (peak + d - neutral_start);
    const double g = 1.0 - 1.0 / (neutral_desaturation * (peak - rolled) + 1.0);
    for (double& v : c) {
        v = (v * rolled / peak) * (1.0 - g) + rolled * g;
    }
    return c;
}

using Rgb8 = std::array<int, 3>;
using Bytes = std::vector<Rgb8>;

// An image, and its pyramid's base B_0 after step 3.
struct Toned {
    Bytes image;
    Field base;
};

// Step 1's clamp of a read of E.
Color clamped(Color c) {
    for (double& v : c) {
        v = std::min(v, ceiling);
    }
    return c;
}

// Step 1: E, radiance times 2^exposure.
Field exposed(const Field& radiance, double exposure) {
    const double factor = std::exp2(exposure);
    Field e = radiance;
    for (Color& t : e.texels) {
        for (double& v : t) {
            v = factor * v;
        }
    }
    return e;
}

// Step 2: the pyramid down from E, B_0 divided by the level count.
std::vector<Field> pyramid_down(const Field& e) {
    std::vector<Field> pyramid;
    pyramid.reserve(levels);
    const Field* source = &e;
    for (unsigned k = 0; k < levels; ++k) {
        const frame::Extent size = half_size({source->width, source->height});
        Field level = field(size.width, size.height, {});
        const auto read = [&](double u, double v) {
            const Color c = bilinear(*source, u, v);
            return k == 0 ? clamped(c) : c;
        };
        const double scale = k == 0 ? 1.0 / levels : 1.0;
        for (std::uint32_t y = 0; y < level.height; ++y) {
            for (std::uint32_t x = 0; x < level.width; ++x) {
                const Color c = down13(read, (x + 0.5) / level.width, (y + 0.5) / level.height, *source);
                level.at(x, y) = {half(scale * c[0]), half(scale * c[1]), half(scale * c[2])};
            }
        }
        pyramid.push_back(std::move(level));
        source = &pyramid.back();
    }
    return pyramid;
}

// Step 3: each level, from the second smallest up, plus the tent of the one
// below it.
void pyramid_up(std::vector<Field>& pyramid) {
    for (std::size_t k = pyramid.size() - 1; k-- > 0;) {
        Field& level = pyramid[k];
        const Field& below = pyramid[k + 1];
        for (std::uint32_t y = 0; y < level.height; ++y) {
            for (std::uint32_t x = 0; x < level.width; ++x) {
                const Color t = tent(below, (x + 0.5) / level.width, (y + 0.5) / level.height);
                for (std::size_t c = 0; c < 3; ++c) {
                    level.at(x, y)[c] = half(level.at(x, y)[c] + t[c]);
                }
            }
        }
    }
}

// Steps 4 to 6: the composite of E and B_0's glare, PBR Neutral, sRGB.
Bytes composite(const Field& e, const Field& base, double bloom) {
    Bytes out;
    out.reserve(e.texels.size());
    for (std::uint32_t y = 0; y < e.height; ++y) {
        for (std::uint32_t x = 0; x < e.width; ++x) {
            const Color glare = tent(base, (x + 0.5) / e.width, (y + 0.5) / e.height);
            Color c{};
            for (std::size_t n = 0; n < 3; ++n) {
                c[n] = (1.0 - bloom) * clamped(e.at(x, y))[n] + bloom * glare[n];
            }
            const Color shown = neutral(c);
            out.push_back({tests::srgb8(shown[0]), tests::srgb8(shown[1]), tests::srgb8(shown[2])});
        }
    }
    return out;
}

// Steps 1 to 6 over `radiance`.
Toned reference(const Field& radiance, passes::ToneMap settings) {
    const Field e = exposed(radiance, settings.exposure);
    std::vector<Field> pyramid = pyramid_down(e);
    pyramid_up(pyramid);
    return {composite(e, pyramid[0], settings.bloom), pyramid[0]};
}

// ---- The pass, on the GPU -----------------------------------------------

// Which pyramid the pass runs over: the renderer's own (frame_images.h), or
// levels made here that the CPU can read, B_0 read back after step 3.
enum class Pyramid { renderers, readable };

// A texture of `format` at `size`, in shared storage, resident.
NS::SharedPtr<MTL::Texture> shared_texture(metal::Device& device, metal::Submission& submission,
                                           MTL::PixelFormat format, frame::Extent size, MTL::TextureUsage usage) {
    auto descriptor = NS::TransferPtr(MTL::TextureDescriptor::alloc()->init());
    descriptor->setPixelFormat(format);
    descriptor->setWidth(size.width);
    descriptor->setHeight(size.height);
    descriptor->setStorageMode(MTL::StorageModeShared);
    descriptor->setUsage(usage);
    auto texture = NS::TransferPtr(device.handle()->newTexture(descriptor.get()));
    REQUIRE(texture);
    submission.make_resident(texture.get());
    return texture;
}

// The tone-map pass alone over `radiance`, through the renderer's own pass
// type, into an 8-bit target.
Toned tone_map(const Field& radiance, passes::ToneMap settings, Pyramid pyramid = Pyramid::renderers) {
    metal::Device device;
    metal::Submission submission(device);
    metal::Library library(device, metallib::shaders);
    const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    const frame::Extent size{radiance.width, radiance.height};

    auto image = shared_texture(device, submission, MTL::PixelFormatRGBA32Float, size, MTL::TextureUsageShaderRead);
    std::vector<float> texels;
    texels.reserve(radiance.texels.size() * 4);
    for (const Color& t : radiance.texels) {
        texels.insert(texels.end(),
                      {static_cast<float>(t[0]), static_cast<float>(t[1]), static_cast<float>(t[2]), 1.0f});
    }
    image->replaceRegion(MTL::Region(0, 0, size.width, size.height), 0, texels.data(), size.width * 4 * sizeof(float));

    metal::FrameImages renderers(device, submission, metal::FrameImages::Radiance::image,
                                 metal::FrameImages::Bloom::pyramid);
    renderers.prepare(size);
    std::array<NS::SharedPtr<MTL::Texture>, passes::bloom_levels> readable;
    if (pyramid == Pyramid::readable) {
        frame::Extent level = size;
        for (auto& texture : readable) {
            level = half_size(level);
            texture = shared_texture(device, submission, MTL::PixelFormatRGBA16Float, level,
                                     MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
        }
    }
    metal::Offscreen target(device, submission, size);
    const metal::ToneMapPass pass(device, library, submission, settings);

    auto table_descriptor = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
    table_descriptor->setMaxBufferBindCount(1);
    table_descriptor->setMaxTextureBindCount(3);
    NS::Error* error = nullptr;
    auto table = NS::TransferPtr(device.handle()->newArgumentTable(table_descriptor.get(), &error));
    INFO("argument table: " << tests::reason(error));
    REQUIRE(table);

    metal::FrameResources resources{};
    resources.arguments = table.get();
    resources.target = target.texture();
    resources.size = size;
    resources.radiance = image.get();
    for (std::uint32_t k = 0; k < passes::bloom_levels; ++k) {
        resources.bloom[k] = pyramid == Pyramid::readable ? readable[k].get() : renderers.bloom(k);
    }
    const metal::FrameSlot slot = submission.begin();
    MTL4::ComputeCommandEncoder* encoder = slot.commands->computeCommandEncoder();
    REQUIRE(encoder != nullptr);
    encoder->setArgumentTable(table.get());
    pass.record(encoder, resources);
    encoder->endEncoding();
    submission.commit();
    (void)submission.wait_until_complete(slot.sequence);

    Toned out;
    if (pyramid == Pyramid::readable) {
        MTL::Texture* b0 = readable[0].get();
        const auto width = static_cast<std::uint32_t>(b0->width());
        const auto height = static_cast<std::uint32_t>(b0->height());
        std::vector<Half> halves(std::size_t{width} * height * 4);
        b0->getBytes(halves.data(), width * 4 * sizeof(Half), MTL::Region(0, 0, width, height), 0);
        out.base = field(width, height, {});
        for (std::size_t i = 0; i < out.base.texels.size(); ++i) {
            out.base.texels[i] = {static_cast<double>(halves[4 * i]), static_cast<double>(halves[4 * i + 1]),
                                  static_cast<double>(halves[4 * i + 2])};
        }
    }
    const std::vector<std::uint8_t> rgba = tests::read_back(target);
    out.image.reserve(rgba.size() / 4);
    for (std::size_t i = 0; i < rgba.size(); i += 4) {
        out.image.push_back({rgba[i], rgba[i + 1], rgba[i + 2]});
    }
    return out;
}

// The largest difference between two images, in 8-bit levels.
int farthest(const Bytes& a, const Bytes& b) {
    REQUIRE(a.size() == b.size());
    int most = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t c = 0; c < 3; ++c) {
            most = std::max(most, std::abs(a[i][c] - b[i][c]));
        }
    }
    return most;
}

// Pixel (x, y) of an image `width` wide.
const Rgb8& pixel(const Bytes& image, std::uint32_t width, std::uint32_t x, std::uint32_t y) {
    return image[std::size_t{y} * width + x];
}

// A scene in small: a dim gradient, a few fireflies of hundreds, and a
// brass-colored patch near the roll-off's start.
Field scene_like(std::uint32_t width, std::uint32_t height) {
    Field f = field(width, height, {});
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const double u = static_cast<double>(x) / width;
            const double v = static_cast<double>(y) / height;
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

constexpr passes::ToneMap settings(float exposure, float bloom) {
    return passes::ToneMap{exposure, bloom, {0.0f, 0.0f}};
}

}  // namespace

TEST_CASE("the tone map computes the core's steps, at even and odd sizes") {
    // Within 2 levels of 255: the pyramid is stored as half floats and the
    // hardware's bilinear weights are fixed point, where the reference has
    // doubles.
    for (const frame::Extent size : {frame::Extent{64, 32}, frame::Extent{61, 37}, frame::Extent{200, 113}}) {
        const Field radiance = scene_like(size.width, size.height);
        for (const passes::ToneMap s : {settings(0.0f, 0.04f), settings(1.5f, 0.3f), settings(-2.0f, 0.0f)}) {
            INFO(size.width << " x " << size.height << ", exposure " << s.exposure << ", bloom " << s.bloom);
            CHECK(farthest(tone_map(radiance, s).image, reference(radiance, s).image) <= 2);
        }
    }
    // The check can see the bloom: with it and without, the images differ
    // far more than the tolerance.
    const Field radiance = scene_like(64, 32);
    CHECK(farthest(reference(radiance, settings(0.0f, 0.3f)).image, reference(radiance, settings(0.0f, 0.0f)).image) >
          20);
}

TEST_CASE("a uniform image stays uniform: bloom moves light and makes none") {
    for (const frame::Extent size : {frame::Extent{50, 30}, frame::Extent{33, 65}}) {
        const Color c{0.3, 0.2, 0.1};
        const Bytes out = tone_map(field(size.width, size.height, c), settings(0.0f, 0.5f)).image;
        const Color shown = neutral(c);
        const Rgb8 expected{tests::srgb8(shown[0]), tests::srgb8(shown[1]), tests::srgb8(shown[2])};
        INFO(size.width << " x " << size.height);
        CHECK(farthest(out, Bytes(out.size(), expected)) <= 1);
        CHECK(farthest(out, Bytes(out.size(), out.front())) == 0);
    }
}

TEST_CASE("frames as small as a pixel have every level") {
    for (const frame::Extent size : {frame::Extent{1, 1}, frame::Extent{2, 1}, frame::Extent{1, 3},
                                     frame::Extent{5, 3}, frame::Extent{33, 1}}) {
        INFO(size.width << " x " << size.height);
        const Field radiance = scene_like(size.width, size.height);
        const passes::ToneMap s = settings(0.0f, 0.2f);
        CHECK(farthest(tone_map(radiance, s).image, reference(radiance, s).image) <= 2);
    }
}

TEST_CASE("light past the half-float range glares as if at the ceiling, and nothing turns to NaN") {
    constexpr std::uint32_t side = 48;
    constexpr tests::Pixel hot{24, 24};
    Field radiance = field(side, side, {0.01, 0.01, 0.01});
    radiance.at(hot.x, hot.y) = {1e30, 1e30, 1e30};
    radiance.at(10, 30) = {3e38, 0.0, 0.0};
    for (const passes::ToneMap s : {settings(10.0f, 0.04f), settings(0.0f, 0.5f)}) {
        const Bytes out = tone_map(radiance, s).image;
        INFO("exposure " << s.exposure << ", bloom " << s.bloom);
        CHECK(farthest(out, reference(radiance, s).image) <= 2);
        CHECK(pixel(out, side, hot.x, hot.y) == Rgb8{255, 255, 255});
    }
}

TEST_CASE("broad light past the half-float range: six levels at the ceiling sum within it") {
    // Every level of the pyramid at the ceiling over most of the image: a
    // sum of six of them, undivided, is past what a half float holds.
    constexpr frame::Extent size{96, 64};
    constexpr std::uint32_t dim_from = 40;  // the rows from here down are dim
    Field radiance = field(size.width, size.height, {1e6, 1e6, 1e6});
    for (std::uint32_t y = dim_from; y < size.height; ++y) {
        for (std::uint32_t x = 0; x < size.width; ++x) {
            radiance.at(x, y) = {0.05, 0.04, 0.03};
        }
    }
    for (const passes::ToneMap s : {settings(0.0f, 0.5f), settings(-10.0f, 0.04f), settings(10.0f, 0.999f)}) {
        INFO("exposure " << s.exposure << ", bloom " << s.bloom);
        const Toned gpu = tone_map(radiance, s, Pyramid::readable);
        const Toned cpu = reference(radiance, s);
        CHECK(farthest(gpu.image, cpu.image) <= 2);
        // B_0 itself, which the image, white wherever it is near the
        // ceiling, cannot show: finite, within the ceiling, and the mean of
        // the blurs, within half floats' and the filter's rounding.
        REQUIRE(gpu.base.texels.size() == cpu.base.texels.size());
        double worst = 0.0;
        for (std::size_t i = 0; i < gpu.base.texels.size(); ++i) {
            for (std::size_t c = 0; c < 3; ++c) {
                const double got = gpu.base.texels[i][c];
                const double want = cpu.base.texels[i][c];
                REQUIRE(std::isfinite(got));
                CHECK(got <= ceiling);
                worst = std::max(worst, std::abs(got - want) / std::max(want, 1e-3));
            }
        }
        CHECK(worst < 0.01);
        if (s.exposure >= 0.0f) {
            CHECK(pixel(gpu.image, size.width, size.width / 2, dim_from / 4) == Rgb8{255, 255, 255});
        }
    }
    // A field past the ceiling everywhere, the most any level holds: B_0
    // finite, within the ceiling and within a part in a thousand of it
    // (the header works out both roundings), and the image white
    // everywhere, none of it NaN.
    const Toned white = tone_map(field(40, 24, {1e6, 1e6, 1e6}), settings(0.0f, 0.5f), Pyramid::readable);
    CHECK(farthest(white.image, Bytes(white.image.size(), {255, 255, 255})) == 0);
    for (const Color& t : white.base.texels) {
        for (const double v : t) {
            INFO("B_0 holds " << v);
            REQUIRE(std::isfinite(v));
            CHECK(v <= ceiling);
            CHECK(v >= 0.999 * ceiling);
        }
    }
}

TEST_CASE("a lone firefly glares: a halo falling off with distance, its color kept, its core white") {
    constexpr std::uint32_t side = 129;
    constexpr std::uint32_t c = side / 2;  // the firefly, in the middle
    Field radiance = field(side, side, {0.0, 0.0, 0.0});
    radiance.at(c, c) = {100.0, 80.0, 20.0};

    // Without bloom, the light stays in its pixel.
    const Bytes sharp = tone_map(radiance, settings(0.0f, 0.0f)).image;
    CHECK(std::ranges::count_if(sharp, [](const Rgb8& p) { return p != Rgb8{0, 0, 0}; }) == 1);

    const Bytes glare = tone_map(radiance, settings(0.0f, 0.04f)).image;
    const auto at = [&](std::uint32_t x, std::uint32_t y) { return pixel(glare, side, x, y); };
    // White-hot at its core.
    for (std::size_t channel = 0; channel < 3; ++channel) {
        CHECK(at(c, c)[channel] >= 240);
    }
    // Falling off, never rising, along a row, and reaching far: B_5's texels
    // are 64 frame pixels apart.
    for (std::uint32_t d = 1; d < 60; ++d) {
        INFO("distance " << d);
        CHECK(at(c + d, c)[0] <= at(c + d - 1, c)[0]);
    }
    CHECK(at(c + 8, c)[0] > 0);
    CHECK(at(c + 24, c)[0] > 0);
    // The glare keeps the firefly's yellow.
    const Rgb8 near = at(c + 4, c);
    CHECK(near[0] > near[1]);
    CHECK(near[1] > near[2]);
    // Round: the same at the same distance in each direction.
    CHECK(at(c + 6, c) == at(c, c + 6));
    CHECK(at(c - 6, c) == at(c, c - 6));
}
