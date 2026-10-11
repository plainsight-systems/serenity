// The naive path tracer (metal/integrator/path.metal.h, metal/passes/path/),
// on the GPU, converged over frames and checked against what it must
// converge to, worked out here from first principles:
//
//   - furnaces: under a sky of one radiance L, a white Lambert sphere, a
//     white coated one and a glass sphere each reflect and pass every bit of
//     light they receive, so every pixel converges to L. A wrong weight, or
//     a doubled count, shows here, through the BSDF sampling and the mean;
//     seen from outside, a convex sphere sends each path to the sky after
//     one bounce (glass, a few), so these say nothing of depth;
//   - the integrating sphere: inside a white sphere every path bounces
//     until the roulette ends it, so a depth limit, or a roulette that does
//     not reweigh what it keeps, loses light there;
//   - next event estimation: a floor under one firefly, and under two, in a
//     black sky, converges to albedo L sin^2 cos summed over the lights:
//     the uniform selection's 1 / N, the emitter's pdf, and principle 7
//     (the light sample and the bounce that reaches a firefly, each
//     weighted, counting it once between them) all at once;
//   - multiple importance sampling (principle 7): the two strategies'
//     weights for one direction sum to 1, from the densities the path
//     tracer asks, from the points it asks them from, at a firefly far off
//     and one millimeters across and near; the power heuristic is finite
//     and complementary at any densities, and a light 10^-10 m across,
//     whose density's square no float holds, lights a floor finitely; a
//     glossy floor's highlight of a firefly converges to the light's
//     integral over the BSDF, worked out on the GPU from the BSDF itself,
//     where each strategy carries a large share of it, so counting the
//     bounce's share in full, or not at all, fails; a Lambert and a glossy
//     floor under a firefly millimeters across converge to its light seen
//     from where rays leave them; and a firefly that a delta lobe, a black
//     coated sphere's coat, mirrors is counted in full: F0 L;
//   - the accumulated image: what it holds, what it refuses, and starting
//     over;
//   - principle 2: the same inputs give the same image, and frame k rendered
//     alone is frame k rendered after others.
//
// Every converged render also checks that no sample was left out for not
// being finite (metal/film/non_finite.h): such a sample is dropped from its
// pixel's mean, so a bug that made some paths NaN would still converge near
// the expected value, biased by the survivors.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <doctest/doctest.h>

#include "core/camera/thin_lens.h"
#include "core/contracts/bsdf.h"
#include "core/frame/graph_file.h"
#include "core/scene/scene.h"
#include "gpu/kernels/path_probe.h"
#include "gpu/support/probe_light.h"
#include "gpu/support/probe_runner.h"
#include "gpu/support/rendering.h"
#include "metal/device/device.h"
#include "metal/device/error.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"
#include "support/references.h"
#include "support/scene_text.h"
#include "support/vector.h"

using namespace serenity;
using tests::camera_text;
using tests::sky;
using tests::Vec3;

namespace {

// A renderer of the path graph over a scene, at a size, rendering frames
// that accumulate from frame 0. A fixture: the renderer reads the scene
// it holds, so it is neither copied nor moved (C.8).
class Rig {
public:
    Rig(const std::string& text, frame::Extent image) : description_(scene::parse(text, "test scene")), size_(image) {}

    // Renders `frames` more frames and returns the last, as displayed, with
    // every sample of them finite.
    std::vector<std::uint8_t> render(std::uint64_t frames) {
        std::uint64_t sequence = 0;
        for (std::uint64_t i = 0; i < frames; ++i, ++next_) {
            sequence = metal::render_to_offscreen(submission_, target_, renderer_,
                                                  tests::frame_at(next_, 0, 0.0, description_.camera));
        }
        (void)submission_.wait_until_complete(sequence);
        CHECK(renderer_.non_finite_samples() == 0);
        return tests::read_back(target_);
    }

    // The accumulated image of the frames rendered so far, in floats.
    std::vector<float> accumulated() {
        std::vector<float> floats(std::size_t{size_.width} * size_.height * 4);
        renderer_.read_accumulated(floats);
        return floats;
    }
    void accumulated_into(std::span<float> out) { renderer_.read_accumulated(out); }

    const scene::SceneDescription& description() const noexcept { return description_; }
    frame::Extent size() const noexcept { return size_; }

private:
    scene::SceneDescription description_;
    frame::Extent size_;
    metal::Device device_;
    metal::Submission submission_{device_};
    metal::Offscreen target_{device_, submission_, size_};
    metal::Renderer renderer_{device_, submission_, tests::path_graph(), &description_};
    std::uint64_t next_ = 0;
};

// A square of pixels: its middle, and how far it reaches either way.
struct Patch {
    std::uint32_t x;
    std::uint32_t y;
    std::uint32_t half;
};

// The mean linear radiance of `channel` over `patch`, which must lie within
// the image.
double mean_channel(const std::vector<std::uint8_t>& rgba, frame::Extent size, Patch patch, std::size_t channel) {
    REQUIRE(patch.x >= patch.half);
    REQUIRE(patch.y >= patch.half);
    REQUIRE(patch.x + patch.half < size.width);
    REQUIRE(patch.y + patch.half < size.height);
    double sum = 0.0;
    std::uint32_t count = 0;
    for (std::uint32_t y = patch.y - patch.half; y <= patch.y + patch.half; ++y) {
        for (std::uint32_t x = patch.x - patch.half; x <= patch.x + patch.half; ++x) {
            sum += tests::linear_of(rgba[(std::size_t{y} * size.width + x) * 4 + channel]);
            ++count;
        }
    }
    return sum / count;
}

// The red channel's.
double mean_red(const std::vector<std::uint8_t>& rgba, frame::Extent size, Patch patch) {
    return mean_channel(rgba, size, patch, 0);
}

// A sphere of radius 1 at the origin of `material`, under a sky of 0.5
// every way, seen from 4 away: the furnace.
std::string furnace(const std::string& material) {
    return camera_text("[0, 0, 4]", "[0, 0, 0]", 30) + sky("[0.5, 0.5, 0.5]", "[0.5, 0.5, 0.5]") +
           "[materials.m]\n" + material +
           "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 1\nmaterial = \"m\"\n";
}

constexpr frame::Extent square{64, 64};

// The square of half-width `half` in the middle of an image of `size`.
constexpr Patch middle_of(frame::Extent size, std::uint32_t half) {
    return {size.width / 2, size.height / 2, half};
}

constexpr Patch middle = middle_of(square, 6);

}  // namespace

TEST_CASE("furnace: a white Lambert sphere under a uniform sky converges to the sky") {
    // Albedo 1 and a sky of 0.5 every way: what the sphere receives it
    // returns.
    Rig rig(furnace("kind = \"rough\"\ncolor = [1, 1, 1]\n"), square);
    const auto image = rig.render(1024);
    // The sphere's middle, and the sky beside it.
    CHECK(mean_red(image, rig.size(), middle) == doctest::Approx(0.5).scale(0).epsilon(0.02));
    CHECK(mean_red(image, rig.size(), {3, 3, 2}) == doctest::Approx(0.5).scale(0).epsilon(0.01));
}

TEST_CASE("integrating sphere: light bounced many times inside a white sphere converges to its closed form") {
    // Inside a hollow sphere of radius R and reflectance rho, a diffuse
    // reflection spreads evenly over the whole inner wall (the form factor
    // between any two points of a sphere is the same), so a small light of
    // radiance L and radius r at the centre gives every wall point
    //   direct irradiance  pi L (r / R)^2,  and with every bounce after it
    //   total              pi L (r / R)^2 / (1 - rho),
    // and the wall's radiance is rho / pi of that: rho L (r / R)^2 / (1 - rho).
    // With rho = 0.9, nine tenths of it has bounced at least twice and the
    // mean path is ten bounces long, so the roulette, the BSDF sampling and
    // principle 7 (the light sample and a bounce that reaches the light,
    // each weighted) are all in play: a path cut at 8 bounces would keep
    // 1 - 0.9^8 = 57% of it. The
    // light itself absorbs what reaches it, a part in a few hundred per
    // bounce, inside the tolerance.
    constexpr double rho = 0.9;
    constexpr double radiance = 20.0;
    constexpr double ratio = 0.05;  // the light's radius over the wall's
    const std::string text = camera_text("[0, 0, 0.5]", "[0, 0, 1]", 30) + sky("[0, 0, 0]", "[0, 0, 0]") +
                             "[materials.wall]\nkind = \"rough\"\ncolor = [0.9, 0.9, 0.9]\n"
                             "[materials.glow]\nkind = \"emissive\"\nradiance = [20, 20, 20]\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 1\nmaterial = \"wall\"\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 0.05\nmaterial = \"glow\"\n";
    Rig rig(text, square);
    const auto image = rig.render(2048);
    const double expected = rho * radiance * ratio * ratio / (1.0 - rho);  // 0.45, within what the display shows
    CHECK(mean_red(image, rig.size(), middle_of(square, 8)) == doctest::Approx(expected).scale(0).epsilon(0.03));
}

TEST_CASE("furnace: a glass sphere under a uniform sky converges to the sky") {
    // Smooth glass loses nothing: reflected or refracted, every path leaves
    // with the sky's radiance, the 1 / eta^2 of entering undone by leaving.
    Rig rig(furnace("kind = \"dielectric\"\nior = 1.5\n"), square);
    const auto image = rig.render(512);
    CHECK(mean_red(image, rig.size(), middle) == doctest::Approx(0.5).scale(0).epsilon(0.02));
}

TEST_CASE("furnace: a white coated sphere under a uniform sky converges to the sky") {
    // A white base under a clear coat reflects all it receives, the coat's
    // share and the base's (materials/coated.h).
    Rig rig(furnace("kind = \"coated\"\ncolor = [1, 1, 1]\nior = 1.5\n"), square);
    const auto image = rig.render(1024);
    CHECK(mean_red(image, rig.size(), middle) == doctest::Approx(0.5).scale(0).epsilon(0.02));
}

namespace {

// Glass's reflectance head on, F0 = ((1.5 - 1) / (1.5 + 1))^2, and what a
// sphere of it filled with a medium returns through its middle under a sky
// of L: every exit, front or back, sees L, L (F + (1 - F)^2 tau /
// (1 - F tau)), tau what one crossing keeps.
constexpr double glass_f0 = 0.04;

double through_filled_glass(double sky_radiance, double tau) {
    return sky_radiance *
           (glass_f0 + (1.0 - glass_f0) * (1.0 - glass_f0) * tau / (1.0 - glass_f0 * tau));
}

}  // namespace

TEST_CASE("an absorbing medium at a bead's scale: its stretch measured from the surface, not the ray's start") {
    Rig rig(tests::tinted_bead(), square);
    const auto image = rig.render(2048);
    const double expected = through_filled_glass(0.5, std::exp(-1.4));  // 2 mm of 700 per meter
    INFO("expected red " << expected);
    CHECK(mean_red(image, rig.size(), middle_of(square, 1)) == doctest::Approx(expected).scale(0).epsilon(0.03));
}

TEST_CASE("a light inside the same medium as the surface it lights: its light dimmed over the shadow ray") {
    Rig rig(tests::core_lit_inside_glass(), square);
    const auto image = rig.render(2048);
    const Patch core = middle_of(square, 2);
    const double ratio = mean_channel(image, rig.size(), core, 0) / mean_channel(image, rig.size(), core, 1);
    const double sigma = std::log(2.0);
    INFO("red over green " << ratio);
    CHECK(ratio == doctest::Approx(std::exp(-sigma * tests::camera_to_core) * std::exp(-sigma * tests::light_to_core))
                       .scale(0)
                       .epsilon(0.06));
}

TEST_CASE("a coated sphere of black base shows its coat alone: F0 of the sky at its middle") {
    const std::string text = camera_text("[0, 0, 4]", "[0, 0, 0]", 30) + sky("[1, 1, 1]", "[1, 1, 1]") +
                             "[materials.black]\nkind = \"coated\"\ncolor = [0, 0, 0]\nior = 1.5\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 1\nmaterial = \"black\"\n";
    // The coat is drawn 4% of the time, so its mean is slow to settle:
    // over 4096 frames each pixel's has some 8% of error, and the mean of
    // 25 pixels some 1.5%, with 8-bit rounding beside it.
    Rig rig(text, square);
    const auto image = rig.render(4096);
    CHECK(mean_red(image, rig.size(), middle_of(square, 2)) == doctest::Approx(glass_f0).scale(0).epsilon(0.07));
}

TEST_CASE("an absorbing medium in glass: through its middle, what Beer and Lambert and Fresnel leave") {
    // A glass sphere of radius 1 filled with a medium that keeps half its red
    // light over 1 m, under a sky of L every way. Along the middle every
    // crossing is 2 m, keeping tau = 0.25 of red. Green and blue it keeps
    // whole: the sky.
    const std::string text = camera_text("[0, 0, 4]", "[0, 0, 0]", 30) + sky("[0.5, 0.5, 0.5]", "[0.5, 0.5, 0.5]") +
                             "[materials.glass]\nkind = \"dielectric\"\nior = 1.5\n"
                             "[media.red_out]\nkind = \"absorbing\"\ntint = [0.5, 1, 1]\ntint_distance = 1\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 1\nmaterial = \"glass\"\n"
                             "interior = \"red_out\"\n";
    Rig rig(text, square);
    const auto image = rig.render(2048);
    const double expected = through_filled_glass(0.5, 0.25);
    INFO("expected red " << expected);
    const Patch through = middle_of(square, 1);
    CHECK(mean_channel(image, rig.size(), through, 0) == doctest::Approx(expected).scale(0).epsilon(0.03));
    CHECK(mean_channel(image, rig.size(), through, 1) == doctest::Approx(0.5).scale(0).epsilon(0.02));
    CHECK(mean_channel(image, rig.size(), through, 2) == doctest::Approx(0.5).scale(0).epsilon(0.02));
}

namespace {

constexpr double floor_albedo = 0.8;

std::string lit_floor(const std::string& fireflies) {
    return camera_text("[0, 2, 3]", "[0, 0, 0]", 50) + sky("[0, 0, 0]", "[0, 0, 0]") +
           "[materials.floor]\nkind = \"rough\"\ncolor = [0.8, 0.8, 0.8]\n"
           "[materials.glow]\nkind = \"emissive\"\nradiance = [200, 200, 200]\n"
           "[[shapes]]\nkind = \"box\"\nmin = [-50, -1, -50]\nmax = [50, 0, 50]\nmaterial = \"floor\"\n" +
           fireflies;
}

std::string firefly(const char* center) {
    return std::string("[[shapes]]\nkind = \"sphere\"\ncenter = ") + center + "\nradius = 0.05\nmaterial = \"glow\"\n";
}

const std::string one_firefly = firefly("[0.3, 2.5, -0.5]");

// albedo x L sin^2 cos, summed over the lights, at the floor point the
// image's point (x, y) sees, in pixels from the top left: pixel (i, j)'s
// center is (i + 0.5, j + 0.5).
double expected_floor(const scene::SceneDescription& description, frame::Extent size, double x, double y) {
    const contracts::CameraData c = camera::shader_form(description.camera, size);
    const Vec3 d = tests::camera_direction(c, x, y, size);
    const Vec3 p = tests::vec(c.origin) + (-c.origin.y / d.y) * d;
    double sum = 0.0;
    for (const lights::SphereLightData& light : description.sphere_lights) {
        // Where its shape is, and how big: the transform's translation and
        // scale (contracts/transform.h).
        const contracts::Transform& placed = description.shapes.transforms.at(light.shape);
        const double radius = placed.m[0][0];
        const Vec3 to = tests::vec(contracts::translation(placed)) - p;
        const double d2 = dot(to, to);
        sum += floor_albedo * light.radiance.x * (radius * radius / d2) * (to.y / std::sqrt(d2));
    }
    return sum;
}

void check_floor(const std::string& text, std::uint64_t frames) {
    Rig rig(text, {64, 48});
    const auto image = rig.render(frames);
    for (const tests::Pixel pixel : {tests::Pixel{32, 30}, tests::Pixel{14, 38}, tests::Pixel{50, 26}}) {
        // A 3 x 3 patch's mean against the formula at its center: the
        // formula varies little over three pixels of floor.
        const double expected = expected_floor(rig.description(), rig.size(), pixel.x + 0.5, pixel.y + 0.5);
        INFO("pixel " << pixel.x << ", " << pixel.y << ": expected " << expected);
        CHECK(mean_red(image, rig.size(), {pixel.x, pixel.y, 1}) == doctest::Approx(expected).scale(0).epsilon(0.03));
    }
}

}  // namespace

TEST_CASE("next event estimation: a floor under one firefly converges to albedo L sin^2 cos") {
    check_floor(lit_floor(one_firefly), 512);
}

TEST_CASE("next event estimation: under two fireflies, chosen one at a time, the floor converges to their sum") {
    check_floor(lit_floor(one_firefly + firefly("[-1.2, 1.5, 0.4]")), 1024);
}

TEST_CASE("MIS: a firefly millimeters across, just over a floor, converges to albedo L sin^2 cos at the floor") {
    // A firefly of radius 3 mm, 2 cm over the floor. Both strategies draw
    // and weigh a direction from the point its ray leaves from
    // (path.metal.h, "One origin"), the floor's point offset by its own
    // float rounding (integrator/surface.metal.h): at y = 0, within
    // offset_ray()'s band about the origin, 15 um up, 0.08% of the
    // firefly's height, so the floor converges to the closed form at its
    // own point, to 0.5%. With the 0.1 mm step before it, the light seen
    // from 0.1 mm up, the floor converged 0.3 to 0.4% darker to the
    // firefly's side and 0.9% brighter below it, at (32, 24), outside the
    // tolerance (docs/research/2026-10-11-ray-offset.md); with step 5 drawn
    // from the floor's point and its rays leaving from 0.1 mm up, as 7751a65
    // drew, 1% short below it. Each pixel's mean against the closed form
    // averaged over 4 x 4 points of it.
    const std::string text = camera_text("[0, 0.1, 0.1]", "[0, 0, 0]", 60) + sky("[0, 0, 0]", "[0, 0, 0]") +
                             "[materials.floor]\nkind = \"rough\"\ncolor = [0.8, 0.8, 0.8]\n"
                             "[materials.glow]\nkind = \"emissive\"\nradiance = [30, 30, 30]\n"
                             "[[shapes]]\nkind = \"box\"\nmin = [-50, -1, -50]\nmax = [50, 0, 50]\nmaterial = \"floor\"\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0.02, 0]\nradius = 0.003\nmaterial = \"glow\"\n";
    constexpr frame::Extent size{64, 48};
    Rig rig(text, size);
    (void)rig.render(65536);
    const std::vector<float> image = rig.accumulated();
    constexpr std::uint32_t within = 4;  // points a side in each pixel
    for (const tests::Pixel pixel : {tests::Pixel{16, 24}, tests::Pixel{48, 30}, tests::Pixel{32, 40},
                                     tests::Pixel{24, 12}, tests::Pixel{32, 24}}) {
        double expected = 0.0;
        for (std::uint32_t j = 0; j < within; ++j) {
            for (std::uint32_t i = 0; i < within; ++i) {
                expected += expected_floor(rig.description(), size, pixel.x + (i + 0.5) / within,
                                           pixel.y + (j + 0.5) / within);
            }
        }
        expected /= within * within;
        const double rendered = image[(std::size_t{pixel.y} * size.width + pixel.x) * 4];
        INFO("pixel " << pixel.x << ", " << pixel.y << ": rendered " << rendered << ", expected " << expected);
        MESSAGE("pixel " << pixel.x << ", " << pixel.y << ": rendered " << rendered << ", expected " << expected
                         << ", rendered / expected - 1 = " << (rendered / expected - 1.0) * 100.0 << "%");
        CHECK(rendered == doctest::Approx(expected).scale(0).epsilon(0.005));
    }
}

TEST_CASE("MIS: a light 10^-10 m across, its density past what a float squares, lights a floor finitely") {
    // A sphere light of radius 10^-10 m 2.5 m over the floor: its cone's
    // solid angle is some 5 x 10^-21 sr, its density 2 x 10^20, whose
    // square is past a float's largest. The scene accepts it (any finite
    // radius above 0). Squared directly, the power heuristic made its every
    // light sample's weight NaN, and the frame's samples not finite (Rig's
    // render checks there are none); computed from the ratio of the
    // densities (path.metal.h, power_heuristic()), the floor converges to
    // its closed form. Its radiance, 10^20, makes that closed form some 0.1.
    check_floor(lit_floor(std::string("[materials.pinpoint]\nkind = \"emissive\"\nradiance = [1e20, 1e20, 1e20]\n") +
                          "[[shapes]]\nkind = \"sphere\"\ncenter = [0.3, 2.5, -0.5]\nradius = 1e-10\n"
                          "material = \"pinpoint\"\n"),
                512);
}

TEST_CASE("the same inputs give the same image") {
    // Compared in the accumulated floats, bit for bit, not as displayed: a
    // difference below what 8 bits a channel can show still fails.
    const std::string text = lit_floor(one_firefly);
    Rig first(text, {32, 24});
    Rig second(text, {32, 24});
    (void)first.render(8);
    (void)second.render(8);
    const std::vector<float> a = first.accumulated();
    const std::vector<float> b = second.accumulated();
    REQUIRE(a.size() == b.size());
    CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
    // And the comparison can fail: one frame more is another image.
    (void)second.render(1);
    CHECK(std::memcmp(a.data(), second.accumulated().data(), a.size() * sizeof(float)) != 0);
}

TEST_CASE("the accumulated image reads back only whole, and only when there is one") {
    Rig rig(lit_floor(one_firefly), {8, 6});
    std::vector<float> wrong(8 * 6 * 4 - 1);
    // Before any frame there is no image to read.
    CHECK_THROWS_WITH_AS(rig.accumulated_into(wrong), doctest::Contains("no frame has made"), metal::MetalError);
    (void)rig.render(1);
    CHECK_THROWS_WITH_AS(rig.accumulated_into(wrong), doctest::Contains("floats given"), metal::MetalError);
    metal::Device device;
    metal::Submission submission(device);
    metal::Renderer preview(device, submission, tests::preview_graph(), &rig.description());
    std::vector<float> any(4);
    CHECK_THROWS_WITH_AS(preview.read_accumulated(any), doctest::Contains("accumulates nothing"), metal::MetalError);
}

TEST_CASE("frame k rendered alone is frame k rendered after others (principle 2)") {
    // Frames 0 to 4, then frame 5 started over (accumulated since 5), on one
    // renderer; and frame 5 alone on a renderer of its own: the same image,
    // whatever the first renderer did before. And frames 5 to 7 accumulated,
    // after frames 0 to 4, against 5 to 7 alone.
    const scene::SceneDescription description = scene::parse(lit_floor(one_firefly), "test scene");
    constexpr frame::Extent size{40, 30};
    const auto run = [&](std::uint64_t first, std::uint64_t last, std::uint64_t since_from) {
        metal::Device device;
        metal::Submission submission(device);
        metal::Offscreen target(device, submission, size);
        metal::Renderer renderer(device, submission, tests::path_graph(), &description);
        std::uint64_t sequence = 0;
        for (std::uint64_t i = first; i <= last; ++i) {
            const std::uint64_t since = i < since_from ? i : since_from;
            sequence = metal::render_to_offscreen(submission, target, renderer,
                                                  tests::frame_at(i, since, 0.0, description.camera));
        }
        (void)submission.wait_until_complete(sequence);
        CHECK(renderer.non_finite_samples() == 0);
        std::vector<float> floats(std::size_t{size.width} * size.height * 4);
        renderer.read_accumulated(floats);
        return floats;
    };
    // Each of frames 0 to 4 starting over; 5 to 7 accumulated from 5:
    // compared bit for bit in the accumulated floats.
    const std::vector<float> after_others = run(0, 7, 5);
    const std::vector<float> alone = run(5, 7, 5);
    REQUIRE(after_others.size() == alone.size());
    CHECK(std::memcmp(after_others.data(), alone.data(), alone.size() * sizeof(float)) == 0);
    // And they are not every frame alike: frame 5 alone differs from 6 alone.
    const std::vector<float> five = run(5, 5, 5);
    CHECK(std::memcmp(five.data(), run(6, 6, 6).data(), five.size() * sizeof(float)) != 0);
}

TEST_CASE("the accumulated image holds what the inputs claim, or the frame is refused") {
    const scene::SceneDescription description = scene::parse(lit_floor(one_firefly), "test scene");
    metal::Device device;
    metal::Submission submission(device);
    metal::Offscreen small(device, submission, {16, 16});
    metal::Offscreen large(device, submission, {24, 16});
    metal::Renderer renderer(device, submission, tests::path_graph(), &description);
    const auto inputs = [&](std::uint64_t index, std::uint64_t since) {
        return tests::frame_at(index, since, 0.0, description.camera);
    };

    (void)metal::render_to_offscreen(submission, small, renderer, inputs(0, 0));
    (void)metal::render_to_offscreen(submission, small, renderer, inputs(1, 0));
    // A frame skipped.
    CHECK_THROWS_AS(metal::render_to_offscreen(submission, small, renderer, inputs(3, 0)), metal::MetalError);
    // A size changed without starting over.
    CHECK_THROWS_AS(metal::render_to_offscreen(submission, large, renderer, inputs(2, 0)), metal::MetalError);
    // accumulated_since moved without starting over.
    CHECK_THROWS_AS(metal::render_to_offscreen(submission, small, renderer, inputs(2, 1)), metal::MetalError);
    // accumulated_since after the frame.
    CHECK_THROWS_AS(metal::render_to_offscreen(submission, small, renderer, inputs(2, 5)), metal::MetalError);
    // Starting over at a new size remakes the image, frames in flight
    // drained, and accumulates on from there.
    (void)metal::render_to_offscreen(submission, large, renderer, inputs(2, 2));
    const auto last = metal::render_to_offscreen(submission, large, renderer, inputs(3, 2));
    (void)submission.wait_until_complete(last);
    CHECK(renderer.non_finite_samples() == 0);

    // More frames than an image holds.
    metal::Renderer fresh(device, submission, tests::path_graph(), &description);
    CHECK_THROWS_AS(fresh.prepare(inputs(frame::max_accumulated_frames + 1, 0), {16, 16}), metal::MetalError);
}

TEST_CASE("a frame recorded without being prepared is refused, and so is a graph with two accumulating passes") {
    const scene::SceneDescription description = scene::parse(lit_floor(one_firefly), "test scene");
    metal::Device device;
    metal::Submission submission(device);
    metal::Offscreen target(device, submission, {16, 16});
    metal::Renderer renderer(device, submission, tests::path_graph(), &description);
    const metal::FrameSlot slot = submission.begin();
    CHECK_THROWS_AS(renderer.record(slot, tests::frame_at(0, 0, 0.0, description.camera), target.texture(),
                                    target.size()),
                    metal::MetalError);

    metal::Submission other(device);
    const frame::Schedule twice{{frame::PassKind::path, frame::PassKind::path, frame::PassKind::display}, std::nullopt};
    CHECK_THROWS_AS(metal::Renderer(device, other, twice, &description), metal::MetalError);
}

TEST_CASE("the camera sees a light's glow, through the emitter") {
    // A camera ray that meets a light counts its emission (step 3): the
    // emitter's radiance, (0.6, 0.3, 0.1), every frame alike, each channel.
    const std::array<double, 3> radiance{0.6, 0.3, 0.1};
    const std::string text = camera_text("[0, 0, 3]", "[0, 0, 0]", 30) + sky("[0, 0, 0]", "[0, 0, 0]") +
                             "[materials.glow]\nkind = \"emissive\"\nradiance = [0.6, 0.3, 0.1]\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 0.5\nmaterial = \"glow\"\n";
    Rig rig(text, {33, 33});
    const auto image = rig.render(4);
    for (std::size_t c = 0; c < 3; ++c) {
        INFO("channel " << c);
        CHECK(mean_channel(image, rig.size(), middle_of(rig.size(), 2), c) ==
              doctest::Approx(radiance[c]).scale(0).epsilon(0.01));
    }
}

TEST_CASE("an infinity a path computes is counted as not finite, under the shaders' fast math") {
    // A light at the most radiance a float holds inside a closed white
    // sphere: a path's sum over its surfaces overflows
    // (support/scene_text.h). Film's finite() test of the sample (metal/film/
    // accumulate.metal.h) must still see it, under -fmetal-math-mode=fast
    // (cmake/MetalLibrary.cmake), whose compiler may assume a value it
    // computed is finite.
    const scene::SceneDescription description = scene::parse(tests::blinding_lights(), "test scene");
    metal::Device device;
    metal::Submission submission(device);
    metal::Offscreen target(device, submission, {16, 16});
    metal::Renderer renderer(device, submission, tests::path_graph(), &description);
    std::uint64_t last = 0;
    for (std::uint64_t i = 0; i < 2; ++i) {
        last = metal::render_to_offscreen(submission, target, renderer, tests::frame_at(i, 0, 0.0, description.camera));
    }
    (void)submission.wait_until_complete(last);
    CHECK(renderer.non_finite_samples() > 0);
}

TEST_CASE("a frame time past what the shaders' float holds is refused before it is recorded") {
    const scene::SceneDescription description = scene::parse(lit_floor(one_firefly), "test scene");
    metal::Device device;
    metal::Submission submission(device);
    metal::Offscreen target(device, submission, {16, 16});
    metal::Renderer renderer(device, submission, tests::path_graph(), &description);
    for (const double seconds : {1e100, -1e39, std::numeric_limits<double>::infinity()}) {
        INFO("time " << seconds);
        CHECK_THROWS_AS(metal::render_to_offscreen(submission, target, renderer,
                                                   tests::frame_at(0, 0, seconds, description.camera)),
                        metal::MetalError);
    }
    const auto fine =
        metal::render_to_offscreen(submission, target, renderer, tests::frame_at(0, 0, 3.0e38, description.camera));
    (void)submission.wait_until_complete(fine);
}

TEST_CASE("before any light wakes, under a black sky, the image is black: the opening's first frames") {
    // Two fireflies over a floor, under no sky, neither awake before 10 s
    // (core/animation/glow.h, Wake): every accumulated value exactly 0, the
    // camera's view of the dark lights and every surface they would light
    // alike. A light's next event adds nothing while it sleeps, its shadow
    // ray not traced (path.metal.h, step 5).
    const auto sleeping = [](const char* center, double at) {
        return firefly(center) + "glow = { kind = \"rhythm\", period = 5.5, flash = 0.35, dim = 0.5, seed = 3, " +
               "wake = { at = " + std::to_string(at) + ", ramp = 1 } }\n";
    };
    Rig dark(lit_floor(sleeping("[0.3, 2.5, -0.5]", 10.0) + sleeping("[-0.2, 0.8, 0.1]", 10.0)), {48, 36});
    (void)dark.render(8);
    // Each pixel's red, green and blue; its fourth number is its count of
    // samples, not light.
    const auto light_in = [](const std::vector<float>& rgba) {
        std::size_t lit = 0;
        for (std::size_t i = 0; i < rgba.size(); i += 4) {
            lit += (rgba[i] != 0.0f || rgba[i + 1] != 0.0f || rgba[i + 2] != 0.0f) ? 1 : 0;
        }
        return lit;
    };
    CHECK(light_in(dark.accumulated()) == 0);
    // And the check can fail: one of them awake at 0 lights the floor.
    Rig waking(lit_floor(sleeping("[0.3, 2.5, -0.5]", 10.0) + sleeping("[-0.2, 0.8, 0.1]", -1.0)), {48, 36});
    (void)waking.render(8);
    CHECK(light_in(waking.accumulated()) > 0);
}

// ---- multiple importance sampling (path.metal.h, steps 3 and 5) -----------

namespace {

// A surface at the origin facing +y, and fireflies where a mirror there
// would show them to wo: what the MIS probe asks about. mis_light is 0.3
// across at 1.6, near_light 3 mm across at 2 cm, near enough that the step
// a ray leaves the surface from (integrator/surface.metal.h, 15 um at the
// origin) moves the cone the light fills.
const contracts::Float3 mis_wo{0.0f, 0.5547002f, 0.8320503f};  // (0, 2, 3) normalized
const tests::ProbeLight mis_light{{0.0f, 0.9f, -1.35f}, 0.3f, 10.0f};
const tests::ProbeLight near_light{{0.0f, 0.011094004f, -0.016641006f}, 0.003f, 10.0f};  // 0.02 along (0, 2, -3)

contracts::Bsdf floor_bsdf(contracts::BsdfKind kind, float alpha) {
    contracts::Bsdf bsdf{};
    bsdf.normal = {0.0f, 1.0f, 0.0f};
    bsdf.kind = kind;
    bsdf.color = {0.8f, 0.8f, 0.8f};
    bsdf.alpha = alpha;
    return bsdf;
}

// What the MIS probe found over its draws of both strategies.
struct WeightSums {
    std::size_t both = 0;          // directions both strategies draw
    std::size_t by_bsdf_both = 0;  // of them, the BSDF's draws: bounces that strike the light
    double worst = 0.0;            // the largest |w_l + w_b - 1| among them
    double worst_alone = 0.0;      // the largest |w_b - 1| where only the BSDF draws the direction

    double worst_either() const { return std::max(worst, worst_alone); }
};

// Where step 5 draws from, for the probe: as the path tracer draws, or from
// the surface's point, as 7751a65 did.
enum class LightFrom : std::uint32_t { path_tracer = 0, surface_point = 1 };

WeightSums weight_sums(tests::ProbeRunner& gpu, const contracts::Bsdf& bsdf, const tests::ProbeLight& light,
                       std::uint32_t light_count, std::uint32_t struck_light_count,
                       LightFrom from = LightFrom::path_tracer) {
    constexpr std::uint32_t draws = 4096;
    const tests::MisQuery query{{0.0f, 0.0f, 0.0f}, draws,   mis_wo, light_count, {0.0f, 1.0f, 0.0f},
                                struck_light_count, static_cast<std::uint32_t>(from), {0u, 0u, 0u}};
    const std::vector<tests::MisDraw> results = tests::run_with_light<tests::MisDraw>(
        gpu, "mis_weights", 2 * draws, light, {tests::Binding::of(query), tests::Binding::of(bsdf)});
    WeightSums sums;
    for (const tests::MisDraw& d : results) {
        if (d.light_density > 0.0f && d.bsdf_density > 0.0f) {
            ++sums.both;
            sums.by_bsdf_both += d.by_bsdf;
            sums.worst = std::max(sums.worst, std::abs(static_cast<double>(d.light_weight) + d.bsdf_weight - 1.0));
        } else if (d.by_bsdf == 1u && d.bsdf_density > 0.0f) {
            sums.worst_alone = std::max(sums.worst_alone, std::abs(static_cast<double>(d.bsdf_weight) - 1.0));
        }
    }
    return sums;
}

}  // namespace

TEST_CASE("MIS: for a direction both strategies draw, the light sample's weight and the bounce's sum to 1") {
    // From the densities each step asks, from the point each asks them from
    // (path.metal.h, "One origin"): step 5 the selection's 1 / N, the
    // light's own sample pdf from next_event_origin() and the BSDF's pdf();
    // step 3 the bounce's density as step 6 hands it on (handed_on()), the
    // selection's 1 / N and light_pdf() from the point the bounce leaves
    // from (leave()). At a glossy conductor and a Lambert surface, under a
    // firefly 0.3 across at 1.6 and one 3 mm across at 2 cm, over each
    // strategy's draws. The sum is 1 only where the two steps ask the same
    // densities of the same direction from the same point; to 10^-4, the
    // rounding of a conductor's half vector, found from the drawn
    // microfacet normal by sample() and from wo + wi by pdf(). A direction
    // only the BSDF draws, one that misses the light, is the bounce's
    // alone: w_b 1, to the rounding of fast math's division.
    constexpr double tolerance = 1e-4;
    tests::ProbeRunner gpu;
    const std::array<std::pair<std::string, contracts::Bsdf>, 2> surfaces{
        {{"conductor, alpha 0.1", floor_bsdf(contracts::BsdfKind::conductor, 0.1f)},
         {"lambert", floor_bsdf(contracts::BsdfKind::lambert, 0.0f)}}};
    const std::array<std::pair<std::string, tests::ProbeLight>, 2> lights{
        {{"firefly 0.3 across at 1.6", mis_light}, {"firefly 3 mm across at 2 cm", near_light}}};
    for (const auto& [light_name, light] : lights) {
        for (const auto& [name, bsdf] : surfaces) {
            const WeightSums sums = weight_sums(gpu, bsdf, light, 4, 4);
            INFO(light_name << ", " << name << ": " << sums.both << " directions both draw (" << sums.by_bsdf_both
                            << " of them the BSDF's), worst |w_l + w_b - 1| " << sums.worst
                            << ", worst |w_b - 1| alone " << sums.worst_alone);
            MESSAGE(light_name << ", " << name << ": " << sums.both << " directions both draw (" << sums.by_bsdf_both
                               << " of them the BSDF's), worst |w_l + w_b - 1| " << sums.worst
                               << ", worst |w_b - 1| alone " << sums.worst_alone);
            // Not empty: every light sample is above the surface, and some
            // bounces strike the light.
            CHECK(sums.both > 4096);
            CHECK(sums.by_bsdf_both > 0);
            CHECK(sums.worst < tolerance);
            CHECK(sums.worst_alone < 1e-6);
            // And the check can fail: step 3 weighing the struck light as
            // if it were the only one (P = 1) while step 5 chose among 4.
            // Off by far more than the tolerance at either surface: by most
            // of the weight at the conductor, where the strategies share
            // it; less at the Lambert surface, where the light's density
            // dominates both.
            const WeightSums wrong = weight_sums(gpu, bsdf, light, 4, 1);
            MESSAGE(light_name << ", " << name << ", step 3's P of 1 against step 5's 1 / 4: worst |w_l + w_b - 1| "
                               << wrong.worst);
            CHECK(wrong.worst > 10.0 * tolerance);
        }
    }
    // And it fails where step 5 draws from the surface's point, as 7751a65
    // drew, while step 3 asks from the point the bounce leaves from: at the
    // near firefly, whose cone the 15 um between the two points moves, the
    // two densities of one direction differ, and in the crescent one
    // point's cone holds and the other's does not, only one is 0. At the far
    // firefly the two points' densities differ within the rounding: why the
    // check above asks at the near one too.
    for (const auto& [name, bsdf] : surfaces) {
        const WeightSums wrong = weight_sums(gpu, bsdf, near_light, 4, 4, LightFrom::surface_point);
        MESSAGE("firefly 3 mm across at 2 cm, " << name << ", step 5 from the surface's point: worst |w_l + w_b - 1| "
                                                << wrong.worst << ", worst |w_b - 1| alone " << wrong.worst_alone);
        CHECK(wrong.worst_either() > 10.0 * tolerance);
    }
}

TEST_CASE("MIS: the power heuristic is finite and complementary at any densities a float holds") {
    // Densities from 0 to infinity, the pairs the path tracer can meet: a
    // light's P p_l past 1.8 x 10^19 (a sphere of 10^-10 m seen from a
    // meter: 3.2 x 10^19), whose square a float does not hold; the largest
    // float; an infinite density beside a finite one, and two infinite; a
    // density beside 0; and ordinary ones. Each weight is checked against
    // f^2 / (f^2 + g^2) worked out in doubles (its limit where a density is
    // infinite, 1/2 for two), and the two ways round must sum to 1.
    constexpr float inf = std::numeric_limits<float>::infinity();
    constexpr float largest = std::numeric_limits<float>::max();
    const std::vector<tests::HeuristicQuery> queries{
        {2.0f, 1.0f},     {1.0f, 1.0f},    {0.3f, 0.0f},         {3.2e19f, 0.3f},  {0.3f, 3.2e19f},
        {1e20f, 1e20f},   {1e30f, 3e29f},  {largest, 1e-3f},     {largest, largest}, {inf, 1.0f},
        {1.0f, inf},      {inf, inf},      {1e-30f, 1e-20f},     {1e-20f, 0.0f}};
    tests::ProbeRunner gpu;
    const auto count = static_cast<std::uint32_t>(queries.size());
    const std::vector<tests::HeuristicWeights> weights = gpu.run<tests::HeuristicWeights>(
        "heuristic_weights", count, {tests::Binding::of(queries), tests::Binding::of(count)});
    std::size_t squared_not_finite = 0;
    for (std::size_t k = 0; k < queries.size(); ++k) {
        const double f = queries[k].drawn;
        const double g = queries[k].other;
        double expected = 0.0;
        if (std::isinf(f) && std::isinf(g)) {
            expected = 0.5;
        } else if (std::isinf(f)) {
            expected = 1.0;
        } else if (std::isinf(g)) {
            expected = 0.0;
        } else {
            expected = f * f / (f * f + g * g);
        }
        const tests::HeuristicWeights& w = weights[k];
        INFO("densities " << f << ", " << g << ": weight " << w.weight << " (expected " << expected << "), swapped "
                          << w.swapped << ", squared directly " << w.squared);
        CHECK(std::isfinite(w.weight));
        CHECK(std::isfinite(w.swapped));
        // Absolute: a weight in [0, 1] to 10^-6 (a float flushes a weight
        // under 10^-38 to 0, as at 0.3 beside 3.2 x 10^19).
        CHECK(w.weight == doctest::Approx(expected).epsilon(1e-6));
        CHECK(static_cast<double>(w.weight) + w.swapped == doctest::Approx(1.0).scale(0).epsilon(1e-6));
        squared_not_finite += std::isfinite(w.squared) ? 0 : 1;
    }
    // And the densities asked are ones that break the heuristic squared
    // directly, as 7751a65 computed it: NaN at some of them.
    MESSAGE(squared_not_finite << " of " << queries.size() << " pairs not finite when squared directly");
    CHECK(squared_not_finite >= 5);
}

namespace {

// A conductor floor, alpha 0.1 (roughness 0.316), seen from (0, 1, 2), and
// a firefly where the floor mirrors it to the camera's middle, out of the
// camera's view: the floor's highlight of it. highlight_light is the
// scene's firefly, as the probe binds it: the same center, radius and
// radiance, written twice.
constexpr float glossy_roughness = 0.31622776f;  // alpha = roughness^2 = 0.1
const tests::ProbeLight highlight_light{{0.0f, 0.7155418f, -1.4310835f}, 0.3f, 10.0f};  // 1.6 along (0, 1, -2)

std::string glossy_floor() {
    return camera_text("[0, 1, 2]", "[0, 0, 0]", 30) + sky("[0, 0, 0]", "[0, 0, 0]") +
           "[materials.floor]\nkind = \"conductor\"\nf0 = [0.8, 0.8, 0.8]\nroughness = 0.31622776\n"
           "[materials.glow]\nkind = \"emissive\"\nradiance = [10, 10, 10]\n"
           "[[shapes]]\nkind = \"box\"\nmin = [-50, -1, -50]\nmax = [50, 0, 50]\nmaterial = \"floor\"\n"
           "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0.7155418, -1.4310835]\nradius = 0.3\nmaterial = \"glow\"\n";
}

}  // namespace

TEST_CASE("MIS: a glossy floor's highlight of a firefly converges to the light's integral, each strategy a large share") {
    // The floor's radiance toward the camera is the light's integral of
    // f |cos| L_e over its cone: nothing is between them, and nothing else
    // is lit. The probe works it out on the GPU from the BSDF and the
    // emitter themselves, by the midpoint rule over the cone at 4 x 4
    // points in each pixel, and the share of it step 5's weighted light
    // samples carry. The render must converge to the integral, to 1.5%
    // (65,536 frames leave some 0.5% of noise at these pixels). At each of
    // them each strategy carries between a third and two thirds of it, so a
    // bounce's light counted in full (1 + share) or not at all (share), or
    // a light sample unweighted, is off by a third or more.
    constexpr frame::Extent size{64, 48};
    Rig rig(glossy_floor(), size);
    const contracts::CameraData c = camera::shader_form(rig.description().camera, size);
    const contracts::Bsdf bsdf = floor_bsdf(contracts::BsdfKind::conductor, glossy_roughness * glossy_roughness);
    constexpr std::uint32_t within = 4;  // points a side in each pixel
    constexpr std::uint32_t cone_grid = 128;
    const std::array<tests::Pixel, 6> pixels{{{32, 24}, {32, 16}, {36, 24}, {28, 20}, {32, 20}, {35, 28}}};
    std::vector<tests::DirectQuery> queries;
    for (const tests::Pixel pixel : pixels) {
        for (std::uint32_t j = 0; j < within; ++j) {
            for (std::uint32_t i = 0; i < within; ++i) {
                const Vec3 d = tests::camera_direction(c, pixel.x + (i + 0.5) / within, pixel.y + (j + 0.5) / within,
                                                       size);
                const Vec3 p = tests::vec(c.origin) + (-c.origin.y / d.y) * d;
                queries.push_back({{static_cast<float>(p.x), 0.0f, static_cast<float>(p.z)},
                                   cone_grid,
                                   {static_cast<float>(-d.x), static_cast<float>(-d.y), static_cast<float>(-d.z)},
                                   1u,
                                   {0.0f, 1.0f, 0.0f},
                                   0u});
            }
        }
    }
    tests::ProbeRunner gpu;
    const auto count = static_cast<std::uint32_t>(queries.size());
    const std::vector<tests::DirectIntegral> integrals = tests::run_with_light<tests::DirectIntegral>(
        gpu, "light_integral", count, highlight_light,
        {tests::Binding::of(queries), tests::Binding::of(count), tests::Binding::of(bsdf)});

    (void)rig.render(65536);
    const std::vector<float> image = rig.accumulated();
    for (std::size_t k = 0; k < pixels.size(); ++k) {
        double expected = 0.0;
        double light_part = 0.0;
        for (std::size_t q = k * within * within; q < (k + 1) * within * within; ++q) {
            expected += integrals[q].direct;
            light_part += static_cast<double>(integrals[q].direct) * integrals[q].light_share;
        }
        const double share = light_part / expected;
        expected /= within * within;
        const double rendered = image[(std::size_t{pixels[k].y} * size.width + pixels[k].x) * 4];
        INFO("pixel " << pixels[k].x << ", " << pixels[k].y << ": rendered " << rendered << ", expected " << expected
                      << ", the light samples' share " << share);
        MESSAGE("pixel " << pixels[k].x << ", " << pixels[k].y << ": rendered " << rendered << ", expected "
                         << expected << ", the light samples' share " << share);
        CHECK(share > 1.0 / 3.0);
        CHECK(share < 2.0 / 3.0);
        CHECK(rendered == doctest::Approx(expected).scale(0).epsilon(0.015));
    }
}

namespace {

// A conductor floor, alpha 0.1, seen from (0, 0.1, 0.1) through a lens of
// 10 degrees, and a firefly 3 mm across 2 cm along the direction the floor
// at the camera's middle mirrors it to: the floor's highlight of it, some
// twenty pixels across, the light close enough that the step a ray leaves
// the floor from moves its cone. near_highlight_light
// is the scene's firefly, as the probe binds it.
const tests::ProbeLight near_highlight_light{{0.0f, 0.014142136f, -0.014142136f}, 0.003f, 30.0f};

std::string near_glossy_floor() {
    return camera_text("[0, 0.1, 0.1]", "[0, 0, 0]", 10) + sky("[0, 0, 0]", "[0, 0, 0]") +
           "[materials.floor]\nkind = \"conductor\"\nf0 = [0.8, 0.8, 0.8]\nroughness = 0.31622776\n"
           "[materials.glow]\nkind = \"emissive\"\nradiance = [30, 30, 30]\n"
           "[[shapes]]\nkind = \"box\"\nmin = [-50, -1, -50]\nmax = [50, 0, 50]\nmaterial = \"floor\"\n"
           "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0.014142136, -0.014142136]\nradius = 0.003\n"
           "material = \"glow\"\n";
}

}  // namespace

TEST_CASE("MIS: a glossy floor's highlight of a firefly millimeters across converges to its integral from where rays leave") {
    // The review of 7751a65: a glossy lobe can show what a Lambert floor's
    // weights cancel, since there each strategy carries a large share. The
    // floor's radiance toward the camera is the light's integral of
    // f |cos| L_e over its cone, seen from the point the path tracer's rays
    // leave the floor from (path.metal.h, "One origin"), which the probe
    // works out on the GPU by the midpoint rule over the cone at 4 x 4
    // points in each pixel, and the share of it step 5's weighted light
    // samples carry: 0.76 to 0.92 at these pixels, the bounce's the rest.
    // The render must converge to it, to 1%. Seen from the floor's point the
    // integral is up to 0.6% away from it, as a light 2 cm off moves for the
    // 15 um step at the origin (in the message; 1.4 to 3.7% for the 0.1 mm
    // step before it). With step 5 drawn from the
    // floor's point, as 7751a65 drew it, the weights do not sum to 1 and the
    // render converges to neither integral: 1 to 2.4% off this one at four
    // of the pixels.
    constexpr frame::Extent size{64, 48};
    Rig rig(near_glossy_floor(), size);
    const contracts::CameraData c = camera::shader_form(rig.description().camera, size);
    const contracts::Bsdf bsdf = floor_bsdf(contracts::BsdfKind::conductor, glossy_roughness * glossy_roughness);
    constexpr std::uint32_t within = 4;  // points a side in each pixel
    constexpr std::uint32_t cone_grid = 128;
    const std::array<tests::Pixel, 6> pixels{{{32, 24}, {32, 20}, {32, 16}, {28, 20}, {32, 28}, {36, 28}}};
    std::vector<tests::DirectQuery> queries;
    for (const std::uint32_t at_point : {0u, 1u}) {
        for (const tests::Pixel pixel : pixels) {
            for (std::uint32_t j = 0; j < within; ++j) {
                for (std::uint32_t i = 0; i < within; ++i) {
                    const Vec3 d = tests::camera_direction(c, pixel.x + (i + 0.5) / within,
                                                           pixel.y + (j + 0.5) / within, size);
                    const Vec3 p = tests::vec(c.origin) + (-c.origin.y / d.y) * d;
                    queries.push_back({{static_cast<float>(p.x), 0.0f, static_cast<float>(p.z)},
                                       cone_grid,
                                       {static_cast<float>(-d.x), static_cast<float>(-d.y), static_cast<float>(-d.z)},
                                       1u,
                                       {0.0f, 1.0f, 0.0f},
                                       at_point});
                }
            }
        }
    }
    tests::ProbeRunner gpu;
    const auto count = static_cast<std::uint32_t>(queries.size());
    const std::vector<tests::DirectIntegral> integrals = tests::run_with_light<tests::DirectIntegral>(
        gpu, "light_integral", count, near_highlight_light,
        {tests::Binding::of(queries), tests::Binding::of(count), tests::Binding::of(bsdf)});

    (void)rig.render(65536);
    const std::vector<float> image = rig.accumulated();
    const std::size_t per_point = pixels.size() * within * within;
    for (std::size_t k = 0; k < pixels.size(); ++k) {
        double expected = 0.0;
        double at_point = 0.0;
        double light_part = 0.0;
        for (std::size_t q = k * within * within; q < (k + 1) * within * within; ++q) {
            expected += integrals[q].direct;
            at_point += integrals[per_point + q].direct;
            light_part += static_cast<double>(integrals[q].direct) * integrals[q].light_share;
        }
        const double share = light_part / expected;
        expected /= within * within;
        at_point /= within * within;
        const double rendered = image[(std::size_t{pixels[k].y} * size.width + pixels[k].x) * 4];
        INFO("pixel " << pixels[k].x << ", " << pixels[k].y << ": rendered " << rendered << ", expected " << expected
                      << ", from the floor's point " << at_point << ", the light samples' share " << share);
        MESSAGE("pixel " << pixels[k].x << ", " << pixels[k].y << ": rendered " << rendered << ", expected "
                         << expected << ", rendered / expected - 1 = " << (rendered / expected - 1.0) * 100.0
                         << "%; from the floor's point " << at_point << ", rendered / it - 1 = "
                         << (rendered / at_point - 1.0) * 100.0 << "%; the light samples' share " << share);
        CHECK(rendered == doctest::Approx(expected).scale(0).epsilon(0.01));
    }
}

TEST_CASE("MIS: a firefly a delta lobe mirrors is counted in full: a black coated sphere's coat shows F0 L") {
    // A coated sphere of black base, its coat of ior 1.5, and behind the
    // camera a large firefly of radiance 25 that the coat mirrors at the
    // sphere's middle: the coat's delta lobe reaches it, and step 3 counts
    // it in full, F0 L = 0.04 x 25 = 1 (the base, black, adds nothing). Had
    // the delta lobe's ray been weighted against the light's density, as a
    // glossy lobe's is, its weight would be some 10^-4: near 0. The coat is
    // drawn 4% of the time, so the mean of 25 pixels over 4096 frames has
    // some 1.6% of noise.
    const std::string text = camera_text("[0, 0, 4]", "[0, 0, 0]", 30) + sky("[0, 0, 0]", "[0, 0, 0]") +
                             "[materials.black]\nkind = \"coated\"\ncolor = [0, 0, 0]\nior = 1.5\n"
                             "[materials.glow]\nkind = \"emissive\"\nradiance = [25, 25, 25]\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 1\nmaterial = \"black\"\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 7]\nradius = 1.5\nmaterial = \"glow\"\n";
    Rig rig(text, square);
    (void)rig.render(4096);
    const std::vector<float> image = rig.accumulated();
    const Patch patch = middle_of(square, 2);
    double sum = 0.0;
    for (std::uint32_t y = patch.y - patch.half; y <= patch.y + patch.half; ++y) {
        for (std::uint32_t x = patch.x - patch.half; x <= patch.x + patch.half; ++x) {
            sum += image[(std::size_t{y} * square.width + x) * 4];
        }
    }
    const double mean = sum / ((2 * patch.half + 1) * (2 * patch.half + 1));
    MESSAGE("the coat's middle: " << mean << ", F0 L " << glass_f0 * 25.0);
    CHECK(mean == doctest::Approx(glass_f0 * 25.0).scale(0).epsilon(0.05));
}
