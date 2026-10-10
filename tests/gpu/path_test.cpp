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
//     the uniform selection's 1 / N, the emitter's pdf, and principle 7 (a
//     bounce that reaches a firefly adds nothing) all at once;
//   - the accumulated image: what it holds, what it refuses, and starting
//     over;
//   - principle 2: the same inputs give the same image, and frame k rendered
//     alone is frame k rendered after others.
//
// Every converged render also checks that no sample was left out for not
// being finite (metal/film/non_finite.h): such a sample is dropped from its
// pixel's mean, so a bug that made some paths NaN would still converge near
// the expected value, biased by the survivors.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "core/camera/thin_lens.h"
#include "core/frame/graph_file.h"
#include "core/scene/scene.h"
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
    // principle 7 (a bounce that reaches the light adds nothing) are all in
    // play: a path cut at 8 bounces would keep 1 - 0.9^8 = 57% of it. The
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
// center of pixel (x, y) sees.
double expected_floor(const scene::SceneDescription& description, frame::Extent size, std::uint32_t x,
                      std::uint32_t y) {
    const contracts::CameraData c = camera::shader_form(description.camera, size);
    const Vec3 d = tests::camera_direction(c, x + 0.5, y + 0.5, size);
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
        const double expected = expected_floor(rig.description(), rig.size(), pixel.x, pixel.y);
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

TEST_CASE("the same inputs give the same image") {
    // Compared as displayed, 8 bits a channel: the renderer gives the
    // accumulated float image to no reader but its passes, so a difference
    // below 1/255 that rounds the same is not seen (needs a readback of the
    // accumulated image, a src change). Every pixel's sample differs from
    // its neighbour's, so a difference anywhere in the frame is unlikely to
    // round the same in all of them.
    const std::string text = lit_floor(one_firefly);
    Rig first(text, {32, 24});
    Rig second(text, {32, 24});
    CHECK(tests::differing(first.render(8), second.render(8)) == 0);
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
        return tests::read_back(target);
    };
    // Each of frames 0 to 4 starting over; 5 to 7 accumulated from 5.
    const std::vector<std::uint8_t> after_others = run(0, 7, 5);
    const std::vector<std::uint8_t> alone = run(5, 7, 5);
    CHECK(tests::differing(after_others, alone) == 0);
    // And they are not every frame alike: frame 5 alone differs from 6 alone.
    CHECK(tests::differing(run(5, 5, 5), run(6, 6, 6)) > 0);
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
    CHECK_THROWS_AS(metal::render_to_offscreen(submission, small, renderer, inputs(3, 0)), metal::Error);
    // A size changed without starting over.
    CHECK_THROWS_AS(metal::render_to_offscreen(submission, large, renderer, inputs(2, 0)), metal::Error);
    // accumulated_since moved without starting over.
    CHECK_THROWS_AS(metal::render_to_offscreen(submission, small, renderer, inputs(2, 1)), metal::Error);
    // accumulated_since after the frame.
    CHECK_THROWS_AS(metal::render_to_offscreen(submission, small, renderer, inputs(2, 5)), metal::Error);
    // Starting over at a new size remakes the image, frames in flight
    // drained, and accumulates on from there.
    (void)metal::render_to_offscreen(submission, large, renderer, inputs(2, 2));
    const auto last = metal::render_to_offscreen(submission, large, renderer, inputs(3, 2));
    (void)submission.wait_until_complete(last);
    CHECK(renderer.non_finite_samples() == 0);

    // More frames than an image holds.
    metal::Renderer fresh(device, submission, tests::path_graph(), &description);
    CHECK_THROWS_AS(fresh.prepare(inputs(frame::max_accumulated_frames + 1, 0), {16, 16}), metal::Error);
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
                    metal::Error);

    metal::Submission other(device);
    const frame::Schedule twice{{frame::PassKind::path, frame::PassKind::path, frame::PassKind::display}, std::nullopt};
    CHECK_THROWS_AS(metal::Renderer(device, other, twice, &description), metal::Error);
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
    // Two lights at the most radiance a float holds, close over a floor:
    // next event estimation's L f cos / pdf, over the selection's 1 / 2,
    // overflows. Film's finite() test of the sample (metal/film/
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
                        metal::Error);
    }
    const auto fine =
        metal::render_to_offscreen(submission, target, renderer, tests::frame_at(0, 0, 3.0e38, description.camera));
    (void)submission.wait_until_complete(fine);
}
