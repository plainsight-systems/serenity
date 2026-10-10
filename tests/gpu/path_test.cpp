// The naive path tracer (metal/integrator/path.metal.h, metal/passes/path/),
// on the GPU, converged over frames and checked against what it must
// converge to, worked out here from first principles:
//
//   - furnaces: under a sky of one radiance L, a white Lambert sphere and a
//     glass sphere each reflect and pass every bit of light they receive, so
//     every pixel converges to L. A path tracer that loses light (a depth
//     limit, a wrong weight) or makes it (a doubled count) shows it here,
//     through the roulette, the BSDF sampling and the mean together;
//   - next event estimation: a floor under one firefly, and under two, in a
//     black sky, converges to albedo L sin^2 cos summed over the lights:
//     the uniform selection's 1 / N, the emitter's pdf, and principle 7 (a
//     bounce that reaches a firefly adds nothing) all at once;
//   - the accumulated image: what it holds, what it refuses, and starting
//     over;
//   - determinism: the same inputs give the same bytes.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "core/camera/thin_lens.h"
#include "core/frame/graph_file.h"
#include "core/scene/scene.h"
#include "metal/device/device.h"
#include "metal/device/error.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"

using namespace serenity;

namespace {

frame::Schedule path_graph() {
    return frame::parse_schedule("passes = [\"path\", \"display\"]\n", "test");
}

// A renderer of the path graph over `text`, at `size`, rendering frames that
// accumulate from frame 0.
struct Rig {
    scene::SceneDescription scene;
    frame::Extent size;
    metal::Device device;
    metal::Submission submission{device};
    metal::Offscreen target{device, submission, size};
    metal::Renderer renderer{device, submission, path_graph(), &scene};
    std::uint64_t next = 0;

    Rig(const std::string& text, frame::Extent image) : scene(scene::parse(text, "test scene")), size(image) {}

    // Renders `frames` more frames and returns the last, as displayed.
    std::vector<std::uint8_t> render(std::uint64_t frames) {
        std::uint64_t sequence = 0;
        for (std::uint64_t i = 0; i < frames; ++i, ++next) {
            sequence = metal::render_to_offscreen(
                submission, target, renderer,
                frame::FrameInputs{.time = frame::Seconds(0.0), .index = next, .accumulated_since = 0,
                                   .camera = scene.camera});
        }
        (void)submission.wait_until_complete(sequence);
        std::vector<std::uint8_t> rgba(std::size_t{size.width} * size.height * 4);
        target.read_rgba(rgba);
        return rgba;
    }
};

// The display's 8 bits back to linear.
double linear(double displayed_value) {
    const double c = displayed_value / 255.0;
    return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
}

std::string camera_text(const char* position, const char* look_at, int fov) {
    return std::string("[camera]\nposition = ") + position + "\nlook_at = " + look_at +
           "\nvertical_fov_degrees = " + std::to_string(fov) + "\n";
}

std::string sky(const char* zenith, const char* horizon) {
    return std::string("[environment]\nkind = \"gradient\"\nzenith = ") + zenith + "\nhorizon = " + horizon + "\n";
}

// The mean linear radiance, red channel, over a square of pixels.
double mean_over(const std::vector<std::uint8_t>& rgba, frame::Extent size, std::uint32_t cx, std::uint32_t cy,
                 std::uint32_t half) {
    double sum = 0.0;
    std::uint32_t count = 0;
    for (std::uint32_t y = cy - half; y <= cy + half; ++y) {
        for (std::uint32_t x = cx - half; x <= cx + half; ++x) {
            sum += linear(rgba[(std::size_t{y} * size.width + x) * 4]);
            ++count;
        }
    }
    return sum / count;
}

}  // namespace

TEST_CASE("furnace: a white Lambert sphere under a uniform sky converges to the sky") {
    // Albedo 1 and a sky of 0.5 every way: what the sphere receives it
    // returns, after however many bounces the roulette allows.
    const std::string text = camera_text("[0, 0, 4]", "[0, 0, 0]", 30) + sky("[0.5, 0.5, 0.5]", "[0.5, 0.5, 0.5]") +
                             "[materials.white]\nkind = \"rough\"\ncolor = [1, 1, 1]\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 1\nmaterial = \"white\"\n";
    Rig rig(text, {64, 64});
    const auto image = rig.render(1024);
    // The sphere's middle, and the sky beside it.
    CHECK(mean_over(image, rig.size, 32, 32, 6) == doctest::Approx(0.5).epsilon(0.02));
    CHECK(mean_over(image, rig.size, 3, 3, 2) == doctest::Approx(0.5).epsilon(0.01));
}

TEST_CASE("integrating sphere: light bounced many times inside a white sphere converges to its closed form") {
    // Inside a hollow sphere of radius R and reflectance rho, a diffuse
    // reflection spreads evenly over the whole inner wall (the form factor
    // between any two points of a sphere is the same), so a small light of
    // radiance L and radius r at the centre gives every wall point
    //   direct irradiance  pi L (r / R)^2,  and with every bounce after it
    //   total              pi L (r / R)^2 / (1 - rho),
    // and the wall's radiance is rho / pi of that: rho L (r / R)^2 / (1 - rho).
    // With rho = 0.8, four fifths of it has bounced at least twice, so the
    // roulette, the BSDF sampling and principle 7 (a bounce that reaches the
    // light adds nothing) are all in play. The light itself absorbs what
    // reaches it, a part in a few hundred per bounce, inside the tolerance.
    const std::string text = camera_text("[0, 0, 0.5]", "[0, 0, 1]", 30) + sky("[0, 0, 0]", "[0, 0, 0]") +
                             "[materials.wall]\nkind = \"rough\"\ncolor = [0.8, 0.8, 0.8]\n"
                             "[materials.glow]\nkind = \"emissive\"\nradiance = [50, 50, 50]\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 1\nmaterial = \"wall\"\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 0.05\nmaterial = \"glow\"\n";
    Rig rig(text, {64, 64});
    const auto image = rig.render(1024);
    const double expected = 0.8 * 50.0 * (0.05 * 0.05) / (1.0 - 0.8);  // 0.5
    CHECK(mean_over(image, rig.size, 32, 32, 8) == doctest::Approx(expected).epsilon(0.03));
}

TEST_CASE("furnace: a glass sphere under a uniform sky converges to the sky") {
    // Smooth glass loses nothing: reflected or refracted, every path leaves
    // with the sky's radiance, the 1 / eta^2 of entering undone by leaving.
    const std::string text = camera_text("[0, 0, 4]", "[0, 0, 0]", 30) + sky("[0.5, 0.5, 0.5]", "[0.5, 0.5, 0.5]") +
                             "[materials.glass]\nkind = \"dielectric\"\nior = 1.5\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 1\nmaterial = \"glass\"\n";
    Rig rig(text, {64, 64});
    const auto image = rig.render(512);
    CHECK(mean_over(image, rig.size, 32, 32, 6) == doctest::Approx(0.5).epsilon(0.02));
}

namespace {

// A floor under the given fireflies, in a black sky: the floor's only light
// is straight from them (a bounce off the floor reaches only the black sky
// or a firefly, which adds nothing, principle 7).
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

// albedo x L sin^2 cos, summed over the lights, at the floor point the
// center of pixel (x, y) sees.
double expected_floor(const scene::SceneDescription& scene, frame::Extent size, std::uint32_t x, std::uint32_t y) {
    const contracts::CameraData c = camera::shader_form(scene.camera, size);
    const double sx = 2.0 * (x + 0.5) / size.width - 1.0;
    const double sy = 1.0 - 2.0 * (y + 0.5) / size.height;
    const std::array<double, 3> d = {c.forward.x + sx * c.right.x + sy * c.up.x,
                                     c.forward.y + sx * c.right.y + sy * c.up.y,
                                     c.forward.z + sx * c.right.z + sy * c.up.z};
    const double t = -c.origin.y / d[1];
    const std::array<double, 3> p = {c.origin.x + t * d[0], 0.0, c.origin.z + t * d[2]};
    double sum = 0.0;
    for (const lights::SphereLightData& light : scene.sphere_lights) {
        // Where its shape is, and how big: the transform's translation and
        // scale (contracts/transform.h).
        const contracts::Transform& placed = scene.shapes.transforms.at(light.shape);
        const contracts::Float3 center = contracts::translation(placed);
        const double radius = placed.m[0][0];
        const std::array<double, 3> to = {center.x - p[0], center.y - p[1], center.z - p[2]};
        const double d2 = to[0] * to[0] + to[1] * to[1] + to[2] * to[2];
        sum += 0.8 * light.radiance.x * (radius * radius / d2) * (to[1] / std::sqrt(d2));
    }
    return sum;
}

void check_floor(const std::string& text, std::uint64_t frames) {
    Rig rig(text, {64, 48});
    const auto image = rig.render(frames);
    for (auto [x, y] : {std::pair{32u, 30u}, {14u, 38u}, {50u, 26u}}) {
        // A 3 x 3 patch's mean against the formula at its center: the
        // formula varies little over three pixels of floor.
        const double expected = expected_floor(rig.scene, rig.size, x, y);
        INFO("pixel " << x << ", " << y << ": expected " << expected);
        CHECK(mean_over(image, rig.size, x, y, 1) == doctest::Approx(expected).epsilon(0.03));
    }
}

}  // namespace

TEST_CASE("next event estimation: a floor under one firefly converges to albedo L sin^2 cos") {
    check_floor(lit_floor(firefly("[0.3, 2.5, -0.5]")), 512);
}

TEST_CASE("next event estimation: under two fireflies, chosen one at a time, the floor converges to their sum") {
    check_floor(lit_floor(firefly("[0.3, 2.5, -0.5]") + firefly("[-1.2, 1.5, 0.4]")), 1024);
}

TEST_CASE("the same inputs give the same image") {
    const std::string text = lit_floor(firefly("[0.3, 2.5, -0.5]"));
    Rig first(text, {32, 24});
    Rig second(text, {32, 24});
    CHECK(first.render(8) == second.render(8));
}

TEST_CASE("the accumulated image holds what the inputs claim, or the frame is refused") {
    const scene::SceneDescription scene = scene::parse(lit_floor(firefly("[0.3, 2.5, -0.5]")), "test scene");
    metal::Device device;
    metal::Submission submission(device);
    metal::Offscreen small(device, submission, {16, 16});
    metal::Offscreen large(device, submission, {24, 16});
    metal::Renderer renderer(device, submission, path_graph(), &scene);
    const auto inputs = [&](std::uint64_t index, std::uint64_t since) {
        return frame::FrameInputs{.time = frame::Seconds(0.0), .index = index, .accumulated_since = since,
                                  .camera = scene.camera};
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
    metal::Renderer fresh(device, submission, path_graph(), &scene);
    CHECK_THROWS_AS(fresh.prepare(inputs(frame::max_accumulated_frames + 1, 0), {16, 16}), metal::Error);
}

TEST_CASE("a frame recorded without being prepared is refused, and so is a graph with two accumulating passes") {
    const scene::SceneDescription scene = scene::parse(lit_floor(firefly("[0.3, 2.5, -0.5]")), "test scene");
    metal::Device device;
    metal::Submission submission(device);
    metal::Offscreen target(device, submission, {16, 16});
    metal::Renderer renderer(device, submission, path_graph(), &scene);
    const metal::FrameSlot frame = submission.begin();
    CHECK_THROWS_AS(renderer.record(frame,
                                    frame::FrameInputs{.time = frame::Seconds(0.0), .index = 0,
                                                       .accumulated_since = 0, .camera = scene.camera},
                                    target.texture(), target.size()),
                    metal::Error);

    metal::Submission other(device);
    const frame::Schedule twice{{frame::PassKind::path, frame::PassKind::path, frame::PassKind::display}, std::nullopt};
    CHECK_THROWS_AS(metal::Renderer(device, other, twice, &scene), metal::Error);
}

TEST_CASE("the camera sees a light's glow, through the emitter") {
    // A camera ray that meets a light counts its emission (step 3): the
    // emitter's radiance, (0.6, 0.3, 0.1), every frame alike.
    const std::string text = camera_text("[0, 0, 3]", "[0, 0, 0]", 30) + sky("[0, 0, 0]", "[0, 0, 0]") +
                             "[materials.glow]\nkind = \"emissive\"\nradiance = [0.6, 0.3, 0.1]\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 0.5\nmaterial = \"glow\"\n";
    Rig rig(text, {33, 33});
    const auto image = rig.render(4);
    CHECK(mean_over(image, rig.size, 16, 16, 2) == doctest::Approx(0.6).epsilon(0.01));
}

TEST_CASE("a frame time past what the shaders' float holds is refused before it is recorded") {
    const scene::SceneDescription scene = scene::parse(lit_floor(firefly("[0.3, 2.5, -0.5]")), "test scene");
    metal::Device device;
    metal::Submission submission(device);
    metal::Offscreen target(device, submission, {16, 16});
    metal::Renderer renderer(device, submission, path_graph(), &scene);
    for (double seconds : {1e100, -1e39, std::numeric_limits<double>::infinity()}) {
        INFO("time " << seconds);
        CHECK_THROWS_AS(metal::render_to_offscreen(submission, target, renderer,
                                                   frame::FrameInputs{.time = frame::Seconds(seconds), .index = 0,
                                                                      .camera = scene.camera}),
                        metal::Error);
    }
    const auto fine = metal::render_to_offscreen(
        submission, target, renderer,
        frame::FrameInputs{.time = frame::Seconds(3.0e38), .index = 0, .camera = scene.camera});
    (void)submission.wait_until_complete(fine);
}
