// The denoiser's features (contract 17, core/contracts/features.h) on the
// GPU, as the features pass writes them and the renderer reads them back:
//
//   - a known surface: a matte floor seen straight down, its albedo, its
//     normal and its depth, the camera's height; the sky beside it, albedo
//     1, no normal, depth 0; motion 0, to rounding, under a still camera;
//   - through glass and a mirror, primary surface replacement: the floor
//     seen through a glass ball is the Fresnel blend of the sky's albedo 1
//     and the floor's, its normal the blend of none and the floor's, its
//     depth the glass's; the floor seen in a mirror is the floor;
//   - motion vectors in NRD's and DLSS's convention, the previous position
//     less the current one, in pixels: a camera moved sideways moves the
//     whole floor by the analytic offset, and the sky not at all; a ball
//     that wanders, by the offset its motion's own positions at the two
//     times project to, and its view depth's change;
//   - the light of a graph with the features pass is the graph's without
//     it, to the bit.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "core/animation/motion.h"
#include "core/contracts/linear_image.h"
#include "core/frame/graph_file.h"
#include "core/scene/scene.h"
#include "gpu/support/rendering.h"
#include "metal/device/device.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"
#include "support/same_bits.h"
#include "support/scene_text.h"
#include "test_paths.h"

using namespace serenity;

namespace {

constexpr frame::Extent size{64, 48};

// The camera every scene here but the mirror's is seen through: 2 m above
// the origin, looking straight down, image up toward -z, 40 degrees high.
// So x runs right and z down the image, and a point at view depth z is
// f / z pixels from the middle per meter, f = H / (2 tan 20 degrees).
constexpr double height = 2.0;
const double focal = size.height / (2.0 * std::tan(20.0 * std::numbers::pi / 180.0));

std::string above() {
    return "[camera]\nposition = [0, 2, 0]\nlook_at = [0, 0, 0]\nup = [0, 0, -1]\nvertical_fov_degrees = 40\n";
}

// What a point that did not move moves, in pixels: float rounding, the
// point placed through its shape's transform and back, and each camera's
// projection (contract 17), some 10^-6 here.
constexpr float still = 1e-4f;

// A matte paint of `paint_color`, written from it.
constexpr std::array<float, 3> paint_color{0.2f, 0.4f, 0.6f};

std::string paint() {
    return "[materials.paint]\nkind = \"rough\"\ncolor = [" + tests::exact(paint_color[0]) + ", " +
           tests::exact(paint_color[1]) + ", " + tests::exact(paint_color[2]) + "]\n";
}

frame::Schedule features_graph() {
    return frame::parse_schedule("passes = [\"features\"]\n", "features graph");
}

// One frame's features, rendered on a renderer of its own and read back.
metal::FeatureImages::Read features_of(const scene::SceneDescription& description, const frame::FrameInputs& inputs) {
    metal::Device device;
    metal::Submission submission(device);
    metal::Offscreen target(device, submission, size);
    metal::Renderer renderer(device, submission, features_graph(), &description);
    (void)submission.wait_until_complete(metal::render_to_offscreen(submission, target, renderer, inputs));
    return renderer.read_features();
}

frame::FrameInputs at(double seconds, const contracts::Camera& camera,
                      std::optional<frame::PreviousFrame> previous = std::nullopt) {
    frame::FrameInputs inputs = tests::frame_at(7, 7, seconds, camera);
    inputs.previous = previous;
    return inputs;
}

// Channel c of pixel (x, y) of `image`.
float value(const contracts::LinearImage& image, std::uint32_t x, std::uint32_t y, std::size_t c) {
    return image.rgb[(std::size_t{y} * image.extent.width + x) * 3 + c];
}

}  // namespace

TEST_CASE("features: a matte floor's albedo, normal and depth, and the sky's") {
    // A floor 1 m wide in x under the camera: the middle columns see it, the
    // outer ones (|x| > 0.5 m, 0.97 m at the image's edge) the sky.
    const scene::SceneDescription description = scene::parse(
        above() + tests::sky("[0, 0, 0]", "[0, 0, 0]") + paint() +
            "[[shapes]]\nkind = \"box\"\nmin = [-0.5, -0.1, -1]\nmax = [0.5, 0, 1]\nmaterial = \"paint\"\n",
        "floor");
    // The frame before at the same instant, through the same camera.
    const metal::FeatureImages::Read read =
        features_of(description, at(0.0, description.camera, frame::PreviousFrame{frame::Seconds{0.0},
                                                                                   description.camera}));
    REQUIRE(read.albedo.extent == size);
    std::size_t floor = 0;
    std::size_t sky = 0;
    for (std::uint32_t y = 0; y < size.height; ++y) {
        for (std::uint32_t x = 0; x < size.width; ++x) {
            INFO("pixel " << x << ", " << y);
            // Motion 0, to float rounding: the same point through the same
            // camera (contract 17).
            for (std::size_t c = 0; c < 3; ++c) {
                REQUIRE(std::abs(value(read.motion, x, y, c)) < still);
            }
            if (x >= 16 && x <= 47) {  // the floor, whatever the jitter (it spans 15.52 to 48.48)
                ++floor;
                for (std::size_t c = 0; c < 3; ++c) {
                    REQUIRE(value(read.albedo, x, y, c) == doctest::Approx(paint_color[c]).epsilon(1e-3));
                    REQUIRE(value(read.normal, x, y, c) == doctest::Approx(c == 1 ? 1.0 : 0.0).epsilon(1e-3));
                    REQUIRE(value(read.depth, x, y, c) == doctest::Approx(height).epsilon(1e-6));
                }
            } else if (x <= 14 || x >= 49) {
                ++sky;
                for (std::size_t c = 0; c < 3; ++c) {
                    REQUIRE(value(read.albedo, x, y, c) == 1.0f);
                    REQUIRE(value(read.normal, x, y, c) == 0.0f);
                    REQUIRE(value(read.depth, x, y, c) == 0.0f);
                }
            }
        }
    }
    CHECK(floor == 32u * size.height);
    CHECK(sky == 30u * size.height);
}

TEST_CASE("features: a floor seen through glass and in a mirror reports the floor (primary surface replacement)") {
    SUBCASE("through a glass ball: the Fresnel blend of the sky above and the floor below") {
        // A glass ball of radius 0.4 at 0.6 m over the floor: the middle of
        // the image sees its top, 1 m below the camera, near normal
        // incidence, where F = ((1.5 - 1) / (1.5 + 1))^2 = 0.04 to within
        // 10^-4 for the pixels below. Reflected, straight up to the black
        // sky (albedo 1, no normal); transmitted, through the ball to the
        // floor under it.
        const scene::SceneDescription description = scene::parse(
            above() + tests::sky("[0, 0, 0]", "[0, 0, 0]") + paint() +
                "[materials.glass]\nkind = \"dielectric\"\nior = 1.5\n"
                "[[shapes]]\nkind = \"box\"\nmin = [-1, -0.1, -1]\nmax = [1, 0, 1]\nmaterial = \"paint\"\n"
                "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0.6, 0]\nradius = 0.4\nmaterial = \"glass\"\n",
            "glass over a floor");
        const metal::FeatureImages::Read read = features_of(description, at(0.0, description.camera));
        constexpr double f = 0.04;
        for (std::uint32_t y = 21; y <= 26; ++y) {
            for (std::uint32_t x = 29; x <= 34; ++x) {
                INFO("pixel " << x << ", " << y);
                for (std::size_t c = 0; c < 3; ++c) {
                    CHECK(value(read.albedo, x, y, c) ==
                          doctest::Approx(f + (1.0 - f) * paint_color[c]).epsilon(2e-3));
                    CHECK(value(read.normal, x, y, c) == doctest::Approx(c == 1 ? 1.0 - f : 0.0).epsilon(2e-3));
                }
                // The glass's own depth, not the floor's: 1 m at the top,
                // under 1.01 m within these pixels.
                CHECK(value(read.depth, x, y, 0) == doctest::Approx(1.0).epsilon(1e-2));
            }
        }
    }
    SUBCASE("in a mirror: the floor it reflects") {
        // A mirror wall (steel smooth enough to be one, f0 1) facing +z, the
        // camera at (0, 1, 2) looking at (0, 0.5, 0) on it: the middle ray
        // reflects down onto the floor at (0, 0, 2).
        const scene::SceneDescription description = scene::parse(
            std::string("[camera]\nposition = [0, 1, 2]\nlook_at = [0, 0.5, 0]\nvertical_fov_degrees = 20\n") +
                tests::sky("[0, 0, 0]", "[0, 0, 0]") + paint() +
                "[materials.mirror]\nkind = \"conductor\"\nf0 = [1, 1, 1]\nroughness = 0.05\n"
                "[[shapes]]\nkind = \"box\"\nmin = [-3, -0.1, -3]\nmax = [3, 0, 6]\nmaterial = \"paint\"\n"
                "[[shapes]]\nkind = \"box\"\nmin = [-3, 0, -0.2]\nmax = [3, 3, 0]\nmaterial = \"mirror\"\n",
            "a mirror over a floor");
        const metal::FeatureImages::Read read = features_of(description, at(0.0, description.camera));
        const double wall = std::sqrt(0.5 * 0.5 + 2.0 * 2.0);  // the camera to (0, 0.5, 0), along its axis
        for (std::uint32_t y = 21; y <= 26; ++y) {
            for (std::uint32_t x = 29; x <= 34; ++x) {
                INFO("pixel " << x << ", " << y);
                for (std::size_t c = 0; c < 3; ++c) {
                    CHECK(value(read.albedo, x, y, c) == doctest::Approx(paint_color[c]).epsilon(1e-3));
                    CHECK(value(read.normal, x, y, c) == doctest::Approx(c == 1 ? 1.0 : 0.0).epsilon(1e-3));
                }
                CHECK(value(read.depth, x, y, 0) == doctest::Approx(wall).epsilon(2e-2));
            }
        }
    }
}

TEST_CASE("features: a camera moved sideways moves the floor by the analytic offset, the sky not at all") {
    // The camera now 0.1 m right of where it was: every floor point is f x
    // 0.1 / 2 pixels further left now than it was, so the previous position
    // less the current one is +3.30 pixels in x, 0 in y and in depth. The
    // sky, at infinity, does not move under a camera that only moves.
    const scene::SceneDescription description = scene::parse(
        above() + tests::sky("[0, 0, 0]", "[0, 0, 0]") + paint() +
            "[[shapes]]\nkind = \"box\"\nmin = [-0.5, -0.1, -1]\nmax = [0.5, 0, 1]\nmaterial = \"paint\"\n",
        "floor");
    contracts::Camera before = description.camera;
    before.position.x -= 0.1f;
    before.look_at.x -= 0.1f;
    const metal::FeatureImages::Read read =
        features_of(description, at(0.0, description.camera, frame::PreviousFrame{frame::Seconds{0.0}, before}));
    const double expected = focal * 0.1 / height;
    for (std::uint32_t y = 0; y < size.height; ++y) {
        for (std::uint32_t x = 0; x < size.width; ++x) {
            INFO("pixel " << x << ", " << y);
            if (x >= 16 && x <= 47) {
                REQUIRE(value(read.motion, x, y, 0) == doctest::Approx(expected).epsilon(3e-3));
                REQUIRE(std::abs(value(read.motion, x, y, 1)) < 1e-3f);
                REQUIRE(std::abs(value(read.motion, x, y, 2)) < 1e-5f);
            } else if (x <= 14 || x >= 49) {
                REQUIRE(std::abs(value(read.motion, x, y, 0)) < still);
                REQUIRE(std::abs(value(read.motion, x, y, 1)) < still);
            }
        }
    }
}

TEST_CASE("features: a wandering ball's motion is where its motion placed it a frame before") {
    // A matte ball of radius 0.2 wandering about (0, 0.5, 0), fast enough to
    // move several pixels in a sixtieth of a second, over a floor, the
    // camera still. A point P of the ball now, at view depth z, was at
    // P - d a frame before, d its center's move, at depth z + d_y: so its
    // previous position less its current one, with u its current place
    // right of the image's middle and v below it, is
    //   dx = (u z - f d_x) / (z + d_y) - u,  dy = (v z - f d_z) / (z + d_y) - v,
    // and its depth's change d_y. u and v are taken at the pixel's middle;
    // the sample's jitter, within half a pixel of it, moves the answer by at
    // most |d_y| / (2 z) pixels, which the tolerance allows.
    const scene::SceneDescription description = scene::parse(
        above() + tests::sky("[0, 0, 0]", "[0, 0, 0]") + paint() +
            "[[shapes]]\nkind = \"box\"\nmin = [-2, -0.1, -2]\nmax = [2, 0, 2]\nmaterial = \"paint\"\n"
            "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0.5, 0]\nradius = 0.2\nmaterial = \"paint\"\n"
            "motion = { kind = \"wander\", reach = 0.1, speed = 4, seed = 3 }\n",
        "a wandering ball");
    REQUIRE(description.animation.movers.size() == 1);
    constexpr double now = 1.0;
    constexpr double then = now - 1.0 / 60.0;
    const animation::MotionRecord motion = description.animation.movers.front().motion;
    const contracts::Float3 a = animation::position(description.animation.motions, motion, frame::Seconds{then});
    const contracts::Float3 b = animation::position(description.animation.motions, motion, frame::Seconds{now});
    const double dx = double{b.x} - a.x;
    const double dy = double{b.y} - a.y;
    const double dz = double{b.z} - a.z;
    INFO("the ball moved " << dx << ", " << dy << ", " << dz);
    REQUIRE(std::hypot(dx, dz) * focal / height > 1.0);  // a move of more than a pixel

    const metal::FeatureImages::Read read =
        features_of(description, at(now, description.camera, frame::PreviousFrame{frame::Seconds{then},
                                                                                   description.camera}));
    std::size_t on_ball = 0;
    for (std::uint32_t y = 0; y < size.height; ++y) {
        for (std::uint32_t x = 0; x < size.width; ++x) {
            INFO("pixel " << x << ", " << y);
            const double z = value(read.depth, x, y, 0);
            if (std::abs(z - height) < 1e-4) {
                // The still floor, under a still camera.
                REQUIRE(std::abs(value(read.motion, x, y, 0)) < still);
                REQUIRE(std::abs(value(read.motion, x, y, 1)) < still);
                continue;
            }
            ++on_ball;
            const double u = x + 0.5 - size.width / 2.0;
            const double v = y + 0.5 - size.height / 2.0;
            const double tolerance = std::abs(dy) / (2.0 * z) + 0.01;
            REQUIRE(std::abs(value(read.motion, x, y, 0) - ((u * z - focal * dx) / (z + dy) - u)) < tolerance);
            REQUIRE(std::abs(value(read.motion, x, y, 1) - ((v * z - focal * dz) / (z + dy) - v)) < tolerance);
            REQUIRE(std::abs(value(read.motion, x, y, 2) - dy) < 2e-4);
        }
    }
    CHECK(on_ball > 100u);
}

TEST_CASE("features: a graph's light is the same to the bit with the features pass beside it") {
    // ReSTIR PT, its reuse running across three frames of the moving brass
    // scene, with and without the features pass.
    const scene::SceneDescription description = scene::load(tests::scenes_dir / "brass_sphere_wander.toml");
    const auto accumulated = [&](const char* graph) {
        metal::Device device;
        metal::Submission submission(device);
        metal::Offscreen target(device, submission, size);
        metal::Renderer renderer(device, submission, frame::load_schedule(tests::graphs_dir / graph), &description);
        std::uint64_t last = 0;
        for (std::uint64_t i = 0; i < 3; ++i) {
            // Its reuse history kept from frame 0 (reused_since, the default).
            const frame::FrameInputs inputs =
                tests::frame_at(i, i, static_cast<double>(i) / 60.0, description.camera);
            last = metal::render_to_offscreen(submission, target, renderer, inputs);
        }
        (void)submission.wait_until_complete(last);
        CHECK(renderer.non_finite_samples() == 0);
        std::vector<float> image(std::size_t{size.width} * size.height * 4);
        renderer.read_accumulated(image);
        return image;
    };
    const std::vector<float> without = accumulated("restir_pt.toml");
    const std::vector<float> with = accumulated("restir_pt_features.toml");
    CHECK(tests::same_bits(with, without));
    // And it holds light: not two black images alike.
    CHECK(std::ranges::max(with) > 0.0f);
}
