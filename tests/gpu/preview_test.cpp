// The preview (metal/passes/preview/preview.h), on the GPU, against what its
// header says it computes, worked out here on the CPU from first principles.
// Each scene is built so that the quantity checked has one value the
// sampling cannot change: a sky that is the same in every direction, a light
// that nothing hides, a point that nothing but a sphere hides.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "core/camera/pinhole.h"
#include "core/frame/graph_file.h"
#include "core/scene/scene.h"
#include "metal/device/device.h"
#include "metal/device/error.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"

using namespace serenity;

namespace {

struct Vec {
    double x, y, z;
};

Vec operator+(Vec a, Vec b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
Vec operator-(Vec a, Vec b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
Vec operator*(double s, Vec a) {
    return {s * a.x, s * a.y, s * a.z};
}
double dot(Vec a, Vec b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
Vec normalized(Vec a) {
    return (1.0 / std::sqrt(dot(a, a))) * a;
}
Vec vec(contracts::Float3 f) {
    return {f.x, f.y, f.z};
}

// The display transform in preview.h, to 8 bits.
std::array<int, 3> displayed(Vec linear) {
    const double largest = std::max({linear.x, linear.y, linear.z});
    const Vec c = largest > 1.0 ? (1.0 / largest) * linear : linear;
    const auto encode = [](double v) {
        v = std::clamp(v, 0.0, 1.0);
        const double e = v <= 0.0031308 ? 12.92 * v : 1.055 * std::pow(v, 1.0 / 2.4) - 0.055;
        return static_cast<int>(std::lround(e * 255.0));
    };
    return {encode(c.x), encode(c.y), encode(c.z)};
}

// The rotated grid of positions within a pixel (preview.metal).
constexpr double positions[4][2] = {{0.375, 0.125}, {0.875, 0.375}, {0.625, 0.875}, {0.125, 0.625}};

// The camera ray through `position`, in pixels, of an image of `size`
// (contracts/camera.h).
Vec ray_direction(const contracts::CameraData& c, double px, double py, frame::Extent size) {
    const double sx = 2.0 * px / size.width - 1.0;
    const double sy = 1.0 - 2.0 * py / size.height;
    return normalized(vec(c.forward) + sx * vec(c.right) + sy * vec(c.up));
}

struct Image {
    frame::Extent size;
    std::vector<std::uint8_t> rgba;

    std::array<int, 3> at(std::uint32_t x, std::uint32_t y) const {
        const std::size_t i = (std::size_t{y} * size.width + x) * 4;
        return {rgba[i], rgba[i + 1], rgba[i + 2]};
    }
};

Image render(const std::string& scene_text, frame::Extent size) {
    const scene::SceneDescription scene = scene::parse(scene_text, "test scene");
    metal::Device device;
    metal::Submission submission(device);
    metal::Offscreen target(device, submission, size);
    metal::Renderer renderer(device, submission, frame::parse_schedule("passes = [\"preview\"]\n", "test"), &scene);
    const auto sequence = metal::render_to_offscreen(submission, target, renderer,
                                                     frame::FrameInputs{frame::Seconds(0.0), 0, scene.camera});
    (void)submission.wait_until_complete(sequence);
    Image image{size, std::vector<std::uint8_t>(std::size_t{size.width} * size.height * 4)};
    target.read_rgba(image.rgba);
    return image;
}

void check_near(std::array<int, 3> actual, std::array<int, 3> expected, int tolerance) {
    INFO("actual " << actual[0] << ", " << actual[1] << ", " << actual[2] << "; expected " << expected[0] << ", "
                   << expected[1] << ", " << expected[2]);
    for (int c = 0; c < 3; ++c) {
        CHECK(std::abs(actual[c] - expected[c]) <= tolerance);
    }
}

// What a pixel that sees the floor's top (y = 0) shows, lit by `light`
// alone, unhidden, under a black sky: the mean over the pixel's positions of
// albedo L sin^2 cos (lights/sphere_light.h).
Vec lit_floor(const contracts::CameraData& framed, frame::Extent size, std::uint32_t x, std::uint32_t y,
              const lights::SphereLightData& light, Vec albedo) {
    Vec sum{0, 0, 0};
    for (const auto& p : positions) {
        const Vec d = ray_direction(framed, x + p[0], y + p[1], size);
        const Vec hit = vec(framed.origin) + (-framed.origin.y / d.y) * d;
        const Vec to_light = vec(light.center) - hit;
        const double distance2 = dot(to_light, to_light);
        const double sin2 = light.radius * light.radius / distance2;
        const double cos_t = to_light.y / std::sqrt(distance2);
        const double e = light.radiance.x * sin2 * cos_t;
        sum = sum + Vec{albedo.x * e, albedo.y * e, albedo.z * e};
    }
    return 0.25 * sum;
}

std::string camera_text(const char* position, const char* look_at, int fov) {
    return std::string("[camera]\nposition = ") + position + "\nlook_at = " + look_at +
           "\nvertical_fov_degrees = " + std::to_string(fov) + "\n";
}

std::string sky(const char* zenith, const char* horizon) {
    return std::string("[environment]\nkind = \"gradient\"\nzenith = ") + zenith + "\nhorizon = " + horizon + "\n";
}

}  // namespace

TEST_CASE("a ray that leaves the scene shows the sky in its direction") {
    // Looking up and away from a lone sphere behind the camera: every pixel
    // in the top rows sees the sky.
    const std::string text = camera_text("[0, 0, 0]", "[0, 1, -1]", 60) + sky("[0.1, 0.3, 0.9]", "[0.8, 0.5, 0.1]") +
                             "[materials.grey]\nkind = \"rough\"\ncolor = [0.5, 0.5, 0.5]\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 5]\nradius = 1\nmaterial = \"grey\"\n";
    const frame::Extent size{48, 32};
    const Image image = render(text, size);
    const scene::SceneDescription scene = scene::parse(text, "test scene");
    const contracts::CameraData framed = camera::shader_form(scene.camera, size);

    for (auto [x, y] : {std::pair{0u, 0u}, {24u, 3u}, {47u, 10u}}) {
        Vec sum{0, 0, 0};
        for (const auto& p : positions) {
            const Vec d = ray_direction(framed, x + p[0], y + p[1], size);
            const double t = std::clamp(d.y, 0.0, 1.0);
            const double s = t * t * (3.0 - 2.0 * t);
            const Vec zenith = vec(scene.environment.zenith);
            const Vec horizon = vec(scene.environment.horizon);
            sum = sum + (horizon + s * (zenith - horizon));
        }
        INFO("pixel " << x << ", " << y);
        check_near(image.at(x, y), displayed(0.25 * sum), 1);
    }
}

TEST_CASE("a glowing sphere shows its radiance, its hue kept when brighter than the display") {
    const std::string text = camera_text("[0, 0, 3]", "[0, 0, 0]", 30) + sky("[0, 0, 0]", "[0, 0, 0]") +
                             "[materials.glow]\nkind = \"emissive\"\nradiance = [8, 4, 2]\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 0.5\nmaterial = \"glow\"\n";
    const Image image = render(text, {33, 33});
    check_near(image.at(16, 16), displayed({1.0, 0.5, 0.25}), 0);
    check_near(image.at(0, 0), {0, 0, 0}, 0);  // the black sky beside it
}

TEST_CASE("a rough surface under an unhidden light: albedo times L sin^2 cos") {
    // A floor, a small light high above it, a black sky and nothing else to
    // hide the light: every shadow ray reaches it, so the fraction visible is
    // exactly 1 and the floor is the formula in lights/sphere_light.h.
    const std::string text = camera_text("[0, 2, 3]", "[0, 0, 0]", 50) + sky("[0, 0, 0]", "[0, 0, 0]") +
                             "[materials.floor]\nkind = \"rough\"\ncolor = [0.8, 0.6, 0.4]\n"
                             "[materials.glow]\nkind = \"emissive\"\nradiance = [200, 200, 200]\n"
                             "[[shapes]]\nkind = \"box\"\nmin = [-50, -1, -50]\nmax = [50, 0, 50]\n"
                             "material = \"floor\"\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0.3, 2.5, -0.5]\nradius = 0.05\n"
                             "material = \"glow\"\n";
    const frame::Extent size{64, 48};
    const Image image = render(text, size);
    const scene::SceneDescription scene = scene::parse(text, "test scene");
    const contracts::CameraData framed = camera::shader_form(scene.camera, size);
    const lights::SphereLightData light = scene.sphere_lights.at(0);

    for (auto [x, y] : {std::pair{32u, 30u}, {10u, 40u}, {50u, 26u}}) {
        INFO("pixel " << x << ", " << y);
        check_near(image.at(x, y), displayed(lit_floor(framed, size, x, y, light, {0.8, 0.6, 0.4})), 1);
    }
}

TEST_CASE("a point that a sphere hides wholly from the light is in shadow") {
    // The light straight above a sphere that is far wider than it: the floor
    // right below the sphere sees none of the light, and the black sky adds
    // nothing. Far from the sphere, the floor is lit.
    const std::string text = camera_text("[0, 3, 4]", "[0, 0, 0]", 50) + sky("[0, 0, 0]", "[0, 0, 0]") +
                             "[materials.floor]\nkind = \"rough\"\ncolor = [0.8, 0.8, 0.8]\n"
                             "[materials.glow]\nkind = \"emissive\"\nradiance = [500, 500, 500]\n"
                             "[[shapes]]\nkind = \"box\"\nmin = [-50, -1, -50]\nmax = [50, 0, 50]\n"
                             "material = \"floor\"\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 1.2, 0]\nradius = 0.4\n"
                             "material = \"floor\"\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 3, 0]\nradius = 0.02\nmaterial = \"glow\"\n";
    const frame::Extent size{96, 72};
    const Image image = render(text, size);
    const scene::SceneDescription scene = scene::parse(text, "test scene");
    const contracts::CameraData framed = camera::shader_form(scene.camera, size);

    // Find the pixel whose center sees the floor at the origin, under the
    // sphere, and one that sees it 1.5 to the side, outside the shadow.
    const auto pixel_of = [&](Vec point) {
        const Vec d = normalized(point - vec(framed.origin));
        const Vec f = vec(framed.forward);
        const Vec on_plane = (1.0 / dot(d, f)) * d;
        const double sx = dot(on_plane, vec(framed.right)) / dot(vec(framed.right), vec(framed.right));
        const double sy = dot(on_plane, vec(framed.up)) / dot(vec(framed.up), vec(framed.up));
        return std::pair{static_cast<std::uint32_t>((sx + 1.0) * 0.5 * size.width),
                         static_cast<std::uint32_t>((1.0 - sy) * 0.5 * size.height)};
    };
    // Just in front of the sphere's contact with the floor, where the camera
    // sees the floor, not the sphere.
    const auto [sx, sy] = pixel_of({0.0, 0.0, 0.25});
    check_near(image.at(sx, sy), {0, 0, 0}, 0);
    const auto [lx, ly] = pixel_of({1.5, 0.0, 0.0});
    check_near(image.at(lx, ly), displayed(lit_floor(framed, size, lx, ly, scene.sphere_lights.at(0), {0.8, 0.8, 0.8})),
               1);
}

TEST_CASE("glass: the Fresnel term splits a ray, and what is refracted goes on") {
    // A glass sphere in a sky of one color L, seen through its middle: the
    // ray reflects F0 of L where it enters and passes (1 - F0)^2 of L
    // through the two surfaces. F0 = ((1.5 - 1) / (1.5 + 1))^2 = 0.04.
    // At 40 degrees the sphere (14.5 degrees across its radius) leaves the
    // corners to the sky.
    const std::string text = camera_text("[0, 0, 4]", "[0, 0, 0]", 40) + sky("[0.5, 0.5, 0.5]", "[0.5, 0.5, 0.5]") +
                             "[materials.glass]\nkind = \"dielectric\"\nior = 1.5\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 1\nmaterial = \"glass\"\n";
    const Image image = render(text, {33, 33});
    const double f0 = 0.04;
    const double l = 0.5 * (f0 + (1.0 - f0) * (1.0 - f0));
    check_near(image.at(16, 16), displayed({l, l, l}), 1);
    check_near(image.at(0, 0), displayed({0.5, 0.5, 0.5}), 1);
}

TEST_CASE("metal: a near-mirror of f0 = 1 reflects a uniform sky as it is") {
    // Every reflected ray sees the sky's one color; with F = 1 and a
    // roughness this low, the masking term is 1 to within the tolerance.
    const std::string text = camera_text("[0, 0, 4]", "[0, 0, 0]", 20) + sky("[0.4, 0.4, 0.4]", "[0.4, 0.4, 0.4]") +
                             "[materials.mirror]\nkind = \"conductor\"\nf0 = [1, 1, 1]\nroughness = 0.05\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 1\nmaterial = \"mirror\"\n";
    const Image image = render(text, {33, 33});
    for (auto [x, y] : {std::pair{16u, 16u}, {12u, 18u}, {20u, 13u}}) {
        INFO("pixel " << x << ", " << y);
        check_near(image.at(x, y), displayed({0.4, 0.4, 0.4}), 1);
    }
}

TEST_CASE("a graph that reads a scene refuses to run without one, and a frame without a camera") {
    metal::Device device;
    metal::Submission submission(device);
    const frame::Schedule preview = frame::parse_schedule("passes = [\"preview\"]\n", "test");
    CHECK_THROWS_AS(metal::Renderer(device, submission, preview, nullptr), metal::Error);

    const scene::SceneDescription scene = scene::parse(
        camera_text("[0, 0, 3]", "[0, 0, 0]", 30) + sky("[0, 0, 0]", "[0, 0, 0]") +
            "[materials.grey]\nkind = \"rough\"\ncolor = [0.5, 0.5, 0.5]\n"
            "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 0.5\nmaterial = \"grey\"\n",
        "test scene");
    metal::Offscreen target(device, submission, {8, 8});
    metal::Renderer renderer(device, submission, preview, &scene);
    CHECK_THROWS_AS(metal::render_to_offscreen(submission, target, renderer,
                                               frame::FrameInputs{frame::Seconds(0.0), 0, std::nullopt}),
                    metal::Error);
}

TEST_CASE("start-up work is settled before the first frame, so it is never measured as one") {
    const scene::SceneDescription scene = scene::load(SERENITY_SCENES_DIR "/brass_sphere.toml");
    metal::Device device;
    metal::Submission submission(device);
    metal::Offscreen target(device, submission, {16, 16});
    metal::Renderer renderer(device, submission, frame::parse_schedule("passes = [\"preview\"]\n", "test"), &scene);
    const std::uint64_t first_frame = submission.next_sequence();
    CHECK(first_frame > 0);  // the acceleration structure's build came first

    std::vector<std::uint64_t> settled;
    for (std::uint64_t i = 0; i < 5; ++i) {
        const metal::FrameSlot frame = submission.begin();
        if (frame.settled) {
            settled.push_back(frame.settled->sequence);
            CHECK(frame.settled->gpu_end >= frame.settled->gpu_start);
        }
        renderer.record(frame, frame::FrameInputs{frame::Seconds(0.0), i, scene.camera}, target.texture(),
                        target.size());
        submission.commit();
    }
    for (const metal::Completed& done : submission.finish()) {
        settled.push_back(done.sequence);
    }
    // Every frame once, in order, and nothing else.
    REQUIRE(settled.size() == 5);
    for (std::uint64_t i = 0; i < 5; ++i) {
        CHECK(settled[i] == first_frame + i);
    }
}
