// The preview (metal/passes/preview/preview.h), on the GPU, against what its
// header says it computes, worked out here on the CPU from first principles.
// Each scene is built so that the quantity checked has one value the
// sampling cannot change: a sky that is the same in every direction, a light
// that nothing hides, a point that nothing but a sphere hides.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
#include "test_paths.h"

using namespace serenity;
using tests::camera_text;
using tests::sky;
using tests::Vec3;

namespace {

using Rgb8 = std::array<int, 3>;

// The display transform in preview.h, to 8 bits: a color brighter than
// the display scaled to its largest channel, its hue kept.
Rgb8 displayed(Vec3 linear) {
    const double largest = std::max({linear.x, linear.y, linear.z});
    const Vec3 c = largest > 1.0 ? (1.0 / largest) * linear : linear;
    return {tests::srgb8(c.x), tests::srgb8(c.y), tests::srgb8(c.z)};
}

// The rotated grid of positions within a pixel (preview.metal).
constexpr std::array<std::array<double, 2>, 4> positions{
    {{0.375, 0.125}, {0.875, 0.375}, {0.625, 0.875}, {0.125, 0.625}}};

struct Image {
    frame::Extent size;
    std::vector<std::uint8_t> rgba;

    Rgb8 at(std::uint32_t x, std::uint32_t y) const {
        const std::size_t i = (std::size_t{y} * size.width + x) * 4;
        return {rgba[i], rgba[i + 1], rgba[i + 2]};
    }
};

Image render(const std::string& scene_text, frame::Extent size) {
    const scene::SceneDescription description = scene::parse(scene_text, "test scene");
    return {size, tests::render_once(description, tests::preview_graph(), size,
                                     tests::frame_at(0, 0, 0.0, description.camera))};
}

void check_near(Rgb8 actual, Rgb8 expected, int tolerance) {
    INFO("actual " << actual[0] << ", " << actual[1] << ", " << actual[2] << "; expected " << expected[0] << ", "
                   << expected[1] << ", " << expected[2]);
    for (std::size_t c = 0; c < 3; ++c) {
        CHECK(std::abs(actual[c] - expected[c]) <= tolerance);
    }
}

// What a pixel that sees the floor's top (y = 0) shows, lit by `light`
// alone, unhidden, under a black sky: the mean over the pixel's positions of
// albedo L sin^2 cos (lights/sphere_light.h). The light is where its shape's
// transform places the unit sphere: center its translation, radius its
// scale.
Vec3 lit_floor(const contracts::CameraData& framed, frame::Extent size, tests::Pixel pixel,
               const lights::SphereLightData& light, const contracts::Transform& placed, Vec3 albedo) {
    const Vec3 center = tests::vec(contracts::translation(placed));
    const double radius = placed.m[0][0];
    Vec3 sum{};
    for (const auto& p : positions) {
        const Vec3 d = tests::camera_direction(framed, pixel.x + p[0], pixel.y + p[1], size);
        const Vec3 hit = tests::vec(framed.origin) + (-framed.origin.y / d.y) * d;
        const Vec3 to_light = center - hit;
        const double distance2 = dot(to_light, to_light);
        const double sin2 = radius * radius / distance2;
        const double cos_t = to_light.y / std::sqrt(distance2);
        const double e = light.radiance.x * sin2 * cos_t;
        sum = sum + Vec3{albedo.x * e, albedo.y * e, albedo.z * e};
    }
    return (1.0 / positions.size()) * sum;
}

// The glass sphere's middle, under a sky of L every way: F0 of L reflected
// where the ray enters, and (1 - F0)^2 tau of L through both surfaces, tau
// the medium's transmittance along the way. F0 = ((1.5 - 1) / (1.5 + 1))^2.
constexpr double glass_f0 = 0.04;

double through_glass(double sky_radiance, double tau) {
    return sky_radiance * (glass_f0 + (1.0 - glass_f0) * (1.0 - glass_f0) * tau);
}

}  // namespace

TEST_CASE("a ray that leaves the scene shows the sky in its direction") {
    // Looking up and away from a lone sphere behind the camera: every pixel
    // in the top rows sees the sky.
    const std::string text = camera_text("[0, 0, 0]", "[0, 1, -1]", 60) + sky("[0.1, 0.3, 0.9]", "[0.8, 0.5, 0.1]") +
                             "[materials.grey]\nkind = \"rough\"\ncolor = [0.5, 0.5, 0.5]\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 5]\nradius = 1\nmaterial = \"grey\"\n";
    constexpr frame::Extent size{48, 32};
    const Image image = render(text, size);
    const scene::SceneDescription description = scene::parse(text, "test scene");
    const contracts::CameraData framed = camera::shader_form(description.camera, size);

    for (const tests::Pixel pixel : {tests::Pixel{0, 0}, tests::Pixel{24, 3}, tests::Pixel{47, 10}}) {
        Vec3 sum{};
        for (const auto& p : positions) {
            const Vec3 d = tests::camera_direction(framed, pixel.x + p[0], pixel.y + p[1], size);
            const double t = std::clamp(d.y, 0.0, 1.0);
            const double s = t * t * (3.0 - 2.0 * t);
            const Vec3 zenith = tests::vec(description.environment.zenith);
            const Vec3 horizon = tests::vec(description.environment.horizon);
            sum = sum + (horizon + s * (zenith - horizon));
        }
        INFO("pixel " << pixel.x << ", " << pixel.y);
        check_near(image.at(pixel.x, pixel.y), displayed((1.0 / positions.size()) * sum), 1);
    }
}

TEST_CASE("a glowing sphere shows its radiance, its hue kept when brighter than the display") {
    const std::string text = camera_text("[0, 0, 3]", "[0, 0, 0]", 30) + sky("[0, 0, 0]", "[0, 0, 0]") +
                             "[materials.glow]\nkind = \"emissive\"\nradiance = [8, 4, 2]\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 0.5\nmaterial = \"glow\"\n";
    constexpr frame::Extent size{33, 33};
    const Image image = render(text, size);
    check_near(image.at(size.width / 2, size.height / 2), displayed({1.0, 0.5, 0.25}), 0);
    check_near(image.at(0, 0), {0, 0, 0}, 0);  // the black sky beside it
}

TEST_CASE("a rough surface under an unhidden light: albedo times L sin^2 cos") {
    // A floor, a small light high above it, a black sky and nothing else to
    // hide the light: every shadow ray reaches it, so the fraction visible is
    // exactly 1 and the floor is the formula in lights/sphere_light.h.
    const Vec3 albedo{0.8, 0.6, 0.4};
    const std::string text = camera_text("[0, 2, 3]", "[0, 0, 0]", 50) + sky("[0, 0, 0]", "[0, 0, 0]") +
                             "[materials.floor]\nkind = \"rough\"\ncolor = [0.8, 0.6, 0.4]\n"
                             "[materials.glow]\nkind = \"emissive\"\nradiance = [200, 200, 200]\n"
                             "[[shapes]]\nkind = \"box\"\nmin = [-50, -1, -50]\nmax = [50, 0, 50]\n"
                             "material = \"floor\"\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0.3, 2.5, -0.5]\nradius = 0.05\n"
                             "material = \"glow\"\n";
    constexpr frame::Extent size{64, 48};
    const Image image = render(text, size);
    const scene::SceneDescription description = scene::parse(text, "test scene");
    const contracts::CameraData framed = camera::shader_form(description.camera, size);
    const lights::SphereLightData light = description.sphere_lights.at(0);
    const contracts::Transform& placed = description.shapes.transforms.at(light.shape);

    for (const tests::Pixel pixel : {tests::Pixel{32, 30}, tests::Pixel{10, 40}, tests::Pixel{50, 26}}) {
        INFO("pixel " << pixel.x << ", " << pixel.y);
        check_near(image.at(pixel.x, pixel.y), displayed(lit_floor(framed, size, pixel, light, placed, albedo)), 1);
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
    constexpr frame::Extent size{96, 72};
    const Image image = render(text, size);
    const scene::SceneDescription description = scene::parse(text, "test scene");
    const contracts::CameraData framed = camera::shader_form(description.camera, size);

    // Just in front of the sphere's contact with the floor, where the camera
    // sees the floor, not the sphere: in shadow.
    const tests::Pixel shadowed = tests::pixel_of(framed, {0.0, 0.0, 0.25}, size);
    check_near(image.at(shadowed.x, shadowed.y), {0, 0, 0}, 0);
    // 1.5 to the side, outside the shadow: lit.
    const tests::Pixel lit = tests::pixel_of(framed, {1.5, 0.0, 0.0}, size);
    const lights::SphereLightData& light = description.sphere_lights.at(0);
    check_near(image.at(lit.x, lit.y),
               displayed(lit_floor(framed, size, lit, light, description.shapes.transforms.at(light.shape),
                                   {0.8, 0.8, 0.8})),
               1);
}

TEST_CASE("glass: the Fresnel term splits a ray, and what is refracted goes on") {
    // A glass sphere in a sky of one color L, seen through its middle: the
    // ray reflects F0 of L where it enters and passes (1 - F0)^2 of L
    // through the two surfaces. At 40 degrees the sphere (14.5 degrees
    // across its radius) leaves the corners to the sky.
    const std::string text = camera_text("[0, 0, 4]", "[0, 0, 0]", 40) + sky("[0.5, 0.5, 0.5]", "[0.5, 0.5, 0.5]") +
                             "[materials.glass]\nkind = \"dielectric\"\nior = 1.5\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 1\nmaterial = \"glass\"\n";
    constexpr frame::Extent size{33, 33};
    const Image image = render(text, size);
    const double l = through_glass(0.5, 1.0);
    check_near(image.at(size.width / 2, size.height / 2), displayed({l, l, l}), 1);
    check_near(image.at(0, 0), displayed({0.5, 0.5, 0.5}), 1);
}

TEST_CASE("an absorbing medium at a bead's scale: its stretch measured from the surface, not the ray's start") {
    constexpr frame::Extent size{33, 33};
    const Image image = render(tests::tinted_bead(), size);
    const double clear = through_glass(0.5, 1.0);
    const double red = through_glass(0.5, std::exp(-1.4));  // 2 mm of 700 per meter
    check_near(image.at(size.width / 2, size.height / 2), displayed({red, clear, clear}), 1);
}

TEST_CASE("a light inside the same medium as the surface it lights: its light dimmed over the shadow ray") {
    constexpr frame::Extent size{65, 65};
    const Image image = render(tests::core_lit_inside_glass(), size);
    double red = 0.0;
    double green = 0.0;
    for (std::uint32_t y = size.height / 2 - 1; y <= size.height / 2 + 1; ++y) {
        for (std::uint32_t x = size.width / 2 - 1; x <= size.width / 2 + 1; ++x) {
            red += tests::linear_of(image.at(x, y)[0]);
            green += tests::linear_of(image.at(x, y)[1]);
        }
    }
    const double sigma = std::log(2.0);
    INFO("red over green " << red / green);
    const double kept = std::exp(-sigma * tests::camera_to_core) * std::exp(-sigma * tests::light_to_core);
    CHECK(red / green == doctest::Approx(kept).scale(0).epsilon(0.05));
}

TEST_CASE("a coated sphere of black base shows its coat's mirror: F0 of the sky at its middle") {
    // Its base takes nothing, so all it shows is the coat's reflection,
    // which the preview takes beside the base (direct.metal.h).
    const std::string text = camera_text("[0, 0, 4]", "[0, 0, 0]", 40) + sky("[1, 1, 1]", "[1, 1, 1]") +
                             "[materials.black]\nkind = \"coated\"\ncolor = [0, 0, 0]\nior = 1.5\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 1\nmaterial = \"black\"\n";
    constexpr frame::Extent size{33, 33};
    const Image image = render(text, size);
    check_near(image.at(size.width / 2, size.height / 2), displayed({glass_f0, glass_f0, glass_f0}), 1);
}

TEST_CASE("glass filled with an absorbing medium: the stretch inside keeps what Beer and Lambert say") {
    // As the clear sphere above, filled with a medium keeping half its red
    // light over 1 m: through the middle the ray crosses 2 m of it, tau =
    // 0.25 of red, and red is L (F0 + (1 - F0)^2 tau); green and blue as
    // clear glass.
    const std::string text = camera_text("[0, 0, 4]", "[0, 0, 0]", 40) + sky("[0.5, 0.5, 0.5]", "[0.5, 0.5, 0.5]") +
                             "[materials.glass]\nkind = \"dielectric\"\nior = 1.5\n"
                             "[media.red_out]\nkind = \"absorbing\"\ntint = [0.5, 1, 1]\ntint_distance = 1\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 1\nmaterial = \"glass\"\n"
                             "interior = \"red_out\"\n";
    constexpr frame::Extent size{33, 33};
    const Image image = render(text, size);
    const double clear = through_glass(0.5, 1.0);
    const double red = through_glass(0.5, 0.25);
    check_near(image.at(size.width / 2, size.height / 2), displayed({red, clear, clear}), 1);
}

TEST_CASE("metal: a near-mirror of f0 = 1 reflects a uniform sky as it is") {
    // Every reflected ray sees the sky's one color; with F = 1 and a
    // roughness this low, the masking term is 1 to within the tolerance.
    const std::string text = camera_text("[0, 0, 4]", "[0, 0, 0]", 20) + sky("[0.4, 0.4, 0.4]", "[0.4, 0.4, 0.4]") +
                             "[materials.mirror]\nkind = \"conductor\"\nf0 = [1, 1, 1]\nroughness = 0.05\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 1\nmaterial = \"mirror\"\n";
    const Image image = render(text, {33, 33});
    for (const tests::Pixel pixel : {tests::Pixel{16, 16}, tests::Pixel{12, 18}, tests::Pixel{20, 13}}) {
        INFO("pixel " << pixel.x << ", " << pixel.y);
        check_near(image.at(pixel.x, pixel.y), displayed({0.4, 0.4, 0.4}), 1);
    }
}

TEST_CASE("a graph that reads a scene refuses to run without one, and a frame without a camera") {
    metal::Device device;
    metal::Submission submission(device);
    CHECK_THROWS_AS(metal::Renderer(device, submission, tests::preview_graph(), nullptr), metal::MetalError);

    const scene::SceneDescription description = scene::parse(
        camera_text("[0, 0, 3]", "[0, 0, 0]", 30) + sky("[0, 0, 0]", "[0, 0, 0]") +
            "[materials.grey]\nkind = \"rough\"\ncolor = [0.5, 0.5, 0.5]\n"
            "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 0.5\nmaterial = \"grey\"\n",
        "test scene");
    metal::Offscreen target(device, submission, {8, 8});
    metal::Renderer renderer(device, submission, tests::preview_graph(), &description);
    CHECK_THROWS_AS(metal::render_to_offscreen(submission, target, renderer, tests::frame_at(0, 0, 0.0, std::nullopt)),
                    metal::MetalError);
}

TEST_CASE("start-up work is settled before the first frame, so it is never measured as one") {
    constexpr std::uint64_t frames = 5;
    const scene::SceneDescription description = scene::load(tests::scenes_dir / "brass_sphere.toml");
    metal::Device device;
    metal::Submission submission(device);
    metal::Offscreen target(device, submission, {16, 16});
    metal::Renderer renderer(device, submission, tests::preview_graph(), &description);
    const std::uint64_t first_frame = submission.next_sequence();
    CHECK(first_frame > 0);  // the acceleration structure's build came first

    std::vector<std::uint64_t> settled;
    for (std::uint64_t i = 0; i < frames; ++i) {
        const frame::FrameInputs inputs = tests::frame_at(i, i, 0.0, description.camera);
        renderer.prepare(inputs, target.size());
        const metal::FrameSlot slot = submission.begin();
        if (slot.settled) {
            settled.push_back(slot.settled->sequence);
            CHECK(slot.settled->gpu_end >= slot.settled->gpu_start);
        }
        renderer.record(slot, inputs, target.texture(), target.size());
        submission.commit();
    }
    for (const metal::Completed& done : submission.finish()) {
        settled.push_back(done.sequence);
    }
    // Every frame once, in order, and nothing else.
    REQUIRE(settled.size() == frames);
    for (std::uint64_t i = 0; i < frames; ++i) {
        CHECK(settled[i] == first_frame + i);
    }
}
