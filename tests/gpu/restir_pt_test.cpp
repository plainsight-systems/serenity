// ReSTIR PT, Phases 1 and 2 (metal/passes/restir_pt/restir_pt.metal.h), on
// the GPU, as the passes run it:
//
//   - initial sampling alone converges to the closed forms the path tests
//     hold the path tracer to (tests/gpu/path_test.cpp), within their
//     tolerances: a floor under one firefly and under two, and the white,
//     glass furnaces (the sky reached after one bounce, and only through
//     delta lobes);
//   - spatial reuse keeps those means: the floor under a firefly, and the
//     glass furnace, whose paths shift by replay alone; and temporal reuse,
//     alone and with spatial reuse and the duplication map, as the graph
//     runs it;
//   - the same inputs give the same image, to the bit, with each reuse;
//   - the temporal merge counts last frame's path at most c_cap times the
//     own tree's count, c_cap lowered by the duplication score as Enhanced
//     §5 states, and weighs and keeps by Talbot MIS;
//   - the duplication map counts copies of a pixel's path in its 17 x 17
//     neighborhood, on constructed images;
//   - a shift from the G-buffer's record gives the shift from the retraced
//     primary hit, to the record's packing;
//   - the reconnection criteria take and refuse the pairs they should: a
//     mirror before x_k refused, a far diffuse pair taken, a pair a
//     centimeter apart taken under a near camera and refused under a far
//     one, which a fixed distance could not do, and a mirror at x_k
//     refused by its inverse footprint;
//   - each frame's pairing is an involution over the image, its offsets'
//     standard deviation sigma;
//   - a path shifted to its own pixel keeps its F, its Jacobian 1; a
//     reconnection's Jacobian is supplemental eq. 4's, on a floor under a
//     firefly whose densities and geometry terms are worked out here.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <numbers>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/camera/thin_lens.h"
#include "core/contracts/bsdf.h"
#include "core/contracts/light_tile.h"
#include "core/contracts/path_reservoir.h"
#include "core/contracts/path_reuse_settings.h"
#include "core/frame/graph_file.h"
#include "core/passes/restir_pt.h"
#include "core/sampling/pairing.h"
#include "core/scene/scene.h"
#include "gpu/kernels/restir_pt_probe.h"
#include "gpu/support/probe_runner.h"
#include "gpu/support/rendering.h"
#include "metal/acceleration/scene_acceleration.h"
#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"
#include "metal/scene/light_glows.h"
#include "metal/scene/scene_buffers.h"
#include "metal/scene/shape_transforms.h"
#include "serenity/metallib/probes.h"
#include "support/references.h"
#include "support/scene_text.h"
#include "support/vector.h"

using namespace serenity;
using tests::camera_text;
using tests::sky;
using tests::Vec3;

namespace {

constexpr const char* initial_only =
    "passes = [\"restir_pt_initial\", \"restir_pt_shade\", \"display\"]\n"
    "[restir_pt]\nnee_candidates = 32\nmax_surfaces = 9\nfootprint = 0.02\nroughness = 0.2\n";
constexpr const char* with_spatial =
    "passes = [\"restir_pt_initial\", \"restir_pt_spatial\", \"restir_pt_shade\", \"display\"]\n"
    "[restir_pt]\nnee_candidates = 32\nmax_surfaces = 9\nfootprint = 0.02\nroughness = 0.2\nneighbors = 3\n"
    "sigma = 16.0\n";
constexpr const char* temporal_alone =
    "passes = [\"restir_pt_initial\", \"restir_pt_temporal\", \"restir_pt_shade\", \"display\"]\n"
    "[restir_pt]\nnee_candidates = 32\nmax_surfaces = 9\nfootprint = 0.02\nroughness = 0.2\n"
    "confidence_cap = 20.0\n";
// graphs/restir_pt.toml's, presenting with display.
constexpr const char* whole_graph =
    "passes = [\"restir_pt_initial\", \"restir_pt_temporal\", \"restir_pt_spatial\", \"restir_pt_shade\", "
    "\"restir_pt_duplication\", \"display\"]\n"
    "[restir_pt]\nnee_candidates = 32\nmax_surfaces = 9\nfootprint = 0.02\nroughness = 0.2\nneighbors = 3\n"
    "sigma = 16.0\nconfidence_cap = 20.0\ncap_floor = 1.0\ncap_exponent = 0.1\n";

// A renderer of a ReSTIR PT graph over a scene, at a size, its frames
// accumulating from frame 0 (as path_test.cpp's Rig).
class Rig {
public:
    Rig(const std::string& text, frame::Extent image, const char* graph)
        : description_(scene::parse(text, "test scene")), size_(image),
          renderer_(device_, submission_, frame::parse_schedule(graph, "graph"), &description_) {}

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

    // The accumulated image's floats (Renderer::read_accumulated).
    std::vector<float> accumulated() {
        std::vector<float> floats(std::size_t{size_.width} * size_.height * 4);
        renderer_.read_accumulated(floats);
        return floats;
    }

    const scene::SceneDescription& description() const noexcept { return description_; }
    frame::Extent size() const noexcept { return size_; }

private:
    scene::SceneDescription description_;
    frame::Extent size_;
    metal::Device device_;
    metal::Submission submission_{device_};
    metal::Offscreen target_{device_, submission_, size_};
    metal::Renderer renderer_;
    std::uint64_t next_ = 0;
};

double mean_red(const std::vector<std::uint8_t>& rgba, frame::Extent size, tests::Pixel at, std::uint32_t half) {
    double sum = 0.0;
    std::uint32_t count = 0;
    for (std::uint32_t y = at.y - half; y <= at.y + half; ++y) {
        for (std::uint32_t x = at.x - half; x <= at.x + half; ++x) {
            sum += tests::linear_of(rgba[(std::size_t{y} * size.width + x) * 4]);
            ++count;
        }
    }
    return sum / count;
}

// path_test.cpp's floor under fireflies, and its closed form.
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

double expected_floor(const scene::SceneDescription& description, frame::Extent size, double x, double y) {
    const contracts::CameraData c = camera::shader_form(description.camera, size);
    const Vec3 d = tests::camera_direction(c, x, y, size);
    const Vec3 p = tests::vec(c.origin) + (-c.origin.y / d.y) * d;
    double sum = 0.0;
    for (const lights::SphereLightData& light : description.sphere_lights) {
        const contracts::Transform& placed = description.shapes.transforms.at(light.shape);
        const double radius = placed.m[0][0];
        const Vec3 to = tests::vec(contracts::translation(placed)) - p;
        const double d2 = dot(to, to);
        sum += floor_albedo * light.radiance.x * (radius * radius / d2) * (to.y / std::sqrt(d2));
    }
    return sum;
}

void check_floor(const std::string& text, std::uint64_t frames, const char* graph) {
    Rig rig(text, {64, 48}, graph);
    const auto image = rig.render(frames);
    for (const tests::Pixel pixel : {tests::Pixel{32, 30}, tests::Pixel{14, 38}, tests::Pixel{50, 26}}) {
        const double expected = expected_floor(rig.description(), rig.size(), pixel.x + 0.5, pixel.y + 0.5);
        INFO("pixel " << pixel.x << ", " << pixel.y << ": expected " << expected);
        CHECK(mean_red(image, rig.size(), pixel, 1) == doctest::Approx(expected).scale(0).epsilon(0.03));
    }
}

std::string furnace(const std::string& material) {
    return camera_text("[0, 0, 4]", "[0, 0, 0]", 30) + sky("[0.5, 0.5, 0.5]", "[0.5, 0.5, 0.5]") +
           "[materials.m]\n" + material +
           "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 1\nmaterial = \"m\"\n";
}

void check_furnace(const std::string& material, std::uint64_t frames, const char* graph) {
    Rig rig(furnace(material), {64, 64}, graph);
    const auto image = rig.render(frames);
    CHECK(mean_red(image, rig.size(), {32, 32}, 6) == doctest::Approx(0.5).scale(0).epsilon(0.02));
    CHECK(mean_red(image, rig.size(), {3, 3}, 2) == doctest::Approx(0.5).scale(0).epsilon(0.01));
}

}  // namespace

TEST_CASE("ReSTIR PT's initial sampling alone: a floor under one firefly converges to albedo L sin^2 cos") {
    check_floor(lit_floor(one_firefly), 512, initial_only);
}

TEST_CASE("ReSTIR PT's initial sampling alone: under two fireflies, the floor converges to their sum") {
    check_floor(lit_floor(one_firefly + firefly("[-1.2, 1.5, 0.4]")), 1024, initial_only);
}

TEST_CASE("ReSTIR PT's initial sampling alone: a white Lambert sphere and a glass one under a uniform sky "
          "converge to the sky") {
    check_furnace("kind = \"rough\"\ncolor = [1, 1, 1]\n", 1024, initial_only);
    check_furnace("kind = \"dielectric\"\nior = 1.5\n", 512, initial_only);
}

TEST_CASE("ReSTIR PT's spatial reuse keeps a still lit floor's mean, and the glass furnace's") {
    // Within the path tests' tolerances: 3% at three patches of floor, 2%
    // through glass, whose paths shift by replay alone.
    check_floor(lit_floor(one_firefly + firefly("[-1.2, 1.5, 0.4]")), 512, with_spatial);
    check_furnace("kind = \"dielectric\"\nior = 1.5\n", 512, with_spatial);
}

TEST_CASE("ReSTIR PT's temporal reuse keeps a still lit floor's mean, and the glass furnace's") {
    // As the spatial reuse's test, the same tolerances: temporal reuse
    // alone, and the whole graph, the duplication map lowering the cap. The
    // map's bias (Enhanced §5) is where copies crowd, which a floor under
    // two fireflies, resampled from the start, has few of.
    check_floor(lit_floor(one_firefly + firefly("[-1.2, 1.5, 0.4]")), 512, temporal_alone);
    check_floor(lit_floor(one_firefly + firefly("[-1.2, 1.5, 0.4]")), 512, whole_graph);
    check_furnace("kind = \"dielectric\"\nior = 1.5\n", 512, whole_graph);
}

TEST_CASE("ReSTIR PT: the same inputs give the same image, to the bit, with its spatial reuse at work") {
    // A floor seen from above under six fireflies and beside a glass ball,
    // so most pixels' partners are similar and merged, and some paths
    // replay through the glass.
    // The fireflies dim, so the floor stays below what the display clips.
    const std::string text = camera_text("[0, 3, 0.5]", "[0, 0, 0]", 50) + sky("[0, 0, 0]", "[0, 0, 0]") +
                             "[materials.floor]\nkind = \"rough\"\ncolor = [0.8, 0.8, 0.8]\n"
                             "[materials.glow]\nkind = \"emissive\"\nradiance = [2, 2, 2]\n"
                             "[materials.glass]\nkind = \"dielectric\"\nior = 1.5\n"
                             "[[shapes]]\nkind = \"box\"\nmin = [-50, -1, -50]\nmax = [50, 0, 50]\nmaterial = \"floor\"\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0.3, 0.3, 0]\nradius = 0.3\nmaterial = \"glass\"\n" +
                             firefly("[0.8, 0.6, 0.4]") + firefly("[-0.9, 0.9, -0.3]") + firefly("[0.2, 1.1, -0.9]") +
                             firefly("[-0.4, 0.4, 0.8]") + firefly("[1.2, 0.8, -0.6]") + firefly("[-1.1, 0.5, 0.9]");
    constexpr frame::Extent image{96, 72};
    Rig a(text, image, with_spatial);
    Rig b(text, image, with_spatial);
    Rig alone(text, image, initial_only);
    (void)a.render(3);
    (void)b.render(3);
    (void)alone.render(3);
    const std::vector<float> first = a.accumulated();
    const std::vector<float> second = b.accumulated();
    const std::vector<float> without = alone.accumulated();
    std::size_t differ = 0;
    std::size_t merged = 0;
    for (std::size_t i = 0; i < first.size(); ++i) {
        differ += std::memcmp(&first[i], &second[i], sizeof(float)) != 0 ? 1u : 0u;
        merged += first[i] != without[i] ? 1u : 0u;
    }
    CHECK(differ == 0u);
    // And the merge is at work: the same frames without it differ.
    INFO(merged << " of " << first.size() << " values changed by the merge");
    CHECK(merged > first.size() / 4);

    // The whole graph, temporal reuse and the duplication map at work over
    // four frames: the same to the bit, and not spatial reuse's image.
    Rig c(text, image, whole_graph);
    Rig d(text, image, whole_graph);
    (void)c.render(4);
    (void)d.render(4);
    (void)a.render(1);
    const std::vector<float> third = c.accumulated();
    const std::vector<float> fourth = d.accumulated();
    const std::vector<float> spatial_only = a.accumulated();
    std::size_t differ_whole = 0;
    std::size_t reused = 0;
    for (std::size_t i = 0; i < third.size(); ++i) {
        differ_whole += std::memcmp(&third[i], &fourth[i], sizeof(float)) != 0 ? 1u : 0u;
        reused += third[i] != spatial_only[i] ? 1u : 0u;
    }
    CHECK(differ_whole == 0u);
    INFO(reused << " of " << third.size() << " values changed by temporal reuse");
    CHECK(reused > third.size() / 4);
}

namespace {

// A pair at the marbles' scale: the camera 0.39 m above a table point x1,
// x_{k-1} on the table, x_k on a marble beside it, both diffuse.
tests::CriteriaQuery diffuse_pair(double camera_height, double apart, double scale) {
    const auto f = [](Vec3 v) {
        return contracts::Float3{static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z)};
    };
    const Vec3 up{0.0, 1.0, 0.0};
    const Vec3 x1 = scale * Vec3{0.0, 0.75, 0.0};
    const Vec3 camera = x1 + scale * Vec3{0.0, camera_height, camera_height};  // seen at 45 degrees
    const Vec3 before = scale * Vec3{0.01, 0.75, 0.0};
    const Vec3 at = before + scale * Vec3{0.0, apart, 0.0};  // straight up: cos 1 at both ends, x_k facing down
    const double cos_before = 1.0;
    tests::CriteriaQuery q{};
    q.camera = f(camera);
    q.footprint = 0.02f;
    q.primary = f(x1);
    q.min_roughness = 0.2f;
    q.primary_normal = f(up);
    q.lobe_before = contracts::lobe_reflection | contracts::lobe_diffuse;
    q.before = f(before);
    q.density_before = static_cast<float>(cos_before / std::numbers::pi);
    q.before_normal = f(up);
    q.lobe_at = contracts::lobe_reflection | contracts::lobe_diffuse;
    q.at = f(at);
    q.density_at = static_cast<float>(1.0 / std::numbers::pi);
    q.at_normal = f(-up);
    q.alpha_before = 0.0f;
    return q;
}

}  // namespace

TEST_CASE("the reconnection criteria take a far diffuse pair, refuse a mirror, and scale with the primary footprint") {
    tests::ProbeRunner runner;
    std::vector<tests::CriteriaQuery> queries;
    // 0: diffuse x_{k-1} and x_k 20 cm apart, the camera 39 cm off: taken.
    queries.push_back(diffuse_pair(0.39 / std::numbers::sqrt2, 0.2, 1.0));
    // 1: the same with x_{k-1} a mirror (a conductor of alpha 10^-3, the
    // marbles' steel; roughness 0.03): refused by the roughness test.
    tests::CriteriaQuery mirror = queries[0];
    mirror.lobe_before = contracts::lobe_reflection | contracts::lobe_glossy;
    mirror.alpha_before = 1.0e-3f;
    mirror.density_before = 1.0e4f;
    queries.push_back(mirror);
    // 2: a pair 5 mm apart under the camera 39 cm off: refused, its ray
    // footprint 7.9 x 10^-5 m^2 under the threshold's 5.4 x 10^-4.
    queries.push_back(diffuse_pair(0.39 / std::numbers::sqrt2, 0.005, 1.0));
    // 3: the same pair under a camera 3 cm off: taken. A fixed distance
    // could not tell 2 from 3.
    queries.push_back(diffuse_pair(0.03 / std::numbers::sqrt2, 0.005, 1.0));
    // 4: case 2 ten times larger, camera and all: refused as case 2 is.
    queries.push_back(diffuse_pair(0.39 / std::numbers::sqrt2, 0.005, 10.0));
    // 5: case 0 with a near mirror at x_k, its glossy lobe's density 10^4:
    // refused by its inverse footprint.
    tests::CriteriaQuery mirror_at = queries[0];
    mirror_at.lobe_at = contracts::lobe_reflection | contracts::lobe_glossy;
    mirror_at.density_at = 1.0e4f;
    queries.push_back(mirror_at);
    // 6: case 5 with x_k's direction drawn by NEE (lobe 0): the inverse test
    // skipped, taken.
    tests::CriteriaQuery nee_at = mirror_at;
    nee_at.lobe_at = 0u;
    queries.push_back(nee_at);
    const auto count = static_cast<std::uint32_t>(queries.size());
    const auto results = runner.run<tests::CriteriaResult>(
        "pt_criteria", count, {tests::Binding::of(queries), tests::Binding::of(count)});
    const std::vector<std::uint32_t> expected = {1u, 0u, 0u, 1u, 0u, 0u, 1u};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        INFO("case " << i << ": threshold " << results[i].threshold << ", forward " << results[i].forward
                     << ", inverse " << results[i].inverse);
        CHECK(results[i].connectable == expected[i]);
    }
    // The threshold scales as the primary footprint does, the square of
    // the distance: 100 times for 10 times the scene.
    CHECK(results[4].threshold == doctest::Approx(100.0 * results[2].threshold).epsilon(1e-4));
}

TEST_CASE("each frame's pairing is an involution over the image, its offsets' deviation sigma") {
    tests::ProbeRunner runner;
    const std::vector<std::uint16_t> textures = sampling::reuse_textures(16.0, 0x5e2e417f);
    constexpr std::uint32_t width = 512;
    constexpr std::uint32_t height = 384;
    for (std::uint32_t frame = 0; frame < 4; ++frame) {
        for (std::uint32_t texture = 0; texture < contracts::max_path_neighbors; ++texture) {
            const tests::PairingQuery query{frame, texture, width, height};
            const auto results = runner.run<tests::PairingResult>(
                "pt_pairing", std::size_t{width} * height, {tests::Binding::of(query), tests::Binding::of(textures)});
            std::size_t broken = 0;
            double squares = 0.0;
            for (std::uint32_t i = 0; i < width * height; ++i) {
                const int x = static_cast<int>(i % width);
                const int y = static_cast<int>(i / width);
                broken += (results[i].back_x != x || results[i].back_y != y) ? 1u : 0u;
                const double dx = results[i].x - x;
                const double dy = results[i].y - y;
                squares += dx * dx + dy * dy;
            }
            INFO("frame " << frame << ", texture " << texture);
            CHECK(broken == 0u);
            const double deviation = std::sqrt(squares / (2.0 * width * height));
            CHECK(deviation == doctest::Approx(16.0).epsilon(0.1));
        }
    }
    // The map changes with the frame: frame 0's partners are not frame 1's.
    const auto at = [&](std::uint32_t frame) {
        return runner.run<tests::PairingResult>("pt_pairing", std::size_t{width} * height,
                                               {tests::Binding::of(tests::PairingQuery{frame, 0u, width, height}),
                                                tests::Binding::of(textures)});
    };
    const auto zero = at(0);
    const auto one = at(1);
    std::size_t same = 0;
    for (std::size_t i = 0; i < zero.size(); ++i) {
        same += (zero[i].x == one[i].x && zero[i].y == one[i].y) ? 1u : 0u;
    }
    CHECK(same < zero.size() / 10);
}

namespace {

// Shifts paths drawn at queries' source pixels to their destinations, in
// `text`'s still scene seen at `size`, through pt_shift (the passes' own
// functions), the scene bound as a pass binds it.
std::vector<tests::ShiftResult> shift(const std::string& text, frame::Extent size,
                                      const std::vector<tests::ShiftQuery>& queries,
                                      std::uint32_t nee_candidates = passes::written_nee_candidates) {
    const scene::SceneDescription description = scene::parse(text, "test scene");
    metal::Device device;
    metal::Submission submission(device);
    const metal::SceneBuffers buffers(device, submission, description);
    metal::ShapeTransforms transforms(device, submission, description.shapes.transforms,
                                      metal::FrameArray::Copies::one);
    metal::LightGlows glows(device, submission, description.light_counts.spheres, false);
    metal::SceneAcceleration acceleration(device, submission, description.shapes, {});
    metal::Library library(device, metallib::probes);
    const auto pipeline = library.compute_pipeline("pt_shift");
    const auto fill = library.compute_pipeline("pt_light_tiles");
    const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    MTL::Device* mtl = device.handle();
    const contracts::CameraData camera = camera::shader_form(description.camera, size);
    contracts::PathReuseSettings settings{};
    settings.nee_candidates = nee_candidates;
    settings.max_surfaces = passes::written_max_surfaces;
    settings.footprint = passes::written_footprint;
    settings.roughness = passes::written_roughness;
    const auto count = static_cast<std::uint32_t>(queries.size());
    const auto buffer_of = [&](const void* bytes, std::size_t size_of) {
        auto buffer = NS::TransferPtr(mtl->newBuffer(std::max<std::size_t>(size_of, 16), MTL::ResourceStorageModeShared));
        REQUIRE(buffer);
        std::memset(buffer->contents(), 0, std::max<std::size_t>(size_of, 16));
        if (bytes != nullptr) {
            std::memcpy(buffer->contents(), bytes, size_of);
        }
        submission.make_resident(buffer.get());
        return buffer;
    };
    const auto out = buffer_of(nullptr, queries.size() * sizeof(tests::ShiftResult));
    const auto in = buffer_of(queries.data(), queries.size() * sizeof(tests::ShiftQuery));
    const auto framed = buffer_of(&camera, sizeof(camera));
    const auto set = buffer_of(&settings, sizeof(settings));
    const auto counted = buffer_of(&count, sizeof(count));
    constexpr std::size_t tile_entries = std::size_t{contracts::light_tiles} * contracts::light_tile_size;
    const auto tiles = buffer_of(nullptr, tile_entries * sizeof(contracts::LightTileEntry));
    auto descriptor = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
    descriptor->setMaxBufferBindCount(10);
    NS::Error* error = nullptr;
    auto table = NS::TransferPtr(mtl->newArgumentTable(descriptor.get(), &error));
    REQUIRE(table);
    const metal::FrameSlot slot = submission.begin();
    MTL4::ComputeCommandEncoder* encoder = slot.commands->computeCommandEncoder();
    REQUIRE(encoder != nullptr);
    table->setAddress(out->gpuAddress(), 0);
    table->setAddress(in->gpuAddress(), 1);
    table->setAddress(framed->gpuAddress(), 2);
    table->setResource(acceleration.resource(slot.slot), 3);
    table->setAddress(buffers.block_address(), 4);
    table->setAddress(glows.address(slot.slot), 5);
    table->setAddress(transforms.address(slot.slot), 6);
    table->setAddress(set->gpuAddress(), 7);
    table->setAddress(counted->gpuAddress(), 8);
    table->setAddress(tiles->gpuAddress(), 9);
    encoder->setArgumentTable(table.get());
    encoder->setComputePipelineState(fill.get());
    encoder->dispatchThreads(MTL::Size(tile_entries, 1, 1), MTL::Size(fill->threadExecutionWidth(), 1, 1));
    encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    encoder->setComputePipelineState(pipeline.get());
    encoder->dispatchThreads(MTL::Size(queries.size(), 1, 1), MTL::Size(pipeline->threadExecutionWidth(), 1, 1));
    encoder->endEncoding();
    submission.commit();
    (void)submission.wait_until_complete(slot.sequence);
    std::vector<tests::ShiftResult> results(queries.size());
    std::memcpy(results.data(), out->contents(), results.size() * sizeof(tests::ShiftResult));
    return results;
}

double luminance(contracts::Float3 c) {
    return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
}

}  // namespace

TEST_CASE("a path shifted to its own pixel keeps its F and has a Jacobian of 1, in every case the shift has") {
    // A floor under a firefly and a large light, a glass ball (replay
    // through it), a diffuse ball and a brass mirror; one NEE candidate, so
    // rays the BSDF draws to a light are kept often. Every case of
    // shift.metal.h's step 6: replayed whole; reconnected to a light NEE
    // drew (forced) or a ray struck; at a surface whose direction NEE drew;
    // at a surface whose BSDF drew it. The floor 6 m across, so a point on
    // it is stored to 0.05 mm (contract 15).
    const std::string text = camera_text("[0, 2, 3]", "[0, 0, 0]", 50) + sky("[0, 0, 0]", "[0, 0, 0]") +
                             "[materials.floor]\nkind = \"rough\"\ncolor = [0.8, 0.8, 0.8]\n"
                             "[materials.glow]\nkind = \"emissive\"\nradiance = [200, 200, 200]\n"
                             "[materials.lamp]\nkind = \"emissive\"\nradiance = [4, 4, 4]\n"
                             "[[shapes]]\nkind = \"box\"\nmin = [-3, -1, -3]\nmax = [3, 0, 3]\nmaterial = \"floor\"\n" +
                             one_firefly +
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [-1.2, 1.8, -1.0]\nradius = 0.6\nmaterial = \"lamp\"\n"
                             "[materials.glass]\nkind = \"dielectric\"\nior = 1.5\n"
                             "[materials.white]\nkind = \"rough\"\ncolor = [0.7, 0.7, 0.7]\n"
                             "[materials.brass]\nkind = \"conductor\"\nf0 = [0.9, 0.7, 0.3]\nroughness = 0.05\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0.5, 0]\nradius = 0.5\nmaterial = \"glass\"\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [1, 0.4, -0.3]\nradius = 0.4\nmaterial = \"white\"\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [-0.9, 0.4, 0.2]\nradius = 0.4\nmaterial = \"brass\"\n";
    constexpr frame::Extent size{96, 72};
    std::vector<tests::ShiftQuery> queries;
    for (std::uint32_t frame = 0; frame < 16; ++frame) {
        for (std::uint32_t y = 0; y < size.height; y += 2) {
            for (std::uint32_t x = 0; x < size.width; x += 2) {
                queries.push_back({x, y, x, y, frame, size.width, size.height, 0u});
            }
        }
    }
    const auto results = shift(text, size, queries, 1u);
    // The cases, and in each the paths held and those shifted off.
    enum Case { whole, forced, struck, nee_at_k, bsdf_at_k, cases };
    const char* names[cases] = {"replayed whole", "to NEE's light", "to a struck light", "at x_k, NEE on",
                                "at x_k, BSDF on"};
    std::array<std::size_t, cases> held{};
    std::array<std::size_t, cases> off{};
    for (const tests::ShiftResult& r : results) {
        if (contracts::path_end(r.flags) == contracts::PathEnd::none || luminance(r.stored) <= 0.0) {
            continue;
        }
        const std::uint32_t k = contracts::path_rc(r.flags);
        const bool at_light = k == contracts::path_length(r.flags);
        const Case c = k == 0u ? whole
                       : at_light ? (contracts::path_end(r.flags) == contracts::PathEnd::nee ? forced : struck)
                                  : (contracts::path_lobe_at(r.flags) == 0u ? nee_at_k : bsdf_at_k);
        ++held[c];
        // To the stored point's rounding (contract 15) and the G-buffer's
        // packing of y1 (contract 5: its normal to 10^-4 radians, which
        // turns every replayed direction after it by as much), which a
        // glossy x_k magnifies in its density and a deep replay carries
        // along: 2% (1% before the G-buffer, the retraced y1 exact).
        const bool same = r.succeeded == 1u && std::abs(r.jacobian - 1.0) < 2.0e-2 &&
                          std::abs(luminance(r.shifted) - luminance(r.stored)) <= 2.0e-2 * luminance(r.stored);
        off[c] += same ? 0u : 1u;
    }
    for (int c = 0; c < cases; ++c) {
        // A pair at the criteria's edge may fall the other way at the
        // stored point, and the shift fail: at most 1 in 200.
        INFO(std::string(names[c]) << ": " << held[static_cast<std::size_t>(c)] << " paths, " << off[static_cast<std::size_t>(c)]
                      << " off");
        CHECK(held[static_cast<std::size_t>(c)] >= 20u);
        CHECK(off[static_cast<std::size_t>(c)] * 200u <= held[static_cast<std::size_t>(c)]);
    }
}

TEST_CASE("a reconnection's Jacobian is supplemental eq. 4's on a floor under a firefly") {
    // The floor alone under one firefly: every path is x0, x1 on the floor,
    // x2 on the light, reconnected from y1 (NEE's, forced; or the BSDF's,
    // where the footprints allow). Eq. 4's ratio, worked out here in
    // double: the densities of the light's point and of its direction,
    //   NEE:   p = P / (its cone's solid angle from the point), and
    //   BSDF:  p = cos / pi at the floor,
    // each times G = cos at the light / distance^2.
    const std::string text = lit_floor(one_firefly);
    constexpr frame::Extent size{64, 48};
    std::vector<tests::ShiftQuery> queries;
    for (std::uint32_t frame = 0; frame < 4; ++frame) {
        for (std::uint32_t y = 4; y < size.height - 4; y += 2) {
            for (std::uint32_t x = 4; x < size.width - 4; x += 2) {
                queries.push_back({x, y, x + 3, y - 2, frame, size.width, size.height, 0u});
            }
        }
    }
    const scene::SceneDescription description = scene::parse(text, "test scene");
    const lights::SphereLightData& light = description.sphere_lights.at(0);
    const contracts::Transform& placed = description.shapes.transforms.at(light.shape);
    const Vec3 center = tests::vec(contracts::translation(placed));
    const double radius = placed.m[0][0];
    const auto results = shift(text, size, queries);
    // The density of the point x2 seen from the floor point p, by each
    // strategy, in the light's area measure.
    const auto area_density = [&](Vec3 p, Vec3 x2, bool nee) {
        const Vec3 to = x2 - p;
        const double d2 = dot(to, to);
        const Vec3 w = (1.0 / std::sqrt(d2)) * to;
        const double cos_light = dot(tests::normalized(x2 - center), -w);
        const double dc2 = dot(center - p, center - p);
        const double solid_angle = 2.0 * std::numbers::pi * (1.0 - std::sqrt(1.0 - radius * radius / dc2));
        const double direction_density = nee ? 1.0 / solid_angle : w.y / std::numbers::pi;  // one light: P = 1
        return direction_density * cos_light / d2;
    };
    std::size_t checked = 0;
    for (const tests::ShiftResult& r : results) {
        // A path the BSDF drew to the light past the footprints' bound is
        // replayed whole, its Jacobian 1: not this test's.
        if (contracts::path_end(r.flags) == contracts::PathEnd::none || r.succeeded == 0u ||
            contracts::path_rc(r.flags) == 0u) {
            continue;
        }
        REQUIRE(contracts::path_length(r.flags) == 2u);
        REQUIRE(contracts::path_rc(r.flags) == 2u);
        const bool nee = contracts::path_end(r.flags) == contracts::PathEnd::nee;
        const Vec3 x2 = tests::vec(r.reconnection);
        const double expected = area_density(tests::vec(r.destination_hit), x2, nee) /
                                area_density(tests::vec(r.source_hit), x2, nee);
        INFO(std::string(nee ? "NEE" : "BSDF") << ": J " << r.jacobian << ", eq. 4 " << expected);
        CHECK(r.jacobian == doctest::Approx(expected).epsilon(2.0e-3));
        ++checked;
    }
    CHECK(checked > 500u);
}

namespace {

// A reservoir holding a path of `luminance` grey F, weight W, count M,
// drawn at (pixel, frame), ending on NEE's light at x2 (k = d = 2), its
// cached Jacobian denominator `jacobian`.
contracts::PathReservoir held_path(float f, float weight, std::uint32_t count, std::uint32_t pixel,
                                   std::uint32_t frame, float jacobian) {
    contracts::PathReservoir r{};
    r.weight = weight;
    r.integrand = contracts::Float3{f, f, f};
    r.seed_pixel = pixel;
    r.seed_frame = frame;
    r.flags = contracts::pack_path_flags(contracts::PathShape{count, 2u, 2u, contracts::PathEnd::nee, 0u, 0u, 0u});
    r.jacobian = jacobian;
    return r;
}

}  // namespace

TEST_CASE("the temporal merge counts last frame's path at most c_cap times the own tree, c_cap lowered by "
          "duplication, and weighs by Talbot MIS") {
    // Grey F, so p-hat is F's value. The own path: F 2, W 0.5, count 1,
    // shifted there to F 1.5 with J 0.8. Last frame's: F 3, W 0.25, count
    // 40, shifted here to F 2.5 with J 1.2 (its cached denominator 0.7).
    const contracts::PathReservoir own = held_path(2.0f, 0.5f, 1u, 11u, 100u, 0.3f);
    const contracts::PathReservoir last = held_path(3.0f, 0.25f, 40u, 22u, 99u, 0.7f);
    const auto query = [&](float cap, float floor, float exponent, float duplication, float u1) {
        tests::PathMergeQuery q{};
        q.own = own;
        q.last = last;
        q.here = contracts::Float3{2.5f, 2.5f, 2.5f};
        q.here_jacobian = 1.2f;
        q.there = contracts::Float3{1.5f, 1.5f, 1.5f};
        q.there_jacobian = 0.8f;
        q.cap = cap;
        q.floor = floor;
        q.exponent = exponent;
        q.duplication = duplication;
        q.u0 = 0.5f;
        q.u1 = u1;
        return q;
    };
    // c_cap with Enhanced's settings at D = 0.25: lerp(20, 1, 0.25^0.1).
    const double quarter_cap = 20.0 + (1.0 - 20.0) * std::pow(0.25, 0.1);
    struct Case {
        tests::PathMergeQuery query;
        double cap;  // c_cap expected
    };
    const std::vector<Case> cases = {
        {query(20.0f, 0.0f, 0.0f, 0.9f, 0.99f), 20.0},  // no duplication map: the cap itself, D unread
        {query(20.0f, 1.0f, 0.1f, 0.0f, 0.99f), 20.0},  // D = 0
        {query(20.0f, 1.0f, 0.1f, 1.0f, 0.99f), 1.0},   // D = 1: the floor
        {query(20.0f, 1.0f, 0.1f, 0.25f, 0.99f), quarter_cap},
        {query(20.0f, 1.0f, 0.1f, 0.25f, 0.0f), quarter_cap},  // the second number keeps last's path
    };
    std::vector<tests::PathMergeQuery> queries;
    for (const Case& c : cases) {
        queries.push_back(c.query);
    }
    tests::ProbeRunner probe;
    const auto results = probe.run<tests::PathMergeResult>(
        "pt_temporal_merge", queries.size(),
        {tests::Binding::of(queries), tests::Binding::of(static_cast<std::uint32_t>(queries.size()))});
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const double cap = cases[i].cap;
        const tests::PathMergeResult& r = results[i];
        INFO("case " << i << ": c_cap " << cap);
        CHECK(r.cap == doctest::Approx(cap).epsilon(1.0e-5));
        // Talbot (path_merge.metal.h), in double: c_c = 1, c_t = min(40, c_cap).
        const double c_own = 1.0;
        const double c_last = std::min(40.0, cap * c_own);
        const double m_own = c_own * 2.0 / (c_own * 2.0 + c_last * 1.5 * 0.8);
        const double m_last = c_last * 3.0 / (c_last * 3.0 + c_own * 2.5 * 1.2);
        const double w_own = m_own * 2.0 * 0.5;
        const double w_last = m_last * 2.5 * 1.2 * 0.25;
        CHECK(r.estimate.x == doctest::Approx(w_own + w_last).epsilon(1.0e-5));
        // Its count: c_c + c_t, rounded, never last's 40 beside the own 1.
        CHECK(contracts::path_count(r.kept.flags) == static_cast<std::uint32_t>(std::lround(c_own + c_last)));
        // The kept path: last's where the second number is under its share
        // of the sum, its F the shifted one, its denominator times J.
        const bool last_kept = queries[i].u1 * (w_own + w_last) < w_last;
        CHECK(r.kept.seed_pixel == (last_kept ? 22u : 11u));
        const double kept_f = last_kept ? 2.5 : 2.0;
        CHECK(r.kept.integrand.x == doctest::Approx(kept_f));
        CHECK(r.kept.jacobian == doctest::Approx(last_kept ? 0.7 * 1.2 : 0.3));
        // W = the weights' sum over the kept path's p-hat: F W is the sum.
        CHECK(r.kept.weight * kept_f == doctest::Approx(w_own + w_last).epsilon(1.0e-5));
    }
    CHECK(contracts::path_count(results[2].kept.flags) == 2u);  // the floor's: 1 + 1
}

namespace {

// The duplication map of a `size` image of reservoirs, through the pass's
// functions, dispatched as the pass dispatches them.
std::vector<float> duplication_map(const std::vector<contracts::PathReservoir>& image, frame::Extent size) {
    metal::Device device;
    metal::Submission submission(device);
    metal::Library library(device, metallib::probes);
    const auto pipeline = library.compute_pipeline("pt_duplication");
    REQUIRE(pipeline->maxTotalThreadsPerThreadgroup() >= contracts::duplication_group * contracts::duplication_group);
    const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    MTL::Device* mtl = device.handle();
    const auto buffer_of = [&](const void* bytes, std::size_t size_of) {
        auto buffer = NS::TransferPtr(mtl->newBuffer(std::max<std::size_t>(size_of, 16), MTL::ResourceStorageModeShared));
        REQUIRE(buffer);
        std::memset(buffer->contents(), 0, std::max<std::size_t>(size_of, 16));
        if (bytes != nullptr) {
            std::memcpy(buffer->contents(), bytes, size_of);
        }
        submission.make_resident(buffer.get());
        return buffer;
    };
    const std::uint32_t extent[2] = {size.width, size.height};
    const auto out = buffer_of(nullptr, image.size() * sizeof(float));
    const auto in = buffer_of(image.data(), image.size() * sizeof(contracts::PathReservoir));
    const auto sized = buffer_of(extent, sizeof(extent));
    auto descriptor = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
    descriptor->setMaxBufferBindCount(3);
    NS::Error* error = nullptr;
    auto table = NS::TransferPtr(mtl->newArgumentTable(descriptor.get(), &error));
    REQUIRE(table);
    table->setAddress(out->gpuAddress(), 0);
    table->setAddress(in->gpuAddress(), 1);
    table->setAddress(sized->gpuAddress(), 2);
    const metal::FrameSlot slot = submission.begin();
    MTL4::ComputeCommandEncoder* encoder = slot.commands->computeCommandEncoder();
    REQUIRE(encoder != nullptr);
    encoder->setArgumentTable(table.get());
    encoder->setComputePipelineState(pipeline.get());
    constexpr std::uint32_t group = contracts::duplication_group;
    encoder->dispatchThreadgroups(MTL::Size((size.width + group - 1) / group, (size.height + group - 1) / group, 1),
                                  MTL::Size(group, group, 1));
    encoder->endEncoding();
    submission.commit();
    (void)submission.wait_until_complete(slot.sequence);
    std::vector<float> map(image.size());
    std::memcpy(map.data(), out->contents(), map.size() * sizeof(float));
    return map;
}

}  // namespace

TEST_CASE("the duplication map counts the copies of each pixel's path in its 17 x 17 neighborhood") {
    // 70 x 50, so groups at the right and bottom are cut by the image.
    constexpr frame::Extent size{70, 50};
    const auto at = [&](std::uint32_t x, std::uint32_t y) { return std::size_t{y} * size.width + x; };
    std::vector<contracts::PathReservoir> image(std::size_t{size.width} * size.height);
    for (std::uint32_t y = 0; y < size.height; ++y) {
        for (std::uint32_t x = 0; x < size.width; ++x) {
            image[at(x, y)] = held_path(1.0f, 1.0f, 1u, x + 65536u * y, 7u, 1.0f);  // each its own path
        }
    }
    // A 5 x 5 block of copies of one path.
    for (std::uint32_t y = 10; y < 15; ++y) {
        for (std::uint32_t x = 10; x < 15; ++x) {
            image[at(x, y)] = held_path(1.0f, 1.0f, 1u, 12u + 65536u * 12u, 7u, 1.0f);
        }
    }
    // Beside it, a pixel holding the same stream with a weight of 0: no
    // path, no one's copy.
    image[at(16, 12)] = held_path(1.0f, 0.0f, 1u, 12u + 65536u * 12u, 7u, 1.0f);
    // Three copies in a row 8 apart: the middle sees both, each end the
    // middle only (17 apart is past the window).
    for (const std::uint32_t x : {40u, 48u, 56u}) {
        image[at(x, 30)] = held_path(1.0f, 1.0f, 1u, 3u, 3u, 1.0f);
    }
    // The same pixel key, another frame: another path.
    image[at(44, 30)] = held_path(1.0f, 1.0f, 1u, 3u, 4u, 1.0f);
    const std::vector<float> map = duplication_map(image, size);
    for (std::uint32_t y = 0; y < size.height; ++y) {
        for (std::uint32_t x = 0; x < size.width; ++x) {
            const bool block = x >= 10 && x < 15 && y >= 10 && y < 15;
            const bool row = y == 30 && (x == 40 || x == 48 || x == 56);
            const double expected = block ? 24.0 / 288.0 : row ? (x == 48 ? 2.0 : 1.0) / 288.0 : 0.0;
            INFO("pixel " << x << ", " << y);
            CHECK(map[at(x, y)] == doctest::Approx(expected).epsilon(1.0e-6));
        }
    }

    // Every pixel one path: 1 where the window is inside the image, the
    // window's share inside it at the edges and corners.
    std::vector<contracts::PathReservoir> same(image.size(), held_path(1.0f, 1.0f, 1u, 5u, 5u, 1.0f));
    const std::vector<float> full = duplication_map(same, size);
    CHECK(full[at(35, 25)] == doctest::Approx(1.0));
    CHECK(full[at(0, 0)] == doctest::Approx(80.0 / 288.0));
    CHECK(full[at(69, 49)] == doctest::Approx(80.0 / 288.0));
    CHECK(full[at(69, 25)] == doctest::Approx((9.0 * 17.0 - 1.0) / 288.0));
}

TEST_CASE("a shift from the G-buffer's record gives the shift from the retraced primary hit, to its packing") {
    // The scene of the self-shift test, each path shifted 3 pixels right and
    // 2 up: through the record the destination's tree packed (contract 5),
    // as the passes shift, and from its primary ray traced and resolved
    // again, as Phase 1 did. They differ by the record's packing (its
    // normal and wo to 10^-4 radians, its BSDF to a half's precision),
    // which replay carries along: whether each succeeds the same, and F and
    // the Jacobian within 2%, for all but 1 in 200.
    const std::string text = camera_text("[0, 2, 3]", "[0, 0, 0]", 50) + sky("[0, 0, 0]", "[0, 0, 0]") +
                             "[materials.floor]\nkind = \"rough\"\ncolor = [0.8, 0.8, 0.8]\n"
                             "[materials.glow]\nkind = \"emissive\"\nradiance = [200, 200, 200]\n"
                             "[materials.lamp]\nkind = \"emissive\"\nradiance = [4, 4, 4]\n"
                             "[[shapes]]\nkind = \"box\"\nmin = [-3, -1, -3]\nmax = [3, 0, 3]\nmaterial = \"floor\"\n" +
                             one_firefly +
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [-1.2, 1.8, -1.0]\nradius = 0.6\nmaterial = \"lamp\"\n"
                             "[materials.glass]\nkind = \"dielectric\"\nior = 1.5\n"
                             "[materials.white]\nkind = \"rough\"\ncolor = [0.7, 0.7, 0.7]\n"
                             "[materials.brass]\nkind = \"conductor\"\nf0 = [0.9, 0.7, 0.3]\nroughness = 0.05\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0.5, 0]\nradius = 0.5\nmaterial = \"glass\"\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [1, 0.4, -0.3]\nradius = 0.4\nmaterial = \"white\"\n"
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [-0.9, 0.4, 0.2]\nradius = 0.4\nmaterial = \"brass\"\n";
    constexpr frame::Extent size{96, 72};
    std::vector<tests::ShiftQuery> queries;
    for (std::uint32_t frame = 0; frame < 8; ++frame) {
        for (std::uint32_t y = 2; y < size.height; y += 2) {
            for (std::uint32_t x = 0; x + 3 < size.width; x += 2) {
                queries.push_back({x, y, x + 3, y - 2, frame, size.width, size.height, 0u});
            }
        }
    }
    const auto results = shift(text, size, queries, 1u);
    std::size_t tried = 0;
    std::size_t succeeded = 0;
    std::size_t through_glass = 0;
    std::size_t off = 0;
    for (const tests::ShiftResult& r : results) {
        if (contracts::path_end(r.flags) == contracts::PathEnd::none || luminance(r.stored) <= 0.0) {
            continue;
        }
        ++tried;
        succeeded += r.succeeded;
        through_glass += r.succeeded == 1u && contracts::path_rc(r.flags) != 2u ? 1u : 0u;
        const double f = luminance(r.retraced);
        const bool same = r.succeeded == r.retraced_succeeded &&
                          (r.succeeded == 0u || (std::abs(r.jacobian - r.retraced_jacobian) <= 2.0e-2 * r.retraced_jacobian &&
                                                 std::abs(luminance(r.shifted) - f) <= 2.0e-2 * f));
        off += same ? 0u : 1u;
    }
    INFO(tried << " paths, " << succeeded << " shifted, " << through_glass << " replayed or reconnected past y2, "
               << off << " off");
    CHECK(succeeded > 1000u);
    CHECK(through_glass > 100u);
    CHECK(off * 200u <= tried);
}
