// ReSTIR PT (metal/passes/restir_pt/restir_pt.metal.h), on the GPU:
//
//   - an integration test: the whole graph, every reuse and the duplication
//     map at work, keeps the closed forms the path tracer is held to
//     (tests/gpu/path_test.cpp), within their tolerances: a floor under two
//     fireflies, and the glass furnace, whose paths shift by replay alone;
//
// and unit tests of its pieces, on the probe kernel (kernels/
// restir_pt_probe.metal), each run on its own:
//
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
//     standard deviation sigma; and with the checkerboard over the frame's
//     traced field, every partner traced, sigma pixels apart;
//   - a path shifted to its own pixel keeps its F, its Jacobian 1; a
//     reconnection's Jacobian is supplemental eq. 4's, on a floor under a
//     firefly whose densities and geometry terms are worked out here.

#include "core/passes/restir_pt.h"

#include <doctest/doctest.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <numbers>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/camera/thin_lens.h"
#include "core/contracts/bsdf.h"
#include "core/contracts/checkerboard.h"
#include "core/contracts/light_tile.h"
#include "core/contracts/mnee_request.h"
#include "core/contracts/path_reservoir.h"
#include "core/contracts/path_reuse_settings.h"
#include "core/frame/graph_file.h"
#include "core/sampling/pairing.h"
#include "core/scene/scene.h"
#include "gpu/kernels/restir_pt_probe.h"
#include "gpu/support/probe_runner.h"
#include "gpu/support/rendering.h"
#include "gpu/support/shared_resources.h"
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
#include "support/lit_floor.h"
#include "support/references.h"
#include "support/scene_text.h"
#include "support/schedules.h"
#include "support/vector.h"

namespace camera = serenity::camera;
namespace contracts = serenity::contracts;
namespace frame = serenity::frame;
namespace lights = serenity::lights;
namespace metal = serenity::metal;
namespace metallib = serenity::metallib;
namespace passes = serenity::passes;
namespace sampling = serenity::sampling;
namespace scene = serenity::scene;
namespace tests = serenity::tests;
using tests::camera_text;
using tests::firefly;
using tests::furnace;
using tests::lit_floor;
using tests::one_firefly;
using tests::patch_mean;
using tests::sky;
using tests::two_fireflies;
using tests::Vec3;

namespace {

using frame::PassKind;
using tests::restir_pt_graph;

// The whole graph, its settings the written ones (support/schedules.h),
// shown through the tone map.
const frame::Schedule whole_graph = restir_pt_graph(tests::restir_pt_passes);

using Rig = tests::AccumulatingRig;

void check_floor(std::string_view text, std::uint64_t frames,
                 const frame::Schedule& graph) {
  Rig rig(text, graph, {64, 48});
  const auto image = rig.render_accumulated(frames);
  for (const tests::Pixel pixel :
       {tests::Pixel{32, 30}, tests::Pixel{14, 38}, tests::Pixel{50, 26}}) {
    const double expected = tests::expected_floor(rig.description(), rig.size(),
                                                  pixel.x + 0.5, pixel.y + 0.5);
    INFO("pixel " << pixel.x << ", " << pixel.y << ": expected " << expected);
    CHECK(patch_mean(image, rig.size(), {pixel.x, pixel.y, 1}) ==
          doctest::Approx(expected).scale(0).epsilon(0.03));
  }
}

void check_furnace(std::string_view material, std::uint64_t frames,
                   const frame::Schedule& graph) {
  Rig rig(furnace(material), graph, {64, 64});
  const auto image = rig.render_accumulated(frames);
  CHECK(patch_mean(image, rig.size(), {32, 32, 6}) ==
        doctest::Approx(0.5).scale(0).epsilon(0.02));
  CHECK(patch_mean(image, rig.size(), {3, 3, 2}) ==
        doctest::Approx(0.5).scale(0).epsilon(0.01));
}

}  // namespace

TEST_CASE(
    "ReSTIR PT, the whole graph, keeps the path tracer's means: a lit floor, "
    "and the glass furnace") {
  // The path tests' tolerances: 3% at three patches of floor, 2% through
  // glass. The duplication map's bias (Enhanced §5) is where copies
  // crowd, which a floor under two fireflies, resampled from the start,
  // has few of.
  check_floor(lit_floor(two_fireflies), 512, whole_graph);
  // Two of unequal power, 5.76 to 1, so the light tiles draw them
  // unequally (contracts/light_tile.h): the means hold only if every
  // density weighs by the draw's own probability.
  check_floor(lit_floor(tests::one_firefly +
                        "[[shapes]]\nkind = \"sphere\"\ncenter = [-1.2, 1.5, "
                        "0.4]\nradius = 0.12\nmaterial = \"glow\"\n"),
              512, whole_graph);
  check_furnace("kind = \"dielectric\"\nior = 1.5\n", 512, whole_graph);
}

namespace {

// A pair at the marbles' scale: the camera 0.39 m above a table point x1,
// x_{k-1} on the table, x_k on a marble beside it, both diffuse.
tests::CriteriaQuery diffuse_pair(double camera_height, double apart,
                                  double scale) {
  const Vec3 up{0.0, 1.0, 0.0};
  const Vec3 x1 = scale * Vec3{0.0, 0.75, 0.0};
  const Vec3 camera = x1 + scale * Vec3{0.0, camera_height,
                                        camera_height};  // seen at 45 degrees
  const Vec3 before = scale * Vec3{0.01, 0.75, 0.0};
  const Vec3 at =
      before +
      scale * Vec3{0.0, apart,
                   0.0};  // straight up: cos 1 at both ends, x_k facing down
  const double cos_before = 1.0;
  tests::CriteriaQuery q{};
  q.camera = tests::float3(camera);
  q.footprint = 0.02f;
  q.primary = tests::float3(x1);
  q.min_roughness = 0.2f;
  q.primary_normal = tests::float3(up);
  q.lobe_before = contracts::lobe_reflection | contracts::lobe_diffuse;
  q.before = tests::float3(before);
  q.density_before = static_cast<float>(cos_before / std::numbers::pi);
  q.before_normal = tests::float3(up);
  q.lobe_at = contracts::lobe_reflection | contracts::lobe_diffuse;
  q.at = tests::float3(at);
  q.density_at = static_cast<float>(1.0 / std::numbers::pi);
  q.at_normal = tests::float3(-up);
  q.alpha_before = 0.0f;
  return q;
}

}  // namespace

TEST_CASE(
    "the reconnection criteria take a far diffuse pair, refuse a mirror, and "
    "scale with the primary footprint") {
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
      "pt_criteria", count,
      {tests::Binding::of(queries), tests::Binding::of(count)});
  const std::vector<std::uint32_t> expected = {1u, 0u, 0u, 1u, 0u, 0u, 1u};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    INFO("case " << i << ": threshold " << results[i].threshold << ", forward "
                 << results[i].forward << ", inverse " << results[i].inverse);
    CHECK(results[i].connectable == expected[i]);
  }
  // The threshold scales as the primary footprint does, the square of
  // the distance: 100 times for 10 times the scene.
  CHECK(results[4].threshold ==
        doctest::Approx(100.0 * results[2].threshold).epsilon(1e-4));
}

TEST_CASE(
    "each frame's pairing is an involution over the image, its offsets' "
    "deviation sigma") {
  tests::ProbeRunner runner;
  const std::vector<std::uint16_t> textures =
      sampling::reuse_textures(16.0, 0x5e2e417f);
  constexpr std::uint32_t width = 512;
  constexpr std::uint32_t height = 384;
  for (std::uint32_t frame = 0; frame < 4; ++frame) {
    for (std::uint32_t texture = 0; texture < contracts::max_path_neighbors;
         ++texture) {
      const tests::PairingQuery query{frame, texture, width, height};
      const auto results = runner.run<tests::PairingResult>(
          "pt_pairing", std::size_t{width} * height,
          {tests::Binding::of(query), tests::Binding::of(textures)});
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
    return runner.run<tests::PairingResult>(
        "pt_pairing", std::size_t{width} * height,
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

TEST_CASE(
    "with the checkerboard, each frame's pairing is an involution over the "
    "traced field, sigma pixels apart") {
  tests::ProbeRunner runner;
  // As the renderer builds them with the checkerboard: at sigma over the
  // lattice's step (metal/frame/renderer.h).
  const std::vector<std::uint16_t> textures = sampling::reuse_textures(
      16.0 / static_cast<double>(contracts::checkerboard_lattice_step),
      0x5e2e417f);
  constexpr std::uint32_t width = 512;
  constexpr std::uint32_t height = 384;
  for (std::uint32_t frame = 0; frame < 4; ++frame) {
    for (std::uint32_t texture = 0; texture < contracts::max_path_neighbors;
         ++texture) {
      const tests::PairingQuery query{frame, texture, width, height};
      const auto results = runner.run<tests::PairingResult>(
          "pt_lattice_pairing", std::size_t{width} * height,
          {tests::Binding::of(query), tests::Binding::of(textures)});
      std::size_t broken = 0;
      std::size_t untraced = 0;
      std::size_t traced = 0;
      double squares = 0.0;
      for (std::uint32_t i = 0; i < width * height; ++i) {
        const int x = static_cast<int>(i % width);
        const int y = static_cast<int>(i / width);
        if (!contracts::checkerboard_traced(static_cast<std::uint32_t>(x),
                                            static_cast<std::uint32_t>(y),
                                            frame)) {
          continue;
        }
        ++traced;
        const tests::PairingResult r = results[i];
        broken += (r.back_x != x || r.back_y != y) ? 1u : 0u;
        // The partner, inside the image or not, a pixel of the field.
        untraced += contracts::checkerboard_traced(
                        static_cast<std::uint32_t>(r.x),
                        static_cast<std::uint32_t>(r.y), frame)
                        ? 0u
                        : 1u;
        const double dx = r.x - x;
        const double dy = r.y - y;
        squares += dx * dx + dy * dy;
      }
      INFO("frame " << frame << ", texture " << texture);
      CHECK(broken == 0u);
      CHECK(untraced == 0u);
      const double deviation =
          std::sqrt(squares / (2.0 * static_cast<double>(traced)));
      CHECK(deviation == doctest::Approx(16.0).epsilon(0.1));
    }
  }
}

namespace {

// Shifts paths drawn at queries' source pixels to their destinations, in
// `text`'s still scene seen at `size`, through pt_shift (the passes' own
// functions), the scene bound as a pass binds it.
std::vector<tests::ShiftResult> shift(
    const std::string& text, frame::Extent size,
    const std::vector<tests::ShiftQuery>& queries,
    std::uint32_t nee_candidates = passes::written_nee_candidates) {
  const scene::SceneDescription description = scene::parse(text, "test scene");
  metal::Device device;
  metal::Submission submission(device);
  const metal::SceneBuffers buffers(device, submission, description);
  metal::ShapeTransforms transforms(device, submission,
                                    description.shapes.transforms,
                                    metal::FrameArray::Copies::one);
  metal::LightGlows glows(device, submission, description.light_counts.spheres,
                          false);
  const std::vector<std::uint32_t> lights = scene::light_shapes(description);
  metal::SceneAcceleration acceleration(device, submission, description.shapes,
                                        {.moving = {}, .lights = lights});
  metal::Library library(device, metallib::probes);
  const auto pipeline = library.compute_pipeline("pt_shift");
  const auto fill = library.compute_pipeline("pt_light_tiles");
  const auto power = library.compute_pipeline("pt_light_power");
  const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  MTL::Device* mtl = device.handle();
  const contracts::CameraData camera =
      camera::shader_form(description.camera, size);
  contracts::PathReuseSettings settings{};
  settings.nee_candidates = nee_candidates;
  settings.max_surfaces = passes::written_max_surfaces;
  settings.footprint = passes::written_footprint;
  settings.roughness = passes::written_roughness;
  const auto count = static_cast<std::uint32_t>(queries.size());
  const auto buffer_of = [&](const tests::Binding& binding) {
    return tests::shared_buffer(device, submission, binding.bytes());
  };
  const auto out = buffer_of(
      tests::Binding::zeroed(queries.size() * sizeof(tests::ShiftResult)));
  const auto in = buffer_of(tests::Binding::of(queries));
  const auto framed = buffer_of(tests::Binding::of(camera));
  const auto set = buffer_of(tests::Binding::of(settings));
  const auto counted = buffer_of(tests::Binding::of(count));
  constexpr std::size_t tile_entries =
      std::size_t{contracts::light_tiles} * contracts::light_tile_size;
  const auto tiles = buffer_of(
      tests::Binding::zeroed(tile_entries * sizeof(contracts::LightTileEntry)));
  const auto pyramid = buffer_of(tests::Binding::zeroed(
      contracts::light_power_floats(
          contracts::light_power_levels(description.light_counts.lights)) *
      sizeof(float)));
  // MNEE's list, of capacity 0 (its header zeroed): the trees' requests
  // are dropped (kernels/restir_pt_probe.metal, pt_shift).
  const auto mnee_header =
      buffer_of(tests::Binding::zeroed(contracts::mnee_header_bytes));
  auto descriptor =
      NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
  descriptor->setMaxBufferBindCount(13);
  NS::Error* error = nullptr;
  auto table = NS::TransferPtr(mtl->newArgumentTable(descriptor.get(), &error));
  REQUIRE(table);
  const metal::FrameSlot slot = submission.begin();
  MTL4::ComputeCommandEncoder* encoder = slot.commands->computeCommandEncoder();
  REQUIRE(encoder != nullptr);
  table->setAddress(out->gpuAddress(), 0);
  table->setAddress(in->gpuAddress(), 1);
  table->setAddress(framed->gpuAddress(), 2);
  table->setAddress(acceleration.address(slot.slot), 3);
  table->setAddress(buffers.block_address(), 4);
  table->setAddress(glows.address(slot.slot), 5);
  table->setAddress(transforms.address(slot.slot), 6);
  table->setAddress(set->gpuAddress(), 7);
  table->setAddress(counted->gpuAddress(), 8);
  table->setAddress(tiles->gpuAddress(), 9);
  table->setAddress(mnee_header->gpuAddress(), 10);
  table->setAddress(mnee_header->gpuAddress(), 11);
  table->setAddress(pyramid->gpuAddress(), 12);
  encoder->setArgumentTable(table.get());
  encoder->setComputePipelineState(power.get());
  encoder->dispatchThreadgroups(
      MTL::Size(1, 1, 1), MTL::Size(contracts::light_power_threads, 1, 1));
  encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch,
                                     MTL4::VisibilityOptionDevice);
  encoder->setComputePipelineState(fill.get());
  encoder->dispatchThreads(MTL::Size(tile_entries, 1, 1),
                           MTL::Size(fill->threadExecutionWidth(), 1, 1));
  encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch,
                                     MTL4::VisibilityOptionDevice);
  encoder->setComputePipelineState(pipeline.get());
  encoder->dispatchThreads(MTL::Size(queries.size(), 1, 1),
                           MTL::Size(pipeline->threadExecutionWidth(), 1, 1));
  encoder->endEncoding();
  submission.commit();
  (void)submission.wait_until_complete(slot.sequence);
  std::vector<tests::ShiftResult> results(queries.size());
  std::memcpy(results.data(), out->contents(),
              results.size() * sizeof(tests::ShiftResult));
  return results;
}

double luminance(contracts::Float3 c) {
  return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
}

// `draws` lights drawn by power in `text`'s still scene (kernels/
// restir_pt_probe.metal, pt_power_draws), the pyramid built as the initial
// pass builds it.
std::vector<tests::PowerDraw> power_draws(const std::string& text,
                                          std::uint32_t draws) {
  const scene::SceneDescription description = scene::parse(text, "test scene");
  metal::Device device;
  metal::Submission submission(device);
  const metal::SceneBuffers buffers(device, submission, description);
  metal::ShapeTransforms transforms(device, submission,
                                    description.shapes.transforms,
                                    metal::FrameArray::Copies::one);
  metal::LightGlows glows(device, submission, description.light_counts.spheres,
                          false);
  const std::vector<std::uint32_t> lights = scene::light_shapes(description);
  metal::SceneAcceleration acceleration(device, submission, description.shapes,
                                        {.moving = {}, .lights = lights});
  metal::Library library(device, metallib::probes);
  const auto build = library.compute_pipeline("pt_light_power");
  const auto draw = library.compute_pipeline("pt_power_draws");
  const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  const auto buffer_of = [&](const tests::Binding& binding) {
    return tests::shared_buffer(device, submission, binding.bytes());
  };
  const auto out = buffer_of(
      tests::Binding::zeroed(std::size_t{draws} * sizeof(tests::PowerDraw)));
  const auto counted = buffer_of(tests::Binding::of(draws));
  const auto pyramid = buffer_of(tests::Binding::zeroed(
      contracts::light_power_floats(
          contracts::light_power_levels(description.light_counts.lights)) *
      sizeof(float)));
  auto descriptor =
      NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
  descriptor->setMaxBufferBindCount(13);
  NS::Error* error = nullptr;
  auto table = NS::TransferPtr(
      device.handle()->newArgumentTable(descriptor.get(), &error));
  REQUIRE(table);
  const metal::FrameSlot slot = submission.begin();
  MTL4::ComputeCommandEncoder* encoder = slot.commands->computeCommandEncoder();
  REQUIRE(encoder != nullptr);
  table->setAddress(out->gpuAddress(), 0);
  table->setAddress(counted->gpuAddress(), 1);
  table->setAddress(acceleration.address(slot.slot), 3);
  table->setAddress(buffers.block_address(), 4);
  table->setAddress(glows.address(slot.slot), 5);
  table->setAddress(transforms.address(slot.slot), 6);
  table->setAddress(pyramid->gpuAddress(), 12);
  encoder->setArgumentTable(table.get());
  encoder->setComputePipelineState(build.get());
  encoder->dispatchThreadgroups(
      MTL::Size(1, 1, 1), MTL::Size(contracts::light_power_threads, 1, 1));
  encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch,
                                     MTL4::VisibilityOptionDevice);
  encoder->setComputePipelineState(draw.get());
  encoder->dispatchThreads(MTL::Size(draws, 1, 1),
                           MTL::Size(draw->threadExecutionWidth(), 1, 1));
  encoder->endEncoding();
  submission.commit();
  (void)submission.wait_until_complete(slot.sequence);
  std::vector<tests::PowerDraw> results(draws);
  std::memcpy(results.data(), out->contents(),
              results.size() * sizeof(tests::PowerDraw));
  return results;
}

// A floor under a firefly and a large light, a glass ball (replay through
// it), a diffuse ball and a brass mirror; the floor 6 m across, so a point
// on it is stored to 0.05 mm (contract 15). The shift tests' scene.
std::string shift_scene() {
  return camera_text("[0, 2, 3]", "[0, 0, 0]", 50) +
         sky("[0, 0, 0]", "[0, 0, 0]") +
         "[materials.floor]\nkind = \"rough\"\ncolor = [0.8, 0.8, 0.8]\n"
         "[materials.glow]\nkind = \"emissive\"\nradiance = [200, 200, 200]\n"
         "[materials.lamp]\nkind = \"emissive\"\nradiance = [4, 4, 4]\n"
         "[[shapes]]\nkind = \"box\"\nmin = [-3, -1, -3]\nmax = [3, 0, "
         "3]\nmaterial = \"floor\"\n" +
         one_firefly +
         "[[shapes]]\nkind = \"sphere\"\ncenter = [-1.2, 1.8, -1.0]\nradius = "
         "0.6\nmaterial = \"lamp\"\n"
         "[materials.glass]\nkind = \"dielectric\"\nior = 1.5\n"
         "[materials.white]\nkind = \"rough\"\ncolor = [0.7, 0.7, 0.7]\n"
         "[materials.brass]\nkind = \"conductor\"\nf0 = [0.9, 0.7, "
         "0.3]\nroughness = 0.05\n"
         "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0.5, 0]\nradius = "
         "0.5\nmaterial = \"glass\"\n"
         "[[shapes]]\nkind = \"sphere\"\ncenter = [1, 0.4, -0.3]\nradius = "
         "0.4\nmaterial = \"white\"\n"
         "[[shapes]]\nkind = \"sphere\"\ncenter = [-0.9, 0.4, 0.2]\nradius = "
         "0.4\nmaterial = \"brass\"\n";
}

}  // namespace

TEST_CASE(
    "the light tiles' draw takes each light in proportion to its power, with "
    "the probability MIS asks") {
  // Five fireflies of powers 1 : 4 : 1/4 : 9 : 1 (radius^2 x luminance,
  // contracts/light_tile.h), so 8 leaves, three of no light: RTXDI's
  // presampling, by descent of the power pyramid
  // (light_selection/power_light.metal.h).
  const std::string text =
      camera_text("[0, 2, 3]", "[0, 0, 0]", 50) +
      sky("[0, 0, 0]", "[0, 0, 0]") +
      "[materials.floor]\nkind = \"rough\"\ncolor = [0.8, 0.8, 0.8]\n"
      "[materials.bright]\nkind = \"emissive\"\nradiance = [200, 200, 200]\n"
      "[materials.dim]\nkind = \"emissive\"\nradiance = [50, 50, 50]\n"
      "[[shapes]]\nkind = \"box\"\nmin = [-3, -1, -3]\nmax = [3, 0, "
      "3]\nmaterial = \"floor\"\n"
      "[[shapes]]\nkind = \"sphere\"\ncenter = [0.3, 2.5, -0.5]\nradius = "
      "0.05\nmaterial = \"bright\"\n"
      "[[shapes]]\nkind = \"sphere\"\ncenter = [-1.2, 1.5, 0.4]\nradius = "
      "0.1\nmaterial = \"bright\"\n"
      "[[shapes]]\nkind = \"sphere\"\ncenter = [1.2, 1.5, 0.4]\nradius = "
      "0.05\nmaterial = \"dim\"\n"
      "[[shapes]]\nkind = \"sphere\"\ncenter = [0.0, 1.0, 1.0]\nradius = "
      "0.15\nmaterial = \"bright\"\n"
      "[[shapes]]\nkind = \"sphere\"\ncenter = [0.0, 1.5, -1.0]\nradius = "
      "0.05\nmaterial = \"bright\"\n";
  const std::array<double, 5> power = {1.0, 4.0, 0.25, 9.0, 1.0};
  const double total = 15.25;
  constexpr std::uint32_t draws = 1u << 20;
  const std::vector<tests::PowerDraw> drawn = power_draws(text, draws);
  std::array<std::uint32_t, 5> counts{};
  for (const tests::PowerDraw& d : drawn) {
    REQUIRE(d.index < power.size());  // never a leaf of no light
    ++counts.at(d.index);
    const double expected = power.at(d.index) / total;
    CHECK(d.probability == doctest::Approx(expected).epsilon(1e-5));
    CHECK(d.asked == d.probability);  // MIS's and the draw's are one number
  }
  for (std::size_t i = 0; i < power.size(); ++i) {
    const double p = power.at(i) / total;
    const double sigma = std::sqrt(draws * p * (1.0 - p));
    INFO("light " << i << ": " << counts.at(i) << " draws, " << draws * p
                  << " expected");
    CHECK(std::abs(counts.at(i) - draws * p) < 5.0 * sigma);
  }
}

TEST_CASE(
    "a path shifted to its own pixel keeps its F and has a Jacobian of 1, in "
    "every case the shift has") {
  // shift_scene(), with one NEE candidate, so rays the BSDF draws to a
  // light are kept often. Every case of shift.metal.h's step 6: replayed
  // whole; reconnected to a light NEE drew (forced) or a ray struck; at a
  // surface whose direction NEE drew; at a surface whose BSDF drew it.
  const std::string text = shift_scene();
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
  enum class Case : std::uint8_t { whole, forced, struck, nee_at_k, bsdf_at_k };
  constexpr std::array<std::string_view, 5> names = {
      "replayed whole", "to NEE's light", "to a struck light", "at x_k, NEE on",
      "at x_k, BSDF on"};
  std::array<std::size_t, names.size()> held{};
  std::array<std::size_t, names.size()> off{};
  for (const tests::ShiftResult& r : results) {
    if (contracts::path_end(r.flags) == contracts::PathEnd::none ||
        luminance(r.stored) <= 0.0) {
      continue;
    }
    const std::uint32_t k = contracts::path_rc(r.flags);
    const bool at_light = k == contracts::path_length(r.flags);
    const Case c =
        k == 0u    ? Case::whole
        : at_light ? (contracts::path_end(r.flags) == contracts::PathEnd::nee
                          ? Case::forced
                          : Case::struck)
                   : (contracts::path_lobe_at(r.flags) == 0u ? Case::nee_at_k
                                                             : Case::bsdf_at_k);
    ++held.at(static_cast<std::size_t>(c));
    // To the stored point's rounding (contract 15) and the G-buffer's
    // packing of y1 (contract 5: its normal to 10^-4 radians, which
    // turns every replayed direction after it by as much), which a
    // glossy x_k magnifies in its density and a deep replay carries
    // along: 2% (1% before the G-buffer, the retraced y1 exact).
    const bool same = r.succeeded == 1u &&
                      std::abs(r.jacobian - 1.0) < 2.0e-2 &&
                      std::abs(luminance(r.shifted) - luminance(r.stored)) <=
                          2.0e-2 * luminance(r.stored);
    off.at(static_cast<std::size_t>(c)) += same ? 0u : 1u;
  }
  for (std::size_t c = 0; c < names.size(); ++c) {
    // A pair at the criteria's edge may fall the other way at the
    // stored point, and the shift fail: at most 1 in 200.
    INFO(names.at(c) << ": " << held.at(c) << " paths, " << off.at(c)
                     << " off");
    CHECK(held.at(c) >= 20u);
    CHECK(off.at(c) * 200u <= held.at(c));
  }
}

TEST_CASE(
    "a reconnection's Jacobian is supplemental eq. 4's on a floor under a "
    "firefly") {
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
        queries.push_back(
            {x, y, x + 3, y - 2, frame, size.width, size.height, 0u});
      }
    }
  }
  const scene::SceneDescription description = scene::parse(text, "test scene");
  const lights::SphereLightData& light = description.sphere_lights.at(0);
  const contracts::Transform& placed =
      description.shapes.transforms.at(light.shape);
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
    const double solid_angle =
        2.0 * std::numbers::pi * (1.0 - std::sqrt(1.0 - radius * radius / dc2));
    const double direction_density =
        nee ? 1.0 / solid_angle : w.y / std::numbers::pi;  // one light: P = 1
    return direction_density * cos_light / d2;
  };
  std::size_t checked = 0;
  for (const tests::ShiftResult& r : results) {
    // A path the BSDF drew to the light past the footprints' bound is
    // replayed whole, its Jacobian 1: not this test's.
    if (contracts::path_end(r.flags) == contracts::PathEnd::none ||
        r.succeeded == 0u || contracts::path_rc(r.flags) == 0u) {
      continue;
    }
    REQUIRE(contracts::path_length(r.flags) == 2u);
    REQUIRE(contracts::path_rc(r.flags) == 2u);
    const bool nee = contracts::path_end(r.flags) == contracts::PathEnd::nee;
    const Vec3 x2 = tests::vec(r.reconnection);
    const double expected =
        area_density(tests::vec(r.destination_hit), x2, nee) /
        area_density(tests::vec(r.source_hit), x2, nee);
    INFO(std::string(nee ? "NEE" : "BSDF")
         << ": J " << r.jacobian << ", eq. 4 " << expected);
    CHECK(r.jacobian == doctest::Approx(expected).epsilon(2.0e-3));
    ++checked;
  }
  CHECK(checked > 500u);
}

namespace {

// A path a reservoir holds, ending on NEE's light at x2 (k = d = 2): its
// grey F, its weight W and count M, the (pixel, frame) it was drawn at, and
// its cached Jacobian denominator. Named, so no two can be swapped (I.24).
struct Held {
  float f = 1.0f;
  float weight = 1.0f;
  std::uint32_t count = 1;
  std::uint32_t pixel = 0;
  std::uint32_t frame = 0;
  float jacobian = 1.0f;
};

contracts::PathReservoir held_path(const Held& held) {
  contracts::PathReservoir r{};
  r.weight = held.weight;
  r.integrand = contracts::Float3{held.f, held.f, held.f};
  r.seed_pixel = held.pixel;
  r.seed_frame = held.frame;
  r.flags = contracts::pack_path_flags(contracts::PathShape{
      held.count, 2u, 2u, contracts::PathEnd::nee, 0u, 0u, 0u});
  r.jacobian = held.jacobian;
  return r;
}

}  // namespace

TEST_CASE(
    "the temporal merge counts last frame's path at most c_cap times the own "
    "tree, c_cap lowered by "
    "duplication, and weighs by Talbot MIS") {
  // Grey F, so p-hat is F's value. The own path: F 2, W 0.5, count 1,
  // shifted there to F 1.5 with J 0.8. Last frame's: F 3, W 0.25, count
  // 40, shifted here to F 2.5 with J 1.2 (its cached denominator 0.7).
  const contracts::PathReservoir own = held_path({.f = 2.0f,
                                                  .weight = 0.5f,
                                                  .count = 1,
                                                  .pixel = 11,
                                                  .frame = 100,
                                                  .jacobian = 0.3f});
  const contracts::PathReservoir last = held_path({.f = 3.0f,
                                                   .weight = 0.25f,
                                                   .count = 40,
                                                   .pixel = 22,
                                                   .frame = 99,
                                                   .jacobian = 0.7f});
  const auto query = [&](float cap, float floor, float exponent,
                         float duplication, float u1) {
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
      {query(20.0f, 0.0f, 0.0f, 0.9f, 0.99f),
       20.0},  // no duplication map: the cap itself, D unread
      {query(20.0f, 1.0f, 0.1f, 0.0f, 0.99f), 20.0},  // D = 0
      {query(20.0f, 1.0f, 0.1f, 1.0f, 0.99f), 1.0},   // D = 1: the floor
      {query(20.0f, 1.0f, 0.1f, 0.25f, 0.99f), quarter_cap},
      {query(20.0f, 1.0f, 0.1f, 0.25f, 0.0f),
       quarter_cap},  // the second number keeps last's path
  };
  std::vector<tests::PathMergeQuery> queries;
  for (const Case& c : cases) {
    queries.push_back(c.query);
  }
  tests::ProbeRunner probe;
  const auto results = probe.run<tests::PathMergeResult>(
      "pt_temporal_merge", queries.size(),
      {tests::Binding::of(queries),
       tests::Binding::of(static_cast<std::uint32_t>(queries.size()))});
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
    CHECK(contracts::path_count(r.kept.flags) ==
          static_cast<std::uint32_t>(std::lround(c_own + c_last)));
    // The kept path: last's where the second number is under its share
    // of the sum, its F the shifted one, its denominator times J.
    const bool last_kept = queries[i].u1 * (w_own + w_last) < w_last;
    CHECK(r.kept.seed_pixel == (last_kept ? 22u : 11u));
    const double kept_f = last_kept ? 2.5 : 2.0;
    CHECK(r.kept.integrand.x == doctest::Approx(kept_f));
    CHECK(r.kept.jacobian == doctest::Approx(last_kept ? 0.7 * 1.2 : 0.3));
    // W = the weights' sum over the kept path's p-hat: F W is the sum.
    CHECK(r.kept.weight * kept_f ==
          doctest::Approx(w_own + w_last).epsilon(1.0e-5));
  }
  CHECK(contracts::path_count(results[2].kept.flags) ==
        2u);  // the floor's: 1 + 1
}

namespace {

// The duplication map of a `size` image of reservoirs, through the pass's
// functions, dispatched as the pass dispatches them.
std::vector<float> duplication_map(
    const std::vector<contracts::PathReservoir>& image, frame::Extent size) {
  metal::Device device;
  metal::Submission submission(device);
  metal::Library library(device, metallib::probes);
  const auto pipeline = library.compute_pipeline("pt_duplication");
  REQUIRE(pipeline->maxTotalThreadsPerThreadgroup() >=
          contracts::duplication_group * contracts::duplication_group);
  const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  MTL::Device* mtl = device.handle();
  const auto buffer_of = [&](const tests::Binding& binding) {
    return tests::shared_buffer(device, submission, binding.bytes());
  };
  const std::array<std::uint32_t, 2> extent{size.width, size.height};
  const auto out =
      buffer_of(tests::Binding::zeroed(image.size() * sizeof(float)));
  const auto in = buffer_of(tests::Binding::of(image));
  const auto sized = buffer_of(tests::Binding::of(extent));
  auto descriptor =
      NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
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
  encoder->dispatchThreadgroups(MTL::Size((size.width + group - 1) / group,
                                          (size.height + group - 1) / group, 1),
                                MTL::Size(group, group, 1));
  encoder->endEncoding();
  submission.commit();
  (void)submission.wait_until_complete(slot.sequence);
  std::vector<float> map(image.size());
  std::memcpy(map.data(), out->contents(), map.size() * sizeof(float));
  return map;
}

}  // namespace

TEST_CASE(
    "the duplication map counts the copies of each pixel's path in its 17 x 17 "
    "neighborhood") {
  // 70 x 50, so groups at the right and bottom are cut by the image.
  constexpr frame::Extent size{70, 50};
  const auto at = [&](std::uint32_t x, std::uint32_t y) {
    return std::size_t{y} * size.width + x;
  };
  std::vector<contracts::PathReservoir> image(std::size_t{size.width} *
                                              size.height);
  for (std::uint32_t y = 0; y < size.height; ++y) {
    for (std::uint32_t x = 0; x < size.width; ++x) {
      image[at(x, y)] = held_path(
          {.pixel = x + 65536u * y, .frame = 7});  // each its own path
    }
  }
  // A 5 x 5 block of copies of one path.
  for (std::uint32_t y = 10; y < 15; ++y) {
    for (std::uint32_t x = 10; x < 15; ++x) {
      image[at(x, y)] = held_path({.pixel = 12u + 65536u * 12u, .frame = 7});
    }
  }
  // Beside it, a pixel holding the same stream with a weight of 0: no
  // path, no one's copy.
  image[at(16, 12)] =
      held_path({.weight = 0.0f, .pixel = 12u + 65536u * 12u, .frame = 7});
  // Three copies in a row 8 apart: the middle sees both, each end the
  // middle only (17 apart is past the window).
  for (const std::uint32_t x : {40u, 48u, 56u}) {
    image[at(x, 30)] = held_path({.pixel = 3, .frame = 3});
  }
  // The same pixel key, another frame: another path.
  image[at(44, 30)] = held_path({.pixel = 3, .frame = 4});
  const std::vector<float> map = duplication_map(image, size);
  for (std::uint32_t y = 0; y < size.height; ++y) {
    for (std::uint32_t x = 0; x < size.width; ++x) {
      const bool block = x >= 10 && x < 15 && y >= 10 && y < 15;
      const bool row = y == 30 && (x == 40 || x == 48 || x == 56);
      const double expected = block ? 24.0 / 288.0
                              : row ? (x == 48 ? 2.0 : 1.0) / 288.0
                                    : 0.0;
      INFO("pixel " << x << ", " << y);
      CHECK(map[at(x, y)] == doctest::Approx(expected).epsilon(1.0e-6));
    }
  }

  // Every pixel one path: 1 where the window is inside the image, the
  // window's share inside it at the edges and corners.
  std::vector<contracts::PathReservoir> same(
      image.size(), held_path({.pixel = 5, .frame = 5}));
  const std::vector<float> full = duplication_map(same, size);
  CHECK(full[at(35, 25)] == doctest::Approx(1.0));
  CHECK(full[at(0, 0)] == doctest::Approx(80.0 / 288.0));
  CHECK(full[at(69, 49)] == doctest::Approx(80.0 / 288.0));
  CHECK(full[at(69, 25)] == doctest::Approx((9.0 * 17.0 - 1.0) / 288.0));
}

TEST_CASE(
    "a shift from the G-buffer's record gives the shift from the retraced "
    "primary hit, to its packing") {
  // shift_scene(), as the self-shift test, each path shifted 3 pixels right and
  // 2 up: through the record the destination's tree packed (contract 5),
  // as the passes shift, and from its primary ray traced and resolved
  // again, as Phase 1 did. They differ by the record's packing (its
  // normal and wo to 10^-4 radians, its BSDF to a half's precision),
  // which replay carries along: whether each succeeds the same, and F and
  // the Jacobian within 2%, for all but 1 in 200.
  const std::string text = shift_scene();
  constexpr frame::Extent size{96, 72};
  std::vector<tests::ShiftQuery> queries;
  for (std::uint32_t frame = 0; frame < 8; ++frame) {
    for (std::uint32_t y = 2; y < size.height; y += 2) {
      for (std::uint32_t x = 0; x + 3 < size.width; x += 2) {
        queries.push_back(
            {x, y, x + 3, y - 2, frame, size.width, size.height, 0u});
      }
    }
  }
  const auto results = shift(text, size, queries, 1u);
  std::size_t tried = 0;
  std::size_t succeeded = 0;
  std::size_t through_glass = 0;
  std::size_t off = 0;
  for (const tests::ShiftResult& r : results) {
    if (contracts::path_end(r.flags) == contracts::PathEnd::none ||
        luminance(r.stored) <= 0.0) {
      continue;
    }
    ++tried;
    succeeded += r.succeeded;
    through_glass +=
        r.succeeded == 1u && contracts::path_rc(r.flags) != 2u ? 1u : 0u;
    const double f = luminance(r.retraced);
    const bool same = r.succeeded == r.retraced_succeeded &&
                      (r.succeeded == 0u ||
                       (std::abs(r.jacobian - r.retraced_jacobian) <=
                            2.0e-2 * r.retraced_jacobian &&
                        std::abs(luminance(r.shifted) - f) <= 2.0e-2 * f));
    off += same ? 0u : 1u;
  }
  INFO(tried << " paths, " << succeeded << " shifted, " << through_glass
             << " replayed or reconnected past y2, " << off << " off");
  CHECK(succeeded > 1000u);
  CHECK(through_glass > 100u);
  CHECK(off * 200u <= tried);
}
