// The denoiser's features (contract 17, core/contracts/features.h) and the
// light beside them, two integration tests. The training data is what the
// network sees (logical-overview.md, principle 13): the data set's input
// graph, ReSTIR PT with the features pass and no network
// (graphs/restir_pt_features.toml), writes for a frame the very light and
// features the denoised graph hands its network for that frame, to the bit,
// and the jitter the network is given; and every ray of such a frame passes
// through the frame's camera jitter.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/animation/animate.h"
#include "core/camera/thin_lens.h"
#include "core/frame/schedule.h"
#include "core/sampling/jitter.h"
#include "core/scene/scene.h"
#include "gpu/support/rendering.h"
#include "support/lit_floor.h"
#include "support/same_bits.h"
#include "support/scene_text.h"
#include "support/schedules.h"

// the backend's frames, by the include path (SF.12)
#include <frame_rig.h>

namespace animation = serenity::animation;
namespace camera = serenity::camera;
namespace contracts = serenity::contracts;
namespace frame = serenity::frame;
namespace sampling = serenity::sampling;
namespace scene = serenity::scene;
namespace tests = serenity::tests;

namespace {

constexpr frame::Extent size{64, 48};

// ReSTIR PT's whole graph, the features pass beside it where `features`
// says, shown through the tone map, its settings the written ones.
frame::Schedule light_graph(bool features) {
  std::vector<frame::PassKind> kinds = tests::restir_pt_passes;
  if (features) {
    kinds.push_back(frame::PassKind::features);
  }
  kinds.push_back(frame::PassKind::tone_map);
  frame::Schedule s{kinds, tests::look};
  s.restir_pt = tests::written_restir_pt(kinds);
  return s;
}

}  // namespace

TEST_CASE(
    "the data set's input graph writes, frame by frame, the light, features "
    "and jitter the demo hands its "
    "network") {
  // Wandering fireflies about a brass ball, so the fill moves pixels and
  // the reuse carries history; no glass, so no MNEE request is made and
  // the light is reproducible to the bit (MNEE's float atomics add in no
  // fixed order: docs/process/QUEUE.md, 0b). The input graph at the light
  // size, the demo (the fixture's network: rendering.h) at twice it, the
  // same frames from the same history.
  const scene::SceneDescription description =
      scene::parse(tests::wandering_fireflies(), "wandering fireflies");
  REQUIRE(animation::moves(description.animation));
  const frame::Schedule input = light_graph(true);
  const frame::Schedule demo = tests::fixture_denoised_graph();
  REQUIRE(frame::feeds_network(input));
  REQUIRE_FALSE(frame::denoises(input));
  tests::FrameRig rig;
  const auto input_target = rig.target(size);
  const auto demo_target = rig.target({2 * size.width, 2 * size.height});
  const auto input_renderer = rig.renderer(input, &description);
  const auto demo_renderer = rig.renderer(demo, &description);
  for (std::uint64_t i = 0; i < 5; ++i) {
    // Its reuse history kept from frame 0 (reused_since, the default);
    // the frame before, which the features' motion points to.
    frame::FrameInputs inputs = tests::frame_at(
        i, i, static_cast<double>(i) / 60.0, description.camera);
    if (i > 0) {
      inputs.previous = frame::PreviousFrame{
          frame::Seconds(static_cast<double>(i - 1) / 60.0),
          description.camera};
    }
    rig.render(*input_target, *input_renderer, inputs);
    rig.render(*demo_target, *demo_renderer, inputs);
    INFO("frame " << i);
    const auto written = input_renderer->read_network_input();
    const auto handed = demo_renderer->read_network_input();
    REQUIRE(written.size == size);
    REQUIRE(handed.size == size);
    CHECK(tests::same_bits(written.rgba, handed.rgba));
    // What the headless renderer writes of the input graph is that light.
    std::vector<float> shown(written.rgba.size());
    input_renderer->read_accumulated(shown);
    CHECK(tests::same_bits(shown, written.rgba));
    const auto a = input_renderer->read_features();
    const auto b = demo_renderer->read_features();
    CHECK(tests::same_bits(a.albedo.rgb, b.albedo.rgb));
    CHECK(tests::same_bits(a.normal.rgb, b.normal.rgb));
    CHECK(tests::same_bits(a.depth.rgb, b.depth.rgb));
    CHECK(tests::same_bits(a.motion.rgb, b.motion.rgb));
    // The jitter the network reads, the data set's script's (pinned to
    // the same values, tests/jitter_test.cpp).
    const sampling::CameraJitter jitter = sampling::camera_jitter(i);
    CHECK(written.jitter.x == jitter.x);
    CHECK(written.jitter.y == jitter.y);
    CHECK(handed.jitter.x == jitter.x);
    CHECK(handed.jitter.y == jitter.y);
    // And it holds light: not two black images alike.
    CHECK(std::ranges::max(written.rgba) > 0.0f);
  }
  CHECK(input_renderer->non_finite_samples() == 0);
}

TEST_CASE(
    "a ReSTIR PT frame's camera rays, the features' with them, all pass "
    "through the frame's jitter") {
  // The features' depth of the lit floor at each pixel is the floor's
  // depth along the ray through the pixel's center plus the frame's
  // jitter (core/sampling/jitter.h), every pixel the same offset: a
  // pixel's own random point would be up to half a pixel away, and the
  // floor's depth changes some 1% a pixel.
  const scene::SceneDescription description =
      scene::parse(tests::lit_floor(tests::one_firefly), "lit floor");
  tests::FrameRig rig;
  const auto target = rig.target(size);
  const auto renderer = rig.renderer(light_graph(true), &description);
  const contracts::CameraData camera =
      camera::shader_form(description.camera, size);
  const tests::Vec3 origin = tests::vec(camera.origin);
  const tests::Vec3 forward = tests::vec(camera.forward);
  for (std::uint64_t i = 0; i < 3; ++i) {
    rig.render(*target, *renderer,
               tests::frame_at(i, i, static_cast<double>(i) / 60.0,
                               description.camera));
    const contracts::LinearImage depth = renderer->read_features().depth;
    const sampling::CameraJitter jitter = sampling::camera_jitter(i);
    double worst = 0.0;
    std::size_t floor_pixels = 0;
    for (std::uint32_t y = 0; y < size.height; ++y) {
      for (std::uint32_t x = 0; x < size.width; ++x) {
        const float seen = depth.rgb[(std::size_t{y} * size.width + x) * 3];
        if (seen <= 0.0f) {
          continue;  // the sky
        }
        const tests::Vec3 at = tests::floor_point(
            description, size, x + 0.5 + jitter.x, y + 0.5 + jitter.y);
        const double expected = tests::dot(at - origin, forward);
        worst = std::max(worst, std::abs(seen - expected) / expected);
        ++floor_pixels;
      }
    }
    INFO("frame " << i << ": jitter (" << jitter.x << ", " << jitter.y
                  << "), worst relative depth error " << worst);
    CHECK(floor_pixels > size.width * size.height / 2);
    CHECK(worst < 1e-5);
  }
}
