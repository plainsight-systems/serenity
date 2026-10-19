// What a scene or a frame may not be (core/frame/refusals.h): each refusal
// fires past its contract's limit and not at it, and says what it refused.

#include <doctest/doctest.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include "core/frame/refusals.h"
#include "core/scene/scene.h"
#include "support/scene_text.h"
#include "support/schedules.h"

namespace serenity::frame {
namespace {

bool says(const std::optional<std::string>& reason, const std::string& text) {
  return reason && reason->find(text) != std::string::npos;
}

scene::SceneDescription fireflies() {
  return scene::parse(tests::flying_fireflies(), "flying fireflies");
}

FrameInputs frame(std::uint64_t index, double seconds) {
  const scene::SceneDescription s = fireflies();
  return FrameInputs{
      .time = Seconds(seconds), .index = index, .camera = s.camera};
}

TEST_CASE("a time the shaders' float holds is not refused; past it, it is") {
  CHECK_FALSE(time_refusal(Seconds(0.0), 3, "its time"));
  CHECK_FALSE(time_refusal(Seconds(-1.0e30), 3, "its time"));
  const double past = 2.0 * std::numeric_limits<float>::max();
  CHECK(says(time_refusal(Seconds(past), 3, "its time"), "frame 3: its time"));
  CHECK(time_refusal(Seconds(std::numeric_limits<double>::infinity()), 3,
                     "its time"));
  CHECK(time_refusal(Seconds(std::numeric_limits<double>::quiet_NaN()), 3,
                     "its time"));
}

TEST_CASE("a graph that reads a scene is refused without one") {
  const Schedule path{{PassKind::path, PassKind::tone_map}, tests::look};
  const Schedule pattern{{PassKind::test_pattern}, tests::look};
  const scene::SceneDescription s = fireflies();
  CHECK(says(scene_refusal(path, nullptr), "none was given"));
  CHECK_FALSE(scene_refusal(path, &s));
  CHECK_FALSE(scene_refusal(pattern, nullptr));
}

TEST_CASE("a frame of a graph that reads a scene needs a camera") {
  FrameInputs inputs = frame(4, 0.0);
  CHECK_FALSE(camera_refusal(inputs, true));
  inputs.camera.reset();
  CHECK(says(camera_refusal(inputs, true), "frame 4:"));
  CHECK_FALSE(camera_refusal(inputs, false));
}

TEST_CASE("a frame before whose camera cannot be framed is refused") {
  FrameInputs inputs = frame(5, 0.0);
  CHECK_FALSE(previous_camera_refusal(inputs));  // none before
  inputs.previous = PreviousFrame{Seconds(0.0), *inputs.camera};
  CHECK_FALSE(previous_camera_refusal(inputs));
  inputs.previous->camera = contracts::Camera{};  // looks at itself
  CHECK(says(previous_camera_refusal(inputs), "frame 5: the frame before's"));
}

TEST_CASE("ReSTIR PT refuses what its G-buffer and light tiles cannot pack") {
  CHECK_FALSE(restir_pt_refusal(fireflies()));

  scene::SceneDescription media = fireflies();
  media.media.resize(0xfffe);  // the most a 16-bit index holds with "none"
  CHECK_FALSE(restir_pt_refusal(media));
  media.media.resize(0xffff);
  CHECK(says(restir_pt_refusal(media), "65535 media"));

  scene::SceneDescription shapes = fireflies();
  shapes.shapes.records.resize(65536);
  CHECK_FALSE(restir_pt_refusal(shapes));
  shapes.shapes.records.resize(65537);
  CHECK(says(restir_pt_refusal(shapes), "65537 shapes"));

  scene::SceneDescription lights = fireflies();
  lights.light_counts.lights = 65537;
  CHECK(says(restir_pt_refusal(lights), "65537 lights"));
}

TEST_CASE("MNEE refuses a shape its requests cannot name") {
  scene::SceneDescription s = fireflies();
  s.shapes.records.resize(65536);
  CHECK_FALSE(mnee_refusal(s));
  s.shapes.records.resize(65537);
  CHECK(says(mnee_refusal(s), "65537 shapes"));
}

TEST_CASE("a glowing light must be one of the scene's sphere lights") {
  scene::SceneDescription s = fireflies();
  REQUIRE_FALSE(s.animation.glowers.empty());
  CHECK_FALSE(glower_refusal(s));
  s.animation.glowers.back().target = s.light_counts.spheres;
  CHECK(says(glower_refusal(s), "is not one of the scene's"));
}

}  // namespace
}  // namespace serenity::frame
