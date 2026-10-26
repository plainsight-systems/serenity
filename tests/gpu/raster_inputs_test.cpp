// Metal's rule for the frame it traces (metal/frame/raster_inputs.h), a unit
// with no Metal call: through a pinhole in any graph whose features Metal
// rasterizes, beside the core's rule (a graph that feeds the network), the
// rest of each camera and of the frame as given; the scene's lens in a graph
// with neither.

#include "metal/frame/raster_inputs.h"

#include <doctest/doctest.h>

#include <vector>

#include "core/frame/frame_inputs.h"
#include "core/frame/schedule.h"
#include "support/required.h"
#include "support/schedules.h"

namespace serenity::metal {
namespace {

using frame::PassKind;

frame::Schedule schedule_of(const std::vector<PassKind>& kinds) {
  frame::Schedule s{kinds, tests::look};
  s.restir_pt = tests::written_restir_pt(kinds);
  return s;
}

TEST_CASE(
    "Metal traces a graph whose features it rasterizes through a pinhole, "
    "and keeps the lens elsewhere") {
  contracts::Camera lensed;
  lensed.position = {0.0f, 0.0f, 0.0f};
  lensed.look_at = {0.0f, 0.0f, -1.0f};
  lensed.up = {0.0f, 1.0f, 0.0f};
  lensed.vertical_fov_degrees = 30.0f;
  lensed.lens_radius = 0.01f;
  lensed.focus_distance = 2.0f;
  contracts::Camera before = lensed;
  before.position = {0.1f, 0.0f, 0.0f};
  frame::FrameInputs inputs{.time = frame::Seconds(1.0),
                            .index = 7,
                            .accumulated_since = 7,
                            .camera = lensed};
  inputs.previous = frame::PreviousFrame{frame::Seconds(0.9), before};

  // The features alone (the data set's full-size albedo): the core keeps
  // the lens; Metal, which rasterizes them, takes it off.
  const frame::Schedule alone = schedule_of({PassKind::features});
  CHECK(
      tests::required(frame::traced_inputs(alone, inputs).camera).lens_radius ==
      lensed.lens_radius);
  const frame::FrameInputs rasterized = raster_inputs(alone, inputs);
  CHECK(tests::required(rasterized.camera).lens_radius == 0.0f);
  CHECK(tests::required(rasterized.previous).camera.lens_radius == 0.0f);
  CHECK(tests::required(rasterized.camera).focus_distance ==
        lensed.focus_distance);
  CHECK(tests::required(rasterized.previous).camera.position.x ==
        before.position.x);
  CHECK(rasterized.index == inputs.index);
  CHECK(rasterized.time == inputs.time);

  // Without the features pass, the core's rule alone: the lens kept.
  const frame::Schedule plain =
      schedule_of({PassKind::restir_pt_initial, PassKind::restir_pt_shade,
                   PassKind::tone_map});
  const frame::FrameInputs kept = raster_inputs(plain, inputs);
  CHECK(tests::required(kept.camera).lens_radius == lensed.lens_radius);
  CHECK(tests::required(kept.previous).camera.lens_radius ==
        before.lens_radius);
}

}  // namespace
}  // namespace serenity::metal
