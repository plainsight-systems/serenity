#include "metal/frame/raster_inputs.h"

namespace serenity::metal {

frame::FrameInputs raster_inputs(const frame::Schedule& schedule,
                                 const frame::FrameInputs& inputs) {
  frame::FrameInputs traced = frame::traced_inputs(schedule, inputs);
  if (!frame::writes_features(schedule)) {
    return traced;
  }
  if (traced.camera) {
    traced.camera->lens_radius = 0.0f;
  }
  if (traced.previous) {
    traced.previous->camera.lens_radius = 0.0f;
  }
  return traced;
}

}  // namespace serenity::metal
