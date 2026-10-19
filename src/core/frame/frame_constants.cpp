#include "core/frame/frame_constants.h"

#include "core/camera/thin_lens.h"

namespace serenity::frame {

contracts::FrameConstants frame_constants(
    const FrameInputs& inputs, Extent at, std::uint32_t accumulated_frames,
    std::optional<sampling::CameraJitter> jitter) {
  const sampling::CameraJitter offset =
      jitter.value_or(sampling::CameraJitter{});
  return contracts::FrameConstants{
      static_cast<float>(inputs.time.count()),
      static_cast<std::uint32_t>(inputs.index),
      at.width,
      at.height,
      accumulated_frames,
      offset.x,
      offset.y,
      jitter ? 1u : 0u,
  };
}

contracts::FeatureConstants feature_constants(
    const std::optional<PreviousFrame>& previous, Extent size) {
  contracts::FeatureConstants features{};
  features.previous = previous ? 1u : 0u;
  if (previous) {
    features.previous_camera = camera::shader_form(previous->camera, size);
  }
  return features;
}

contracts::PathReuseConstants path_reuse_constants(
    const std::optional<contracts::Camera>& previous_camera, Extent light) {
  contracts::PathReuseConstants reuse{};
  reuse.previous = previous_camera ? 1u : 0u;
  if (previous_camera) {
    reuse.previous_camera = camera::shader_form(*previous_camera, light);
  }
  return reuse;
}

contracts::DenoiseConstants denoise_constants(const contracts::Camera& camera,
                                              const DenoiseFrame& frame) {
  const camera::ViewAxes axes = camera::view_axes(camera);
  contracts::DenoiseConstants k{};
  k.right = axes.right;
  k.up = axes.up;
  k.backward = axes.backward;
  k.light_width = frame.light.width;
  k.light_height = frame.light.height;
  k.history = frame.history ? 1u : 0u;
  k.padded_width = frame.padded.width;
  k.padded_height = frame.padded.height;
  k.output_width = frame.output.width;
  k.output_height = frame.output.height;
  k.features = frame.features;
  k.jitter_x = frame.jitter.x;
  k.jitter_y = frame.jitter.y;
  return k;
}

contracts::CheckerboardConstants checkerboard_constants(bool history) {
  return contracts::CheckerboardConstants{history ? 1u : 0u, {0u, 0u, 0u}};
}

Extent traced_grid(Extent light, bool checkerboard) {
  return checkerboard
             ? Extent{contracts::checkerboard_width(light.width), light.height}
             : light;
}

}  // namespace serenity::frame
