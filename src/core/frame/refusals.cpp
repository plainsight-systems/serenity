#include "core/frame/refusals.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "core/camera/thin_lens.h"
#include "core/contracts/light_tile.h"
#include "core/contracts/mnee_request.h"
#include "core/contracts/primary_hit.h"
#include "core/scene/packed_range.h"

namespace serenity::frame {

std::optional<std::string> time_refusal(Seconds time, std::uint64_t index,
                                        std::string_view which) {
  const double seconds = time.count();
  if (std::isfinite(seconds) &&
      std::abs(seconds) <= std::numeric_limits<float>::max()) {
    return std::nullopt;
  }
  return "frame " + std::to_string(index) + ": " + std::string(which) + ", " +
         std::to_string(seconds) + " s, is past what the shaders' float holds";
}

std::optional<std::string> scene_refusal(const Schedule& schedule,
                                         const scene::SceneDescription* scene) {
  if (scene != nullptr || !std::ranges::any_of(schedule.passes, needs_scene)) {
    return std::nullopt;
  }
  return "the frame graph's passes read a scene, and none was given "
         "(--scene)";
}

std::optional<std::string> camera_refusal(const FrameInputs& inputs,
                                          bool needs_scene) {
  if (!needs_scene || inputs.camera) {
    return std::nullopt;
  }
  return "frame " + std::to_string(inputs.index) +
         ": the frame graph reads a scene, and the frame has no camera";
}

std::optional<std::string> previous_camera_refusal(const FrameInputs& inputs) {
  if (!inputs.previous) {
    return std::nullopt;
  }
  if (const std::optional<std::string_view> reason =
          camera::invalid(inputs.previous->camera)) {
    return "frame " + std::to_string(inputs.index) +
           ": the frame before's camera cannot be framed: " +
           std::string(*reason);
  }
  return std::nullopt;
}

std::optional<std::string> restir_pt_refusal(
    const scene::SceneDescription& scene) {
  // A medium's 16-bit index with its "none" (contracts/primary_hit.h).
  constexpr std::size_t media_limit = contracts::primary_no_medium;
  if (scene.media.size() >= media_limit) {
    return "the scene has " + std::to_string(scene.media.size()) +
           " media, and ReSTIR PT's G-buffer keeps a medium's index in 16 "
           "bits, under " +
           std::to_string(media_limit);
  }
  if (const std::optional<std::string> outside =
          scene::outside_packed_bsdf(scene)) {
    return "the scene's " + *outside;
  }
  if (scene.shapes.records.size() > contracts::tile_largest_id + 1u ||
      scene.light_counts.lights > contracts::tile_largest_id + 1u) {
    return "the scene has " + std::to_string(scene.shapes.records.size()) +
           " shapes and " + std::to_string(scene.light_counts.lights) +
           " lights, and ReSTIR PT's light tiles keep each index in 16 bits, "
           "at most " +
           std::to_string(contracts::tile_largest_id + 1u) + " of either";
  }
  return std::nullopt;
}

std::optional<std::string> mnee_refusal(const scene::SceneDescription& scene) {
  if (scene.shapes.records.size() <= contracts::mnee_largest_shape + 1u) {
    return std::nullopt;
  }
  return "the scene has " + std::to_string(scene.shapes.records.size()) +
         " shapes, and MNEE's requests keep a shape's index in 16 bits, at "
         "most " +
         std::to_string(contracts::mnee_largest_shape + 1u);
}

std::optional<std::string> glower_refusal(
    const scene::SceneDescription& scene) {
  for (const animation::Glower& glower : scene.animation.glowers) {
    if (glower.target >= scene.light_counts.spheres) {
      return "a glowing light's target, " + std::to_string(glower.target) +
             ", is not one of the scene's " +
             std::to_string(scene.light_counts.spheres) + " sphere lights";
    }
  }
  return std::nullopt;
}

}  // namespace serenity::frame
