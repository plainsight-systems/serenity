#include "core/passes/depth_of_field.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "core/camera/thin_lens.h"

namespace serenity::passes {

namespace {

// `n` over `d`, rounded up.
constexpr std::uint32_t divided_up(std::uint32_t n, std::uint32_t d) noexcept {
  return n / d + (n % d != 0 ? 1u : 0u);
}

}  // namespace

frame::Extent depth_of_field_half(frame::Extent size) noexcept {
  return {divided_up(size.width, contracts::dof_scale),
          divided_up(size.height, contracts::dof_scale)};
}

frame::Extent depth_of_field_tiles(frame::Extent size) noexcept {
  const frame::Extent half = depth_of_field_half(size);
  return {divided_up(half.width, contracts::dof_tile),
          divided_up(half.height, contracts::dof_tile)};
}

contracts::DepthOfFieldConstants depth_of_field_constants(
    const contracts::Camera& camera, frame::Extent size) {
  // The scale's checks are the camera's and the size's (thin_lens.h).
  const float scale = camera::circle_of_confusion_scale(camera, size);
  contracts::DepthOfFieldConstants k{};
  k.coc_scale = scale;
  k.focus_distance = camera.focus_distance;
  k.max_radius = contracts::dof_max_radius * static_cast<float>(size.width);
  k.width = size.width;
  k.height = size.height;
  const frame::Extent half = depth_of_field_half(size);
  const frame::Extent tiles = depth_of_field_tiles(size);
  k.half_width = half.width;
  k.half_height = half.height;
  k.tiles_x = tiles.width;
  k.tiles_y = tiles.height;
  const double reach = static_cast<double>(k.max_radius) /
                       contracts::dof_scale / contracts::dof_tile;
  k.dilation = std::max(1u, static_cast<std::uint32_t>(std::ceil(reach)));
  return k;
}

}  // namespace serenity::passes
