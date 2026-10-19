#pragma once

#include <algorithm>

#include "core/frame/extent.h"

namespace serenity::passes {

// Axis: Pass (tone map: the bloom pyramid's sizes, for every backend).
//
// The size of each level of the tone-map pass's bloom pyramid
// (core/passes/tone_map.h, step 2): the level before's, from the frame's
// size, halved on each axis, rounded up, at least 1, so a frame of any
// size, 1 x 1 included, has every level. A backend makes its images at
// these sizes and dispatches over them (Metal's: metal/frame/
// frame_images.h). The core's (A.1), a pure function (F.8). Apart from
// tone_map.h, which the shaders include, since the shaders take each
// level's size from its image.
constexpr frame::Extent bloom_level(frame::Extent before) noexcept {
  return {std::max(1u, (before.width + 1) / 2),
          std::max(1u, (before.height + 1) / 2)};
}

}  // namespace serenity::passes
