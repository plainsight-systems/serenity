#pragma once

#include "core/contracts/float3.h"

namespace serenity::animation {

// Axis: Animation.
//
// The axis-aligned box a motion keeps what it moves inside, for all time:
// what the scene reader checks against the shapes that do not move
// (core/scene/scene.h). Every motion kind has one; a kind whose reach is
// not bounded has no place in a scene of fixed shapes.
struct Extent {
    contracts::Float3 min;
    contracts::Float3 max;
};

}  // namespace serenity::animation
