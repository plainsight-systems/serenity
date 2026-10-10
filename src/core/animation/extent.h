#pragma once

#include "core/contracts/obstacles.h"

namespace serenity::animation {

// Axis: Animation.
//
// The axis-aligned box a motion keeps what it moves inside, for all time:
// what the scene reader checks against the shapes that do not move
// (core/scene/scene.h). Every motion kind has one; a kind whose reach is
// not bounded has no place in a scene of fixed shapes.
//
// The box of contract 11, which this family owns (contracts/obstacles.h),
// not a second type of the same two corners (ES.3): a motion's extent is
// what it asks the still shapes about.
using Extent = contracts::Box;

}  // namespace serenity::animation
