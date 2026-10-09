#pragma once

#include <cstdint>
#include <vector>

#include "core/animation/extent.h"
#include "core/animation/wander.h"
#include "core/contracts/float3.h"
#include "core/frame/frame_inputs.h"

namespace serenity::animation {

// Axis: Animation.
//
// The motion kinds, and the record that says which motion a moving thing
// follows: a kind and an index into that kind's array (Enum.2, C.181), as
// for shapes, materials, textures and lights. The family's one view of all
// its kinds, so what places things at a time, the Animate step
// (animate.h), names no kind.
//
// A motion places one point, the center of what it moves, at a time t: a
// closed form, evaluated directly, never stepped from the frame before
// (logical-overview.md, principle 1). The mechanism is here; which object
// follows which motion, and its numbers, are the scene's (change-axes.md:
// mechanism is code, values are data).
//
// A new kind adds a value here, an array of its parameters, and its
// position() and extent(); it changes no other kind and nothing outside the
// family. Free flight through a volume, a curve through waypoints drawn
// from a seed, is the next: still a closed form in t.
//
// CPU only: motions are evaluated once a frame, on the CPU, and only the
// positions they give reach the GPU, as the translations of the moving
// shapes' transforms (core/animation/animate.h).

enum class MotionKind : std::uint32_t {
    wander = 0,  // a drift about a fixed point (wander.h)
};

struct MotionRecord {
    MotionKind kind;
    std::uint32_t index;  // into that kind's array
};

struct Motions {
    std::vector<Wander> wanders;
};

// Where the motion `record` names places its point at `t`. The record must
// index its kind's array; the scene reader makes it so. The mapping from
// kind to position is a switch with no default, so a kind without one fails
// the build.
contracts::Float3 position(const Motions& motions, MotionRecord record, frame::Seconds t);

// The box the motion `record` names keeps its point inside, for all t.
Extent extent(const Motions& motions, MotionRecord record);

}  // namespace serenity::animation
