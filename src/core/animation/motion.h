#pragma once

#include <cstdint>
#include <vector>

#include "core/animation/extent.h"
#include "core/animation/flight.h"
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
// Every kind is made clear of the scene's still shapes (contract 11,
// contracts/obstacles.h): when it is made, at load, it checks that what it
// moves can touch none of them, for all time, and refuses to be made
// otherwise. So the guarantee is each kind's own, in the terms its path
// allows: the wander's whole reach clear (wander.h), a flight's every stretch
// (flight.h).
//
// A new kind adds a value here, an array of its parameters, its make
// function, and its position() and extent(); it changes no other kind and
// nothing outside the family.
//
// CPU only: motions are evaluated once a frame, on the CPU, and only the
// positions they give reach the GPU, as the translations of the moving
// shapes' transforms (core/animation/animate.h).

enum class MotionKind {
    wander,  // a drift about a fixed point (wander.h)
    flight,  // flying free among the still shapes (flight.h)
};

struct MotionRecord {
    MotionKind kind = MotionKind::wander;
    std::uint32_t index = 0;  // into that kind's array
};

struct Motions {
    std::vector<Wander> wanders;
    std::vector<Flight> flights;
};

// Where the motion `record` names places its point at `t`. The record must
// index its kind's array; the scene reader makes it so. The mapping from
// kind to position is a switch with no default, so a kind without one fails
// the build, and a value no enumerator names is refused by std::logic_error,
// never answered with a point that looks real (P.6).
contracts::Float3 position(const Motions& motions, MotionRecord record, frame::Seconds t);

// The box the motion `record` names keeps its point inside, for all t.
Extent extent(const Motions& motions, MotionRecord record);

}  // namespace serenity::animation
