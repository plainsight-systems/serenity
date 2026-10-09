#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "core/animation/motion.h"
#include "core/contracts/transform.h"
#include "core/frame/frame_inputs.h"

namespace serenity::animation {

// Axis: Animation, and logical-overview.md's Animate step.
//
// Placing what moves at a frame's time: each mover, a target that follows a
// motion, has its transform (contract 10, contracts/transform.h) set to
// where the motion puts it at t. This is how a motion's result changes a
// transform, the Animation family's to decide; which targets move, and by
// which motion, is the scene's, which fills in the movers from its file
// (core/scene/scene.h). A target is an index into the transforms: for a
// scene, its shape (core/shapes/primitive.h). The family names no shape kind
// and no scene type.
//
// A motion places a point (motion.h), and the point is the transform's
// translation: animate() replaces the translation and keeps the rest, the
// scale and the rotation. A motion that also turns what it moves (a marble
// that rolls) will give the whole placement, and the change is here and in
// its kind, nowhere else.
//
// animate() writes into a span the caller owns, the frame's copy of the
// transforms, which the GPU backend keeps in memory the GPU reads
// (metal/scene/shape_transforms.h): the placements are computed once, in one
// place. It writes only the movers' transforms; the rest the caller filled
// at start-up and nothing changes them. Throws std::invalid_argument, before
// writing anything, if a mover's target is not an index into `transforms`
// (I.6).
//
// moves() says whether anything moves at all: what does not move looks the
// same at every t, so a converging pass may average frames of any time
// (metal/frame/accumulation.h), and the GPU backend builds its structures
// once (metal/acceleration/scene_acceleration.h).
//
// Cost, per frame: one motion and one 48-byte write per mover, O(movers),
// nothing allocated (MEM.9). The thousands of fireflies of the goal are some
// hundred microseconds of sines on one core, before anything is spread
// across threads.

struct Mover {
    std::uint32_t target;  // the index of its transform
    MotionRecord motion;
};

struct Animation {
    Motions motions;
    std::vector<Mover> movers;  // in target order
};

inline bool moves(const Animation& animation) {
    return !animation.movers.empty();
}

// Places every mover at `t`; see above.
void animate(const Animation& animation, frame::Seconds t, std::span<contracts::Transform> transforms);

}  // namespace serenity::animation
