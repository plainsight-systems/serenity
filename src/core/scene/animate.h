#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "core/animation/motion.h"
#include "core/frame/frame_inputs.h"
#include "core/shapes/transform.h"

namespace serenity::scene {

// Axis: Scene content (what moves), and logical-overview.md's Animate step.
//
// Which shapes move, and by which motion: the scene's side of animation.
// The Animation family says how a kind of motion places a point at t
// (core/animation/motion.h); this says which shapes follow which motion, and
// places them, at a frame's time, where the GPU will read them.
//
// Only spheres move: a firefly is a sphere, and a sphere moves by its
// center alone, the translation of its transform (core/shapes/transform.h).
// A shape that is a light moves with its light, which is where its shape is
// (core/lights/sphere_light.h): nothing else to place. A marble that rolls
// will turn as well, a rotation in the same transform, when it comes.
//
// animate() places every moving shape at `t`: for each, it sets its
// transform's translation to its motion's position (shapes::moved_to),
// keeping its scale. It writes into a span the caller owns, the frame's copy
// of the shapes' transforms, which the GPU backend keeps in memory the GPU
// reads (metal/scene/shape_transforms.h): the positions are computed once,
// in one place. It writes only the moving shapes' transforms; the rest the
// caller filled at start-up, from the scene, and nothing changes them.
// Throws std::invalid_argument, before writing anything, if `transforms` is
// not the scene's shape count (I.6).
//
// moves() says whether anything moves at all: a scene that does not looks
// the same at every t, so a converging pass may average frames of any time
// (metal/frame/accumulation.h), and the GPU backend builds its structures
// once (metal/acceleration/scene_acceleration.h).
//
// Cost, per frame: one motion and one 48-byte write per moving shape,
// O(moving shapes), nothing allocated (MEM.9). The thousands of fireflies
// of the goal are some hundred microseconds of sines on one core, before
// anything is spread across threads.

struct MovingShape {
    std::uint32_t shape;  // in the scene's order
    animation::MotionRecord motion;
};

struct SceneAnimation {
    animation::Motions motions;
    std::vector<MovingShape> shapes;  // in the scene's order
};

inline bool moves(const SceneAnimation& animation) {
    return !animation.shapes.empty();
}

// Places every moving shape at `t`; see above. `transforms` has the scene's
// shape count.
void animate(const SceneAnimation& animation, frame::Seconds t, std::span<shapes::Transform> transforms);

}  // namespace serenity::scene
