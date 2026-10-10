#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "core/animation/glow.h"
#include "core/animation/motion.h"
#include "core/contracts/transform.h"
#include "core/frame/frame_inputs.h"

namespace serenity::animation {

// Axis: Animation, and logical-overview.md's Animate step.
//
// Placing what moves and lighting what glows at a frame's time:
//
//   - each mover, a target that follows a motion (motion.h), has its
//     transform (contract 10, contracts/transform.h) set to where the motion
//     puts it at t. A motion places a point, the transform's translation;
//     animate() replaces the translation and keeps the scale and rotation. A
//     motion that also turns what it moves (a marble that rolls) will give
//     the whole placement, and the change is here and in its kind;
//   - each glower, a target that follows a glow (glow.h), has its factor set
//     to the glow's at t: the factor on its light's radiance.
//
// How a motion or a glow changes its target is this family's to decide;
// which targets move or glow, and by which kind, is the scene's, which fills
// in the movers and glowers from its file (core/scene/scene.h). A mover's
// target is an index into the transforms: for a scene, its shape. A
// glower's is an index into the glows: for a scene, its sphere light
// (core/lights/sphere_light.h). The family names no shape kind, no light
// kind and no scene type.
//
// animate() writes into spans the caller owns, the frame's copies of the
// transforms and the glows, which the GPU backend keeps in memory the GPU
// reads (metal/scene/shape_transforms.h, metal/scene/light_glows.h): each is
// computed once, in one place. It writes only the movers' transforms and the
// glowers' factors; the rest the caller filled at start-up (every glow
// starts at 1, its light's full radiance) and nothing changes them. Throws
// std::invalid_argument, before writing anything, if a target is not an
// index into its span: a precondition stated and checked (I.5, E.2).
//
// moves() says whether anything moves, which is what the acceleration
// structure is rebuilt for (metal/acceleration/scene_acceleration.h);
// changes() whether anything moves or glows, which is what makes frames at
// two times two different scenes, so a converging pass may average frames
// of any time only when nothing changes (core/frame/history.h).
//
// Cost, per frame: one motion and one 48-byte write per mover, one glow and
// one 4-byte write per glower, O(movers + glowers), nothing allocated
// (MEM.9).

struct Mover {
    std::uint32_t target = 0;  // the index of its transform
    MotionRecord motion;
};

struct Glower {
    std::uint32_t target = 0;  // the index of its glow factor
    GlowRecord glow;
};

struct Animation {
    Motions motions;
    Glows glows;
    std::vector<Mover> movers;    // in target order
    std::vector<Glower> glowers;  // in target order
};

inline bool moves(const Animation& animation) noexcept {
    return !animation.movers.empty();
}

inline bool changes(const Animation& animation) noexcept {
    return !animation.movers.empty() || !animation.glowers.empty();
}

// Places every mover and lights every glower at `t`; see above.
void animate(const Animation& animation, frame::Seconds t, std::span<contracts::Transform> transforms,
             std::span<float> glows);

}  // namespace serenity::animation
