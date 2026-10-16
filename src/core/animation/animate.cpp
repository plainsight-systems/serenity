#include "core/animation/animate.h"

#include <stdexcept>

namespace serenity::animation {

namespace {

void check_movers(const Animation& animation, std::span<const contracts::Transform> transforms) {
    for (const Mover& mover : animation.movers) {
        if (mover.target >= transforms.size()) {
            throw std::invalid_argument("animate: a mover's target is not one of the transforms");
        }
    }
}

void place_checked(const Animation& animation, frame::Seconds t, std::span<contracts::Transform> transforms) {
    for (const Mover& mover : animation.movers) {
        contracts::Transform& placed = transforms[mover.target];
        placed = contracts::moved_to(placed, position(animation.motions, mover.motion, t));
    }
}

}  // namespace

void place(const Animation& animation, frame::Seconds t, std::span<contracts::Transform> transforms) {
    check_movers(animation, transforms);
    place_checked(animation, t, transforms);
}

void animate(const Animation& animation, frame::Seconds t, std::span<contracts::Transform> transforms,
             std::span<float> glows) {
    // Every target checked before anything is written (animate.h).
    check_movers(animation, transforms);
    for (const Glower& glower : animation.glowers) {
        if (glower.target >= glows.size()) {
            throw std::invalid_argument("animate: a glower's target is not one of the glows");
        }
    }
    place_checked(animation, t, transforms);
    for (const Glower& glower : animation.glowers) {
        glows[glower.target] = glow(animation.glows, glower.glow, t);
    }
}

}  // namespace serenity::animation
