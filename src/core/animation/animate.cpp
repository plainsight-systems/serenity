#include "core/animation/animate.h"

#include <stdexcept>

namespace serenity::animation {

void animate(const Animation& animation, frame::Seconds t, std::span<contracts::Transform> transforms,
             std::span<float> glows) {
    for (const Mover& mover : animation.movers) {
        if (mover.target >= transforms.size()) {
            throw std::invalid_argument("animate: a mover's target is not one of the transforms");
        }
    }
    for (const Glower& glower : animation.glowers) {
        if (glower.target >= glows.size()) {
            throw std::invalid_argument("animate: a glower's target is not one of the glows");
        }
    }
    for (const Mover& mover : animation.movers) {
        contracts::Transform& placed = transforms[mover.target];
        placed = contracts::moved_to(placed, position(animation.motions, mover.motion, t));
    }
    for (const Glower& glower : animation.glowers) {
        glows[glower.target] = glow(animation.glows, glower.glow, t);
    }
}

}  // namespace serenity::animation
