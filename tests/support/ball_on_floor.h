#pragma once

// The scene the flights and the swarms are made in (flight_test.cpp,
// swarm_test.cpp), once (ES.3): a matte ball named "ball" on a floor, the
// still shapes a flier keeps clear of, and how far a point is from them.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

#include "core/contracts/obstacles.h"
#include "core/scene/scene.h"
#include "core/shapes/shapes.h"
#include "support/scene_text.h"

namespace serenity::tests {

// The shape indices of the floor and the ball.
inline constexpr std::uint32_t floor_shape = 0;
inline constexpr std::uint32_t ball_shape = 1;

// A matte ball of radius 0.5 named "ball", resting on a floor 20 m across,
// in a black sky, and a glowing material "glow" of radiance
// `glow_radiance`. 25 lines; what is appended starts on line 26.
inline std::string ball_on_floor(std::string_view glow_radiance) {
    return camera_text("[0, 1.5, 4]", "[0, 0.5, 0]", 40) + sky("[0, 0, 0]", "[0, 0, 0]") +
           "[materials.matte]\nkind = \"rough\"\ncolor = [0.5, 0.5, 0.5]\n"
           "[materials.glow]\nkind = \"emissive\"\nradiance = " + std::string(glow_radiance) + "\n" +
           "[[shapes]]\nkind = \"box\"\nmin = [-10, -0.1, -10]\nmax = [10, 0, 10]\nmaterial = \"matte\"\n"
           "[[shapes]]\nkind = \"sphere\"\nname = \"ball\"\ncenter = [0, 0.5, 0]\nradius = 0.5\nmaterial = \"matte\"\n";
}

// How far `p` is from the floor and the ball.
inline double distance_to_still(const scene::SceneDescription& s, contracts::Float3 p) {
    return std::min(shapes::distance(s.shapes, floor_shape, p), shapes::distance(s.shapes, ball_shape, p));
}

// A scene's first `count` shapes, which do not move, as contract 11 asks of
// them: each shape kind's exact tests (core/shapes/shapes.h). Holds the
// scene it asks by pointer, so it copies as its fakes do (C.12).
class StillShapesOf : public contracts::Obstacles {
public:
    StillShapesOf(const scene::SceneDescription& s, std::uint32_t count) : scene_(&s), count_(count) {}
    double distance(contracts::Float3 p) const override {
        double nearest = std::numeric_limits<double>::infinity();
        for (std::uint32_t k = 0; k < count_; ++k) {
            nearest = std::min(nearest, shapes::distance(scene_->shapes, k, p));
        }
        return nearest;
    }
    bool touches(const contracts::Box& box) const override {
        for (std::uint32_t k = 0; k < count_; ++k) {
            if (shapes::touches(scene_->shapes, k, {box.min, box.max})) {
                return true;
            }
        }
        return false;
    }

private:
    const scene::SceneDescription* scene_;
    std::uint32_t count_;
};

// The ball and the floor as contract 11 sees them: ball_on_floor()'s first
// two shapes.
class BallAndFloor final : public StillShapesOf {
public:
    explicit BallAndFloor(const scene::SceneDescription& s) : StillShapesOf(s, 2) {}
};
static_assert(floor_shape < 2 && ball_shape < 2, "BallAndFloor is the scene's first two shapes");

}  // namespace serenity::tests
