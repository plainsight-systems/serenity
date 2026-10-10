#pragma once

// The scene the flights and the swarms are made in (flight_test.cpp,
// swarm_test.cpp), once (ES.3): a matte ball named "ball" on a floor, the
// still shapes a flier keeps clear of, and how far a point is from them.

#include <algorithm>
#include <cstdint>
#include <string>

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
inline std::string ball_on_floor(const char* glow_radiance) {
    return camera_text("[0, 1.5, 4]", "[0, 0.5, 0]", 40) + sky("[0, 0, 0]", "[0, 0, 0]") +
           "[materials.matte]\nkind = \"rough\"\ncolor = [0.5, 0.5, 0.5]\n"
           "[materials.glow]\nkind = \"emissive\"\nradiance = " + glow_radiance + "\n" +
           "[[shapes]]\nkind = \"box\"\nmin = [-10, -0.1, -10]\nmax = [10, 0, 10]\nmaterial = \"matte\"\n"
           "[[shapes]]\nkind = \"sphere\"\nname = \"ball\"\ncenter = [0, 0.5, 0]\nradius = 0.5\nmaterial = \"matte\"\n";
}

// How far `p` is from the floor and the ball.
inline double distance_to_still(const scene::SceneDescription& s, contracts::Float3 p) {
    return std::min(shapes::distance(s.shapes, floor_shape, p), shapes::distance(s.shapes, ball_shape, p));
}

// The ball and the floor as contract 11 sees them. Holds the scene it asks
// by pointer, so it copies as its fakes do (C.12).
class BallAndFloor final : public contracts::Obstacles {
public:
    explicit BallAndFloor(const scene::SceneDescription& s) : scene_(&s) {}
    double distance(contracts::Float3 p) const override { return distance_to_still(*scene_, p); }
    bool touches(const contracts::Box& box) const override {
        return shapes::touches(scene_->shapes, floor_shape, {box.min, box.max}) ||
               shapes::touches(scene_->shapes, ball_shape, {box.min, box.max});
    }

private:
    const scene::SceneDescription* scene_;
};

}  // namespace serenity::tests
