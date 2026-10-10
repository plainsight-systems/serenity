// The wander, and Animate: a closed form in time, drawn from a seed, placing
// only what moves.

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <doctest/doctest.h>

#include "core/animation/animate.h"
#include "core/animation/motion_error.h"
#include "core/animation/wander.h"

using namespace serenity;
using animation::Wander;
using frame::Seconds;

namespace {

constexpr contracts::Float3 anchor{1.0f, 2.0f, -3.0f};

// A scene with no still shapes, as contract 11 asks of one: nothing to keep
// clear of. A stand-in for the scene's own answer (core/scene/scene.cpp),
// which the scene tests exercise.
class Nothing final : public contracts::Obstacles {
public:
    double distance(contracts::Float3) const override { return std::numeric_limits<double>::infinity(); }
    bool touches(const contracts::Box&) const override { return false; }
};
const Nothing nothing;

double axis(contracts::Float3 v, int a) {
    return a == 0 ? v.x : (a == 1 ? v.y : v.z);
}

}  // namespace

TEST_CASE("step 2: each axis's amplitudes sum to the reach") {
    const Wander w = animation::make_wander({anchor, 0.25f, 0.3f, 7}, 0.0f, nothing);
    for (int a = 0; a < 3; ++a) {
        double sum = 0.0;
        for (int k = 0; k < 3; ++k) {
            CHECK(w.amplitude[a][k] > 0.0);
            sum += w.amplitude[a][k];
        }
        CHECK(sum == doctest::Approx(0.25).epsilon(1e-12));
    }
}

TEST_CASE("the path never leaves its extent, anchor +/- reach") {
    const Wander w = animation::make_wander({anchor, 0.25f, 0.3f, 7}, 0.0f, nothing);
    const animation::Extent e = animation::extent(w);
    CHECK(e.min.x <= 0.75f);
    CHECK(e.max.y >= 2.25f);
    double reached = 0.0;  // the farthest it strays, along any axis
    int outside = 0;
    for (int i = 0; i < 200000; ++i) {
        const contracts::Float3 p = animation::position(w, Seconds(i * 0.01));
        for (int a = 0; a < 3; ++a) {
            outside += axis(p, a) < axis(e.min, a) || axis(p, a) > axis(e.max, a);
            reached = std::max(reached, std::abs(axis(p, a) - axis(anchor, a)));
        }
    }
    CHECK(outside == 0);
    // It uses its room: over 2000 s it strays most of the way to the reach.
    CHECK(reached > 0.6 * 0.25);
}

TEST_CASE("step 3: its root-mean-square speed is the speed") {
    const float speed = 0.3f;
    const Wander w = animation::make_wander({anchor, 0.25f, speed, 7}, 0.0f, nothing);
    // The mean of |velocity|^2 over a long time, the velocity by central
    // differences of the exact path.
    const double h = 1e-3;
    double total = 0.0;
    const int samples = 400000;
    for (int i = 0; i < samples; ++i) {
        const double t = i * 0.05;
        double v2 = 0.0;
        for (int a = 0; a < 3; ++a) {
            double ahead = axis(anchor, a), behind = axis(anchor, a);
            for (int k = 0; k < 3; ++k) {
                ahead += w.amplitude[a][k] * std::sin(2.0 * M_PI * w.frequency[a][k] * (t + h) + w.phase[a][k]);
                behind += w.amplitude[a][k] * std::sin(2.0 * M_PI * w.frequency[a][k] * (t - h) + w.phase[a][k]);
            }
            const double v = (ahead - behind) / (2.0 * h);
            v2 += v * v;
        }
        total += v2;
    }
    CHECK(std::sqrt(total / samples) == doctest::Approx(speed).epsilon(0.02));
}

TEST_CASE("a function of the seed alone: the same seed the same path, another seed another") {
    const Wander a = animation::make_wander({anchor, 0.25f, 0.3f, 7}, 0.0f, nothing);
    const Wander b = animation::make_wander({anchor, 0.25f, 0.3f, 7}, 0.0f, nothing);
    const Wander c = animation::make_wander({anchor, 0.25f, 0.3f, 8}, 0.0f, nothing);
    for (double t : {0.0, 1.5, 3600.0}) {
        const contracts::Float3 pa = animation::position(a, Seconds(t));
        const contracts::Float3 pb = animation::position(b, Seconds(t));
        const contracts::Float3 pc = animation::position(c, Seconds(t));
        CHECK(std::memcmp(&pa, &pb, sizeof pa) == 0);
        CHECK(pa.x != pc.x);
    }
    // Pinned: splitmix64 and the steps, not <random>, so these bits are the
    // same under every standard library. A change to how the path is drawn
    // changes them.
    const contracts::Float3 p = animation::position(a, Seconds(1.0));
    CHECK(p.x == 1.13084936f);
    CHECK(p.y == 2.10570359f);
    CHECK(p.z == -2.98401809f);
}

TEST_CASE("numbers it cannot make a path of are refused") {
    CHECK_THROWS_AS(animation::make_wander({anchor, 0.0f, 0.3f, 7}, 0.0f, nothing), std::invalid_argument);
    CHECK_THROWS_AS(animation::make_wander({anchor, 0.25f, std::numeric_limits<float>::infinity(), 7}, 0.0f, nothing),
                    std::invalid_argument);
    CHECK_THROWS_AS(animation::make_wander({{3e38f, 0.0f, 0.0f}, 1e38f, 0.3f, 7}, 0.0f, nothing), std::invalid_argument);
}

TEST_CASE("the extent is rounded outward, so it holds every point the path reaches") {
    const Wander w = animation::make_wander({{0.1f, 0.1f, 0.1f}, 0.2f, 0.3f, 7}, 0.0f, nothing);
    const animation::Extent e = animation::extent(w);
    CHECK(static_cast<double>(e.max.x) >= static_cast<double>(0.1f) + static_cast<double>(0.2f));
    CHECK(static_cast<double>(e.min.x) <= static_cast<double>(0.1f) - static_cast<double>(0.2f));
}

TEST_CASE("Animate places only the movers: the translation replaced, the scale kept") {
    animation::Animation anim;
    anim.motions.wanders.push_back(animation::make_wander({anchor, 0.25f, 0.3f, 7}, 0.0f, nothing));
    anim.movers.push_back({1, {animation::MotionKind::wander, 0}});
    CHECK(animation::moves(anim));

    std::vector<contracts::Transform> transforms = {contracts::placed({9.0f, 9.0f, 9.0f}, 2.0f),
                                                    contracts::placed(anchor, 0.05f)};
    animation::animate(anim, Seconds(1.0), transforms, {});
    const contracts::Float3 p = animation::position(anim.motions.wanders[0], Seconds(1.0));
    CHECK(transforms[1].m[0][3] == p.x);
    CHECK(transforms[1].m[1][3] == p.y);
    CHECK(transforms[1].m[2][3] == p.z);
    CHECK(transforms[1].m[0][0] == 0.05f);
    CHECK(transforms[1].m[2][2] == 0.05f);
    // The still one is untouched.
    CHECK(transforms[0].m[0][3] == 9.0f);
    CHECK(transforms[0].m[0][0] == 2.0f);

    // A target past the transforms is refused before anything is written.
    anim.movers.insert(anim.movers.begin(), animation::Mover{0, {animation::MotionKind::wander, 0}});
    anim.movers.push_back({5, {animation::MotionKind::wander, 0}});
    CHECK_THROWS_AS(animation::animate(anim, Seconds(2.0), transforms, {}), std::invalid_argument);
    CHECK(transforms[0].m[0][3] == 9.0f);
}

TEST_CASE("nothing to move, nothing moves") {
    CHECK_FALSE(animation::moves(animation::Animation{}));
}

namespace {

// Everywhere is a still shape.
class Everything final : public contracts::Obstacles {
public:
    double distance(contracts::Float3) const override { return -1.0; }
    bool touches(const contracts::Box&) const override { return true; }
};

}  // namespace

TEST_CASE("a wander's refusals: its numbers say which is wrong, a still shape in reach is a MotionError") {
    const Nothing nothing;
    const contracts::Float3 anchor{1.0f, 1.5f, -0.5f};
    try {
        (void)animation::make_wander({anchor, 0.25f, 0.3f, 7}, -1.0f, nothing);
        FAIL("expected an error");
    } catch (const animation::MotionError&) {
        FAIL("numbers out of range are not a refusal");
    } catch (const std::invalid_argument& error) {
        CHECK(std::string(error.what()).find("body") != std::string::npos);
    }
    const Everything everything;
    CHECK_THROWS_AS(animation::make_wander({anchor, 0.25f, 0.3f, 7}, 0.0f, everything), animation::MotionError);
}

TEST_CASE("a motion or glow record whose kind no enumerator names is refused, never placed") {
    animation::Animation anim;
    anim.motions.wanders.push_back(animation::make_wander({{0.0f, 1.0f, 0.0f}, 0.25f, 0.3f, 7}, 0.0f, Nothing{}));
    const animation::MotionRecord stray{static_cast<animation::MotionKind>(5), 0};
    CHECK_THROWS_AS(animation::position(anim.motions, stray, Seconds(1.0)), std::logic_error);
    CHECK_THROWS_AS(animation::extent(anim.motions, stray), std::logic_error);
    const animation::GlowRecord glow{static_cast<animation::GlowKind>(5), 0};
    CHECK_THROWS_AS(animation::glow(anim.glows, glow, Seconds(1.0)), std::logic_error);
}
