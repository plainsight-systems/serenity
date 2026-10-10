// The wander, and Animate: a closed form in time, drawn from a seed, placing
// only what moves.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <doctest/doctest.h>

#include "core/animation/animate.h"
#include "core/animation/motion_error.h"
#include "core/animation/wander.h"
#include "support/text.h"

using namespace serenity;
using animation::Wander;
using animation::WanderParams;
using frame::Seconds;

namespace {

constexpr contracts::Float3 anchor{1.0f, 2.0f, -3.0f};
constexpr float reach = 0.25f;
constexpr float speed = 0.3f;
constexpr std::uint64_t seed = 7;

// The wander the tests share: about `anchor`, its reach and speed, seed 7.
constexpr WanderParams wandering{.anchor = anchor, .reach = reach, .speed = speed, .seed = seed};

// A scene with no still shapes, as contract 11 asks of one: nothing to keep
// clear of. A stand-in for the scene's own answer (core/scene/scene.cpp),
// which the scene tests exercise.
class Nothing final : public contracts::Obstacles {
public:
    double distance(contracts::Float3) const override { return std::numeric_limits<double>::infinity(); }
    bool touches(const contracts::Box&) const override { return false; }
};
const Nothing nothing;

// Everywhere is a still shape.
class Everything final : public contracts::Obstacles {
public:
    double distance(contracts::Float3) const override { return -1.0; }
    bool touches(const contracts::Box&) const override { return true; }
};

}  // namespace

TEST_CASE("step 2: each axis's amplitudes sum to the reach") {
    const Wander w = animation::make_wander(wandering, 0.0f, nothing);
    for (const auto& axis : w.amplitude) {
        double sum = 0.0;
        for (const double a : axis) {
            CHECK(a > 0.0);
            sum += a;
        }
        CHECK(sum == doctest::Approx(reach).scale(0).epsilon(1e-12));
    }
}

TEST_CASE("the path never leaves its extent, anchor +/- reach") {
    const Wander w = animation::make_wander(wandering, 0.0f, nothing);
    const animation::Extent e = animation::extent(w);
    CHECK(e.min.x <= anchor.x - reach);
    CHECK(e.max.y >= anchor.y + reach);
    double reached = 0.0;  // the farthest it strays, along any axis
    int outside = 0;
    for (int i = 0; i < 200000; ++i) {
        const contracts::Float3 p = animation::position(w, Seconds(i * 0.01));
        for (int a = 0; a < 3; ++a) {
            const float v = contracts::component(p, a);
            outside += v < contracts::component(e.min, a) || v > contracts::component(e.max, a) ? 1 : 0;
            reached = std::max(reached, static_cast<double>(std::abs(v - contracts::component(anchor, a))));
        }
    }
    CHECK(outside == 0);
    // It uses its room: over 2000 s it strays most of the way to the reach.
    CHECK(reached > 0.6 * reach);
}

TEST_CASE("step 3: its root-mean-square speed is the speed") {
    const Wander w = animation::make_wander(wandering, 0.0f, nothing);
    // The mean of |velocity|^2 over a long time, the velocity by central
    // differences of the exact path.
    constexpr double h = 1e-3;
    constexpr int samples = 400000;
    const auto along = [&](std::size_t a, double t) {
        double x = contracts::component(anchor, static_cast<int>(a));
        for (std::size_t k = 0; k < 3; ++k) {
            x += w.amplitude[a][k] * std::sin(2.0 * std::numbers::pi * w.frequency[a][k] * t + w.phase[a][k]);
        }
        return x;
    };
    double total = 0.0;
    for (int i = 0; i < samples; ++i) {
        const double t = i * 0.05;
        for (std::size_t a = 0; a < 3; ++a) {
            const double v = (along(a, t + h) - along(a, t - h)) / (2.0 * h);
            total += v * v;
        }
    }
    CHECK(std::sqrt(total / samples) == doctest::Approx(speed).scale(0).epsilon(0.02));
}

TEST_CASE("a function of the seed alone: the same seed the same path, another seed another") {
    const Wander a = animation::make_wander(wandering, 0.0f, nothing);
    const Wander b = animation::make_wander(wandering, 0.0f, nothing);
    WanderParams reseeded = wandering;
    reseeded.seed = seed + 1;
    const Wander c = animation::make_wander(reseeded, 0.0f, nothing);
    for (const double t : {0.0, 1.5, 3600.0}) {
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
    WanderParams no_reach = wandering;
    no_reach.reach = 0.0f;
    CHECK_THROWS_AS(animation::make_wander(no_reach, 0.0f, nothing), std::invalid_argument);
    WanderParams endless = wandering;
    endless.speed = std::numeric_limits<float>::infinity();
    CHECK_THROWS_AS(animation::make_wander(endless, 0.0f, nothing), std::invalid_argument);
    CHECK_THROWS_AS(animation::make_wander({{3e38f, 0.0f, 0.0f}, 1e38f, speed, seed}, 0.0f, nothing),
                    std::invalid_argument);
}

TEST_CASE("the extent is rounded outward, so it holds every point the path reaches") {
    const Wander w = animation::make_wander({{0.1f, 0.1f, 0.1f}, 0.2f, speed, seed}, 0.0f, nothing);
    const animation::Extent e = animation::extent(w);
    CHECK(static_cast<double>(e.max.x) >= static_cast<double>(0.1f) + static_cast<double>(0.2f));
    CHECK(static_cast<double>(e.min.x) <= static_cast<double>(0.1f) - static_cast<double>(0.2f));
}

TEST_CASE("Animate places only the movers: the translation replaced, the scale kept") {
    animation::Animation anim;
    anim.motions.wanders.push_back(animation::make_wander(wandering, 0.0f, nothing));
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

TEST_CASE("a wander's refusals: its numbers say which is wrong, a still shape in reach is a Refusal") {
    const WanderParams near_floor{.anchor = {1.0f, 1.5f, -0.5f}, .reach = reach, .speed = speed, .seed = seed};
    // A negative body: an invalid_argument naming it, not a Refusal (which
    // is one too, so it is caught first).
    try {
        (void)animation::make_wander(near_floor, -1.0f, nothing);
        FAIL("expected an error");
    } catch (const animation::MotionError&) {
        FAIL("numbers out of range are not a refusal");
    } catch (const std::invalid_argument& error) {
        CHECK(tests::contains(error.what(), "body"));
    }
    const Everything everything;
    CHECK_THROWS_AS(animation::make_wander(near_floor, 0.0f, everything), animation::MotionError);
}

TEST_CASE("a motion or glow record whose kind no enumerator names is refused, never placed") {
    animation::Animation anim;
    anim.motions.wanders.push_back(animation::make_wander({{0.0f, 1.0f, 0.0f}, reach, speed, seed}, 0.0f, nothing));
    const animation::MotionRecord stray{static_cast<animation::MotionKind>(5), 0};
    CHECK_THROWS_AS(animation::position(anim.motions, stray, Seconds(1.0)), std::logic_error);
    CHECK_THROWS_AS(animation::extent(anim.motions, stray), std::logic_error);
    const animation::GlowRecord glow{static_cast<animation::GlowKind>(5), 0};
    CHECK_THROWS_AS(animation::glow(anim.glows, glow, Seconds(1.0)), std::logic_error);
}
