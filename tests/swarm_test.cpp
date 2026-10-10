// Swarms (core/scene/swarm.h): many fireflies from one entry, each what a
// written firefly is, its seed and start a function of the swarm's seed,
// its number and its draw; its fireflies' flights, openings and all; and a
// refused firefly drawn again (swarm.h, step 5). How many flights are made
// at once, and what they throw, is the flight's (tests/flight_test.cpp).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <doctest/doctest.h>

#include "core/animation/draw.h"
#include "core/animation/flight.h"
#include "core/scene/scene.h"
#include "core/scene/swarm.h"
#include "core/shapes/shapes.h"
#include "support/ball_jobs.h"
#include "support/ball_on_floor.h"
#include "support/text.h"

using namespace serenity;
using frame::Seconds;
using tests::clear_start;
using tests::contains;
using tests::job;
using tests::replaced;

namespace {

// Firefly i's first draw, d = 0: the one every firefly had before redraws
// (swarm.h, step 1).
constexpr scene::FireflyDraw first(std::uint32_t i) {
    return {.index = i, .draw = 0};
}

// A swarm's numbers, as swarm_of() writes them: the shared job's
// (support/ball_jobs.h), whose flights its fireflies fly.
constexpr float firefly_radius = tests::job_body;
constexpr float swarm_clearance = tests::job_clearance;
constexpr float volume_low = tests::job_volume.min.y;
constexpr float volume_high = tests::job_volume.max.y;
constexpr float volume_reach = tests::job_volume.max.x;  // the volume's x and z bounds, either way

// The ball on its floor, and a swarm about it: the [[swarms]] entry on line
// 26, its count on 27.
std::string swarmed(const std::string& swarm) {
    return tests::ball_on_floor("[10, 8, 2]") + "[[swarms]]\n" + swarm + "\n";
}

std::string swarm_of(int count, int seed = 4, const std::string& extra = "") {
    return "count = " + std::to_string(count) +
           "\nradius = 0.02\nmaterial = \"glow\"\nmin = [-2, 0.05, -2]\nmax = [2, 2.2, 2]\n"
           "targets = [\"ball\"]\nspeed = 0.4\nclearance = 0.05\ncircle = 2\nswoop = 1\ndrift = 2\n"
           "flash = 0.35\ndim = 0.25\nseed = " +
           std::to_string(seed) + "\n" + extra;
}

std::string error_of(const std::string& text) {
    return tests::error_of<scene::SceneError>([&] { return scene::parse(text, "s.toml"); });
}

// The shapes before a swarm's: the floor and the ball.
constexpr std::uint32_t written_shapes = 2;

}  // namespace

TEST_CASE("a swarm becomes its count of fireflies, each a moving, flashing sphere light") {
    constexpr std::uint32_t count = 24;
    const scene::SceneDescription s = scene::parse(swarmed(swarm_of(static_cast<int>(count))), "s");
    REQUIRE(s.shapes.records.size() == written_shapes + count);
    CHECK(s.sphere_lights.size() == count);
    REQUIRE(s.animation.movers.size() == count);
    REQUIRE(s.animation.glowers.size() == count);
    for (std::uint32_t i = 0; i < count; ++i) {
        // After the written shapes, in order; each a light, its own flight
        // and a glow of the schedule kind on that flight's flashes.
        CHECK(s.animation.movers[i].target == written_shapes + i);
        CHECK(s.animation.movers[i].motion.kind == animation::MotionKind::flight);
        CHECK(s.shape_lights[written_shapes + i] == i);
        CHECK(s.animation.glowers[i].target == i);
        const animation::ScheduleGlow& glow = s.animation.glows.schedules[s.animation.glowers[i].glow.index];
        CHECK(glow.flash == doctest::Approx(0.35).scale(0).epsilon(1e-6));
        CHECK(glow.dim == doctest::Approx(0.25f).scale(0).epsilon(1e-6));
        CHECK(glow.schedule.starts == s.animation.motions.flights[i].flashes.starts);
        CHECK(s.shapes.transforms[written_shapes + i].m[0][0] ==
              doctest::Approx(firefly_radius).scale(0).epsilon(1e-6));
    }
}

TEST_CASE("each firefly starts clear, with room for its first drift, and its flight keeps clear for all time") {
    constexpr std::uint32_t count = 24;
    const scene::SceneDescription s = scene::parse(swarmed(swarm_of(static_cast<int>(count))), "s");
    const double needed =
        swarm_clearance + firefly_radius + animation::flight_delta + animation::first_drift_reach * std::sqrt(3.0);
    const double margin = firefly_radius + animation::flight_delta + animation::first_drift_reach;
    for (std::uint32_t i = 0; i < count; ++i) {
        const contracts::Float3 start = contracts::translation(s.shapes.transforms[written_shapes + i]);
        CHECK(tests::distance_to_still(s, start) >= needed);
        CHECK(start.x >= -volume_reach + margin);
        CHECK(start.y <= volume_high - margin);
        // The flight, sampled every 10 ms of its loop.
        const animation::Flight& f = s.animation.motions.flights[i];
        double nearest = std::numeric_limits<double>::infinity();
        const auto samples = static_cast<int>(f.loop / 0.01);
        for (int k = 0; k < samples; ++k) {
            nearest = std::min(nearest,
                               tests::distance_to_still(s, animation::position(f, Seconds(k * 0.01))) - firefly_radius);
        }
        CHECK(nearest >= swarm_clearance);
    }
}

TEST_CASE("a firefly is a function of its swarm's seed and its number alone") {
    const scene::SceneDescription a = scene::parse(swarmed(swarm_of(8)), "a");
    const scene::SceneDescription b = scene::parse(swarmed(swarm_of(16)), "b");
    const scene::SceneDescription c = scene::parse(swarmed(swarm_of(8, 5)), "c");
    // More fireflies at the end move none of the first eight.
    for (std::uint32_t i = 0; i < 8; ++i) {
        const contracts::Float3 pa = animation::position(a.animation.motions.flights[i], Seconds(30.0));
        const contracts::Float3 pb = animation::position(b.animation.motions.flights[i], Seconds(30.0));
        CHECK(pa.x == pb.x);
        CHECK(pa.y == pb.y);
        CHECK(pa.z == pb.z);
    }
    // Another seed, other fireflies.
    CHECK(a.shapes.transforms[written_shapes].m[0][3] != c.shapes.transforms[written_shapes].m[0][3]);
    CHECK(scene::firefly_seed(4, first(0)) != scene::firefly_seed(4, first(1)));
    CHECK(scene::firefly_seed(4, first(0)) != scene::firefly_seed(5, first(0)));
}

TEST_CASE("every mistake in a swarm is refused, naming the file and the line") {
    CHECK(contains(error_of(swarmed(swarm_of(0))), "s.toml:27: swarm 1's count must be an integer from 1 to 4096"));
    CHECK(contains(error_of(swarmed(swarm_of(4097))), "s.toml:27: swarm 1's count must be an integer from 1 to 4096"));
    CHECK(contains(error_of(swarmed(replaced(swarm_of(4), "\"glow\"", "\"matte\""))),
                   "s.toml:29: swarm 1's material must be emissive"));
    CHECK(contains(error_of(swarmed(swarm_of(4, 4, "colour = 1\n"))), "s.toml:41: unknown key 'colour' in swarm 1"));
    CHECK(contains(error_of(swarmed(replaced(swarm_of(4), "\"ball\"", "\"cup\""))),
                   "s.toml:32: swarm 1 circles 'cup', which is not defined"));
    CHECK(contains(error_of(swarmed(replaced(swarm_of(4), "flash = 0.35", "flash = 1.5"))),
                   "s.toml:38: swarm 1's flash must be under a second"));
    CHECK(contains(error_of(swarmed(replaced(swarm_of(4), "max = [2, 2.2, 2]", "max = [2, 2.2, 1000000]"))),
                   "swarm 1's volume, grown by its radius, must lie within 1000 km"));
    // A volume the ball fills: no start is clear.
    const std::string full =
        replaced(swarm_of(4), "min = [-2, 0.05, -2]\nmax = [2, 2.2, 2]",
                 "min = [-0.3, 0.3, -0.3]\nmax = [0.3, 0.7, 0.3]");
    const std::string no_room = error_of(swarmed(full));
    INFO(no_room);
    CHECK(contains(no_room, "s.toml:26: swarm 1: firefly 0: no start clear of the still shapes in 64 draws"));
    CHECK(contains(error_of(swarmed("count = 4\n[[swarms]]\n" + swarm_of(4))), "s.toml:26: swarm 1 has no 'radius'"));
    CHECK(contains(error_of(swarmed(swarm_of(4)) + "\n[[swarms]]\nfoo = 1\n"), "unknown key 'foo' in swarm 2"));
}

// Starts and wakes (swarm.h, steps 2, 2p and 3).

namespace {

// A swarm about the ball, as swarm_of() writes it, of `count` fireflies,
// with `start` and `wake`.
scene::Swarm swarm_with(const scene::SwarmStart& start, std::optional<scene::SwarmWake> wake,
                        std::uint32_t count = 64) {
    return {.count = count,
            .radius = firefly_radius,
            .flight = job(4, clear_start).params,
            .flash = 0.35f,
            .dim = 0.25f,
            .start = start,
            .wake = wake};
}

// Step 2 as it was before starts had kinds: the reference an air swarm with
// no wake must still draw.
contracts::Float3 start_before_kinds(const scene::Swarm& swarm, std::uint32_t i,
                                     const contracts::Obstacles& obstacles) {
    const double r = swarm.radius;
    const double margin = r + animation::flight_delta + animation::first_drift_reach;
    const double needed = static_cast<double>(swarm.flight.clearance) + r + animation::flight_delta +
                          animation::first_drift_reach * std::sqrt(3.0);
    std::array<double, 3> low{};
    std::array<double, 3> high{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        low[axis] = contracts::component(swarm.flight.volume.min, static_cast<int>(axis)) + margin;
        high[axis] = contracts::component(swarm.flight.volume.max, static_cast<int>(axis)) - margin;
    }
    const std::uint64_t seed = scene::firefly_seed(swarm.flight.seed, first(i));
    for (int attempt = 0; attempt < scene::start_attempts; ++attempt) {
        std::array<double, 3> p{};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const double u =
                animation::draw(seed, static_cast<std::uint64_t>(attempt), static_cast<std::uint64_t>(axis));
            p[axis] = low[axis] + u * (high[axis] - low[axis]);
        }
        const contracts::Float3 start{static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2])};
        if (obstacles.distance(start) >= needed) {
            return start;
        }
    }
    FAIL("no start");
    return {};
}

// The margin step 2 shrinks the volume by, and the distance a start keeps.
constexpr double start_margin =
    static_cast<double>(firefly_radius) + animation::flight_delta + animation::first_drift_reach;

// A perch box over the ball's top, and one on the floor beside it.
constexpr contracts::Box over_ball{{-0.2f, 0.9f, -0.2f}, {0.2f, 1.2f, 0.2f}};
constexpr contracts::Box on_floor{{1.0f, -0.05f, 1.0f}, {1.5f, 0.3f, 1.5f}};

constexpr scene::SwarmWake a_wake{.from = 2.0, .to = 30.0, .power = 2.0, .ramp = 1.5};

}  // namespace

TEST_CASE("an air swarm with no wake draws exactly the starts it drew before starts had kinds") {
    const scene::SceneDescription s = scene::parse(swarmed(swarm_of(1)), "s");
    const tests::BallAndFloor obstacles(s);
    const scene::Swarm swarm = swarm_with(scene::AirStart{}, std::nullopt);
    for (std::uint32_t i = 0; i < swarm.count; ++i) {
        CAPTURE(i);
        const scene::Firefly f = scene::make_firefly(swarm, first(i), obstacles);
        const contracts::Float3 before = start_before_kinds(swarm, i, obstacles);
        CHECK(f.start.x == before.x);
        CHECK(f.start.y == before.y);
        CHECK(f.start.z == before.z);
        CHECK(std::holds_alternative<animation::NoPrelude>(f.prelude));
        CHECK_FALSE(f.wake.has_value());
    }
    // A wake draws apart from the starts: they are the same with one.
    const scene::Swarm woken = swarm_with(scene::AirStart{}, a_wake);
    for (std::uint32_t i = 0; i < 8; ++i) {
        const scene::Firefly f = scene::make_firefly(woken, first(i), obstacles);
        const contracts::Float3 before = start_before_kinds(swarm, i, obstacles);
        CHECK(f.start.x == before.x);
        CHECK(f.start.y == before.y);
        CHECK(f.start.z == before.z);
        REQUIRE(f.wake.has_value());
        CHECK(f.wake->ramp == a_wake.ramp);
    }
}

TEST_CASE("an above swarm starts in its volume's top slab and holds there until its wake") {
    constexpr double depth = 0.3;
    const scene::SceneDescription s = scene::parse(swarmed(swarm_of(1)), "s");
    const tests::BallAndFloor obstacles(s);
    const double top = static_cast<double>(volume_high) - start_margin;
    for (const std::optional<scene::SwarmWake>& wake : {std::optional<scene::SwarmWake>{}, std::optional{a_wake}}) {
        const scene::Swarm swarm = swarm_with(scene::AboveStart{.depth = depth}, wake);
        for (std::uint32_t i = 0; i < swarm.count; ++i) {
            CAPTURE(i);
            const scene::Firefly f = scene::make_firefly(swarm, first(i), obstacles);
            CHECK(static_cast<double>(f.start.y) >= static_cast<double>(static_cast<float>(top - depth)));
            CHECK(static_cast<double>(f.start.y) <= static_cast<double>(static_cast<float>(top)));
            REQUIRE(std::holds_alternative<animation::Hold>(f.prelude));
            CHECK(std::get<animation::Hold>(f.prelude).until == (wake ? f.wake->at : 0.0));
        }
    }
    // A depth past the whole range: all of it.
    const scene::Swarm deep = swarm_with(scene::AboveStart{.depth = 100.0}, std::nullopt);
    const scene::Swarm air = swarm_with(scene::AirStart{}, std::nullopt);
    for (std::uint32_t i = 0; i < 8; ++i) {
        const contracts::Float3 a = scene::make_firefly(deep, first(i), obstacles).start;
        const contracts::Float3 b = scene::make_firefly(air, first(i), obstacles).start;
        CHECK((a.x == b.x && a.y == b.y && a.z == b.z));
    }
}

namespace {

// Step 2's range, the volume shrunk by the margin, on axis `axis`.
bool in_range(float v, int axis) {
    const double low = static_cast<double>(contracts::component(job(0, clear_start).params.volume.min, axis));
    const double high = static_cast<double>(contracts::component(job(0, clear_start).params.volume.max, axis));
    return static_cast<double>(v) >= low + start_margin && static_cast<double>(v) <= high - start_margin;
}

// For each of `swarm`'s fireflies, perched: its loop's first point as its
// made flight places it, position() at its loop's begin, has its perch's x
// and z, bit for bit, and lies above it. Returns how many fail.
int not_straight_above(const scene::Swarm& swarm, const contracts::Obstacles& obstacles) {
    std::vector<animation::FlightJob> jobs;
    std::vector<contracts::Float3> perches;
    for (std::uint32_t i = 0; i < swarm.count; ++i) {
        const scene::Firefly f = scene::make_firefly(swarm, first(i), obstacles);
        animation::FlightJob j = job(scene::firefly_seed(swarm.flight.seed, first(i)), f.start);
        j.prelude = f.prelude;
        jobs.push_back(j);
        perches.push_back(std::get<animation::Perch>(f.prelude).at);
    }
    const std::vector<animation::Flight> flights = animation::make_flights(jobs, obstacles, 8);
    int failing = 0;
    for (std::size_t k = 0; k < flights.size(); ++k) {
        const contracts::Float3 first = animation::position(flights[k], Seconds(flights[k].begin));
        failing += first.x == perches[k].x && first.z == perches[k].z && first.y > perches[k].y ? 0 : 1;
    }
    return failing;
}

}  // namespace

TEST_CASE("a perched firefly rests a perch_gap above an upward-facing surface in its box, its loop straight "
          "above") {
    const scene::SceneDescription s = scene::parse(swarmed(swarm_of(1)), "s");
    const tests::BallAndFloor obstacles(s);
    const double height = static_cast<double>(firefly_radius) + animation::perch_gap;
    const double needed = static_cast<double>(swarm_clearance) + firefly_radius + animation::flight_delta +
                          animation::first_drift_reach * std::sqrt(3.0);
    for (const contracts::Box& box : {over_ball, on_floor}) {
        const scene::Swarm swarm =
            swarm_with(scene::PerchStart{.box = box, .linger_least = 2.0, .linger_most = 8.0}, a_wake);
        // Its loop's first point, as its flight places it, straight above.
        CHECK(not_straight_above(swarm, obstacles) == 0);
        for (std::uint32_t i = 0; i < swarm.count; ++i) {
            CAPTURE(i);
            const scene::Firefly f = scene::make_firefly(swarm, first(i), obstacles);
            REQUIRE(std::holds_alternative<animation::Perch>(f.prelude));
            const animation::Perch& perch = std::get<animation::Perch>(f.prelude);
            const contracts::Float3 p = perch.at;
            // A perch_gap above the nearest surface, within the trace's
            // tolerance, perch_gap / 10.
            const double d = obstacles.distance(p);
            CHECK(std::abs(d - height) <= animation::perch_gap / 10.0);
            // That surface faces up: the ball's normal there, or the floor's.
            const double to_ball = shapes::distance(s.shapes, tests::ball_shape, p);
            const double to_floor = shapes::distance(s.shapes, tests::floor_shape, p);
            const double ny = to_ball < to_floor ? (static_cast<double>(p.y) - 0.5) / (to_ball + 0.5) : 1.0;
            CHECK(ny >= std::cos(scene::perch_steepest) - 1e-6);
            // Inside its box.
            CHECK((p.x >= box.min.x && p.x <= box.max.x && p.y >= box.min.y && p.y <= box.max.y &&
                   p.z >= box.min.z && p.z <= box.max.z));
            // Its start, the first point less the flight's first_offset(), in
            // step 2's range and clear as any start.
            const std::array<double, 3> offset = animation::first_offset(scene::firefly_seed(4, first(i)));
            CHECK(static_cast<float>(static_cast<double>(f.start.x) + offset[0]) == p.x);
            CHECK(static_cast<float>(static_cast<double>(f.start.z) + offset[2]) == p.z);
            CHECK((in_range(f.start.x, 0) && in_range(f.start.y, 1) && in_range(f.start.z, 2)));
            CHECK(obstacles.distance(f.start) >= needed);
            // It rests until its wake and its linger.
            REQUIRE(f.wake.has_value());
            CHECK(perch.until - f.wake->at >= 2.0);
            CHECK(perch.until - f.wake->at <= 8.0);
        }
    }
}

TEST_CASE("a perch box with no upward-facing surface, or its top inside a shape, is refused, naming the firefly") {
    const scene::SceneDescription s = scene::parse(swarmed(swarm_of(1)), "s");
    const tests::BallAndFloor obstacles(s);
    const auto refusal = [&](const contracts::Box& box) {
        const scene::Swarm swarm = swarm_with(scene::PerchStart{.box = box, .linger_least = 1.0, .linger_most = 2.0},
                                              std::nullopt);
        return tests::error_of<animation::MotionError>([&] { return scene::make_firefly(swarm, first(0), obstacles); });
    };
    // In the air, nothing below it above its floor.
    CHECK(contains(refusal({{1.0f, 1.0f, 1.0f}, {1.5f, 1.2f, 1.5f}}), "firefly 0: no perch found"));
    // Its top inside the ball.
    CHECK(contains(refusal({{-0.1f, 0.4f, -0.1f}, {0.1f, 0.6f, 0.1f}}), "firefly 0: no perch found"));
    // On the ball's flank: surfaces, all steeper than perch_steepest.
    CHECK(contains(refusal({{0.49f, 0.35f, -0.01f}, {0.53f, 0.65f, 0.01f}}), "firefly 0: no perch found"));
    // Numbers make_firefly takes as given are refused, not drawn from.
    CHECK_THROWS_AS((void)scene::make_firefly(swarm_with(scene::AboveStart{.depth = 0.0}, std::nullopt), first(0),
                                              obstacles),
                    std::invalid_argument);
    CHECK_THROWS_AS((void)scene::make_firefly(
                        swarm_with(scene::AirStart{}, scene::SwarmWake{.from = 5.0, .to = 1.0, .power = 1.0}), first(0),
                        obstacles),
                    std::invalid_argument);
}

TEST_CASE("make_firefly refuses every wait past most_wait and a perch box not finite") {
    const scene::SceneDescription s = scene::parse(swarmed(swarm_of(1)), "s");
    const tests::BallAndFloor obstacles(s);
    const auto make = [&](const scene::SwarmStart& start, std::optional<scene::SwarmWake> wake) {
        return scene::make_firefly(swarm_with(start, wake), first(0), obstacles);
    };
    constexpr double most = animation::most_wait;
    const auto perch = [](double least, double most_linger) {
        return scene::PerchStart{.box = over_ball, .linger_least = least, .linger_most = most_linger};
    };
    // make_firefly's own refusal of its numbers, not a MotionError (which is
    // one of std::invalid_argument) from failing to draw with them.
    const auto numbers_refused = doctest::Contains("make_firefly: a swarm's");
    // A wake's to past most_wait, with any start.
    CHECK_THROWS_WITH_AS((void)make(scene::AboveStart{.depth = 0.3},
                                    scene::SwarmWake{.from = most + 1.0, .to = most + 1.0, .power = 1.0}),
                         numbers_refused, std::invalid_argument);
    CHECK_THROWS_WITH_AS((void)make(scene::AirStart{}, scene::SwarmWake{.from = 0.0, .to = 1e300, .power = 1.0}),
                         numbers_refused, std::invalid_argument);
    CHECK_NOTHROW((void)make(scene::AboveStart{.depth = 0.3}, scene::SwarmWake{.from = most, .to = most}));
    // A linger's most past most_wait, alone or with the wake's to; a sum
    // that would overflow is refused, not added.
    CHECK_THROWS_WITH_AS((void)make(perch(most + 1.0, most + 1.0), std::nullopt), numbers_refused,
                         std::invalid_argument);
    CHECK_THROWS_WITH_AS((void)make(perch(0.0, 700.0), scene::SwarmWake{.from = 0.0, .to = 3000.0}),
                         numbers_refused, std::invalid_argument);
    CHECK_THROWS_WITH_AS((void)make(perch(0.0, std::numeric_limits<double>::max()),
                                    scene::SwarmWake{.from = 0.0, .to = std::numeric_limits<double>::max()}),
                         numbers_refused, std::invalid_argument);
    CHECK_NOTHROW((void)make(perch(0.0, 600.0), scene::SwarmWake{.from = 0.0, .to = 3000.0}));
    // A perch box with a coordinate not finite.
    for (const float bad : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN()}) {
        CAPTURE(bad);
        scene::PerchStart minned = perch(1.0, 2.0);
        minned.box.min.x = bad;
        CHECK_THROWS_WITH_AS((void)make(minned, std::nullopt), numbers_refused, std::invalid_argument);
        scene::PerchStart maxed = perch(1.0, 2.0);
        maxed.box.max.z = bad;
        CHECK_THROWS_WITH_AS((void)make(maxed, std::nullopt), numbers_refused, std::invalid_argument);
    }
}

namespace {

// The floor's top at y = 0 and a thin shelf over it, 0.6 m square and 1 cm
// thick, its top at 1.5 m: a perch on the shelf has open air straight below
// it as well as above, where a start could be clear and its loop's first
// point below the perch. Exact distances, as contract 11 asks.
class Shelf final : public contracts::Obstacles {
public:
    double distance(contracts::Float3 p) const override {
        const double qx = std::abs(static_cast<double>(p.x)) - 0.3;
        const double qy = std::abs(static_cast<double>(p.y) - 1.495) - 0.005;
        const double qz = std::abs(static_cast<double>(p.z)) - 0.3;
        const double outside = std::hypot(std::max(qx, 0.0), std::max(qy, 0.0), std::max(qz, 0.0));
        const double inside = std::min(std::max({qx, qy, qz}), 0.0);
        return std::min(static_cast<double>(p.y), outside + inside);
    }
    bool touches(const contracts::Box&) const override { return true; }
};

}  // namespace

TEST_CASE("a perched firefly's loop begins above its perch, never below it") {
    // On the shelf, starts are drawn from 0.18 to 2.07 m and many below it
    // are clear: each accepted perch has its loop's first point above it.
    const Shelf shelf;
    const scene::Swarm swarm = swarm_with(
        scene::PerchStart{.box = {{-0.2f, 1.45f, -0.2f}, {0.2f, 1.8f, 0.2f}}, .linger_least = 1.0, .linger_most = 2.0},
        std::nullopt, 64);
    int below = 0;
    for (std::uint32_t i = 0; i < swarm.count; ++i) {
        const scene::Firefly f = scene::make_firefly(swarm, first(i), shelf);
        const std::array<double, 3> offset = animation::first_offset(scene::firefly_seed(swarm.flight.seed, first(i)));
        const contracts::Float3 perch = std::get<animation::Perch>(f.prelude).at;
        CHECK(perch.y > 1.5f);
        below += static_cast<double>(f.start.y) + offset[1] > static_cast<double>(perch.y) ? 0 : 1;
    }
    CHECK(below == 0);
}

TEST_CASE("a perch box partly outside the flight's range perches only where a loop can start straight above") {
    // The volume's x reaches 2 m, its start's range 2 - 0.13 = 1.87 m; a
    // first point is at most first_drift_reach from its start, so no perch
    // past 1.97 m can have one straight above it. A box from 1.6 to 2.4 m on
    // the floor: every perch accepted has its loop's first point straight
    // above; none is moved in, so a box wholly past 1.97 m is refused.
    const scene::SceneDescription s = scene::parse(swarmed(swarm_of(1)), "s");
    const tests::BallAndFloor obstacles(s);
    const scene::Swarm partly = swarm_with(
        scene::PerchStart{.box = {{1.6f, -0.05f, 1.0f}, {2.4f, 0.3f, 1.5f}}, .linger_least = 1.0, .linger_most = 2.0},
        std::nullopt, 32);
    CHECK(not_straight_above(partly, obstacles) == 0);
    int moved = 0;
    for (std::uint32_t i = 0; i < partly.count; ++i) {
        const scene::Firefly f = scene::make_firefly(partly, first(i), obstacles);
        const contracts::Float3 perch = std::get<animation::Perch>(f.prelude).at;
        CHECK(in_range(f.start.x, 0));
        CHECK(perch.x <= 1.97f + 1e-6f);
        // Never moved: the perch is on a line some attempt drew, keyed
        // (seed_i, perch_draws, attempt, axis) (swarm.h, step 2p), its start
        // the drawn x and z less the offset, its line that start plus it.
        const std::uint64_t seed = scene::firefly_seed(partly.flight.seed, first(i));
        const std::array<double, 3> offset = animation::first_offset(seed);
        const auto line_of = [&](std::uint64_t attempt, std::uint64_t axis, float lo, float hi, std::size_t at) {
            const double value = lo + animation::draw(seed, scene::perch_draws, attempt, axis) *
                                          (static_cast<double>(hi) - lo);
            return static_cast<float>(static_cast<double>(static_cast<float>(value - offset[at])) + offset[at]);
        };
        bool drawn = false;
        for (std::uint64_t a = 0; a < static_cast<std::uint64_t>(scene::start_attempts); ++a) {
            drawn = drawn || (line_of(a, 0, 1.6f, 2.4f, 0) == perch.x && line_of(a, 2, 1.0f, 1.5f, 2) == perch.z);
        }
        moved += drawn ? 0 : 1;
    }
    CHECK(moved == 0);
    const scene::Swarm outside = swarm_with(
        scene::PerchStart{.box = {{2.1f, -0.05f, 1.0f}, {2.5f, 0.3f, 1.5f}}, .linger_least = 1.0, .linger_most = 2.0},
        std::nullopt);
    CHECK(contains(
        tests::error_of<animation::MotionError>([&] { return scene::make_firefly(outside, first(0), obstacles); }),
        "firefly 0: no perch found"));
}

TEST_CASE("wake times lie in [from, to], the share awake by t growing as ((t - from) / (to - from))^power") {
    constexpr std::uint32_t count = 4096;
    const scene::SceneDescription s = scene::parse(swarmed(swarm_of(1)), "s");
    const tests::BallAndFloor obstacles(s);
    for (const double power : {1.0, 2.0, 0.5}) {
        CAPTURE(power);
        scene::SwarmWake w = a_wake;
        w.power = power;
        const scene::Swarm swarm = swarm_with(scene::AirStart{}, w, count);
        std::vector<double> at;
        at.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            at.push_back(scene::make_firefly(swarm, first(i), obstacles).wake->at);
        }
        CHECK(std::ranges::all_of(at, [&](double t) { return t >= w.from && t <= w.to; }));
        // The Kolmogorov-Smirnov distance from the stated distribution. The
        // draws are fixed, so this is not a chance of failing; the bound is
        // what 4096 uniform draws stay under with probability 0.999,
        // 1.95 / sqrt(4096), some 0.03.
        std::ranges::sort(at);
        double ks = 0.0;
        for (std::size_t k = 0; k < at.size(); ++k) {
            const double expected = std::pow((at[k] - w.from) / (w.to - w.from), power);
            const double n = static_cast<double>(at.size());
            ks = std::max({ks, std::abs(static_cast<double>(k + 1) / n - expected),
                           std::abs(static_cast<double>(k) / n - expected)});
        }
        INFO("KS distance " << ks);
        CHECK(ks < 1.95 / std::sqrt(static_cast<double>(count)));
    }
}

TEST_CASE("a firefly's wake is drawn apart from its start") {
    // With power 1 a wake's u is (at - from) / (to - from); a start taken
    // at its first draw has its height's u at (y - low) / (high - low). Keyed
    // apart (swarm.h, wake_draws), the two agree, to a float's rounding of
    // y, for no firefly; keyed alike, for nearly all.
    constexpr std::uint32_t count = 512;
    const scene::SceneDescription s = scene::parse(swarmed(swarm_of(1)), "s");
    const tests::BallAndFloor obstacles(s);
    scene::SwarmWake w = a_wake;
    w.power = 1.0;
    const scene::Swarm swarm = swarm_with(scene::AirStart{}, w, count);
    const double low = static_cast<double>(volume_low) + start_margin;
    const double high = static_cast<double>(volume_high) - start_margin;
    int alike = 0;
    for (std::uint32_t i = 0; i < count; ++i) {
        const scene::Firefly f = scene::make_firefly(swarm, first(i), obstacles);
        const double wake_u = (f.wake->at - w.from) / (w.to - w.from);
        const double height_u = (static_cast<double>(f.start.y) - low) / (high - low);
        alike += std::abs(wake_u - height_u) < 1e-6 ? 1 : 0;
    }
    CHECK(alike == 0);
}

TEST_CASE("perched fireflies' flights made in parallel are the ones made one by one, openings and all") {
    const scene::SceneDescription s = scene::parse(swarmed(swarm_of(1)), "s");
    const tests::BallAndFloor obstacles(s);
    const scene::Swarm swarm =
        swarm_with(scene::PerchStart{.box = over_ball, .linger_least = 1.0, .linger_most = 3.0}, a_wake, 12);
    std::vector<animation::FlightJob> jobs;
    for (std::uint32_t i = 0; i < swarm.count; ++i) {
        const scene::Firefly f = scene::make_firefly(swarm, first(i), obstacles);
        animation::FlightJob j = job(scene::firefly_seed(4, first(i)), f.start);
        j.prelude = f.prelude;
        jobs.push_back(j);
    }
    for (const std::size_t workers : {std::size_t{1}, std::size_t{3}, std::size_t{8}}) {
        CAPTURE(workers);
        const std::vector<animation::Flight> together = animation::make_flights(jobs, obstacles, workers);
        for (std::size_t k = 0; k < jobs.size(); ++k) {
            const animation::Flight alone = animation::make_flight(jobs[k], obstacles);
            CHECK(together[k].begin == alone.begin);
            CHECK(together[k].opening.size() == 2);
            CHECK(together[k].flashes.opening == alone.flashes.opening);
            for (const double t : {0.5, together[k].begin - 0.1, together[k].begin + 3.0}) {
                const contracts::Float3 a = animation::position(together[k], Seconds(t));
                const contracts::Float3 b = animation::position(alone, Seconds(t));
                CHECK((a.x == b.x && a.y == b.y && a.z == b.z));
            }
        }
    }
}

// Step 5's redraw (swarm.h).

namespace {

// The marbles' case at a modest count: a table top, nine marbles in a
// cluster on it, a swarm perched on them and the table among them, and a
// swarm held above, both waking.
constexpr float marble_radius = 0.01f;
constexpr std::array<float, 3> marble_rows = {-0.06f, 0.0f, 0.06f};
constexpr contracts::Box table_volume{{-0.25f, 0.795f, -0.25f}, {0.25f, 1.1f, 0.25f}};
constexpr std::uint32_t perched_count = 40;
constexpr std::uint32_t above_count = 24;
constexpr std::uint32_t table_shapes = 10;  // the table and the nine marbles

std::string table_text(std::uint64_t perched_seed, std::uint64_t above_seed) {
    std::string text = tests::camera_text("[0, 0.95, 0.5]", "[0, 0.76, 0]", 40) + tests::sky("[0, 0, 0]", "[0, 0, 0]") +
                       "[materials.wood]\nkind = \"rough\"\ncolor = [0.2, 0.1, 0.05]\n"
                       "[materials.glass]\nkind = \"dielectric\"\nior = 1.5\n"
                       "[materials.glow]\nkind = \"emissive\"\nradiance = [10, 9, 3]\n"
                       "[[shapes]]\nkind = \"box\"\nmin = [-0.5, 0.7, -0.5]\nmax = [0.5, 0.75, 0.5]\n"
                       "material = \"wood\"\n";
    std::string targets;
    int n = 0;
    for (const float x : marble_rows) {
        for (const float z : marble_rows) {
            text += "[[shapes]]\nkind = \"sphere\"\nname = \"m" + std::to_string(n) + "\"\ncenter = [" +
                    std::to_string(x) + ", 0.76, " + std::to_string(z) + "]\nradius = 0.01\nmaterial = \"glass\"\n";
            targets += (n == 0 ? "\"m" : ", \"m") + std::to_string(n) + "\"";
            ++n;
        }
    }
    const std::string flight = "radius = 0.0015\nmaterial = \"glow\"\nmin = [-0.25, 0.795, -0.25]\n"
                               "max = [0.25, 1.1, 0.25]\ntargets = [" +
                               targets +
                               "]\nspeed = 0.05\nclearance = 0.008\ncircle = 6\nswoop = 0\ndrift = 1\n"
                               "flash = 0.9\ndim = 0.1\n";
    text += "[[swarms]]\ncount = " + std::to_string(perched_count) + "\n" + flight +
            "seed = " + std::to_string(perched_seed) +
            "\nstart = { kind = \"perch\", min = [-0.09, 0.74, -0.09], max = [0.09, 0.8, 0.09], linger = [1, 3] }\n"
            "wake = { from = 1, to = 20, power = 2, ramp = 1 }\n";
    text += "[[swarms]]\ncount = " + std::to_string(above_count) + "\n" + flight +
            "seed = " + std::to_string(above_seed) +
            "\nstart = { kind = \"above\", depth = 0.08 }\nwake = { from = 0, to = 10, power = 1, ramp = 1 }\n";
    return text;
}

// The table's swarms as the reader reads them (swarm.h): the perched one,
// or the one held above.
scene::Swarm table_swarm(bool perched, std::uint64_t seed) {
    animation::FlightParams params{
        .volume = table_volume, .speed = 0.05f, .clearance = 0.008f, .weights = {6.0f, 0.0f, 1.0f}, .seed = seed};
    for (const float x : marble_rows) {
        for (const float z : marble_rows) {
            params.targets.push_back({{x, 0.76f, z}, marble_radius});
        }
    }
    scene::Swarm swarm{.count = perched ? perched_count : above_count,
                       .radius = 0.0015f,
                       .flight = params,
                       .flash = 0.9f,
                       .dim = 0.1f};
    if (perched) {
        swarm.start = scene::PerchStart{
            .box = {{-0.09f, 0.74f, -0.09f}, {0.09f, 0.8f, 0.09f}}, .linger_least = 1.0, .linger_most = 3.0};
        swarm.wake = scene::SwarmWake{.from = 1.0, .to = 20.0, .power = 2.0, .ramp = 1.0};
    } else {
        // The reader reads its numbers as floats: 0.08 as the float nearest.
        swarm.start = scene::AboveStart{.depth = static_cast<double>(0.08f)};
        swarm.wake = scene::SwarmWake{.from = 0.0, .to = 10.0, .power = 1.0, .ramp = 1.0};
    }
    return swarm;
}

// A scene's first `count` shapes, which do not move, as contract 11 asks of
// them: each shape kind's exact tests (core/shapes/shapes.h).
class StillShapesOf final : public contracts::Obstacles {
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

// Which of a swarm's fireflies, its first at shape `first_shape`, the
// reader placed somewhere other than its first draw's start: drawn again.
std::vector<std::uint32_t> redrawn(const scene::SceneDescription& s, const scene::Swarm& swarm,
                                   std::uint32_t first_shape, const contracts::Obstacles& still) {
    std::vector<std::uint32_t> again;
    for (std::uint32_t i = 0; i < swarm.count; ++i) {
        const contracts::Float3 placed = contracts::translation(s.shapes.transforms[first_shape + i]);
        const contracts::Float3 drawn = scene::make_firefly(swarm, first(i), still).start;
        if (!(placed.x == drawn.x && placed.y == drawn.y && placed.z == drawn.z)) {
            again.push_back(i);
        }
    }
    return again;
}

}  // namespace

TEST_CASE("a firefly's seed for each draw: its first the one it had before redraws, every one its own") {
    std::vector<std::uint64_t> seeds;
    for (const std::uint64_t swarm_seed : {0ull, 4ull, 0xffffffffffffffffull}) {
        for (std::uint32_t i = 0; i < 64; ++i) {
            // Draw 0 is step 1's seed as it was, bit for bit.
            CHECK(scene::firefly_seed(swarm_seed, first(i)) ==
                  animation::splitmix64(animation::splitmix64(swarm_seed) ^ i));
            for (std::uint32_t d = 0; d < scene::firefly_draws; ++d) {
                seeds.push_back(scene::firefly_seed(swarm_seed, {.index = i, .draw = d}));
            }
        }
    }
    std::ranges::sort(seeds);
    CHECK(std::ranges::adjacent_find(seeds) == seeds.end());
    // A draw past firefly_draws, or a firefly past the count, is refused.
    const scene::SceneDescription s = scene::parse(swarmed(swarm_of(1)), "s");
    const tests::BallAndFloor obstacles(s);
    const scene::Swarm swarm = swarm_with(scene::AirStart{}, std::nullopt, 8);
    CHECK_THROWS_AS((void)scene::make_firefly(swarm, {.index = 0, .draw = scene::firefly_draws}, obstacles),
                    std::invalid_argument);
    CHECK_THROWS_AS((void)scene::make_firefly(swarm, {.index = 8, .draw = 0}, obstacles), std::invalid_argument);
    CHECK_NOTHROW((void)scene::make_firefly(swarm, {.index = 7, .draw = scene::firefly_draws - 1}, obstacles));
}


TEST_CASE("a swarm with no refusals is the swarm it was before redraws, bit for bit") {
    constexpr std::uint32_t count = 24;
    const scene::SceneDescription s = scene::parse(swarmed(swarm_of(static_cast<int>(count))), "s");
    const tests::BallAndFloor obstacles(s);
    const scene::Swarm swarm = swarm_with(scene::AirStart{}, std::nullopt, count);
    int differ = 0;
    for (std::uint32_t i = 0; i < count; ++i) {
        CAPTURE(i);
        // The first draw, as it was made before there were redraws.
        const scene::Firefly f = scene::make_firefly(swarm, first(i), obstacles);
        const animation::Flight alone =
            animation::make_flight(job(scene::firefly_seed(swarm.flight.seed, first(i)), f.start), obstacles);
        const contracts::Transform& placed = s.shapes.transforms[written_shapes + i];
        const contracts::Transform expected = contracts::placed(f.start, firefly_radius);
        for (std::size_t r = 0; r < 3; ++r) {
            for (std::size_t c = 0; c < 4; ++c) {
                differ += placed.m[r][c] == expected.m[r][c] ? 0 : 1;
            }
        }
        const animation::Flight& made = s.animation.motions.flights[i];
        REQUIRE(made.segments.size() == alone.segments.size());
        for (std::size_t k = 0; k < made.segments.size(); ++k) {
            differ += made.segments[k].start == alone.segments[k].start &&
                              made.segments[k].numbers == alone.segments[k].numbers
                          ? 0
                          : 1;
        }
        differ += made.loop == alone.loop && made.flashes.starts == alone.flashes.starts ? 0 : 1;
        const animation::ScheduleGlow& glow = s.animation.glows.schedules[s.animation.glowers[i].glow.index];
        differ += glow.schedule.starts == alone.flashes.starts ? 0 : 1;
    }
    CHECK(differ == 0);
}

namespace {

// Step 5 done one firefly at a time: its first draw whose flight is made,
// and that flight; none if every draw is refused. The reference the
// reader's rounds, made in parallel, must match.
struct Drawn {
    std::uint32_t draw = 0;
    scene::Firefly firefly;
    animation::Flight flight;
};

std::optional<Drawn> drawn_one_by_one(const scene::Swarm& swarm, std::uint32_t i, const contracts::Obstacles& still) {
    for (std::uint32_t d = 0; d < scene::firefly_draws; ++d) {
        const scene::Firefly f = scene::make_firefly(swarm, {.index = i, .draw = d}, still);
        animation::FlightJob j{.params = swarm.flight, .start = f.start, .body = swarm.radius, .prelude = f.prelude};
        j.params.seed = scene::firefly_seed(swarm.flight.seed, {.index = i, .draw = d});
        try {
            return Drawn{.draw = d, .firefly = f, .flight = animation::make_flight(j, still)};
        } catch (const animation::MotionError&) {
            // refused: its next draw
        }
    }
    return std::nullopt;
}

}  // namespace

TEST_CASE("a swarm with refused fireflies loads, each drawn again whole, as step 5 one by one would draw it") {
    // Seeds whose swarms on the table have fireflies refused at their first
    // draw (the marbles' case, below): each refused firefly is drawn again
    // from its next draws, its sphere placed at its new start, its light
    // still its sphere's, its glow woken as its new draw wakes; the rest
    // are their first draws. The reader makes each round in parallel; the
    // reference here, one at a time, is the same whatever the threads.
    int again = 0;
    for (const std::uint64_t seed : {0ull, 6ull}) {
        CAPTURE(seed);
        const scene::SceneDescription s = scene::parse(table_text(seed, seed + 100), "t");
        const StillShapesOf still(s, table_shapes);
        std::uint32_t shape = table_shapes;
        for (const scene::Swarm& swarm : {table_swarm(true, seed), table_swarm(false, seed + 100)}) {
            for (std::uint32_t i = 0; i < swarm.count; ++i, ++shape) {
                CAPTURE(i);
                const std::optional<Drawn> expected = drawn_one_by_one(swarm, i, still);
                REQUIRE(expected.has_value());
                again += expected->draw > 0 ? 1 : 0;
                // Its sphere, placed at its start, and its light, its sphere's.
                const contracts::Float3 at = contracts::translation(s.shapes.transforms[shape]);
                CHECK((at.x == expected->firefly.start.x && at.y == expected->firefly.start.y &&
                       at.z == expected->firefly.start.z));
                const std::uint32_t light = s.shape_lights[shape];
                REQUIRE(light != lights::no_light);
                CHECK(s.sphere_lights[s.lights[light].index].shape == shape);
                // Its flight, and its glow's flashes and wake.
                const std::size_t motion = shape - table_shapes;
                const animation::Flight& made = s.animation.motions.flights[motion];
                CHECK(made.loop == expected->flight.loop);
                CHECK(made.begin == expected->flight.begin);
                for (const double t : {0.0, made.begin + 1.5, 77.25}) {
                    const contracts::Float3 a = animation::position(made, Seconds(t));
                    const contracts::Float3 b = animation::position(expected->flight, Seconds(t));
                    CHECK((a.x == b.x && a.y == b.y && a.z == b.z));
                }
                const animation::ScheduleGlow& glow =
                    s.animation.glows.schedules[s.animation.glowers[motion].glow.index];
                CHECK(glow.schedule.opening == expected->flight.flashes.opening);
                REQUIRE(glow.wake.has_value());
                CHECK(glow.wake->at == expected->firefly.wake->at);
            }
        }
    }
    INFO(again << " fireflies drawn again");
    CHECK(again > 0);
}

TEST_CASE("the marbles' case loads for every seed: perched and held swarms on a table among marbles") {
    // Twenty seeds of each swarm. A refused firefly in any would once have
    // refused the scene; each is drawn again, and none is refused eight
    // times. Some are drawn again in most seeds (shown), so this fails
    // without the redraw.
    int seeds_redrawn = 0;
    for (std::uint64_t seed = 0; seed < 20; ++seed) {
        CAPTURE(seed);
        const std::string text = table_text(seed, seed + 100);
        const scene::SceneDescription s = scene::parse(text, "t");
        CHECK(s.animation.motions.flights.size() == perched_count + above_count);
        const StillShapesOf still(s, table_shapes);
        const std::size_t perched = redrawn(s, table_swarm(true, seed), table_shapes, still).size();
        const std::size_t above =
            redrawn(s, table_swarm(false, seed + 100), table_shapes + perched_count, still).size();
        const std::size_t again = perched + above;
        seeds_redrawn += again > 0 ? 1 : 0;
    }
    CHECK(seeds_redrawn > 10);
}

TEST_CASE("a swarm no draw can fly is refused after firefly_draws draws, naming its line, the firefly and why") {
    // A volume too tight to circle the ball in, and only circling: every
    // draw's flight is refused.
    const std::string tight = replaced(replaced(swarm_of(2), "min = [-2, 0.05, -2]\nmax = [2, 2.2, 2]",
                                                "min = [-0.7, 0.05, -0.7]\nmax = [1.4, 1.3, 1.0]"),
                                       "circle = 2\nswoop = 1\ndrift = 2", "circle = 1\nswoop = 0\ndrift = 0");
    const std::string refused = error_of(swarmed(tight));
    INFO(refused);
    CHECK(contains(refused, "s.toml:26: swarm 1's firefly 0's flight was refused at each of its 8 draws; the last: "));
    CHECK(contains(refused, "could not be drawn clear"));
}

TEST_CASE("a written flight that is refused is reported at its motion's line, with no count of draws") {
    // A swarm beside it that loads, and a written firefly circling in a
    // volume too tight: the error is the written flight's refusal, at its
    // motion's line, as make_flight gave it, not a swarm's "refused at each
    // of its draws". What this shows is the error's kind and place; how many
    // times the flight was asked for is not observable from here, and
    // scene.cpp's fly() is where the first batch refuses it.
    const std::string text = swarmed(swarm_of(4)) +
                             "[[shapes]]\nkind = \"sphere\"\ncenter = [1.2, 1.0, 0.8]\nradius = 0.03\nmaterial = "
                             "\"glow\"\nmotion = { kind = \"flight\", min = [-0.7, 0.05, -0.7], max = [1.4, 1.3, 1.0], "
                             "targets = [\"ball\"], speed = 0.45, clearance = 0.06, circle = 1, swoop = 0, drift = 0, "
                             "seed = 5 }\n";
    const std::string refused = error_of(text);
    INFO(refused);
    CHECK(contains(refused, "s.toml:47: shape 3's motion: flight episode"));
    CHECK(contains(refused, "could not be drawn clear"));
    CHECK_FALSE(contains(refused, "each of its"));
}
