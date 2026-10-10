// Swarms (core/scene/swarm.h): many fireflies from one entry, each what a
// written firefly is, its seed and start a function of the swarm's seed and
// its number; and flights made in parallel (core/animation/flight.h,
// make_flights), the same flights and the same error as made one by one,
// whatever the number of threads.

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
#include "support/ball_on_floor.h"
#include "support/text.h"

using namespace serenity;
using frame::Seconds;
using tests::contains;
using tests::replaced;

namespace {

// A swarm's numbers, as written below.
constexpr float firefly_radius = 0.02f;
constexpr float swarm_clearance = 0.05f;
constexpr float volume_low = 0.05f;
constexpr float volume_high = 2.2f;
constexpr float volume_reach = 2.0f;  // the volume's x and z bounds, either way

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
    CHECK(scene::firefly_seed(4, 0) != scene::firefly_seed(4, 1));
    CHECK(scene::firefly_seed(4, 0) != scene::firefly_seed(5, 0));
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

namespace {

// A flight job about the ball, in the swarm's volume, from `start`.
animation::FlightJob job(std::uint64_t seed, contracts::Float3 start) {
    return {.params = {.volume = {{-volume_reach, volume_low, -volume_reach},
                                  {volume_reach, volume_high, volume_reach}},
                       .targets = {{{0.0f, 0.5f, 0.0f}, 0.5f}},
                       .speed = 0.4f,
                       .clearance = swarm_clearance,
                       .weights = {2.0f, 1.0f, 2.0f},
                       .seed = seed},
            .start = start,
            .body = firefly_radius};
}

constexpr contracts::Float3 clear_start{1.2f, 1.0f, 0.8f};
constexpr contracts::Float3 inside_ball{0.0f, 0.5f, 0.0f};

}  // namespace

TEST_CASE("flights made in parallel are the flights made one by one, in order") {
    constexpr std::uint64_t count = 40;
    const scene::SceneDescription s = scene::parse(swarmed(swarm_of(1)), "s");
    const tests::BallAndFloor obstacles(s);
    std::vector<animation::FlightJob> jobs;
    jobs.reserve(count);
    for (std::uint64_t k = 0; k < count; ++k) {
        jobs.push_back(job(100 + k, {clear_start.x, clear_start.y + 0.01f * static_cast<float>(k), clear_start.z}));
    }
    // One worker, a few, more than the jobs: the same flights every time.
    for (const std::size_t workers : {std::size_t{1}, std::size_t{2}, std::size_t{7}, std::size_t{64}}) {
        CAPTURE(workers);
        const std::vector<animation::Flight> together = animation::make_flights(jobs, obstacles, workers);
        REQUIRE(together.size() == jobs.size());
        for (std::size_t k = 0; k < jobs.size(); ++k) {
            const animation::Flight alone =
                animation::make_flight(jobs[k], obstacles);
            CHECK(together[k].loop == alone.loop);
            REQUIRE(together[k].segments.size() == alone.segments.size());
            const contracts::Float3 a = animation::position(together[k], Seconds(17.5));
            const contracts::Float3 b = animation::position(alone, Seconds(17.5));
            CHECK(a.x == b.x);
            CHECK(a.y == b.y);
            CHECK(a.z == b.z);
        }
        CHECK(animation::make_flights({}, obstacles, workers).empty());
    }
}

TEST_CASE("of flights made in parallel, the lowest that fails is the one reported") {
    const scene::SceneDescription s = scene::parse(swarmed(swarm_of(1)), "s");
    const tests::BallAndFloor obstacles(s);
    std::vector<animation::FlightJob> jobs;
    for (std::uint64_t k = 0; k < 30; ++k) {
        jobs.push_back(job(k, clear_start));
    }
    // Starts inside the ball: 7 and 19 cannot be made.
    jobs[19].start = inside_ball;
    jobs[7].start = {0.1f, 0.5f, 0.0f};
    // Which worker reaches which job first varies with their number; the job
    // reported must not.
    for (const std::size_t workers : {std::size_t{1}, std::size_t{2}, std::size_t{7}, std::size_t{64}}) {
        CAPTURE(workers);
        try {
            (void)animation::make_flights(jobs, obstacles, workers);
            FAIL("expected an error");
        } catch (const animation::FlightsError& error) {
            CHECK(error.job == 7);
            CHECK(std::string(error.what()).starts_with("flight 7: "));
        }
    }
}

namespace {

// The ball and floor, but an answer near one start throws what no flight
// expects: a failure that is not a refusal.
class Breaking final : public contracts::Obstacles {
public:
    explicit Breaking(const scene::SceneDescription& s) : inner_(s) {}
    double distance(contracts::Float3 p) const override {
        if (p.z > 1.7f) {
            throw std::runtime_error("the obstacles broke");
        }
        return inner_.distance(p);
    }
    bool touches(const contracts::Box& box) const override { return inner_.touches(box); }

private:
    tests::BallAndFloor inner_;
};

}  // namespace

TEST_CASE("of flights made in parallel, any exception reaches the caller, the lowest job's") {
    const scene::SceneDescription s = scene::parse(swarmed(swarm_of(1)), "s");
    const Breaking obstacles(s);
    std::vector<animation::FlightJob> jobs;
    for (std::uint64_t k = 0; k < 24; ++k) {
        jobs.push_back(job(k, clear_start));
    }
    jobs[5].start = {1.2f, 1.0f, 1.8f};  // breaks
    jobs[9].start = inside_ball;          // refused
    try {
        (void)animation::make_flights(jobs, obstacles, animation::flight_workers());
        FAIL("expected an error");
    } catch (const animation::FlightsError&) {
        FAIL("the refusal at 9 was reported, not the failure at 5");
    } catch (const std::runtime_error& error) {
        CHECK(std::string(error.what()) == "the obstacles broke");
    }
}

TEST_CASE("a flight too fast or too long to sample is refused, its count never converted past int64") {
    const scene::SceneDescription s = scene::parse(swarmed(swarm_of(1)), "s");
    const tests::BallAndFloor obstacles(s);
    animation::FlightJob fast = job(3, clear_start);
    fast.params.speed = 1e30f;  // within float's range, as the reader accepts
    CHECK_THROWS_AS(animation::make_flight(fast, obstacles), animation::MotionError);
    // The bound is the params': a small one refuses the first drift, the
    // default makes the flight.
    animation::FlightJob bounded = job(3, clear_start);
    CHECK_NOTHROW((void)animation::make_flight(bounded, obstacles));
    bounded.params.most_steps = 2;
    CHECK_THROWS_WITH_AS(animation::make_flight(bounded, obstacles),
                         doctest::Contains("is not clear"), animation::MotionError);
}

namespace {

// Contract 11 answered with an exception of the standard type a refusal
// derives from: not a refusal of any flight.
class Arguing final : public contracts::Obstacles {
public:
    double distance(contracts::Float3) const override { throw std::invalid_argument("the obstacles argued"); }
    bool touches(const contracts::Box&) const override { return false; }
};

}  // namespace

TEST_CASE("a refusal carries its job and its reason; another std::invalid_argument is not taken for one") {
    const scene::SceneDescription s = scene::parse(swarmed(swarm_of(1)), "s");
    const tests::BallAndFloor obstacles(s);
    const std::vector<animation::FlightJob> jobs = {job(1, clear_start), job(2, inside_ball)};
    try {
        (void)animation::make_flights(jobs, obstacles, animation::flight_workers());
        FAIL("expected an error");
    } catch (const animation::FlightsError& error) {
        CHECK(error.job == 1);
        CHECK(error.reason.starts_with("the flight's start"));
        CHECK(std::string(error.what()) == "flight 1: " + error.reason);
    }
    const Arguing arguing;
    try {
        (void)animation::make_flights({job(1, clear_start)}, arguing, animation::flight_workers());
        FAIL("expected an error");
    } catch (const animation::MotionError&) {
        FAIL("an Obstacles' own exception was taken for a refusal");
    } catch (const std::invalid_argument& error) {
        CHECK(std::string(error.what()) == "the obstacles argued");
    }
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
    const std::uint64_t seed = scene::firefly_seed(swarm.flight.seed, i);
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
        const scene::Firefly f = scene::make_firefly(swarm, i, obstacles);
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
        const scene::Firefly f = scene::make_firefly(woken, i, obstacles);
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
            const scene::Firefly f = scene::make_firefly(swarm, i, obstacles);
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
        const contracts::Float3 a = scene::make_firefly(deep, i, obstacles).start;
        const contracts::Float3 b = scene::make_firefly(air, i, obstacles).start;
        CHECK((a.x == b.x && a.y == b.y && a.z == b.z));
    }
}

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
        for (std::uint32_t i = 0; i < swarm.count; ++i) {
            CAPTURE(i);
            const scene::Firefly f = scene::make_firefly(swarm, i, obstacles);
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
            // Its loop starts straight above it, clear as any start.
            CHECK(f.start.x == p.x);
            CHECK(f.start.z == p.z);
            CHECK(f.start.y > p.y);
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
        return tests::error_of<animation::MotionError>([&] { return scene::make_firefly(swarm, 0, obstacles); });
    };
    // In the air, nothing below it above its floor.
    CHECK(contains(refusal({{1.0f, 1.0f, 1.0f}, {1.5f, 1.2f, 1.5f}}), "firefly 0: no perch found"));
    // Its top inside the ball.
    CHECK(contains(refusal({{-0.1f, 0.4f, -0.1f}, {0.1f, 0.6f, 0.1f}}), "firefly 0: no perch found"));
    // On the ball's flank: surfaces, all steeper than perch_steepest.
    CHECK(contains(refusal({{0.49f, 0.35f, -0.01f}, {0.53f, 0.65f, 0.01f}}), "firefly 0: no perch found"));
    // Numbers make_firefly takes as given are refused, not drawn from.
    CHECK_THROWS_AS((void)scene::make_firefly(swarm_with(scene::AboveStart{.depth = 0.0}, std::nullopt), 0,
                                              obstacles),
                    std::invalid_argument);
    CHECK_THROWS_AS((void)scene::make_firefly(
                        swarm_with(scene::AirStart{}, scene::SwarmWake{.from = 5.0, .to = 1.0, .power = 1.0}), 0,
                        obstacles),
                    std::invalid_argument);
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
            at.push_back(scene::make_firefly(swarm, i, obstacles).wake->at);
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
        const scene::Firefly f = scene::make_firefly(swarm, i, obstacles);
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
        const scene::Firefly f = scene::make_firefly(swarm, i, obstacles);
        animation::FlightJob j = job(scene::firefly_seed(4, i), f.start);
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
