// Swarms (core/scene/swarm.h): many fireflies from one entry, each what a
// written firefly is, its seed and start a function of the swarm's seed and
// its number; and flights made in parallel (core/animation/flight.h,
// make_flights), the same flights and the same error as made one by one.
//
// Not tested here: that make_flights() gives the same flights whatever the
// number of threads. It takes its count from the machine
// (std::thread::hardware_concurrency()), not from its caller, so a test
// cannot vary it (I.1); the parallel results are compared with the serial
// ones on this machine's count only.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <doctest/doctest.h>

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
    return tests::error_of<scene::Error>([&] { return scene::parse(text, "s.toml"); });
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
    return {.params = {.volume = {{-volume_reach, volume_low, -volume_reach}, {volume_reach, volume_high, volume_reach}},
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
    const std::vector<animation::Flight> together = animation::make_flights(jobs, obstacles);
    REQUIRE(together.size() == jobs.size());
    for (std::size_t k = 0; k < jobs.size(); ++k) {
        const animation::Flight alone = animation::make_flight(jobs[k].params, jobs[k].start, jobs[k].body, obstacles);
        CHECK(together[k].loop == alone.loop);
        REQUIRE(together[k].segments.size() == alone.segments.size());
        const contracts::Float3 a = animation::position(together[k], Seconds(17.5));
        const contracts::Float3 b = animation::position(alone, Seconds(17.5));
        CHECK(a.x == b.x);
        CHECK(a.y == b.y);
        CHECK(a.z == b.z);
    }
    CHECK(animation::make_flights({}, obstacles).empty());
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
    // Run again and again: which thread reaches which job first varies from
    // run to run, the job reported must not.
    for (int run = 0; run < 5; ++run) {
        try {
            (void)animation::make_flights(jobs, obstacles);
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
        (void)animation::make_flights(jobs, obstacles);
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
    CHECK_THROWS_AS(animation::make_flight(fast.params, fast.start, fast.body, obstacles), animation::Refusal);
    // The bound is the params': a small one refuses the first drift, the
    // default makes the flight.
    animation::FlightJob bounded = job(3, clear_start);
    CHECK_NOTHROW((void)animation::make_flight(bounded.params, bounded.start, bounded.body, obstacles));
    bounded.params.most_steps = 2;
    CHECK_THROWS_WITH_AS(animation::make_flight(bounded.params, bounded.start, bounded.body, obstacles),
                         doctest::Contains("is not clear"), animation::Refusal);
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
        (void)animation::make_flights(jobs, obstacles);
        FAIL("expected an error");
    } catch (const animation::FlightsError& error) {
        CHECK(error.job == 1);
        CHECK(error.reason.starts_with("the flight's start"));
        CHECK(std::string(error.what()) == "flight 1: " + error.reason);
    }
    const Arguing arguing;
    try {
        (void)animation::make_flights({job(1, clear_start)}, arguing);
        FAIL("expected an error");
    } catch (const animation::Refusal&) {
        FAIL("an Obstacles' own exception was taken for a refusal");
    } catch (const std::invalid_argument& error) {
        CHECK(std::string(error.what()) == "the obstacles argued");
    }
}
