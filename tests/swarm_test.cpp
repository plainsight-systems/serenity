// Swarms (core/scene/swarm.h): many fireflies from one entry, each what a
// written firefly is, its seed and start a function of the swarm's seed and
// its number; and flights made in parallel (core/animation/flight.h,
// make_flights), the same flights and the same error whatever the threads.

#include <cmath>
#include <stdexcept>
#include <limits>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "core/animation/flight.h"
#include "core/scene/scene.h"
#include "core/scene/swarm.h"
#include "core/shapes/shapes.h"

using namespace serenity;
using frame::Seconds;

namespace {

// A ball on a floor, and a swarm about it.
std::string swarmed(const std::string& swarm) {
    return std::string(R"(
[camera]
position = [0, 1.5, 4]
look_at = [0, 0.5, 0]
vertical_fov_degrees = 40
[environment]
kind = "gradient"
zenith = [0, 0, 0]
horizon = [0, 0, 0]
[materials.matte]
kind = "rough"
color = [0.5, 0.5, 0.5]
[materials.glow]
kind = "emissive"
radiance = [10, 8, 2]
[[shapes]]
kind = "box"
min = [-10, -0.1, -10]
max = [10, 0, 10]
material = "matte"
[[shapes]]
kind = "sphere"
name = "ball"
center = [0, 0.5, 0]
radius = 0.5
material = "matte"
[[swarms]]
)") + swarm + "\n";
}

std::string swarm_of(int count, int seed = 4, const std::string& extra = "") {
    return "count = " + std::to_string(count) +
           "\nradius = 0.02\nmaterial = \"glow\"\nmin = [-2, 0.05, -2]\nmax = [2, 2.2, 2]\n"
           "targets = [\"ball\"]\nspeed = 0.4\nclearance = 0.05\ncircle = 2\nswoop = 1\ndrift = 2\n"
           "flash = 0.35\ndim = 0.25\nseed = " +
           std::to_string(seed) + "\n" + extra;
}

std::string error_of(const std::string& text) {
    try {
        (void)scene::parse(text, "s.toml");
    } catch (const scene::Error& error) {
        return error.what();
    }
    return "";
}

bool contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

double still_distance(const scene::SceneDescription& s, contracts::Float3 p) {
    return std::min(shapes::distance(s.shapes, 0, p), shapes::distance(s.shapes, 1, p));
}

}  // namespace

TEST_CASE("a swarm becomes its count of fireflies, each a moving, flashing sphere light") {
    const scene::SceneDescription s = scene::parse(swarmed(swarm_of(24)), "s");
    REQUIRE(s.shapes.records.size() == 2 + 24);
    CHECK(s.sphere_lights.size() == 24);
    REQUIRE(s.animation.movers.size() == 24);
    REQUIRE(s.animation.glowers.size() == 24);
    for (std::uint32_t i = 0; i < 24; ++i) {
        // After the written shapes, in order; each a light, its own flight
        // and a glow of the schedule kind on that flight's flashes.
        CHECK(s.animation.movers[i].target == 2 + i);
        CHECK(s.animation.movers[i].motion.kind == animation::MotionKind::flight);
        CHECK(s.shape_lights[2 + i] == i);
        CHECK(s.animation.glowers[i].target == i);
        const animation::ScheduleGlow& glow = s.animation.glows.schedules[s.animation.glowers[i].glow.index];
        CHECK(glow.flash == doctest::Approx(0.35));
        CHECK(glow.dim == doctest::Approx(0.25f));
        CHECK(glow.schedule.starts == s.animation.motions.flights[i].flashes.starts);
        CHECK(s.shapes.transforms[2 + i].m[0][0] == doctest::Approx(0.02f));
    }
}

TEST_CASE("each firefly starts clear, with room for its first drift, and its flight keeps clear for all time") {
    const scene::SceneDescription s = scene::parse(swarmed(swarm_of(24)), "s");
    const double needed = 0.05 + 0.02 + animation::flight_delta + animation::first_drift_reach * std::sqrt(3.0);
    const double margin = 0.02 + animation::flight_delta + animation::first_drift_reach;
    for (std::uint32_t i = 0; i < 24; ++i) {
        const contracts::Float3 start = contracts::translation(s.shapes.transforms[2 + i]);
        CHECK(still_distance(s, start) >= needed);
        CHECK(start.x >= -2.0 + margin);
        CHECK(start.y <= 2.2 - margin);
        // The flight, sampled every 10 ms of its loop.
        const animation::Flight& f = s.animation.motions.flights[i];
        double nearest = std::numeric_limits<double>::infinity();
        for (double t = 0.0; t < f.loop; t += 0.01) {
            nearest = std::min(nearest, still_distance(s, animation::position(f, Seconds(t))) - 0.02);
        }
        CHECK(nearest >= 0.05);
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
    CHECK(a.shapes.transforms[2].m[0][3] != c.shapes.transforms[2].m[0][3]);
    CHECK(scene::firefly_seed(4, 0) != scene::firefly_seed(4, 1));
    CHECK(scene::firefly_seed(4, 0) != scene::firefly_seed(5, 0));
}

TEST_CASE("every mistake in a swarm is refused, naming the file and the line") {
    CHECK(contains(error_of(swarmed(swarm_of(0))), "s.toml:28: swarm 1's count must be an integer from 1 to 4096"));
    CHECK(contains(error_of(swarmed(swarm_of(4097))), "count must be an integer from 1 to 4096"));
    std::string matte = swarm_of(4);
    matte.replace(matte.find("\"glow\""), 6, "\"matte\"");
    CHECK(contains(error_of(swarmed(matte)), "swarm 1's material must be emissive"));
    CHECK(contains(error_of(swarmed(swarm_of(4, 4, "colour = 1\n"))), "unknown key 'colour' in swarm 1"));
    std::string untargeted = swarm_of(4);
    untargeted.replace(untargeted.find("\"ball\""), 6, "\"cup\"");
    CHECK(contains(error_of(swarmed(untargeted)), "swarm 1 circles 'cup', which is not defined"));
    std::string late = swarm_of(4);
    late.replace(late.find("flash = 0.35"), 12, "flash = 1.5");
    CHECK(contains(error_of(swarmed(late)), "swarm 1's flash must be under a second"));
    std::string far = swarm_of(4);
    far.replace(far.find("max = [2, 2.2, 2]"), 17, "max = [2, 2.2, 1000000]");
    CHECK(contains(error_of(swarmed(far)), "swarm 1's volume, grown by its radius, must lie within 1000 km"));
    // A volume the ball fills: no start is clear.
    std::string full = swarm_of(4);
    const std::string volume = "min = [-2, 0.05, -2]\nmax = [2, 2.2, 2]";
    full.replace(full.find(volume), volume.size(), "min = [-0.3, 0.3, -0.3]\nmax = [0.3, 0.7, 0.3]");
    const std::string no_room = error_of(swarmed(full));
    INFO(no_room);
    CHECK(contains(no_room, "s.toml:27: swarm 1: firefly 0: no start clear of the still shapes in 64 draws"));
    CHECK(contains(error_of(swarmed("count = 4\n[[swarms]]\n" + swarm_of(4))), "swarm 1 has no 'radius'"));
    CHECK(contains(error_of(swarmed(swarm_of(4)) + "\n[[swarms]]\nfoo = 1\n"), "unknown key 'foo' in swarm 2"));
}

namespace {

// The ball and floor as contract 11 sees them.
struct BallAndFloor final : contracts::Obstacles {
    explicit BallAndFloor(const scene::SceneDescription& s) : scene(s) {}
    double distance(contracts::Float3 p) const override { return still_distance(scene, p); }
    bool touches(const contracts::Box& box) const override {
        return shapes::touches(scene.shapes, 0, {box.min, box.max}) ||
               shapes::touches(scene.shapes, 1, {box.min, box.max});
    }
    const scene::SceneDescription& scene;
};

animation::FlightJob job(std::uint64_t seed, contracts::Float3 start) {
    animation::FlightJob j;
    j.params.volume = {{-2.0f, 0.05f, -2.0f}, {2.0f, 2.2f, 2.0f}};
    j.params.targets = {{{0.0f, 0.5f, 0.0f}, 0.5f}};
    j.params.speed = 0.4f;
    j.params.clearance = 0.05f;
    j.params.weights = {2.0f, 1.0f, 2.0f};
    j.params.seed = seed;
    j.start = start;
    j.body = 0.02f;
    return j;
}

}  // namespace

TEST_CASE("flights made in parallel are the flights made one by one, in order") {
    const scene::SceneDescription s = scene::parse(swarmed(swarm_of(1)), "s");
    const BallAndFloor obstacles(s);
    std::vector<animation::FlightJob> jobs;
    for (std::uint64_t k = 0; k < 40; ++k) {
        jobs.push_back(job(100 + k, {1.2f, 1.0f + 0.01f * static_cast<float>(k), 0.8f}));
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
    const BallAndFloor obstacles(s);
    std::vector<animation::FlightJob> jobs;
    for (std::uint64_t k = 0; k < 30; ++k) {
        jobs.push_back(job(k, {1.2f, 1.0f, 0.8f}));
    }
    // Starts inside the ball: 7 and 19 cannot be made.
    jobs[19].start = {0.0f, 0.5f, 0.0f};
    jobs[7].start = {0.1f, 0.5f, 0.0f};
    for (int run = 0; run < 5; ++run) {
        try {
            (void)animation::make_flights(jobs, obstacles);
            FAIL("expected an error");
        } catch (const animation::FlightsError& error) {
            CHECK(error.job == 7);
            CHECK(std::string(error.what()).rfind("flight 7: ", 0) == 0);
        }
    }
}

namespace {

// The ball and floor, but an answer near one start throws what no flight
// expects: a failure that is not a refusal.
struct Breaking final : contracts::Obstacles {
    explicit Breaking(const scene::SceneDescription& s) : inner(s) {}
    double distance(contracts::Float3 p) const override {
        if (p.z > 1.7f) {
            throw std::runtime_error("the obstacles broke");
        }
        return inner.distance(p);
    }
    bool touches(const contracts::Box& box) const override { return inner.touches(box); }
    BallAndFloor inner;
};

}  // namespace

TEST_CASE("of flights made in parallel, any exception reaches the caller, the lowest job's, after every thread ends") {
    const scene::SceneDescription s = scene::parse(swarmed(swarm_of(1)), "s");
    const Breaking obstacles(s);
    std::vector<animation::FlightJob> jobs;
    for (std::uint64_t k = 0; k < 24; ++k) {
        jobs.push_back(job(k, {1.2f, 1.0f, 0.8f}));
    }
    jobs[5].start = {1.2f, 1.0f, 1.8f};  // breaks
    jobs[9].start = {0.0f, 0.5f, 0.0f};  // refused: inside the ball
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
    const BallAndFloor obstacles(s);
    animation::FlightJob fast = job(3, {1.2f, 1.0f, 0.8f});
    fast.params.speed = 1e30f;  // within float's range, as the reader accepts
    CHECK_THROWS_AS(animation::make_flight(fast.params, fast.start, fast.body, obstacles), animation::Refusal);
    // The bound is the params': a small one refuses the first drift, the
    // default makes the flight.
    animation::FlightJob bounded = job(3, {1.2f, 1.0f, 0.8f});
    CHECK_NOTHROW((void)animation::make_flight(bounded.params, bounded.start, bounded.body, obstacles));
    bounded.params.most_steps = 2;
    CHECK_THROWS_WITH_AS(animation::make_flight(bounded.params, bounded.start, bounded.body, obstacles),
                         doctest::Contains("is not clear"), animation::Refusal);
}

namespace {

// Contract 11 answered with an exception of the standard type a refusal
// derives from: not a refusal of any flight.
struct Arguing final : contracts::Obstacles {
    double distance(contracts::Float3) const override { throw std::invalid_argument("the obstacles argued"); }
    bool touches(const contracts::Box&) const override { return false; }
};

}  // namespace

TEST_CASE("a refusal carries its job and its reason; another std::invalid_argument is not taken for one") {
    const scene::SceneDescription s = scene::parse(swarmed(swarm_of(1)), "s");
    const BallAndFloor obstacles(s);
    std::vector<animation::FlightJob> jobs = {job(1, {1.2f, 1.0f, 0.8f}), job(2, {0.0f, 0.5f, 0.0f})};
    try {
        (void)animation::make_flights(jobs, obstacles);
        FAIL("expected an error");
    } catch (const animation::FlightsError& error) {
        CHECK(error.job == 1);
        CHECK(error.reason.rfind("the flight's start", 0) == 0);
        CHECK(std::string(error.what()) == "flight 1: " + error.reason);
    }
    const Arguing arguing;
    try {
        (void)animation::make_flights({job(1, {1.2f, 1.0f, 0.8f})}, arguing);
        FAIL("expected an error");
    } catch (const animation::Refusal&) {
        FAIL("an Obstacles' own exception was taken for a refusal");
    } catch (const std::invalid_argument& error) {
        CHECK(std::string(error.what()) == "the obstacles argued");
    }
}
