// The flight (core/animation/flight.h) and glows (core/animation/glow.h):
// a loop made at load that keeps its clearance for all time, smooth through
// every join, a function of its seed; flashes at least a second apart; and
// the pulse of each glow kind. Flights are made through the scene reader, so
// the obstacles are the scene's own answer (contract 11) and every check is
// against the shapes' exact distances.

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include <doctest/doctest.h>

#include "core/animation/glow.h"
#include "core/scene/scene.h"

using namespace serenity;
using frame::Seconds;

namespace {

// A sphere to circle over a floor, and one firefly flying among them.
std::string flying(const std::string& motion) {
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
radiance = [10, 10, 10]
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
[[shapes]]
kind = "sphere"
center = [1.2, 1.0, 0.8]
radius = 0.03
material = "glow"
motion = )") + motion + "\n";
}

const std::string normal = "{ kind = \"flight\", min = [-2, 0.05, -2], max = [2, 2, 2], targets = [\"ball\"], "
                           "speed = 0.45, clearance = 0.06, circle = 3, swoop = 1, drift = 1, seed = 5 }";

double distance_to_still(const scene::SceneDescription& s, contracts::Float3 p) {
    return std::min(shapes::distance(s.shapes, 0, p), shapes::distance(s.shapes, 1, p));
}

}  // namespace

TEST_CASE("a flight keeps its clearance and its volume for all time, smooth through every join") {
    const scene::SceneDescription s = scene::parse(flying(normal), "flight");
    REQUIRE(s.animation.motions.flights.size() == 1);
    const animation::Flight& f = s.animation.motions.flights[0];
    CHECK(f.loop > 60.0);

    // Every millisecond of the loop, and across its end into the next.
    const double body = 0.03, clearance = 0.06;
    double nearest = std::numeric_limits<double>::infinity();
    double widest_step = 0.0;
    int outside = 0;
    contracts::Float3 last = animation::position(f, Seconds(0.0));
    const auto steps = static_cast<long>(std::ceil((f.loop + 1.0) / 0.001));
    for (long i = 1; i <= steps; ++i) {
        const contracts::Float3 p = animation::position(f, Seconds(i * 0.001));
        nearest = std::min(nearest, distance_to_still(s, p) - body);
        outside += p.x - body < -2.0f || p.x + body > 2.0f || p.y - body < 0.05f || p.y + body > 2.0f ||
                   p.z - body < -2.0f || p.z + body > 2.0f;
        const double dx = p.x - last.x, dy = p.y - last.y, dz = p.z - last.z;
        widest_step = std::max(widest_step, std::sqrt(dx * dx + dy * dy + dz * dz));
        last = p;
    }
    INFO("nearest surface " << nearest << " m, widest step " << widest_step << " m in a millisecond");
    CHECK(nearest >= clearance);
    CHECK(outside == 0);
    // No jump at any join or at the loop's end: a millisecond's travel stays
    // under 5 mm, a speed under 5 m/s.
    CHECK(widest_step < 0.005);
}

TEST_CASE("a flight circles its target, swoops and drifts, and repeats its loop exactly") {
    const scene::SceneDescription s = scene::parse(flying(normal), "flight");
    const animation::Flight& f = s.animation.motions.flights[0];
    int circles = 0, swoops = 0, drifts = 0;
    for (const animation::Segment& seg : f.segments) {
        circles += seg.behaviour == animation::Behaviour::circle;
        swoops += seg.behaviour == animation::Behaviour::swoop;
        drifts += seg.behaviour == animation::Behaviour::drift;
        if (seg.behaviour == animation::Behaviour::circle) {
            // About the ball: its center within the ball's radius above the
            // ball's center, on the ball's axis.
            CHECK(std::abs(seg.numbers[0]) < 1e-9);
            CHECK(std::abs(seg.numbers[2]) < 1e-9);
        }
    }
    CHECK(circles > 0);
    CHECK(swoops > 0);
    CHECK(drifts > 0);
    for (double t : {0.0, 3.3, 41.7}) {
        const contracts::Float3 a = animation::position(f, Seconds(t));
        const contracts::Float3 b = animation::position(f, Seconds(t + f.loop));
        CHECK(a.x == doctest::Approx(b.x).epsilon(1e-5));
        CHECK(a.y == doctest::Approx(b.y).epsilon(1e-5));
    }
    // Its start: the shape's center in the file, where its first drift is.
    const contracts::Float3 start = animation::position(f, Seconds(0.0));
    CHECK(std::abs(start.x - 1.2f) < 0.11f);
    CHECK(std::abs(start.y - 1.0f) < 0.11f);
}

TEST_CASE("a flight is a function of its seed") {
    const scene::SceneDescription a = scene::parse(flying(normal), "a");
    const scene::SceneDescription b = scene::parse(flying(normal), "b");
    std::string other = normal;
    other.replace(other.find("seed = 5"), 8, "seed = 6");
    const scene::SceneDescription c = scene::parse(flying(other), "c");
    const animation::Flight& fa = a.animation.motions.flights[0];
    CHECK(fa.loop == b.animation.motions.flights[0].loop);
    const contracts::Float3 pa = animation::position(fa, Seconds(20.0));
    const contracts::Float3 pb = animation::position(b.animation.motions.flights[0], Seconds(20.0));
    const contracts::Float3 pc = animation::position(c.animation.motions.flights[0], Seconds(20.0));
    CHECK(pa.x == pb.x);
    CHECK(pa.z == pb.z);
    CHECK(pa.x != pc.x);
}

TEST_CASE("a flight's flashes: in order, within its loop, a second apart, one on each swoop's climb") {
    const scene::SceneDescription s = scene::parse(flying(normal), "flight");
    const animation::Flight& f = s.animation.motions.flights[0];
    const std::vector<double>& starts = f.flashes.starts;
    REQUIRE(starts.size() > 2);
    CHECK(f.flashes.loop == f.loop);
    for (std::size_t i = 0; i < starts.size(); ++i) {
        CHECK(starts[i] >= 0.0);
        CHECK(starts[i] < f.loop);
        if (i > 0) {
            CHECK(starts[i] - starts[i - 1] >= 1.0);
        }
    }
    CHECK(starts.front() + f.loop - starts.back() >= 1.0);
    // Every swoop's climb flashes, unless a flash a second before it took
    // its place.
    int swoops = 0, flashed = 0;
    for (const animation::Segment& seg : f.segments) {
        if (seg.behaviour != animation::Behaviour::swoop) {
            continue;
        }
        ++swoops;
        const double climb = seg.start + 0.55 * seg.duration;
        flashed += std::any_of(starts.begin(), starts.end(), [&](double t) { return std::abs(t - climb) < 1.0; });
    }
    CHECK(flashed == swoops);
}

TEST_CASE("a flight that cannot be made clear is refused, naming its line") {
    const auto error_of = [](const std::string& text) {
        try {
            (void)scene::parse(text, "s.toml");
        } catch (const scene::Error& e) {
            return std::string(e.what());
        }
        return std::string();
    };
    // A start inside the ball.
    std::string inside = flying(normal);
    inside.replace(inside.find("center = [1.2, 1.0, 0.8]"), 24, "center = [0.1, 0.5, 0.0]");
    CHECK(error_of(inside).find("is not clear") != std::string::npos);
    // A volume too tight to circle the ball in.
    CHECK(error_of(flying("{ kind = \"flight\", min = [-0.7, 0.05, -0.7], max = [1.4, 1.3, 1.0], targets = [\"ball\"], "
                          "speed = 0.45, clearance = 0.06, circle = 1, swoop = 0, drift = 0, seed = 5 }"))
              .find("could not be drawn clear") != std::string::npos);
    // A target that is not a still sphere, or not defined.
    CHECK(error_of(flying("{ kind = \"flight\", min = [-2, 0.05, -2], max = [2, 2, 2], targets = [\"wall\"], "
                          "speed = 0.45, clearance = 0.06, circle = 1, swoop = 1, drift = 1, seed = 5 }"))
              .find("circles 'wall', which is not defined") != std::string::npos);
    CHECK(error_of(flying("{ kind = \"flight\", min = [-2, 0.05, -2], max = [2, 2, 2], targets = [], "
                          "speed = 0.45, clearance = 0.06, circle = 1, swoop = 1, drift = 1, seed = 5 }"))
              .find("circles, and names no targets") != std::string::npos);
    CHECK(error_of(flying("{ kind = \"flight\", min = [-2, 0.05, -2], max = [2, 2, 2], targets = [\"ball\"], "
                          "speed = 0.45, clearance = 0.06, circle = 0, swoop = 0, drift = 0, seed = 5 }"))
              .find("all 0") != std::string::npos);
    CHECK(error_of(flying("{ kind = \"flight\", min = [2, 0.05, -2], max = [-2, 2, 2], targets = [\"ball\"], "
                          "speed = 0.45, clearance = 0.06, circle = 1, swoop = 1, drift = 1, seed = 5 }"))
              .find("min must be below its max") != std::string::npos);
}

TEST_CASE("a rhythm's flashes: dim between, 1 at each peak, one a period") {
    animation::Glows glows;
    glows.rhythms.push_back({5.0, 0.4, 0.1f, 9});
    const animation::GlowRecord r{animation::GlowKind::rhythm, 0};
    // Over 200 periods, the brightness at each millisecond: it reaches 1
    // once a period, and is dim most of the time.
    int peaks = 0;
    int lit = 0;
    float lowest = 1.0f;
    float previous = animation::glow(glows, r, Seconds(0.0));
    bool rising = false;
    for (int i = 1; i < 1000000; ++i) {
        const float g = animation::glow(glows, r, Seconds(i * 0.001));
        lowest = std::min(lowest, g);
        lit += g > 0.1f;
        if (g < previous && rising && previous > 0.99f) {
            ++peaks;
        }
        rising = g > previous;
        previous = g;
    }
    CHECK(lowest == doctest::Approx(0.1f));
    CHECK(peaks >= 199);
    CHECK(peaks <= 201);
    // Lit for a flash's length, 0.4 s, of every 5: 8% of the time.
    CHECK(lit / 1000000.0 == doctest::Approx(0.08).epsilon(0.05));
}

TEST_CASE("a schedule's flashes: lit at each start, across the loop's end too") {
    animation::Glows glows;
    animation::ScheduleGlow glow;
    glow.schedule = {10.0, {1.0, 6.0, 9.8}};
    glow.flash = 0.5;
    glow.dim = 0.0f;
    glows.schedules.push_back(glow);
    const animation::GlowRecord r{animation::GlowKind::schedule, 0};
    CHECK(animation::glow(glows, r, Seconds(1.25)) == doctest::Approx(1.0f));
    CHECK(animation::glow(glows, r, Seconds(3.0)) == 0.0f);
    CHECK(animation::glow(glows, r, Seconds(16.25)) == doctest::Approx(1.0f));
    // The flash at 9.8 runs on past the loop's end, into the next loop's 0.3.
    CHECK(animation::glow(glows, r, Seconds(10.05)) == doctest::Approx(1.0f));
    CHECK(animation::glow(glows, r, Seconds(0.05)) > 0.9f);
    CHECK(animation::glow(glows, r, Seconds(0.4)) == 0.0f);
}
