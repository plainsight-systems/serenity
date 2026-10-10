// The flight (core/animation/flight.h) and glows (core/animation/glow.h):
// a loop made at load that keeps its clearance for all time, smooth through
// every join, a function of its seed; flashes at least a second apart; and
// the pulse of each glow kind. Flights are made through the scene reader, so
// the obstacles are the scene's own answer (contract 11) and every check is
// against the shapes' exact distances.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "core/animation/glow.h"
#include "core/scene/scene.h"
#include "support/ball_on_floor.h"
#include "support/text.h"

using namespace serenity;
using frame::Seconds;
using tests::contains;

namespace {

// A flight's numbers as a scene file writes them.
struct FlightText {
    std::string min = "[-2, 0.05, -2]";
    std::string max = "[2, 2, 2]";
    std::string targets = "[\"ball\"]";
    std::string circle = "3";
    std::string swoop = "1";
    std::string drift = "1";
    std::string seed = "5";
};

// The firefly's body and clearance, and its volume's bounds, as written
// below.
constexpr double body = 0.03;
constexpr double clearance = 0.06;
constexpr float volume_low = 0.05f;   // the volume's floor
constexpr float volume_reach = 2.0f;  // its other bounds, either way

std::string motion(const FlightText& f) {
    return "{ kind = \"flight\", min = " + f.min + ", max = " + f.max + ", targets = " + f.targets +
           ", speed = 0.45, clearance = 0.06, circle = " + f.circle + ", swoop = " + f.swoop +
           ", drift = " + f.drift + ", seed = " + f.seed + " }";
}

// The ball on its floor, and one firefly flying among them.
std::string flying(const FlightText& f = {}) {
    return tests::ball_on_floor("[10, 10, 10]") +
           "[[shapes]]\nkind = \"sphere\"\ncenter = [1.2, 1.0, 0.8]\nradius = 0.03\nmaterial = \"glow\"\nmotion = " +
           motion(f) + "\n";
}

const animation::Flight& the_flight(const scene::SceneDescription& s) {
    REQUIRE(s.animation.motions.flights.size() == 1);
    return s.animation.motions.flights[0];
}

}  // namespace

TEST_CASE("a flight keeps its clearance and its volume for all time, smooth through every join") {
    const scene::SceneDescription s = scene::parse(flying(), "flight");
    const animation::Flight& f = the_flight(s);
    CHECK(f.loop > 60.0);

    // Every millisecond of the loop, and across its end into the next.
    double nearest = std::numeric_limits<double>::infinity();
    double widest_step = 0.0;
    int outside = 0;
    contracts::Float3 last = animation::position(f, Seconds(0.0));
    const auto steps = static_cast<long>(std::ceil((f.loop + 1.0) / 0.001));
    for (long i = 1; i <= steps; ++i) {
        const contracts::Float3 p = animation::position(f, Seconds(static_cast<double>(i) * 0.001));
        nearest = std::min(nearest, tests::distance_to_still(s, p) - body);
        const auto beyond = [](float v, float low, float high) { return v - body < low || v + body > high; };
        outside += beyond(p.x, -volume_reach, volume_reach) || beyond(p.y, volume_low, volume_reach) ||
                           beyond(p.z, -volume_reach, volume_reach)
                       ? 1
                       : 0;
        const double dx = p.x - last.x;
        const double dy = p.y - last.y;
        const double dz = p.z - last.z;
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
    const scene::SceneDescription s = scene::parse(flying(), "flight");
    const animation::Flight& f = the_flight(s);
    int circles = 0;
    int swoops = 0;
    int drifts = 0;
    for (const animation::Segment& seg : f.segments) {
        circles += seg.behaviour == animation::Behaviour::circle ? 1 : 0;
        swoops += seg.behaviour == animation::Behaviour::swoop ? 1 : 0;
        drifts += seg.behaviour == animation::Behaviour::drift ? 1 : 0;
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
    for (const double t : {0.0, 3.3, 41.7}) {
        const contracts::Float3 a = animation::position(f, Seconds(t));
        const contracts::Float3 b = animation::position(f, Seconds(t + f.loop));
        // Positions in meters, within 1e-5 (1 + |x|) m: a coordinate near 0
        // is held to 10 um, not to a part in 10^5 of itself (scale 1).
        CHECK(a.x == doctest::Approx(b.x).scale(1.0).epsilon(1e-5));
        CHECK(a.y == doctest::Approx(b.y).scale(1.0).epsilon(1e-5));
        CHECK(a.z == doctest::Approx(b.z).scale(1.0).epsilon(1e-5));
    }
    // Its start: the shape's center in the file, where its first drift is.
    const contracts::Float3 start = animation::position(f, Seconds(0.0));
    CHECK(std::abs(start.x - 1.2f) < 0.11f);
    CHECK(std::abs(start.y - 1.0f) < 0.11f);
}

TEST_CASE("a flight is a function of its seed") {
    const scene::SceneDescription a = scene::parse(flying(), "a");
    const scene::SceneDescription b = scene::parse(flying(), "b");
    FlightText reseeded;
    reseeded.seed = "6";
    const scene::SceneDescription c = scene::parse(flying(reseeded), "c");
    CHECK(the_flight(a).loop == the_flight(b).loop);
    const contracts::Float3 pa = animation::position(the_flight(a), Seconds(20.0));
    const contracts::Float3 pb = animation::position(the_flight(b), Seconds(20.0));
    const contracts::Float3 pc = animation::position(the_flight(c), Seconds(20.0));
    CHECK(pa.x == pb.x);
    CHECK(pa.y == pb.y);
    CHECK(pa.z == pb.z);
    CHECK(pa.x != pc.x);
}

TEST_CASE("a flight's flashes: in order, within its loop, a second apart, one on each swoop's climb") {
    // Where on a swoop its flash is, a fraction of its duration: on its
    // climb (flight.h). The number is flight.cpp's own, swoop_flash_at,
    // which flight.h does not name.
    constexpr double swoop_flash_at = 0.55;
    const scene::SceneDescription s = scene::parse(flying(), "flight");
    const animation::Flight& f = the_flight(s);
    const std::vector<double>& starts = f.flashes.starts;
    REQUIRE(starts.size() > 2);
    CHECK(f.flashes.loop == f.loop);
    CHECK(std::ranges::all_of(starts, [&](double t) { return t >= 0.0 && t < f.loop; }));
    CHECK(std::ranges::adjacent_find(starts, [](double a, double b) { return b - a < 1.0; }) == starts.end());
    CHECK(starts.front() + f.loop - starts.back() >= 1.0);
    // Every swoop's climb flashes, unless a flash a second before it took
    // its place.
    int swoops = 0;
    int flashed = 0;
    for (const animation::Segment& seg : f.segments) {
        if (seg.behaviour != animation::Behaviour::swoop) {
            continue;
        }
        ++swoops;
        const double climb = seg.start + swoop_flash_at * seg.duration;
        flashed += std::ranges::any_of(starts, [&](double t) { return std::abs(t - climb) < 1.0; }) ? 1 : 0;
    }
    CHECK(flashed == swoops);
}

TEST_CASE("a flight that cannot be made clear is refused, naming its line") {
    // The firefly's motion is on line 31.
    const auto error_of = [](const std::string& text) {
        return tests::error_of<scene::Error>([&] { return scene::parse(text, "s.toml"); });
    };
    // A start inside the ball.
    const std::string inside = tests::replaced(flying(), "center = [1.2, 1.0, 0.8]", "center = [0.1, 0.5, 0.0]");
    CHECK(contains(error_of(inside), "s.toml:"));
    CHECK(contains(error_of(inside), "is not clear"));
    // A volume too tight to circle the ball in.
    CHECK(contains(error_of(flying({.min = "[-0.7, 0.05, -0.7]",
                                    .max = "[1.4, 1.3, 1.0]",
                                    .circle = "1",
                                    .swoop = "0",
                                    .drift = "0"})),
                   "could not be drawn clear"));
    // A target that is not a still sphere, or not defined.
    CHECK(contains(error_of(flying({.targets = "[\"wall\"]", .circle = "1"})),
                   "s.toml:31: shape 3's motion circles 'wall', which is not defined"));
    CHECK(contains(error_of(flying({.targets = "[]", .circle = "1"})), "circles, and names no targets"));
    CHECK(contains(error_of(flying({.circle = "0", .swoop = "0", .drift = "0"})), "all 0"));
    CHECK(contains(error_of(flying({.min = "[2, 0.05, -2]", .max = "[-2, 2, 2]", .circle = "1"})),
                   "min must be below its max"));
}

namespace {

// Over `periods` periods of `period` seconds, the brightness at each
// millisecond of a glow.
std::vector<float> every_millisecond(const animation::Glows& glows, animation::GlowRecord r, double seconds) {
    const auto steps = static_cast<std::size_t>(seconds * 1000.0);
    std::vector<float> g;
    g.reserve(steps);
    for (std::size_t i = 0; i < steps; ++i) {
        g.push_back(animation::glow(glows, r, Seconds(static_cast<double>(i) * 0.001)));
    }
    return g;
}

}  // namespace

TEST_CASE("a rhythm's flashes: dim between, 1 at each peak, one a period") {
    constexpr double period = 5.0;
    constexpr double flash = 0.4;
    constexpr float dim = 0.1f;
    constexpr int periods = 200;
    animation::Glows glows;
    glows.rhythms.push_back({period, flash, dim, 9});
    const animation::GlowRecord r{animation::GlowKind::rhythm, 0};
    // Over 200 periods, the brightness at each millisecond: it reaches 1
    // once a period, and is dim most of the time.
    const std::vector<float> g = every_millisecond(glows, r, periods * period);
    int peaks = 0;
    for (std::size_t i = 2; i < g.size(); ++i) {
        // A peak: risen to it, falling after it, at 1.
        peaks += g[i - 1] > g[i - 2] && g[i] < g[i - 1] && g[i - 1] > 0.99f ? 1 : 0;
    }
    CHECK(*std::ranges::min_element(g) == doctest::Approx(dim).scale(0).epsilon(1e-6));
    CHECK(peaks >= periods - 1);
    CHECK(peaks <= periods + 1);
    // Lit for a flash's length, 0.4 s, of every 5: 8% of the time.
    const auto lit = std::ranges::count_if(g, [](float v) { return v > dim; });
    CHECK(static_cast<double>(lit) / static_cast<double>(g.size()) ==
          doctest::Approx(flash / period).scale(0).epsilon(0.01));
}

TEST_CASE("a schedule's flashes: lit at each start, across the loop's end too") {
    animation::Glows glows;
    animation::ScheduleGlow schedule;
    schedule.schedule = {10.0, {1.0, 6.0, 9.8}};
    schedule.flash = 0.5;
    schedule.dim = 0.0f;
    glows.schedules.push_back(schedule);
    const animation::GlowRecord r{animation::GlowKind::schedule, 0};
    CHECK(animation::glow(glows, r, Seconds(1.25)) == doctest::Approx(1.0f).scale(0).epsilon(1e-6));
    CHECK(animation::glow(glows, r, Seconds(3.0)) == 0.0f);
    CHECK(animation::glow(glows, r, Seconds(16.25)) == doctest::Approx(1.0f).scale(0).epsilon(1e-6));
    // The flash at 9.8 runs on past the loop's end, into the next loop's 0.3.
    CHECK(animation::glow(glows, r, Seconds(10.05)) == doctest::Approx(1.0f).scale(0).epsilon(1e-6));
    CHECK(animation::glow(glows, r, Seconds(0.05)) > 0.9f);
    CHECK(animation::glow(glows, r, Seconds(0.4)) == 0.0f);
}

TEST_CASE("a flight circles a marble on a table: its orbit raised so its bob clears the floor") {
    // A marble 1.6 cm across on a table top at 0.75 m, the volume's floor
    // just above it: an orbit at the marble's height would bob through the
    // table, so every circle would fail without the raise.
    constexpr double firefly_radius = 0.003;
    constexpr double volume_floor = 0.76;
    const std::string text = R"(
[camera]
position = [0, 0.8, 0.5]
look_at = [0, 0.76, 0]
vertical_fov_degrees = 30
[environment]
kind = "gradient"
zenith = [0, 0, 0]
horizon = [0, 0, 0]
[materials.wood]
kind = "rough"
color = [0.1, 0.05, 0.02]
[materials.glass]
kind = "dielectric"
ior = 1.5
[materials.glow]
kind = "emissive"
radiance = [10, 10, 10]
[[shapes]]
kind = "box"
min = [-1, 0.72, -1]
max = [1, 0.75, 1]
material = "wood"
[[shapes]]
kind = "sphere"
name = "marble"
center = [0, 0.758, 0]
radius = 0.008
material = "glass"
[[shapes]]
kind = "sphere"
center = [0.3, 1.0, 0.2]
radius = 0.003
material = "glow"
motion = { kind = "flight", min = [-0.9, 0.76, -0.9], max = [0.9, 2.0, 0.9], targets = ["marble"],)"
                             R"( speed = 0.35, clearance = 0.01, circle = 1, swoop = 0, drift = 0, seed = 3 }
)";
    const scene::SceneDescription s = scene::parse(text, "marble");
    const animation::Flight& f = s.animation.motions.flights.at(0);
    int circles = 0;
    double lowest = std::numeric_limits<double>::infinity();
    for (const animation::Segment& seg : f.segments) {
        if (seg.behaviour != animation::Behaviour::circle) {
            continue;
        }
        ++circles;
        const auto samples = static_cast<int>(seg.duration / 0.005);
        for (int k = 0; k <= samples; ++k) {
            const double tau = k * 0.005;
            lowest = std::min(lowest, static_cast<double>(animation::position(f, Seconds(seg.start + tau)).y));
        }
    }
    CHECK(circles > 10);
    // Its body inside the volume: above the floor by the body and delta.
    CHECK(lowest - firefly_radius >= volume_floor);
}

TEST_CASE("one seed's loop, pinned: the level of determinism flight.h states") {
    // On the development machine's toolchain, libm and flags (flight.h,
    // GDSA.2): a change of any of them that moves the loop fails here.
    const scene::SceneDescription s = scene::parse(flying(), "flight");
    const animation::Flight& f = the_flight(s);
    CHECK(f.loop == 0x1.016e3730c1826p+9);
    CHECK(f.segments.size() == 138);
    CHECK(f.flashes.starts.size() == 63);
    const contracts::Float3 at_20 = animation::position(f, Seconds(20.0));
    CHECK(at_20.x == 0x1.63f23cp-1f);
    CHECK(at_20.y == 0x1.a5bdd8p-1f);
    CHECK(at_20.z == 0x1.89affp-1f);
    const contracts::Float3 at_400 = animation::position(f, Seconds(400.5));
    CHECK(at_400.x == 0x1.ed9c5p-1f);
    CHECK(at_400.y == 0x1.dfdb5cp-1f);
    CHECK(at_400.z == -0x1.2b2918p-1f);
}

TEST_CASE("a flight make_flight did not make is refused, as is a segment of no behaviour") {
    CHECK_THROWS_AS(animation::position(animation::Flight{}, Seconds(1.0)), std::invalid_argument);
    const scene::SceneDescription s = scene::parse(flying(), "flight");
    animation::Flight corrupt = the_flight(s);
    corrupt.segments[0].behaviour = static_cast<animation::Behaviour>(9);
    CHECK_THROWS_AS(animation::position(corrupt, Seconds(0.0)), std::logic_error);
}

TEST_CASE("a glow refuses a time past what its rhythm counts, and a schedule with no loop") {
    animation::Glows glows;
    glows.rhythms.push_back({animation::least_period, animation::least_period / 4.0, 0.1f, 3});
    const animation::GlowRecord rhythm{animation::GlowKind::rhythm, 0};
    CHECK_NOTHROW((void)animation::glow(glows, rhythm, Seconds(86400.0 * 365.0)));  // a year
    CHECK_THROWS_AS((void)animation::glow(glows, rhythm, Seconds(1e17)), std::invalid_argument);
    CHECK_THROWS_AS((void)animation::glow(glows, rhythm, Seconds(std::nan(""))), std::invalid_argument);
    glows.schedules.push_back({});
    const animation::GlowRecord schedule{animation::GlowKind::schedule, 0};
    CHECK_THROWS_AS((void)animation::glow(glows, schedule, Seconds(1.0)), std::invalid_argument);
}
