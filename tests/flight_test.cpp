// The flight (core/animation/flight.h) and glows (core/animation/glow.h):
// a loop made at load that keeps its clearance for all time, smooth through
// every join, a function of its seed; flashes at least a second apart; and
// the pulse of each glow kind. Flights are made through the scene reader, so
// the obstacles are the scene's own answer (contract 11) and every check is
// against the shapes' exact distances.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <optional>
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
    // Where on a swoop its flash is: flight.h's swoop_flash_at.
    using animation::swoop_flash_at;
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
        return tests::error_of<scene::SceneError>([&] { return scene::parse(text, "s.toml"); });
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
    glows.rhythms.push_back({.period = period, .flash = flash, .dim = dim, .seed = 9});
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
    schedule.schedule = {.loop = 10.0, .starts = {1.0, 6.0, 9.8}};
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
    glows.rhythms.push_back(
        {.period = animation::least_period, .flash = animation::least_period / 4.0, .dim = 0.1f, .seed = 3});
    const animation::GlowRecord rhythm{animation::GlowKind::rhythm, 0};
    CHECK_NOTHROW((void)animation::glow(glows, rhythm, Seconds(86400.0 * 365.0)));  // a year
    CHECK_THROWS_AS((void)animation::glow(glows, rhythm, Seconds(1e17)), std::invalid_argument);
    CHECK_THROWS_AS((void)animation::glow(glows, rhythm, Seconds(std::nan(""))), std::invalid_argument);
    glows.schedules.push_back({});
    const animation::GlowRecord schedule{animation::GlowKind::schedule, 0};
    CHECK_THROWS_AS((void)animation::glow(glows, schedule, Seconds(1.0)), std::invalid_argument);
}

// The dark opening (flight.h, preludes; glow.h, wakes; flashes.h, openings).

namespace {

// A glow of the schedule kind with no flashes and `dim`: its glow is dim
// times its wake's factor, so the factor is read off it.
animation::Glows dim_schedule(float dim, std::optional<animation::Wake> wake) {
    animation::Glows glows;
    animation::ScheduleGlow g;
    g.schedule = {.loop = 10.0};
    g.flash = 0.5;
    g.dim = dim;
    g.wake = wake;
    glows.schedules.push_back(g);
    return glows;
}

constexpr animation::GlowRecord first_schedule{animation::GlowKind::schedule, 0};
constexpr animation::GlowRecord first_rhythm{animation::GlowKind::rhythm, 0};

// glow.h's smoothstep, x^2 (3 - 2x).
double smoothstep(double x) {
    return x * x * (3.0 - 2.0 * x);
}

// A schedule glow with no opening as glow.h stated it before openings: one
// loop from 0, repeating for every t, negative too; the pulse of the latest
// start at or before t mod loop, or the loop's last a loop earlier.
float loop_only_glow(const animation::ScheduleGlow& g, double t) {
    const std::vector<double>& starts = g.schedule.starts;
    const double loop = g.schedule.loop;
    double tau = std::fmod(t, loop);
    if (tau < 0.0) {
        tau += loop;
    }
    double start = starts.back() - loop;
    for (const double s : starts) {
        if (s <= tau) {
            start = s;
        }
    }
    double p = 0.0;
    if (tau >= start && tau < start + g.flash) {
        const double s = std::sin(std::numbers::pi * (tau - start) / g.flash);
        p = s * s;
    }
    return static_cast<float>(g.dim + (1.0 - g.dim) * p);
}

}  // namespace

TEST_CASE("a wake: dark before at, a smoothstep over its ramp, its glow after") {
    constexpr float dim = 0.5f;
    constexpr double at = 10.0;
    constexpr double ramp = 2.0;
    const animation::Glows glows = dim_schedule(dim, animation::Wake{.at = at, .ramp = ramp});
    const auto factor = [&](double t) {
        return static_cast<double>(animation::glow(glows, first_schedule, Seconds(t))) / dim;
    };
    // 0 before at, exactly, negative times too.
    for (const double t : {-100.0, 0.0, 5.0, std::nextafter(at, 0.0)}) {
        CAPTURE(t);
        CHECK(factor(t) == 0.0);
    }
    // The smoothstep in the ramp: relative to itself (scale 0), to a float's
    // rounding of the glow.
    for (const double x : {0.1, 0.25, 0.5, 0.75, 0.9}) {
        CAPTURE(x);
        CHECK(factor(at + x * ramp) == doctest::Approx(smoothstep(x)).scale(0).epsilon(1e-6));
    }
    CHECK(factor(at) == 0.0);
    // Its full glow from at + ramp on.
    for (const double t : {at + ramp, at + ramp + 1e-9, 100.0, 1e6}) {
        CAPTURE(t);
        CHECK(animation::glow(glows, first_schedule, Seconds(t)) == dim);
    }
}

TEST_CASE("a wake with a ramp of 0 is a switch at at, with no division") {
    constexpr float dim = 0.25f;
    const animation::Glows glows = dim_schedule(dim, animation::Wake{.at = 3.0, .ramp = 0.0});
    CHECK(animation::glow(glows, first_schedule, Seconds(std::nextafter(3.0, 0.0))) == 0.0f);
    CHECK(animation::glow(glows, first_schedule, Seconds(3.0)) == dim);
    CHECK(animation::glow(glows, first_schedule, Seconds(4.0)) == dim);
}

TEST_CASE("a wake scales a rhythm's flashes as it does its dim") {
    animation::Glows glows;
    glows.rhythms.push_back({.period = 5.0, .flash = 0.4, .dim = 0.1f, .seed = 9});
    animation::Glows woken = glows;
    woken.rhythms[0].wake = animation::Wake{.at = 20.0, .ramp = 4.0};
    int wrong = 0;
    for (int i = -2000; i < 40000; ++i) {
        const double t = static_cast<double>(i) * 0.001;
        const float plain = animation::glow(glows, first_rhythm, Seconds(t));
        const float w = animation::glow(woken, first_rhythm, Seconds(t));
        if (t < 20.0) {
            wrong += w == 0.0f ? 0 : 1;
        } else if (t >= 24.0) {
            wrong += w == plain ? 0 : 1;
        } else {
            const double expected = smoothstep((t - 20.0) / 4.0) * static_cast<double>(plain);
            wrong += std::abs(static_cast<double>(w) - expected) <= 1e-6 * expected ? 0 : 1;
        }
    }
    CHECK(wrong == 0);
}

TEST_CASE("no wake, and no opening: the glow as it was, for every t, negative too") {
    animation::Glows glows;
    animation::ScheduleGlow g;
    g.schedule = {.loop = 10.0, .starts = {1.0, 6.0, 9.8}};
    g.flash = 0.5;
    g.dim = 0.05f;
    glows.schedules.push_back(g);
    // Exactly the old arithmetic, at every millisecond from -25 s to 35 s.
    int differ = 0;
    for (int i = -25000; i < 35000; ++i) {
        const double t = static_cast<double>(i) * 0.001;
        differ += animation::glow(glows, first_schedule, Seconds(t)) == loop_only_glow(g, t) ? 0 : 1;
    }
    CHECK(differ == 0);
    // A wake long past answers what no wake does.
    animation::Glows passed = glows;
    passed.schedules[0].wake = animation::Wake{.at = -1e9, .ramp = 1.0};
    for (int i = -5000; i < 5000; ++i) {
        const double t = static_cast<double>(i) * 0.003;
        differ += animation::glow(passed, first_schedule, Seconds(t)) ==
                          animation::glow(glows, first_schedule, Seconds(t))
                      ? 0
                      : 1;
    }
    CHECK(differ == 0);
}

TEST_CASE("a wake whose at is not finite, or whose ramp is below 0 or not finite, is refused") {
    constexpr double inf = std::numeric_limits<double>::infinity();
    const double nan = std::nan("");
    for (const animation::Wake w : {animation::Wake{.at = nan, .ramp = 1.0}, animation::Wake{.at = inf, .ramp = 1.0},
                                    animation::Wake{.at = -inf, .ramp = 1.0}, animation::Wake{.at = 1.0, .ramp = -1.0},
                                    animation::Wake{.at = 1.0, .ramp = inf}, animation::Wake{.at = 1.0, .ramp = nan}}) {
        CAPTURE(w.at);
        CAPTURE(w.ramp);
        CHECK_THROWS_AS((void)animation::glow(dim_schedule(0.5f, w), first_schedule, Seconds(1.0)),
                        std::invalid_argument);
        animation::Glows rhythm;
        rhythm.rhythms.push_back({.period = 5.0, .flash = 0.4, .dim = 0.1f, .seed = 9, .wake = w});
        CHECK_THROWS_AS((void)animation::glow(rhythm, first_rhythm, Seconds(1.0)), std::invalid_argument);
    }
}

TEST_CASE("a schedule with an opening: its flashes once each, then the loop's from begin") {
    animation::Glows glows;
    animation::ScheduleGlow g;
    // The opening's last, at 9.8, runs past begin, 10; the loop's first is
    // at begin + 1.5, and its last near its end, at begin + 19.8.
    g.schedule = {.begin = 10.0, .opening = {2.0, 6.0, 9.8}, .loop = 20.0, .starts = {1.5, 19.8}};
    g.flash = 0.5;
    g.dim = 0.0f;
    glows.schedules.push_back(g);
    const auto at = [&](double t) { return animation::glow(glows, first_schedule, Seconds(t)); };
    // Nothing before the opening's first.
    CHECK(at(-3.0) == 0.0f);
    CHECK(at(0.25) == 0.0f);
    CHECK(at(1.9) == 0.0f);
    // The opening's flashes, at their peaks.
    CHECK(at(2.25) == doctest::Approx(1.0f).scale(0).epsilon(1e-6));
    CHECK(at(6.25) == doctest::Approx(1.0f).scale(0).epsilon(1e-6));
    CHECK(at(7.0) == 0.0f);
    // Into the first pass, the opening's last still lit, past begin.
    CHECK(at(10.05) == doctest::Approx(1.0f).scale(0).epsilon(1e-6));
    CHECK(at(10.4) == 0.0f);
    // The loop's flashes from begin; its last, into the next pass.
    CHECK(at(11.75) == doctest::Approx(1.0f).scale(0).epsilon(1e-6));
    CHECK(at(30.05) == doctest::Approx(1.0f).scale(0).epsilon(1e-6));
    CHECK(at(30.4) == 0.0f);
    CHECK(at(31.75) == doctest::Approx(1.0f).scale(0).epsilon(1e-6));
    CHECK(at(20.0 * 1000.0 + 11.75) == doctest::Approx(1.0f).scale(0).epsilon(1e-6));
}

TEST_CASE("the first pass of a loop after an opening uses the opening's last, not a phantom of the loop's") {
    animation::Glows glows;
    animation::ScheduleGlow g;
    // A loop whose last, 19.8, a loop before begin would be at 9.8: lit at
    // 10.05 if the loop were taken to run before begin. It does not.
    g.schedule = {.begin = 10.0, .opening = {2.0}, .loop = 20.0, .starts = {0.5, 19.8}};
    g.flash = 0.5;
    g.dim = 0.0f;
    glows.schedules.push_back(g);
    CHECK(animation::glow(glows, first_schedule, Seconds(9.9)) == 0.0f);
    CHECK(animation::glow(glows, first_schedule, Seconds(10.05)) == 0.0f);
    CHECK(animation::glow(glows, first_schedule, Seconds(30.05)) == doctest::Approx(1.0f).scale(0).epsilon(1e-6));
    // With no opening flashes at all, the first pass starts dark too.
    glows.schedules[0].schedule.opening.clear();
    CHECK(animation::glow(glows, first_schedule, Seconds(9.9)) == 0.0f);
    CHECK(animation::glow(glows, first_schedule, Seconds(10.05)) == 0.0f);
    CHECK(animation::glow(glows, first_schedule, Seconds(10.75)) == doctest::Approx(1.0f).scale(0).epsilon(1e-6));
}

TEST_CASE("a schedule whose begin is not finite and 0 or more is refused") {
    animation::Glows glows;
    animation::ScheduleGlow g;
    g.schedule = {.begin = -1.0, .loop = 10.0, .starts = {1.0}};
    g.flash = 0.5;
    glows.schedules.push_back(g);
    CHECK_THROWS_AS((void)animation::glow(glows, first_schedule, Seconds(1.0)), std::invalid_argument);
    glows.schedules[0].schedule.begin = std::numeric_limits<double>::infinity();
    CHECK_THROWS_AS((void)animation::glow(glows, first_schedule, Seconds(1.0)), std::invalid_argument);
    glows.schedules[0].schedule.begin = std::nan("");
    CHECK_THROWS_AS((void)animation::glow(glows, first_schedule, Seconds(1.0)), std::invalid_argument);
}

namespace {

// flying()'s firefly's start, and a start straight above the ball, where a
// firefly perched on its top rises to (flight.h, P2: up, from rest).
constexpr contracts::Float3 flying_start{1.2f, 1.0f, 0.8f};
constexpr contracts::Float3 above_ball{0.0f, 1.5f, 0.0f};

// flying()'s firefly as a job, with `prelude`, from `start`: from
// flying_start, its flight is the scene's, made directly against the ball
// and the floor.
animation::FlightJob job_with(const animation::Prelude& prelude, contracts::Float3 start = flying_start) {
    return {.params = {.volume = {{-volume_reach, volume_low, -volume_reach},
                                  {volume_reach, volume_reach, volume_reach}},
                       .targets = {{{0.0f, 0.5f, 0.0f}, 0.5f}},
                       .speed = 0.45f,
                       .clearance = static_cast<float>(clearance),
                       .weights = {3.0f, 1.0f, 1.0f},
                       .seed = 5},
            .start = start,
            .body = static_cast<float>(body),
            .prelude = prelude};
}

// The ball's top, where a firefly of flying()'s body rests a perch_gap
// above it.
constexpr contracts::Float3 ball_top{0.0f, static_cast<float>(1.0 + body + animation::perch_gap), 0.0f};

bool same(contracts::Float3 a, contracts::Float3 b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

double apart(contracts::Float3 a, contracts::Float3 b) {
    const double dx = static_cast<double>(a.x) - static_cast<double>(b.x);
    const double dy = static_cast<double>(a.y) - static_cast<double>(b.y);
    const double dz = static_cast<double>(a.z) - static_cast<double>(b.z);
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

bool inside(const animation::Extent& box, contracts::Float3 p) {
    return p.x >= box.min.x && p.x <= box.max.x && p.y >= box.min.y && p.y <= box.max.y && p.z >= box.min.z &&
           p.z <= box.max.z;
}

}  // namespace

TEST_CASE("no prelude: the flight it was, the pinned loop's, with no opening and its volume for its reach") {
    const scene::SceneDescription s = scene::parse(flying(), "flight");
    const tests::BallAndFloor obstacles(s);
    const animation::Flight& pinned = the_flight(s);
    const animation::Flight f = animation::make_flight(job_with(animation::NoPrelude{}), obstacles);
    CHECK(f.loop == pinned.loop);
    CHECK(f.loop == 0x1.016e3730c1826p+9);
    REQUIRE(f.segments.size() == pinned.segments.size());
    int differ = 0;
    for (std::size_t k = 0; k < f.segments.size(); ++k) {
        differ += f.segments[k].start == pinned.segments[k].start &&
                          f.segments[k].numbers == pinned.segments[k].numbers
                      ? 0
                      : 1;
    }
    CHECK(differ == 0);
    CHECK(f.flashes.starts == pinned.flashes.starts);
    CHECK(f.opening.empty());
    CHECK(f.begin == 0.0);
    CHECK(f.flashes.begin == 0.0);
    CHECK(f.flashes.opening.empty());
    const animation::Extent reach = animation::extent(f);
    CHECK(same(reach.min, f.volume.min));
    CHECK(same(reach.max, f.volume.max));
    // A hold of no length is no opening either.
    const animation::Flight held = animation::make_flight(job_with(animation::Hold{.until = 0.0}), obstacles);
    CHECK(held.opening.empty());
    CHECK(held.begin == 0.0);
    CHECK(held.loop == f.loop);
    // Before 0, as before preludes: the loop repeats.
    for (const double t : {-0.5, -17.25, -600.0}) {
        CHECK(same(animation::position(f, Seconds(t)), animation::position(f, Seconds(t + f.loop))));
    }
}

TEST_CASE("a hold: still at the loop's first point until `until`, then the flight without one, shifted") {
    constexpr double until = 8.0;
    const scene::SceneDescription s = scene::parse(flying(), "flight");
    const tests::BallAndFloor obstacles(s);
    const animation::Flight plain = animation::make_flight(job_with(animation::NoPrelude{}), obstacles);
    const animation::Flight held = animation::make_flight(job_with(animation::Hold{.until = until}), obstacles);
    REQUIRE(held.opening.size() == 1);
    CHECK(held.opening[0].behaviour == animation::Behaviour::still);
    CHECK(held.begin == until);
    CHECK(held.loop == plain.loop);
    CHECK(held.flashes.starts == plain.flashes.starts);
    CHECK(held.flashes.begin == until);
    const contracts::Float3 first = animation::position(plain, Seconds(0.0));
    // Times a quarter second apart, exact in double, so t + until - until is
    // t, and the shifted flight is the same bit for bit.
    int differ = 0;
    for (int i = -8; i <= 32; ++i) {
        differ += same(animation::position(held, Seconds(static_cast<double>(i) * 0.25)), first) ? 0 : 1;
    }
    for (int i = 0; i < 4 * 700; ++i) {
        const double t = static_cast<double>(i) * 0.25;
        differ += same(animation::position(held, Seconds(t + until)), animation::position(plain, Seconds(t))) ? 0 : 1;
    }
    CHECK(differ == 0);
    // Its reach is its volume: the loop's first point is inside it.
    CHECK(same(animation::extent(held).min, held.volume.min));
    CHECK(same(animation::extent(held).max, held.volume.max));
}

TEST_CASE("a perch: at rest until `until`, then a rise from rest, smooth into the loop, clear of every surface") {
    constexpr double until = 5.0;
    const scene::SceneDescription s = scene::parse(flying(), "flight");
    const tests::BallAndFloor obstacles(s);
    const animation::Flight plain = animation::make_flight(job_with(animation::NoPrelude{}, above_ball), obstacles);
    const animation::Flight f =
        animation::make_flight(job_with(animation::Perch{.at = ball_top, .until = until}, above_ball), obstacles);
    REQUIRE(f.opening.size() == 2);
    CHECK(f.opening[0].behaviour == animation::Behaviour::still);
    CHECK(f.opening[1].behaviour == animation::Behaviour::transit);
    CHECK(f.opening[1].start == until);
    CHECK(f.begin == until + f.opening[1].duration);
    // The loop is the one without a prelude, whatever the prelude.
    CHECK(f.loop == plain.loop);
    CHECK(f.flashes.starts == plain.flashes.starts);

    // At the perch from before 0 to until.
    int away = 0;
    for (int i = -4; i <= 50; ++i) {
        away += same(animation::position(f, Seconds(static_cast<double>(i) * 0.1)), ball_top) ? 0 : 1;
    }
    CHECK(away == 0);
    // Continuous through both joins, in position and in velocity, by finite
    // differences over h: the rise leaves from rest, and arrives at the
    // loop's own velocity. A float's rounding at a meter, some 1e-7 m, over
    // h = 1 ms is some 1e-4 m/s; a jump the size of the drift's speed, some
    // 0.15 m/s, would be far past the bounds.
    constexpr double h = 1e-3;
    const auto velocity = [&](double t0, double t1) {
        const contracts::Float3 a = animation::position(f, Seconds(t0));
        const contracts::Float3 b = animation::position(f, Seconds(t1));
        return std::array<double, 3>{(static_cast<double>(b.x) - static_cast<double>(a.x)) / (t1 - t0),
                                     (static_cast<double>(b.y) - static_cast<double>(a.y)) / (t1 - t0),
                                     (static_cast<double>(b.z) - static_cast<double>(a.z)) / (t1 - t0)};
    };
    const auto norm = [](std::array<double, 3> v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); };
    CHECK(apart(animation::position(f, Seconds(until + 1e-6)), ball_top) < 1e-5);
    CHECK(norm(velocity(until - h, until)) == 0.0);
    CHECK(norm(velocity(until, until + h)) < 0.01);
    CHECK(apart(animation::position(f, Seconds(f.begin - 1e-6)), animation::position(f, Seconds(f.begin))) < 1e-5);
    CHECK(same(animation::position(f, Seconds(f.begin)), animation::position(plain, Seconds(0.0))));
    const std::array<double, 3> in = velocity(f.begin - h, f.begin);
    const std::array<double, 3> out = velocity(f.begin, f.begin + h);
    CHECK(norm({in[0] - out[0], in[1] - out[1], in[2] - out[2]}) < 0.02);
    CHECK(norm(out) > 0.02);  // the loop's start moves: the check above is not of two stills

    // Every point of the rise, every 0.1 ms, keeps its body perch_gap / 2
    // from every still surface, and lies in the flight's reach.
    const animation::Extent reach = animation::extent(f);
    CHECK(inside(reach, ball_top));
    double nearest = std::numeric_limits<double>::infinity();
    int outside = 0;
    const auto samples = static_cast<int>(f.opening[1].duration / 1e-4);
    for (int i = 0; i <= samples; ++i) {
        const contracts::Float3 p = animation::position(f, Seconds(until + static_cast<double>(i) * 1e-4));
        nearest = std::min(nearest, tests::distance_to_still(s, p) - body);
        outside += inside(reach, p) ? 0 : 1;
    }
    INFO("the rise's nearest surface: " << nearest << " m, over " << samples << " samples");
    CHECK(samples > 1000);
    CHECK(nearest >= animation::perch_gap / 2.0);
    CHECK(outside == 0);
    // And the reach holds the volume.
    CHECK(reach.min.x <= f.volume.min.x);
    CHECK(reach.max.y >= f.volume.max.y);
}

TEST_CASE("a flight's reach grows from its volume to hold a perch below it and the whole rise") {
    // The volume's floor at 1.2 m, above the ball's top: the perch and the
    // start of the rise lie outside it, as the marbles' do (flight.h).
    const scene::SceneDescription s = scene::parse(flying(), "flight");
    const tests::BallAndFloor obstacles(s);
    animation::FlightJob raised = job_with(animation::Perch{.at = ball_top, .until = 1.0}, above_ball);
    raised.params.volume.min.y = 1.2f;
    const animation::Flight f = animation::make_flight(raised, obstacles);
    const animation::Extent reach = animation::extent(f);
    CHECK(reach.min.y <= ball_top.y);
    CHECK(reach.min.y < f.volume.min.y);
    CHECK(reach.max.y == f.volume.max.y);
    CHECK(inside(reach, ball_top));
    int outside = 0;
    for (int i = 0; i <= 10000; ++i) {
        const double t = f.opening[1].start + f.opening[1].duration * static_cast<double>(i) / 10000.0;
        outside += inside(reach, animation::position(f, Seconds(t))) ? 0 : 1;
    }
    CHECK(outside == 0);
}

TEST_CASE("an opening's flashes: while it waits, within [0, begin), a second apart, the last a second before "
          "the loop's first") {
    const scene::SceneDescription s = scene::parse(flying(), "flight");
    const tests::BallAndFloor obstacles(s);
    const animation::Flight plain = animation::make_flight(job_with(animation::NoPrelude{}, above_ball), obstacles);
    for (const animation::Prelude& prelude : {animation::Prelude{animation::Perch{.at = ball_top, .until = 600.0}},
                                              animation::Prelude{animation::Hold{.until = animation::most_wait}}}) {
        const animation::Flight f = animation::make_flight(job_with(prelude, above_ball), obstacles);
        const std::vector<double>& opening = f.flashes.opening;
        // At a drifting rate of 0.02 to 0.08 a second: at least 12 over
        // 600 s, at most some 300 over most_wait (flight.h).
        CHECK(opening.size() >= 6);
        CHECK(opening.size() <= 300);
        CHECK(f.flashes.begin == f.begin);
        // On the wait, none on the rise: within the still segment.
        CHECK(std::ranges::all_of(opening, [&](double t) { return t >= 0.0 && t < f.opening[0].duration; }));
        CHECK(std::ranges::is_sorted(opening));
        CHECK(std::ranges::adjacent_find(opening, [](double a, double b) {
                  return b - a < animation::flash_spacing;
              }) == opening.end());
        REQUIRE_FALSE(f.flashes.starts.empty());
        CHECK(f.begin + f.flashes.starts.front() - opening.back() >= animation::flash_spacing);
        // From keys of their own: the loop's flashes are the ones without a
        // prelude.
        CHECK(f.flashes.starts == plain.flashes.starts);
    }
}

TEST_CASE("an opening's flashes keep their spacing for every seed, against each other and the loop's first") {
    // Sixty seeds of hour-long waits, some 70 to 290 flashes each: drawn
    // uniformly, many fall within a second of another, and some within a
    // second of the loop's first; every one of those is left out.
    const scene::SceneDescription s = scene::parse(flying(), "flight");
    const tests::BallAndFloor obstacles(s);
    int crowded = 0;
    int against_loop = 0;
    for (std::uint64_t seed = 0; seed < 60; ++seed) {
        animation::FlightJob held = job_with(animation::Hold{.until = animation::most_wait});
        held.params.seed = seed;
        const animation::Flight f = animation::make_flight(held, obstacles);
        const std::vector<double>& opening = f.flashes.opening;
        crowded += std::ranges::adjacent_find(opening, [](double a, double b) {
                       return b - a < animation::flash_spacing;
                   }) == opening.end()
                       ? 0
                       : 1;
        against_loop += !opening.empty() && !f.flashes.starts.empty() &&
                                f.begin + f.flashes.starts.front() - opening.back() < animation::flash_spacing
                            ? 1
                            : 0;
    }
    CHECK(crowded == 0);
    CHECK(against_loop == 0);
    // An opening's last falls within a second of the loop's first only when
    // the loop flashes in its first second, a few seeds in a thousand. These
    // are the first three of seeds 0 to 3999 whose 10-minute holds draw one
    // there, found by drawing them all with the spacing left out: each must
    // leave it out.
    for (const std::uint64_t seed : {403u, 1085u, 1601u}) {
        CAPTURE(seed);
        animation::FlightJob held = job_with(animation::Hold{.until = 600.0});
        held.params.seed = seed;
        const animation::Flight f = animation::make_flight(held, obstacles);
        REQUIRE_FALSE(f.flashes.opening.empty());
        REQUIRE_FALSE(f.flashes.starts.empty());
        CHECK(f.flashes.starts.front() < animation::flash_spacing);
        CHECK(f.begin + f.flashes.starts.front() - f.flashes.opening.back() >= animation::flash_spacing);
    }
    // Perched for 30 s and then rising, over sixty seeds: a rise of a second
    // or so would draw a flash for some few of them; none is drawn there.
    int on_rise = 0;
    for (std::uint64_t seed = 0; seed < 60; ++seed) {
        animation::FlightJob perched = job_with(animation::Perch{.at = ball_top, .until = 30.0}, above_ball);
        perched.params.seed = seed;
        const animation::Flight f = animation::make_flight(perched, obstacles);
        on_rise += static_cast<int>(std::ranges::count_if(
            f.flashes.opening, [&](double t) { return t >= f.opening[0].start + f.opening[0].duration; }));
    }
    CHECK(on_rise == 0);
}

TEST_CASE("one seed's opening flashes, pinned: drawn from keys of their own") {
    // As the pinned loop (above): a change of the opening's keys, or of how
    // its flashes are drawn, moves these.
    const scene::SceneDescription s = scene::parse(flying(), "flight");
    const tests::BallAndFloor obstacles(s);
    const animation::Flight f =
        animation::make_flight(job_with(animation::Hold{.until = animation::most_wait}), obstacles);
    const std::vector<double>& opening = f.flashes.opening;
    REQUIRE(opening.size() == 241);
    CHECK(opening.front() == 0x1.03e335dc12e8p+5);
    CHECK(opening.back() == 0x1.c1cd1dd6bed78p+11);
}

namespace {

// Every still surface at one distance, wherever it is asked: a fake of
// contract 11 that sets a flight's every sample at a chosen margin from its
// threshold.
class AtDistance final : public contracts::Obstacles {
public:
    explicit AtDistance(double d) : d_(d) {}
    double distance(contracts::Float3) const override { return d_; }
    bool touches(const contracts::Box&) const override { return false; }

private:
    double d_;
};

}  // namespace

TEST_CASE("step 3's thresholds carry the volume's rounding allowance") {
    // A drifting firefly in a volume reaching 2 m, where rho is sqrt 3 x
    // 2^-22 m, some 4e-7 m. Every sample half an allowance past the
    // clearance without it is refused; two allowances past, made.
    const animation::FlightJob drifting{.params = {.volume = {{-2.0f, 0.05f, -2.0f}, {2.0f, 2.0f, 2.0f}},
                                                   .speed = 0.45f,
                                                   .clearance = 0.06f,
                                                   .weights = {0.0f, 0.0f, 1.0f},
                                                   .seed = 5},
                                        .start = {0.0f, 1.0f, 0.0f},
                                        .body = 0.03f};
    const double rho = animation::rounding_allowance(2.0);
    const double bare = static_cast<double>(drifting.params.clearance) + static_cast<double>(drifting.body) +
                        animation::flight_delta;
    CHECK_THROWS_AS(animation::make_flight(drifting, AtDistance(bare + 0.5 * rho)), animation::MotionError);
    CHECK_NOTHROW((void)animation::make_flight(drifting, AtDistance(bare + 2.0 * rho)));
}

TEST_CASE("a perch too near, a wait past most_wait and a rise under an overhang are each refused") {
    const scene::SceneDescription s = scene::parse(flying(), "flight");
    const tests::BallAndFloor obstacles(s);
    const auto make = [&](const animation::Prelude& prelude) {
        return animation::make_flight(job_with(prelude, above_ball), obstacles);
    };
    // Its body 0.3 mm from the ball's top: under 3/4 of perch_gap.
    const contracts::Float3 too_near{0.0f, static_cast<float>(1.0 + body + 0.0003), 0.0f};
    CHECK_THROWS_WITH_AS(make(animation::Perch{.at = too_near, .until = 1.0}),
                         doctest::Contains("the perch is too near a still surface"), animation::MotionError);
    CHECK_THROWS_AS(make(animation::Perch{.at = {0.0f, 0.5f, 0.0f}, .until = 1.0}), animation::MotionError);
    // 0.4 mm, past the 3/4 of perch_gap it needs: made.
    const contracts::Float3 near_enough{0.0f, static_cast<float>(1.0 + body + 0.0004), 0.0f};
    CHECK_NOTHROW((void)make(animation::Perch{.at = near_enough, .until = 1.0}));
    // Waits past most_wait, below 0 or not a number: numbers out of range.
    CHECK_THROWS_AS(make(animation::Hold{.until = animation::most_wait + 1.0}), std::invalid_argument);
    CHECK_THROWS_AS(make(animation::Hold{.until = -1.0}), std::invalid_argument);
    CHECK_THROWS_AS(make(animation::Perch{.at = ball_top, .until = std::nan("")}), std::invalid_argument);
    CHECK_THROWS_AS(make(animation::Perch{.at = ball_top, .until = animation::most_wait * 2.0}),
                    std::invalid_argument);
    CHECK_NOTHROW((void)make(animation::Hold{.until = animation::most_wait}));
    CHECK_THROWS_AS(make(animation::Perch{.at = {std::numeric_limits<float>::infinity(), 1.0f, 0.0f}, .until = 1.0}),
                    std::invalid_argument);
    // On the floor beside the ball, its loop straight above: the ball
    // overhangs the way up.
    animation::FlightJob under =
        job_with(animation::Perch{.at = {0.3f, static_cast<float>(body + animation::perch_gap), 0.0f}, .until = 1.0});
    under.start = {0.3f, 1.5f, 0.0f};
    CHECK_THROWS_WITH_AS(animation::make_flight(under, obstacles),
                         doctest::Contains("the rise from the perch is not clear of the still shapes"),
                         animation::MotionError);
    // The same perch with its loop off to the side rises clear.
    under.start = {1.2f, 1.0f, 0.8f};
    CHECK_NOTHROW((void)animation::make_flight(under, obstacles));
    // A rise needing more samples than a segment may take, some 4 V /
    // perch_gap, is refused before the walk; the loop alone is made.
    animation::FlightJob bounded = job_with(animation::Perch{.at = ball_top, .until = 1.0}, above_ball);
    bounded.params.most_steps = 3000;
    animation::FlightJob loop_only = job_with(animation::NoPrelude{}, above_ball);
    loop_only.params.most_steps = 3000;
    CHECK_NOTHROW((void)animation::make_flight(loop_only, obstacles));
    CHECK_THROWS_WITH_AS(animation::make_flight(bounded, obstacles), doctest::Contains("more than the most samples"),
                         animation::MotionError);
}

namespace {

// The ground plane y = 0, everywhere: a still surface at any distance from
// the origin, for perches far out.
class Ground final : public contracts::Obstacles {
public:
    Ground() = default;
    double distance(contracts::Float3 p) const override { return static_cast<double>(p.y); }
    bool touches(const contracts::Box& box) const override { return box.min.y <= 0.0f; }
};

}  // namespace

TEST_CASE("a perch too far out for its gap is refused; one nearer the origin is made") {
    const Ground ground;
    const auto job_at = [](float x) {
        return animation::FlightJob{
            .params = {.volume = {{x - 10.0f, 0.05f, -10.0f}, {x + 10.0f, 2.0f, 10.0f}},
                       .speed = 0.45f,
                       .clearance = 0.06f,
                       .weights = {0.0f, 1.0f, 1.0f},
                       .seed = 3},
            .start = {x, 1.0f, 0.0f},
            .body = 0.03f,
            .prelude = animation::Perch{.at = {x, 0.03f + static_cast<float>(animation::perch_gap), 0.0f},
                                        .until = 1.0}};
    };
    // At 100 m float's spacing is 7.6e-6 m and rho 1.3e-5 m, under
    // perch_gap / 8, 6.25e-5 m; at 1000 m, 6.1e-5 m and 1.06e-4 m, past it.
    CHECK_NOTHROW((void)animation::make_flight(job_at(100.0f), ground));
    CHECK_THROWS_WITH_AS(animation::make_flight(job_at(1000.0f), ground),
                         doctest::Contains("too far from the origin for its gap"), animation::MotionError);
}

TEST_CASE("a flight's first loop point is its start plus first_offset(seed), bit for bit") {
    const scene::SceneDescription s = scene::parse(flying(), "flight");
    const tests::BallAndFloor obstacles(s);
    int differ = 0;
    int far = 0;
    for (std::uint64_t seed = 0; seed < 24; ++seed) {
        for (const contracts::Float3 start : {flying_start, above_ball, contracts::Float3{-1.3f, 1.4f, -0.9f}}) {
            animation::FlightJob j = job_with(animation::Hold{.until = 2.0}, start);
            j.params.seed = seed;
            const animation::Flight f = animation::make_flight(j, obstacles);
            const std::array<double, 3> offset = animation::first_offset(seed);
            const contracts::Float3 expected{static_cast<float>(static_cast<double>(start.x) + offset[0]),
                                             static_cast<float>(static_cast<double>(start.y) + offset[1]),
                                             static_cast<float>(static_cast<double>(start.z) + offset[2])};
            // Where it holds, and where its loop begins.
            differ += same(animation::position(f, Seconds(0.5)), expected) ? 0 : 1;
            differ += same(animation::position(f, Seconds(f.begin)), expected) ? 0 : 1;
            far += std::ranges::any_of(offset, [](double o) { return std::abs(o) > animation::first_drift_reach; })
                       ? 1
                       : 0;
        }
    }
    CHECK(differ == 0);
    CHECK(far == 0);
    // Not the start itself: the offset is the drift's, up to 10 cm an axis.
    const std::array<double, 3> offset = animation::first_offset(5);
    CHECK(std::abs(offset[0]) + std::abs(offset[1]) + std::abs(offset[2]) > 0.0);
}

TEST_CASE("a wake at a large at keeps its ramp: the end is judged from t - at, not at + ramp") {
    // At 2^100, at + 1 rounds to at; a ramp judged from at + ramp would be
    // gone, the light full at at. Judged from t - at, it is 0 at at, and full
    // a float spacing of at later, 2^48 s, past the ramp.
    constexpr float dim = 0.5f;
    constexpr double at = 0x1p100;
    const animation::Glows glows = dim_schedule(dim, animation::Wake{.at = at, .ramp = 1.0});
    CHECK(animation::glow(glows, first_schedule, Seconds(at)) == 0.0f);
    CHECK(animation::glow(glows, first_schedule, Seconds(std::nextafter(at, 0.0))) == 0.0f);
    CHECK(animation::glow(glows, first_schedule, Seconds(std::nextafter(at, 2.0 * at))) == dim);
    animation::Glows rhythm;
    rhythm.rhythms.push_back({.period = 1e30, .flash = 0.4, .dim = dim, .seed = 9, .wake = animation::Wake{.at = at,
                                                                                                       .ramp = 1.0}});
    CHECK(animation::glow(rhythm, first_rhythm, Seconds(at)) == 0.0f);
}

TEST_CASE("a glow not yet awake is still checked: a bad record is refused whether it is awake or not") {
    const animation::Wake late{.at = 1e6, .ramp = 1.0};
    animation::Glows bad_loop = dim_schedule(0.5f, late);
    bad_loop.schedules[0].schedule.loop = 0.0;
    CHECK_THROWS_AS((void)animation::glow(bad_loop, first_schedule, Seconds(1.0)), std::invalid_argument);
    animation::Glows bad_begin = dim_schedule(0.5f, late);
    bad_begin.schedules[0].schedule.begin = -1.0;
    CHECK_THROWS_AS((void)animation::glow(bad_begin, first_schedule, Seconds(1.0)), std::invalid_argument);
    // A rhythm's t past what its period counts to, while it sleeps.
    animation::Glows rhythm;
    rhythm.rhythms.push_back({.period = animation::least_period,
                              .flash = animation::least_period / 4.0,
                              .dim = 0.1f,
                              .seed = 3,
                              .wake = animation::Wake{.at = 1e18, .ramp = 1.0}});
    CHECK_THROWS_AS((void)animation::glow(rhythm, first_rhythm, Seconds(1e17)), std::invalid_argument);
    // And a wake not yet reached that is itself bad.
    CHECK_THROWS_AS((void)animation::glow(dim_schedule(0.5f, animation::Wake{.at = 5.0, .ramp = -1.0}),
                                          first_schedule, Seconds(1.0)),
                    std::invalid_argument);
}

TEST_CASE("a woken glow is its plain glow times its wake's factor, bit for bit, asleep, waking and awake") {
    // The factor in glow.h's arithmetic; the plain glow is the same record
    // without its wake. Every millisecond from 2 s before at to 2 s past the
    // ramp, for both kinds, a schedule with an opening included.
    constexpr double at = 6.0;
    constexpr double ramp = 3.0;
    const auto factor = [](double t) {
        if (t < at) {
            return 0.0;
        }
        if (t - at >= ramp) {
            return 1.0;
        }
        const double x = (t - at) / ramp;
        return x * x * (3.0 - 2.0 * x);
    };
    animation::Glows plain;
    plain.rhythms.push_back({.period = 0.9, .flash = 0.3, .dim = 0.1f, .seed = 4});
    animation::ScheduleGlow g;
    g.schedule = {.begin = 5.0, .opening = {1.0, 4.5}, .loop = 3.0, .starts = {0.5, 2.0}};
    g.flash = 0.4;
    g.dim = 0.2f;
    plain.schedules.push_back(g);
    animation::Glows woken = plain;
    woken.rhythms[0].wake = animation::Wake{.at = at, .ramp = ramp};
    woken.schedules[0].wake = animation::Wake{.at = at, .ramp = ramp};
    int differ = 0;
    for (int i = 4000; i < 11000; ++i) {
        const double t = static_cast<double>(i) * 0.001;
        for (const animation::GlowRecord r : {first_rhythm, first_schedule}) {
            const double w = factor(t);
            const float p = animation::glow(plain, r, Seconds(t));
            const float expected = w == 0.0 ? 0.0f : static_cast<float>(w * static_cast<double>(p));
            differ += animation::glow(woken, r, Seconds(t)) == expected ? 0 : 1;
        }
    }
    CHECK(differ == 0);
}

TEST_CASE("the rounding allowance: sqrt 3 float spacings at the largest coordinate") {
    constexpr double sqrt3 = std::numbers::sqrt3;
    CHECK(animation::rounding_allowance(1.0) == sqrt3 * 0x1p-23);
    CHECK(animation::rounding_allowance(0.75) == sqrt3 * 0x1p-24);
    CHECK(animation::rounding_allowance(1.5) == sqrt3 * 0x1p-23);
    // A double between floats is taken at the float at or above it: just
    // under 2 rounds to 2, whose spacing above is 2^-22.
    CHECK(animation::rounding_allowance(std::nextafter(2.0, 0.0)) == sqrt3 * 0x1p-22);
    // Nearer the float below 2, 2 - 2^-23, whose spacing is 2^-23; the float
    // at or above it is 2, whose spacing is 2^-22, which holds every point.
    CHECK(animation::rounding_allowance(2.0 - 0x1p-23 + 0x1p-30) == sqrt3 * 0x1p-22);
    CHECK(animation::rounding_allowance(1e6) == sqrt3 * 0.0625);
    CHECK(animation::rounding_allowance(512.0) == sqrt3 * 0x1p-14);
    CHECK(animation::rounding_allowance(0.0) ==
          sqrt3 * static_cast<double>(std::numeric_limits<float>::denorm_min()));
    // The marbles' scale, some 2e-7 m (flight.h).
    CHECK(animation::rounding_allowance(1.7) < 2.1e-7);
    // Past float's range no float holds a point.
    CHECK(animation::rounding_allowance(static_cast<double>(std::numeric_limits<float>::max())) ==
          std::numeric_limits<double>::infinity());
    CHECK(animation::rounding_allowance(1e39) == std::numeric_limits<double>::infinity());
    CHECK_THROWS_AS((void)animation::rounding_allowance(-1.0), std::invalid_argument);
    CHECK_THROWS_AS((void)animation::rounding_allowance(std::nan("")), std::invalid_argument);
}
