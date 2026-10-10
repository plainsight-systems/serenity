#include "core/animation/flight.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

#include "core/animation/draw.h"

namespace serenity::animation {

namespace {

constexpr int episodes = 64;
constexpr int attempts = 16;         // draws of an episode in a round (step 3)
constexpr int rounds = 4;            // rounds of draws an episode may have (step 4)
constexpr int most_backtracks = 64;  // redraws of earlier episodes, in all (step 4)
constexpr double delta = flight_delta;  // the sampling bound, meters (step 3)
constexpr double pi = std::numbers::pi;

// The behaviours' numbers (flight.h), named (ES.45). A range is drawn
// uniformly between its least and its most.

// circle
constexpr double orbit_gap = 0.04;            // meters past the clearance to the nearest orbit
constexpr double orbit_widest = 0.6;          // meters wider than the nearest the orbit may be
constexpr double tilt_most = 25.0 * pi / 180.0;
constexpr double breathe_most = 0.1;          // of the radius
constexpr double breathe_rate_least = 0.1;    // hertz
constexpr double breathe_rate_most = 0.4;
constexpr double bob_most = 0.05;             // meters
constexpr double bob_rate_least = 0.3;        // hertz
constexpr double bob_rate_most = 0.8;
constexpr double circle_shortest = 4.0;       // seconds
constexpr double circle_longest = 9.0;

// swoop: a J of a dip, then a climb, over a reach along its heading
constexpr double dip_least = 0.1;             // meters
constexpr double dip_most = 0.25;
constexpr double climb_least = 0.3;           // meters
constexpr double climb_most = 0.6;
constexpr double swoop_reach_least = 0.3;     // meters along the heading
constexpr double swoop_reach_most = 0.6;
constexpr double swoop_headroom = 0.35;       // meters above the climb, for the transit out of it
constexpr double swoop_low = 1.6;             // the first handle's depth, in dips: the deepest the J goes
constexpr double swoop_mid = 0.8;             // the second handle's depth, in dips
constexpr double swoop_first_along = 0.3;     // the handles' way along the reach
constexpr double swoop_second_along = 0.8;
constexpr double swoop_speed_share = 0.8;     // of the cruising speed, along its chord
constexpr double swoop_shortest = 1.0;        // seconds
constexpr double swoop_longest = 2.5;

// drift
constexpr double drift_least_reach = 0.04;    // meters on each axis, up to first_drift_reach
constexpr double drift_slowdown = 3.0;        // the cruising speed over this
constexpr double drift_room = 0.1;            // meters it is placed with room for on every side
constexpr double drift_shortest = 2.0;        // seconds
constexpr double drift_longest = 4.0;

// transit
constexpr double transit_shortest = 0.4;      // seconds
constexpr double transit_slack = 1.2;         // its duration over its chord's at the cruising speed
constexpr double waypoint_headroom = 0.05;    // meters below the volume's top, past body and delta
constexpr double level_least = 1e-6;         // meters: a level offset shorter than this has no heading

// where a swoop or a drift starts, near where the last episode ended: within
// a meter of it, pulled a third of the way toward the volume's middle
constexpr double episode_start_reach = 1.0;   // meters
constexpr double pull_to_middle = 1.0 / 3.0;

// step 7
constexpr double swoop_flash_at = 0.55;       // of the swoop's duration: on its climb
constexpr double circling_rate_least = 0.1;   // flashes a second
constexpr double circling_rate_most = 0.25;
constexpr double drifting_rate_least = 0.02;
constexpr double drifting_rate_most = 0.08;

// The central difference's step, in seconds: small beside any segment's
// time scale, large beside a double's rounding at a loop's times.
constexpr double velocity_step = 1e-5;

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};
constexpr Vec3 operator+(Vec3 a, Vec3 b) noexcept { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
constexpr Vec3 operator-(Vec3 a, Vec3 b) noexcept { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
constexpr Vec3 operator*(double s, Vec3 a) noexcept { return {s * a.x, s * a.y, s * a.z}; }
constexpr double dot(Vec3 a, Vec3 b) noexcept { return a.x * b.x + a.y * b.y + a.z * b.z; }
double length(Vec3 a) noexcept { return std::sqrt(dot(a, a)); }
constexpr Vec3 cross(Vec3 a, Vec3 b) noexcept {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
Vec3 normalized(Vec3 a) noexcept { return (1.0 / length(a)) * a; }
constexpr Vec3 to_vec(contracts::Float3 f) noexcept { return {f.x, f.y, f.z}; }
constexpr Vec3 world_up{0.0, 1.0, 0.0};

// Where each number of a segment is (Segment::numbers, flight.h), per
// behaviour (P.1): read and written only by these names. A triple takes
// three numbers from its index.
struct TransitAt {  // Hermite: tangents scaled by the duration
    static constexpr std::size_t p0 = 0;
    static constexpr std::size_t p1 = 3;
    static constexpr std::size_t t0 = 6;
    static constexpr std::size_t t1 = 9;
};
struct CircleAt {
    static constexpr std::size_t center = 0;
    static constexpr std::size_t u = 3;
    static constexpr std::size_t v = 6;
    static constexpr std::size_t r0 = 9;
    static constexpr std::size_t theta0 = 10;
    static constexpr std::size_t omega = 11;
    static constexpr std::size_t breathe = 12;
    static constexpr std::size_t breathe_rate = 13;
    static constexpr std::size_t breathe_phase = 14;
    static constexpr std::size_t bob = 15;
    static constexpr std::size_t bob_rate = 16;
    static constexpr std::size_t bob_phase = 17;
};
struct SwoopAt {  // a cubic Bezier's control points
    static constexpr std::size_t b0 = 0;
    static constexpr std::size_t b1 = 3;
    static constexpr std::size_t b2 = 6;
    static constexpr std::size_t b3 = 9;
};
struct DriftAt {  // per axis
    static constexpr std::size_t center = 0;
    static constexpr std::size_t amplitude = 3;
    static constexpr std::size_t frequency = 6;
    static constexpr std::size_t phase = 9;
};

constexpr Vec3 triple_at(const Segment& s, std::size_t i) noexcept {
    return {s.numbers[i], s.numbers[i + 1], s.numbers[i + 2]};
}
constexpr void put_triple(Segment& s, std::size_t i, Vec3 v) noexcept {
    s.numbers[i] = v.x;
    s.numbers[i + 1] = v.y;
    s.numbers[i + 2] = v.z;
}

constexpr Vec3 bezier(Vec3 b0, Vec3 b1, Vec3 b2, Vec3 b3, double u) noexcept {
    const double w = 1.0 - u;
    return (w * w * w) * b0 + (3.0 * w * w * u) * b1 + (3.0 * w * u * u) * b2 + (u * u * u) * b3;
}

[[noreturn]] void no_case(const char* where) {
    // After a switch over every Behaviour: reached only by a value no
    // enumerator names, a corrupt segment, which is refused rather than
    // answered with a point that looks real (P.6).
    throw std::logic_error(std::string(where) + ": a segment whose behaviour is no Behaviour");
}

// Step E3: a segment's closed form at time `tau` into it.
Vec3 evaluate(const Segment& s, double tau) {
    switch (s.behaviour) {
    case Behaviour::transit: {
        // Hermite as Bezier: B1 = P0 + T0 / 3, B2 = P1 - T1 / 3.
        const Vec3 p0 = triple_at(s, TransitAt::p0);
        const Vec3 p1 = triple_at(s, TransitAt::p1);
        const Vec3 t0 = triple_at(s, TransitAt::t0);
        const Vec3 t1 = triple_at(s, TransitAt::t1);
        return bezier(p0, p0 + (1.0 / 3.0) * t0, p1 - (1.0 / 3.0) * t1, p1, tau / s.duration);
    }
    case Behaviour::circle: {
        const Vec3 c = triple_at(s, CircleAt::center);
        const Vec3 u = triple_at(s, CircleAt::u);
        const Vec3 v = triple_at(s, CircleAt::v);
        const Vec3 w = cross(u, v);
        const auto n = [&](std::size_t i) { return s.numbers[i]; };
        const double breathing = std::sin(2.0 * pi * n(CircleAt::breathe_rate) * tau + n(CircleAt::breathe_phase));
        const double r = n(CircleAt::r0) * (1.0 + n(CircleAt::breathe) * breathing);
        const double theta = n(CircleAt::theta0) + n(CircleAt::omega) * tau;
        const double bob = n(CircleAt::bob) * std::sin(2.0 * pi * n(CircleAt::bob_rate) * tau + n(CircleAt::bob_phase));
        return c + (r * std::cos(theta)) * u + (r * std::sin(theta)) * v + bob * w;
    }
    case Behaviour::swoop:
        return bezier(triple_at(s, SwoopAt::b0), triple_at(s, SwoopAt::b1), triple_at(s, SwoopAt::b2),
                      triple_at(s, SwoopAt::b3), tau / s.duration);
    case Behaviour::drift: {
        const Vec3 c = triple_at(s, DriftAt::center);
        const Vec3 a = triple_at(s, DriftAt::amplitude);
        const Vec3 f = triple_at(s, DriftAt::frequency);
        const Vec3 p = triple_at(s, DriftAt::phase);
        return c + Vec3{a.x * std::sin(2.0 * pi * f.x * tau + p.x), a.y * std::sin(2.0 * pi * f.y * tau + p.y),
                        a.z * std::sin(2.0 * pi * f.z * tau + p.z)};
    }
    }
    no_case("evaluate");
}

// The velocity, by a central difference of the closed form, which is
// defined, and smooth, past either end of the segment.
Vec3 velocity(const Segment& s, double tau) {
    return (1.0 / (2.0 * velocity_step)) * (evaluate(s, tau + velocity_step) - evaluate(s, tau - velocity_step));
}

// An upper bound on a segment's speed, for step 3's sample spacing: the
// convex hull of a Bezier's derivative, and each sinusoid's peak.
double speed_bound(const Segment& s) {
    switch (s.behaviour) {
    case Behaviour::transit: {
        const Vec3 p0 = triple_at(s, TransitAt::p0);
        const Vec3 p1 = triple_at(s, TransitAt::p1);
        const Vec3 b1 = p0 + (1.0 / 3.0) * triple_at(s, TransitAt::t0);
        const Vec3 b2 = p1 - (1.0 / 3.0) * triple_at(s, TransitAt::t1);
        return 3.0 * std::max({length(b1 - p0), length(b2 - b1), length(p1 - b2)}) / s.duration;
    }
    case Behaviour::swoop: {
        const Vec3 b0 = triple_at(s, SwoopAt::b0);
        const Vec3 b1 = triple_at(s, SwoopAt::b1);
        const Vec3 b2 = triple_at(s, SwoopAt::b2);
        const Vec3 b3 = triple_at(s, SwoopAt::b3);
        return 3.0 * std::max({length(b1 - b0), length(b2 - b1), length(b3 - b2)}) / s.duration;
    }
    case Behaviour::circle: {
        const auto n = [&](std::size_t i) { return s.numbers[i]; };
        const double r0 = n(CircleAt::r0);
        return r0 * (1.0 + n(CircleAt::breathe)) * std::abs(n(CircleAt::omega)) +
               r0 * n(CircleAt::breathe) * 2.0 * pi * n(CircleAt::breathe_rate) +
               n(CircleAt::bob) * 2.0 * pi * n(CircleAt::bob_rate);
    }
    case Behaviour::drift: {
        const Vec3 a = triple_at(s, DriftAt::amplitude);
        const Vec3 f = triple_at(s, DriftAt::frequency);
        return 2.0 * pi * length(Vec3{a.x * f.x, a.y * f.y, a.z * f.z});
    }
    }
    no_case("speed_bound");
}

// What every step asks of one flight: its numbers, its body and the still
// shapes. Read through references, so never copied (C.12).
struct Context {
    Context(const FlightParams& p, double b, const contracts::Obstacles& o) : params(p), body(b), obstacles(o) {}
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    const FlightParams& params;
    double body;
    const contracts::Obstacles& obstacles;
};

// Step 3: whether every sample of the segment keeps the clearance and keeps
// the body inside the volume, the samples spaced so the path between two is
// within delta of them, and at most the params' most_steps of them.
bool clear(const Context& c, const Segment& s) {
    const double v = speed_bound(s);
    if (!std::isfinite(v) || !(s.duration > 0.0)) {
        return false;
    }
    // Compared as a double, before it is made an integer: a count past
    // int64's range would make the conversion undefined (ES.46).
    const double spans = std::ceil(v * s.duration / delta);
    if (!(spans < static_cast<double>(c.params.most_steps))) {
        return false;  // a path that long is not a firefly's
    }
    const auto steps = static_cast<std::int64_t>(spans) + 1;
    const double margin = c.body + delta;
    const double needed = static_cast<double>(c.params.clearance) + c.body + delta;
    const Extent& box = c.params.volume;
    for (std::int64_t i = 0; i <= steps; ++i) {
        const Vec3 p = evaluate(s, s.duration * static_cast<double>(i) / static_cast<double>(steps));
        if (p.x < box.min.x + margin || p.x > box.max.x - margin || p.y < box.min.y + margin ||
            p.y > box.max.y - margin || p.z < box.min.z + margin || p.z > box.max.z - margin) {
            return false;
        }
        if (c.obstacles.distance({static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z)}) <
            needed) {
            return false;
        }
    }
    return true;
}

// Which number of an episode's draws a draw is: its value is the draw's
// last counter, so the order is the flights' keying, and an enumerator is
// added only at the end.
enum class Purpose : std::uint64_t {
    choose,
    target,
    radius,
    tilt,
    azimuth,
    lift,
    turn,
    direction,
    loops,
    breathe,
    breathe_rate,
    breathe_phase,
    bob,
    bob_rate,
    bob_phase,
    where_x,
    where_y,
    where_z,
    heading,
    dip,
    rise,
    reach_out,
    lasting,
    flashes,
    flash_at,
    rate_circle,
    rate_drift,
    amplitude_x,
    amplitude_y,
    amplitude_z,
    phase_x,
    phase_y,
    phase_z,
};

struct Draws {
    std::uint64_t seed = 0;
    std::uint64_t episode = 0;
    std::uint64_t attempt = 0;
    double operator()(Purpose p) const { return draw(seed, episode, attempt, static_cast<std::uint64_t>(p)); }
    double between(Purpose p, double lo, double hi) const { return lo + (hi - lo) * (*this)(p); }
};

// The room a point drawn near another must leave inside the volume, for
// what is drawn there: below it, above it, and about it on the level.
struct Room {
    double below = 0.0;
    double above = 0.0;
    double around = 0.0;
};

// A point near `from`: within `reach` of it, pulled a third of the way
// toward the volume's middle, so a firefly ranges over its volume rather
// than settling at an edge; and inside the volume with `room` for what is
// drawn there.
Vec3 nearby(const Context& c, const Draws& d, Vec3 from, double reach, Room room) {
    const Extent& box = c.params.volume;
    const double m = c.body + delta + static_cast<double>(c.params.clearance);
    const auto mid = [](float lo, float hi) { return 0.5 * (static_cast<double>(lo) + hi); };
    const Vec3 middle{mid(box.min.x, box.max.x), mid(box.min.y, box.max.y), mid(box.min.z, box.max.z)};
    const Vec3 toward = from + pull_to_middle * (middle - from);
    const auto pick = [&](Purpose p, double at, double lo, double hi) {
        const double v = at + reach * (2.0 * d(p) - 1.0);
        return lo < hi ? std::clamp(v, lo, hi) : 0.5 * (lo + hi);
    };
    return {pick(Purpose::where_x, toward.x, box.min.x + m + room.around, box.max.x - m - room.around),
            pick(Purpose::where_y, toward.y, box.min.y + m + room.below, box.max.y - m - room.above),
            pick(Purpose::where_z, toward.z, box.min.z + m + room.around, box.max.z - m - room.around)};
}

Segment drift_about(const FlightParams& params, const Draws& d, Vec3 center) {
    Segment s;
    s.behaviour = Behaviour::drift;
    s.duration = d.between(Purpose::lasting, drift_shortest, drift_longest);
    put_triple(s, DriftAt::center, center);
    // A reach of at most first_drift_reach on each axis, at a third of the
    // cruising speed: each axis's frequency from its amplitude,
    // f = (speed / 3) / (2 pi a sqrt 3).
    const std::array<double, 3> a = {d.between(Purpose::amplitude_x, drift_least_reach, first_drift_reach),
                                     d.between(Purpose::amplitude_y, drift_least_reach, first_drift_reach),
                                     d.between(Purpose::amplitude_z, drift_least_reach, first_drift_reach)};
    const std::array<Purpose, 3> phases = {Purpose::phase_x, Purpose::phase_y, Purpose::phase_z};
    const double v = static_cast<double>(params.speed) / drift_slowdown;
    for (std::size_t i = 0; i < 3; ++i) {
        s.numbers[DriftAt::amplitude + i] = a[i];
        s.numbers[DriftAt::frequency + i] = v / (2.0 * pi * a[i] * std::sqrt(3.0));
        s.numbers[DriftAt::phase + i] = d.between(phases[i], 0.0, 2.0 * pi);
    }
    return s;
}

Segment circle_about(const Context& c, const Draws& d) {
    const FlightParams& params = c.params;
    const auto pick = static_cast<std::size_t>(d(Purpose::target) * static_cast<double>(params.targets.size()));
    const Target& t = params.targets[std::min(pick, params.targets.size() - 1)];
    const double inner = t.radius + static_cast<double>(params.clearance) + c.body + delta + orbit_gap;
    const double r0 = d.between(Purpose::radius, inner, inner + orbit_widest);
    // The orbit's center: a drawn lift above the target's, and higher where
    // that leaves the bob no room above the volume's floor (a marble on the
    // table): the floor by the body and delta, and the bob's depth, below it.
    const double floor = static_cast<double>(params.volume.min.y) + c.body + delta;
    const double height =
        std::max(static_cast<double>(t.center.y) + d.between(Purpose::lift, 0.0, static_cast<double>(t.radius)),
                 floor + bob_most);
    // The orbit's plane: level tilted by up to tilt_most about a level axis,
    // or less where the floor is near: as far as keeps the orbit's lowest
    // point, its radius breathed out and bobbed down, above the floor.
    const double room = height - bob_most - floor;
    const double steepest = std::min(tilt_most, std::asin(std::clamp(room / ((1.0 + breathe_most) * r0), 0.0, 1.0)));
    const double alpha = d.between(Purpose::tilt, 0.0, steepest);
    const double beta = d.between(Purpose::azimuth, 0.0, 2.0 * pi);
    const Vec3 axis{std::cos(beta), 0.0, std::sin(beta)};
    const Vec3 w = normalized(std::cos(alpha) * world_up + std::sin(alpha) * cross(axis, world_up));
    const Vec3 u = normalized(cross(w, axis));
    const Vec3 v = cross(w, u);
    Segment s;
    s.behaviour = Behaviour::circle;
    put_triple(s, CircleAt::center, Vec3{static_cast<double>(t.center.x), height, static_cast<double>(t.center.z)});
    put_triple(s, CircleAt::u, u);
    put_triple(s, CircleAt::v, v);
    s.numbers[CircleAt::r0] = r0;
    s.numbers[CircleAt::theta0] = d.between(Purpose::turn, 0.0, 2.0 * pi);
    const double omega = static_cast<double>(params.speed) / r0;
    s.numbers[CircleAt::omega] = d(Purpose::direction) < 0.5 ? omega : -omega;
    s.numbers[CircleAt::breathe] = d.between(Purpose::breathe, 0.0, breathe_most);
    s.numbers[CircleAt::breathe_rate] = d.between(Purpose::breathe_rate, breathe_rate_least, breathe_rate_most);
    s.numbers[CircleAt::breathe_phase] = d.between(Purpose::breathe_phase, 0.0, 2.0 * pi);
    s.numbers[CircleAt::bob] = d.between(Purpose::bob, 0.0, bob_most);
    s.numbers[CircleAt::bob_rate] = d.between(Purpose::bob_rate, bob_rate_least, bob_rate_most);
    s.numbers[CircleAt::bob_phase] = d.between(Purpose::bob_phase, 0.0, 2.0 * pi);
    // Four to nine seconds: a wide orbit makes part of a loop, a tight one
    // more than one, at the cruising speed either way.
    s.duration = d.between(Purpose::loops, circle_shortest, circle_longest);
    return s;
}

Segment swoop_from(const Context& c, const Draws& d, Vec3 from) {
    const double h = d.between(Purpose::heading, 0.0, 2.0 * pi);
    const Vec3 forward{std::cos(h), 0.0, std::sin(h)};
    const double depth = d.between(Purpose::dip, dip_least, dip_most);
    const double climb = d.between(Purpose::rise, climb_least, climb_most);
    const double reach = d.between(Purpose::reach_out, swoop_reach_least, swoop_reach_most);
    // Its start, with room for the whole J: its dip below, its reach about,
    // and above, its climb and the way out of it: it ends climbing, and the
    // transit that leaves it rises on before it turns.
    const Vec3 start = nearby(c, d, from, episode_start_reach,
                              Room{.below = swoop_low * depth, .above = climb + swoop_headroom, .around = reach});
    // A J: forward and down, then up steeply.
    const Vec3 b0 = start;
    const Vec3 b1 = start + (swoop_first_along * reach) * forward - (swoop_low * depth) * world_up;
    const Vec3 b2 = start + (swoop_second_along * reach) * forward - (swoop_mid * depth) * world_up;
    const Vec3 b3 = start + reach * forward + climb * world_up;
    Segment s;
    s.behaviour = Behaviour::swoop;
    put_triple(s, SwoopAt::b0, b0);
    put_triple(s, SwoopAt::b1, b1);
    put_triple(s, SwoopAt::b2, b2);
    put_triple(s, SwoopAt::b3, b3);
    const double chord = length(b1 - b0) + length(b2 - b1) + length(b3 - b2);
    s.duration = std::clamp(chord / (swoop_speed_share * static_cast<double>(c.params.speed)), swoop_shortest,
                            swoop_longest);
    return s;
}

// Step 2: episode k's behaviour, drawn near where the last ended.
Segment behaviour_for(const Context& c, const Draws& d, Vec3 from) {
    const auto& w = c.params.weights;
    const double total = static_cast<double>(w[0]) + w[1] + w[2];
    const double pick = d(Purpose::choose) * total;
    if (pick < w[0]) {
        return circle_about(c, d);
    }
    if (pick < static_cast<double>(w[0]) + w[1]) {
        return swoop_from(c, d, from);
    }
    const Room room{.below = drift_room, .above = drift_room, .around = drift_room};
    return drift_about(c.params, d, nearby(c, d, from, episode_start_reach, room));
}

// One end of a transit: where it is, and how fast it moves which way.
struct End {
    Vec3 at;
    Vec3 velocity;
};

// A transit from `from` to `to` over a duration from its length.
Segment hermite(const FlightParams& params, End from, End to) {
    const double chord = length(to.at - from.at);
    Segment s;
    s.behaviour = Behaviour::transit;
    s.duration = std::max(transit_shortest, transit_slack * chord / static_cast<double>(params.speed));
    put_triple(s, TransitAt::p0, from.at);
    put_triple(s, TransitAt::p1, to.at);
    put_triple(s, TransitAt::t0, s.duration * from.velocity);
    put_triple(s, TransitAt::t1, s.duration * to.velocity);
    return s;
}

// Step 4: the transit from `from` to `to`, direct or over the waypoint at
// the volume's top; empty if neither is clear.
std::vector<Segment> transit(const Context& c, const Segment& from, const Segment& to) {
    const End out{evaluate(from, from.duration), velocity(from, from.duration)};
    const End in{evaluate(to, 0.0), velocity(to, 0.0)};
    const Segment direct = hermite(c.params, out, in);
    if (clear(c, direct)) {
        return {direct};
    }
    const double top = c.params.volume.max.y - c.body - delta - waypoint_headroom;
    const Vec3 mid{0.5 * (out.at.x + in.at.x), top, 0.5 * (out.at.z + in.at.z)};
    Vec3 level = in.at - out.at;
    level.y = 0.0;
    const Vec3 across =
        length(level) > level_least ? (static_cast<double>(c.params.speed) / length(level)) * level : Vec3{};
    const Segment rise = hermite(c.params, out, End{mid, across});
    const Segment fall = hermite(c.params, End{mid, across}, in);
    if (clear(c, rise) && clear(c, fall)) {
        return {rise, fall};
    }
    return {};
}

// The episodes, each with the transit that leads into it: transits[k]
// leads into behaviours[k], for k from 1.
struct Episodes {
    std::vector<Segment> behaviours;
    std::vector<std::vector<Segment>> transits;
};

// Episode k drawn from round `round` of its draws, near where k - 1 ended,
// with the transit into it (steps 2 to 4): whether one of the round's draws
// was clear.
bool build(const Context& c, Episodes& e, int k, int round) {
    const auto at = static_cast<std::size_t>(k);
    for (int attempt = round * attempts; attempt < (round + 1) * attempts; ++attempt) {
        const Draws d{c.params.seed, static_cast<std::uint64_t>(k), static_cast<std::uint64_t>(attempt)};
        const Segment& last = e.behaviours[at - 1];
        const Segment next = behaviour_for(c, d, evaluate(last, last.duration));
        if (!clear(c, next)) {
            continue;  // step 3: draw again
        }
        std::vector<Segment> into = transit(c, last, next);
        if (into.empty()) {
            continue;  // step 4: draw again
        }
        if (at < e.behaviours.size()) {
            e.behaviours[at] = next;
        } else {
            e.behaviours.push_back(next);
        }
        e.transits[at] = std::move(into);
        return true;
    }
    return false;
}

// Steps 1 to 4: episode 0, the drift about the start, then every episode in
// order, each with the transit into it. An episode that cannot be drawn may
// be boxed in by where the one before ended: back up and draw that one
// again, from its next round of draws (step 4).
Episodes draw_episodes(const Context& c, Vec3 start) {
    Episodes e;
    e.behaviours.reserve(episodes);
    e.transits.resize(episodes);
    const Segment first = drift_about(c.params, Draws{c.params.seed, 0, 0}, start);
    if (!clear(c, first)) {
        throw Refusal("the flight's start, the shape's center, is not clear of the still shapes and inside the "
                      "volume by the clearance");
    }
    e.behaviours.push_back(first);
    std::array<int, episodes> round{};
    int backtracks = 0;
    int k = 1;
    while (k < episodes) {
        const auto at = static_cast<std::size_t>(k);
        if (build(c, e, k, round[at])) {
            ++k;  // on to the next episode, from its first round
            if (k < episodes) {
                round[at + 1] = 0;
            }
            continue;
        }
        const bool no_earlier = k == 1;
        const bool earlier_spent = !no_earlier && round[at - 1] + 1 >= rounds;
        if (no_earlier || earlier_spent || backtracks == most_backtracks) {
            throw Refusal("flight episode " + std::to_string(k) + " could not be drawn clear of the still shapes, "
                          "after " + std::to_string(backtracks) + " redraws of the episodes before it: a target "
                          "with no room to circle it, or a volume too tight");
        }
        ++round[at - 1];  // back to the episode before, from its next round
        ++backtracks;
        --k;
    }
    return e;
}

// Step 5: the transit that closes the loop, drawing the last episode again
// until it can.
std::vector<Segment> close_loop(const Context& c, Episodes& e) {
    std::vector<Segment> closing = transit(c, e.behaviours.back(), e.behaviours.front());
    for (int attempt = 0; closing.empty() && attempt < attempts; ++attempt) {
        // Draw episode 63 afresh, from attempt numbers past the ones its own
        // build used, so the draws differ.
        for (int a = attempts * (attempt + 1); a < attempts * (attempt + 2); ++a) {
            const Draws d{c.params.seed, episodes - 1, static_cast<std::uint64_t>(a)};
            const Segment& last = e.behaviours[episodes - 2];
            const Segment next = behaviour_for(c, d, evaluate(last, last.duration));
            if (!clear(c, next)) {
                continue;
            }
            std::vector<Segment> into = transit(c, last, next);
            if (!into.empty()) {
                e.behaviours[episodes - 1] = next;
                e.transits[episodes - 1] = std::move(into);
                break;
            }
        }
        closing = transit(c, e.behaviours.back(), e.behaviours.front());
    }
    if (closing.empty()) {
        throw Refusal("the flight's loop could not be closed clear of the still shapes");
    }
    return closing;
}

// Step 6: the segments in time, each starting where the one before ends.
void lay_out(Flight& flight, const Episodes& e, const std::vector<Segment>& closing) {
    double t = 0.0;
    const auto lay = [&](const Segment& s) {
        flight.segments.push_back(s);
        flight.segments.back().start = t;
        t += s.duration;
    };
    lay(e.behaviours[0]);
    for (std::size_t k = 1; k < episodes; ++k) {
        for (const Segment& s : e.transits[k]) {
            lay(s);
        }
        lay(e.behaviours[k]);
    }
    for (const Segment& s : closing) {
        lay(s);
    }
    flight.loop = t;
}

// Step 7: the flashes: each swoop's climb; while circling and drifting, at
// the firefly's own rates; at least flash_spacing apart.
FlashSchedule schedule_flashes(std::uint64_t seed, const std::vector<Segment>& segments, double loop) {
    const Draws rates{seed, episodes, 0};
    const double circling = rates.between(Purpose::rate_circle, circling_rate_least, circling_rate_most);
    const double drifting = rates.between(Purpose::rate_drift, drifting_rate_least, drifting_rate_most);
    const auto rate_of = [&](Behaviour b) {
        switch (b) {
        case Behaviour::circle:
            return circling;
        case Behaviour::drift:
            return drifting;
        case Behaviour::transit:
        case Behaviour::swoop:
            return 0.0;
        }
        no_case("schedule_flashes");
    };
    std::vector<double> starts;
    for (std::size_t i = 0; i < segments.size(); ++i) {
        const Segment& s = segments[i];
        if (s.behaviour == Behaviour::swoop) {
            starts.push_back(s.start + swoop_flash_at * s.duration);
            continue;
        }
        const Draws d{seed, episodes + 1, i};
        const auto count = static_cast<int>(rate_of(s.behaviour) * s.duration + d(Purpose::flashes));
        for (int n = 0; n < count; ++n) {
            starts.push_back(s.start + s.duration * draw(seed, episodes + 2, i, static_cast<std::uint64_t>(n)));
        }
    }
    std::sort(starts.begin(), starts.end());
    FlashSchedule flashes;
    flashes.loop = loop;
    for (double s : starts) {
        if (flashes.starts.empty() || s - flashes.starts.back() >= flash_spacing) {
            flashes.starts.push_back(s);
        }
    }
    // The loop's last and first, as far apart.
    while (flashes.starts.size() > 1 && flashes.starts.front() + loop - flashes.starts.back() < flash_spacing) {
        flashes.starts.pop_back();
    }
    return flashes;
}

void check_numbers(const FlightParams& params, float body) {
    constexpr double float_max = std::numeric_limits<float>::max();
    const auto& w = params.weights;
    if (!(std::isfinite(params.speed) && params.speed > 0.0f) ||
        !(std::isfinite(params.clearance) && params.clearance >= 0.0f) || !(std::isfinite(body) && body > 0.0f) ||
        !(w[0] >= 0.0f && w[1] >= 0.0f && w[2] >= 0.0f && w[0] + w[1] + w[2] > 0.0f) ||
        (w[0] > 0.0f && params.targets.empty()) || !(params.most_steps > 0)) {
        throw std::invalid_argument("make_flight: a speed, clearance, body, weights or most_steps out of range");
    }
    for (int axis = 0; axis < 3; ++axis) {
        const double lo = contracts::component(params.volume.min, axis);
        const double hi = contracts::component(params.volume.max, axis);
        if (!(lo < hi) || std::abs(lo) + body > float_max || std::abs(hi) + body > float_max) {
            throw std::invalid_argument("make_flight: a volume whose min is not below its max, or past float's range");
        }
    }
}

}  // namespace

Flight make_flight(const FlightParams& params, contracts::Float3 start, float body,
                   const contracts::Obstacles& obstacles) {
    check_numbers(params, body);
    const Context c{params, static_cast<double>(body), obstacles};
    Episodes e = draw_episodes(c, to_vec(start));  // steps 1 to 4
    const std::vector<Segment> closing = close_loop(c, e);  // step 5
    Flight flight;
    flight.volume = params.volume;
    lay_out(flight, e, closing);  // step 6
    flight.flashes = schedule_flashes(params.seed, flight.segments, flight.loop);  // step 7
    return flight;
}

contracts::Float3 position(const Flight& flight, frame::Seconds t) {
    if (flight.segments.empty() || !(flight.loop > 0.0)) {
        throw std::invalid_argument("position: a flight make_flight did not make, with no segments or no loop");
    }
    // Step E1: t into the loop.
    double tau = std::fmod(t.count(), flight.loop);
    if (tau < 0.0) {
        tau += flight.loop;
    }
    // Step E2: the segment holding it. The first starts at 0 and tau is at
    // least 0, so `after` is past the first.
    const auto after = std::upper_bound(flight.segments.begin(), flight.segments.end(), tau,
                                        [](double value, const Segment& s) { return value < s.start; });
    const Segment& s = *(after - 1);
    // Step E3: its closed form.
    const Vec3 p = evaluate(s, tau - s.start);
    return {static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z)};
}

std::vector<Flight> make_flights(const std::vector<FlightJob>& jobs, const contracts::Obstacles& obstacles) {
    std::vector<Flight> flights(jobs.size());
    // Each job's failure, whatever it threw, kept to be rethrown here, where
    // it can be reported: catching it in the worker keeps any exception from
    // ending the program there. Taking it cannot throw.
    std::vector<std::exception_ptr> failures(jobs.size());
    // Optimization: relaxed. The counter hands out jobs and publishes
    // nothing; joining the threads publishes every flight and failure
    // (CONC.1). One increment per flight, flights milliseconds apart, so
    // there is no contention to shard (CONC.3), and each slot of `flights`
    // and `failures` is written once, by one thread, at its job's end, so
    // neighbours sharing a cache line cost nothing worth padding (CACHE.1).
    std::atomic<std::size_t> next{0};
    const auto work = [&]() noexcept {
        for (std::size_t k = next.fetch_add(1, std::memory_order_relaxed); k < jobs.size();
             k = next.fetch_add(1, std::memory_order_relaxed)) {
            try {
                flights[k] = make_flight(jobs[k].params, jobs[k].start, jobs[k].body, obstacles);
            } catch (...) {
                failures[k] = std::current_exception();
            }
        }
    };
    // Tasks, not threads (CP.4): each thread takes the next unmade flight.
    // The threads are made once a load (CP.41) and joined as this block
    // ends, however it ends (CP.25).
    {
        const std::size_t threads = std::min<std::size_t>(std::max(1u, std::thread::hardware_concurrency()),
                                                          std::max<std::size_t>(jobs.size(), 1));
        std::vector<std::jthread> workers;
        workers.reserve(threads - 1);
        for (std::size_t t = 1; t < threads; ++t) {
            try {
                workers.emplace_back(work);
            } catch (const std::system_error&) {
                break;  // fewer threads: the same flights, made more slowly
            }
        }
        work();
    }
    for (std::size_t k = 0; k < jobs.size(); ++k) {
        if (failures[k]) {
            try {
                std::rethrow_exception(failures[k]);
            } catch (const Refusal& refused) {
                throw FlightsError(k, refused.what());
            }
        }
    }
    return flights;
}

Extent extent(const Flight& flight) {
    return flight.volume;
}

}  // namespace serenity::animation
