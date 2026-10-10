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

#include "core/animation/draw.h"

namespace serenity::animation {

namespace {

constexpr int episodes = 64;
constexpr int attempts = 16;         // draws of an episode in a round (step 3)
constexpr int rounds = 4;            // rounds of draws an episode may have (step 4)
constexpr int most_backtracks = 64;  // redraws of earlier episodes, in all (step 4)
constexpr double delta = flight_delta;  // the sampling bound, meters (step 3)

// A circle's numbers (flight.h, circle), named (ES.45).
constexpr double orbit_gap = 0.04;            // meters past the clearance to the nearest orbit
constexpr double orbit_widest = 0.6;          // meters wider than the nearest the orbit may be
constexpr double tilt_most = 25.0 * std::numbers::pi / 180.0;
constexpr double breathe_most = 0.1;          // of the radius
constexpr double bob_most = 0.05;             // meters
constexpr double pi = std::numbers::pi;

struct V {
    double x = 0.0, y = 0.0, z = 0.0;
};
V operator+(V a, V b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
V operator-(V a, V b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V operator*(double s, V a) { return {s * a.x, s * a.y, s * a.z}; }
double dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
double length(V a) { return std::sqrt(dot(a, a)); }
V cross(V a, V b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
V normalized(V a) { return (1.0 / length(a)) * a; }
V of(contracts::Float3 f) { return {f.x, f.y, f.z}; }
constexpr V up{0.0, 1.0, 0.0};

// Where a number of a segment is (its `numbers`, flight.h), per behaviour.
//   transit  0-2 P0, 3-5 P1, 6-8 T0, 9-11 T1 (Hermite, tangents scaled by the duration)
//   circle   0-2 center, 3-5 u, 6-8 v, 9 r0, 10 theta0, 11 omega, 12 breathe, 13 breathe frequency,
//            14 breathe phase, 15 bob, 16 bob frequency, 17 bob phase
//   swoop    0-2 B0, 3-5 B1, 6-8 B2, 9-11 B3 (cubic Bezier)
//   drift    0-2 center, 3-5 amplitudes, 6-8 frequencies, 9-11 phases
V at3(const Segment& s, int i) {
    return {s.numbers[i], s.numbers[i + 1], s.numbers[i + 2]};
}
void put3(Segment& s, int i, V v) {
    s.numbers[i] = v.x;
    s.numbers[i + 1] = v.y;
    s.numbers[i + 2] = v.z;
}

V bezier(V b0, V b1, V b2, V b3, double u) {
    const double w = 1.0 - u;
    return (w * w * w) * b0 + (3.0 * w * w * u) * b1 + (3.0 * w * u * u) * b2 + (u * u * u) * b3;
}

// Step E3: a segment's closed form at time `tau` into it.
V evaluate(const Segment& s, double tau) {
    switch (s.behaviour) {
    case Behaviour::transit: {
        // Hermite as Bezier: B1 = P0 + T0 / 3, B2 = P1 - T1 / 3.
        const V p0 = at3(s, 0), p1 = at3(s, 3), t0 = at3(s, 6), t1 = at3(s, 9);
        return bezier(p0, p0 + (1.0 / 3.0) * t0, p1 - (1.0 / 3.0) * t1, p1, tau / s.duration);
    }
    case Behaviour::circle: {
        const V c = at3(s, 0), u = at3(s, 3), v = at3(s, 6);
        const V w = cross(u, v);
        const double r = s.numbers[9] * (1.0 + s.numbers[12] * std::sin(2.0 * pi * s.numbers[13] * tau + s.numbers[14]));
        const double theta = s.numbers[10] + s.numbers[11] * tau;
        const double bob = s.numbers[15] * std::sin(2.0 * pi * s.numbers[16] * tau + s.numbers[17]);
        return c + (r * std::cos(theta)) * u + (r * std::sin(theta)) * v + bob * w;
    }
    case Behaviour::swoop:
        return bezier(at3(s, 0), at3(s, 3), at3(s, 6), at3(s, 9), tau / s.duration);
    case Behaviour::drift: {
        const V c = at3(s, 0), a = at3(s, 3), f = at3(s, 6), p = at3(s, 9);
        return c + V{a.x * std::sin(2.0 * pi * f.x * tau + p.x), a.y * std::sin(2.0 * pi * f.y * tau + p.y),
                     a.z * std::sin(2.0 * pi * f.z * tau + p.z)};
    }
    }
    return {};
}

// The velocity, by a central difference of the closed form, which is
// defined, and smooth, past either end of the segment.
V velocity(const Segment& s, double tau) {
    const double h = 1e-5;
    return (1.0 / (2.0 * h)) * (evaluate(s, tau + h) - evaluate(s, tau - h));
}

// An upper bound on a segment's speed, for step 3's sample spacing: the
// convex hull of a Bezier's derivative, and each sinusoid's peak.
double speed_bound(const Segment& s) {
    switch (s.behaviour) {
    case Behaviour::transit: {
        const V p0 = at3(s, 0), p1 = at3(s, 3), t0 = at3(s, 6), t1 = at3(s, 9);
        const V b1 = p0 + (1.0 / 3.0) * t0, b2 = p1 - (1.0 / 3.0) * t1;
        return 3.0 * std::max({length(b1 - p0), length(b2 - b1), length(p1 - b2)}) / s.duration;
    }
    case Behaviour::swoop: {
        const V b0 = at3(s, 0), b1 = at3(s, 3), b2 = at3(s, 6), b3 = at3(s, 9);
        return 3.0 * std::max({length(b1 - b0), length(b2 - b1), length(b3 - b2)}) / s.duration;
    }
    case Behaviour::circle: {
        const double r0 = s.numbers[9];
        return r0 * (1.0 + s.numbers[12]) * std::abs(s.numbers[11]) + r0 * s.numbers[12] * 2.0 * pi * s.numbers[13] +
               s.numbers[15] * 2.0 * pi * s.numbers[16];
    }
    case Behaviour::drift: {
        const V a = at3(s, 3), f = at3(s, 6);
        return 2.0 * pi * length(V{a.x * f.x, a.y * f.y, a.z * f.z});
    }
    }
    return 0.0;
}

struct Context {
    const FlightParams& params;
    double body;
    const contracts::Obstacles& obstacles;
};

// Step 3: whether every sample of the segment keeps the clearance and keeps
// the body inside the volume, the samples spaced so the path between two is
// within delta of them.
bool clear(const Context& c, const Segment& s) {
    const double v = speed_bound(s);
    if (!std::isfinite(v) || !(s.duration > 0.0)) {
        return false;
    }
    const auto steps = static_cast<std::int64_t>(std::ceil(v * s.duration / delta)) + 1;
    if (steps > 1000000) {
        return false;  // a path that long is not a firefly's
    }
    const double margin = c.body + delta;
    const double needed = static_cast<double>(c.params.clearance) + c.body + delta;
    const Extent& box = c.params.volume;
    for (std::int64_t i = 0; i <= steps; ++i) {
        const V p = evaluate(s, s.duration * static_cast<double>(i) / static_cast<double>(steps));
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

// Which number of an episode's draws a draw is.
enum Purpose : std::uint64_t {
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
    std::uint64_t seed;
    std::uint64_t episode;
    std::uint64_t attempt;
    double operator()(Purpose p) const { return draw(seed, episode, attempt, p); }
    double between(Purpose p, double lo, double hi) const { return lo + (hi - lo) * (*this)(p); }
};

// A point near `from`: within `reach` of it, pulled a third of the way
// toward the volume's middle, so a firefly ranges over its volume rather
// than settling at an edge; and inside the volume with room for what is
// drawn there: `below` and `above` it, and `around` it on the level.
V nearby(const Context& c, const Draws& d, V from, double reach, double below, double above, double around) {
    const Extent& box = c.params.volume;
    const double m = c.body + delta + static_cast<double>(c.params.clearance);
    const V middle{0.5 * (double(box.min.x) + box.max.x), 0.5 * (double(box.min.y) + box.max.y),
                   0.5 * (double(box.min.z) + box.max.z)};
    const V toward = from + (1.0 / 3.0) * (middle - from);
    const auto pick = [&](Purpose p, double at, double lo, double hi) {
        const double v = at + reach * (2.0 * d(p) - 1.0);
        return lo < hi ? std::clamp(v, lo, hi) : 0.5 * (lo + hi);
    };
    return {pick(where_x, toward.x, box.min.x + m + around, box.max.x - m - around),
            pick(where_y, toward.y, box.min.y + m + below, box.max.y - m - above),
            pick(where_z, toward.z, box.min.z + m + around, box.max.z - m - around)};
}

Segment drift_about(const FlightParams& params, const Draws& d, V center) {
    Segment s;
    s.behaviour = Behaviour::drift;
    s.duration = d.between(lasting, 2.0, 4.0);
    put3(s, 0, center);
    // A reach of 10 cm on each axis, at a third of the cruising speed: each
    // axis's frequency from its amplitude, f = (speed / 3) / (2 pi a sqrt 3).
    const double a[3] = {d.between(amplitude_x, 0.04, first_drift_reach),
                         d.between(amplitude_y, 0.04, first_drift_reach),
                         d.between(amplitude_z, 0.04, first_drift_reach)};
    const Purpose phases[3] = {phase_x, phase_y, phase_z};
    const double v = static_cast<double>(params.speed) / 3.0;
    for (int i = 0; i < 3; ++i) {
        s.numbers[3 + i] = a[i];
        s.numbers[6 + i] = v / (2.0 * pi * a[i] * std::sqrt(3.0));
        s.numbers[9 + i] = d.between(phases[i], 0.0, 2.0 * pi);
    }
    return s;
}

Segment circle_about(const Context& c, const Draws& d) {
    const FlightParams& params = c.params;
    const auto pick = static_cast<std::size_t>(d(target) * static_cast<double>(params.targets.size()));
    const Target& t = params.targets[std::min(pick, params.targets.size() - 1)];
    const double inner = t.radius + static_cast<double>(params.clearance) + c.body + delta + orbit_gap;
    const double r0 = d.between(radius, inner, inner + orbit_widest);
    // The orbit's center: a drawn lift above the target's, and higher where
    // that leaves the bob no room above the volume's floor (a marble on the
    // table): the floor by the body and delta, and the bob's depth, below it.
    const double floor = static_cast<double>(params.volume.min.y) + c.body + delta;
    const double height =
        std::max(static_cast<double>(t.center.y) + d.between(lift, 0.0, static_cast<double>(t.radius)),
                 floor + bob_most);
    // The orbit's plane: level tilted by up to tilt_most about a level axis,
    // or less where the floor is near: as far as keeps the orbit's lowest
    // point, its radius breathed out and bobbed down, above the floor.
    const double room = height - bob_most - floor;
    const double steepest = std::min(tilt_most, std::asin(std::clamp(room / ((1.0 + breathe_most) * r0), 0.0, 1.0)));
    const double alpha = d.between(tilt, 0.0, steepest);
    const double beta = d.between(azimuth, 0.0, 2.0 * pi);
    const V axis{std::cos(beta), 0.0, std::sin(beta)};
    const V w = normalized(std::cos(alpha) * up + std::sin(alpha) * cross(axis, up));
    const V u = normalized(cross(w, axis));
    const V v = cross(w, u);
    Segment s;
    s.behaviour = Behaviour::circle;
    put3(s, 0, V{static_cast<double>(t.center.x), height, static_cast<double>(t.center.z)});
    put3(s, 3, u);
    put3(s, 6, v);
    s.numbers[9] = r0;
    s.numbers[10] = d.between(turn, 0.0, 2.0 * pi);
    const double omega = static_cast<double>(params.speed) / r0;
    s.numbers[11] = d(direction) < 0.5 ? omega : -omega;
    s.numbers[12] = d.between(breathe, 0.0, breathe_most);
    s.numbers[13] = d.between(breathe_rate, 0.1, 0.4);
    s.numbers[14] = d.between(breathe_phase, 0.0, 2.0 * pi);
    s.numbers[15] = d.between(bob, 0.0, bob_most);
    s.numbers[16] = d.between(bob_rate, 0.3, 0.8);
    s.numbers[17] = d.between(bob_phase, 0.0, 2.0 * pi);
    // Four to nine seconds: a wide orbit makes part of a loop, a tight one
    // more than one, at the cruising speed either way.
    s.duration = d.between(loops, 4.0, 9.0);
    return s;
}

Segment swoop_from(const Context& c, const Draws& d, V from) {
    const double h = d.between(heading, 0.0, 2.0 * pi);
    const V forward{std::cos(h), 0.0, std::sin(h)};
    const double depth = d.between(dip, 0.1, 0.25);
    const double climb = d.between(rise, 0.3, 0.6);
    const double reach = d.between(reach_out, 0.3, 0.6);
    // Its start, with room for the whole J: its dip below, its reach about,
    // and above, its climb and the way out of it: it ends climbing, and the
    // transit that leaves it rises on before it turns (headroom 35 cm).
    const V start = nearby(c, d, from, 1.0, 1.6 * depth, climb + 0.35, reach);
    // A J: forward and down, then up steeply.
    const V b0 = start;
    const V b1 = start + (0.3 * reach) * forward - (1.6 * depth) * up;
    const V b2 = start + (0.8 * reach) * forward - (0.8 * depth) * up;
    const V b3 = start + reach * forward + climb * up;
    Segment s;
    s.behaviour = Behaviour::swoop;
    put3(s, 0, b0);
    put3(s, 3, b1);
    put3(s, 6, b2);
    put3(s, 9, b3);
    const double chord = length(b1 - b0) + length(b2 - b1) + length(b3 - b2);
    s.duration = std::clamp(chord / (0.8 * static_cast<double>(c.params.speed)), 1.0, 2.5);
    return s;
}

// Step 2: episode k's behaviour, drawn near where the last ended.
Segment behaviour_for(const Context& c, const Draws& d, V from) {
    const auto& w = c.params.weights;
    const double total = static_cast<double>(w[0]) + w[1] + w[2];
    const double pick = d(choose) * total;
    if (pick < w[0]) {
        return circle_about(c, d);
    }
    if (pick < static_cast<double>(w[0]) + w[1]) {
        return swoop_from(c, d, from);
    }
    // A drift's reach is at most 10 cm on each axis.
    return drift_about(c.params, d, nearby(c, d, from, 1.0, 0.1, 0.1, 0.1));
}

// A transit from (p0, v0) to (p1, v1) over a duration from its length.
Segment hermite(const FlightParams& params, V p0, V v0, V p1, V v1) {
    const double chord = length(p1 - p0);
    Segment s;
    s.behaviour = Behaviour::transit;
    s.duration = std::max(0.4, 1.2 * chord / static_cast<double>(params.speed));
    put3(s, 0, p0);
    put3(s, 3, p1);
    put3(s, 6, s.duration * v0);
    put3(s, 9, s.duration * v1);
    return s;
}

// Step 4: the transit from `from` to `to`, direct or over the waypoint at
// the volume's top; empty if neither is clear.
std::vector<Segment> transit(const Context& c, const Segment& from, const Segment& to) {
    const V p0 = evaluate(from, from.duration), v0 = velocity(from, from.duration);
    const V p1 = evaluate(to, 0.0), v1 = velocity(to, 0.0);
    const Segment direct = hermite(c.params, p0, v0, p1, v1);
    if (clear(c, direct)) {
        return {direct};
    }
    const double top = c.params.volume.max.y - c.body - delta - 0.05;
    const V mid{0.5 * (p0.x + p1.x), top, 0.5 * (p0.z + p1.z)};
    V level = p1 - p0;
    level.y = 0.0;
    const V across = length(level) > 1e-6 ? (static_cast<double>(c.params.speed) / length(level)) * level : V{};
    const Segment rise = hermite(c.params, p0, v0, mid, across);
    const Segment fall = hermite(c.params, mid, across, p1, v1);
    if (clear(c, rise) && clear(c, fall)) {
        return {rise, fall};
    }
    return {};
}

double within(contracts::Float3 v, int axis) {
    return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

}  // namespace

Flight make_flight(const FlightParams& params, contracts::Float3 start, float body,
                   const contracts::Obstacles& obstacles) {
    constexpr double float_max = std::numeric_limits<float>::max();
    const auto& w = params.weights;
    if (!(std::isfinite(params.speed) && params.speed > 0.0f) ||
        !(std::isfinite(params.clearance) && params.clearance >= 0.0f) || !(std::isfinite(body) && body > 0.0f) ||
        !(w[0] >= 0.0f && w[1] >= 0.0f && w[2] >= 0.0f && w[0] + w[1] + w[2] > 0.0f) ||
        (w[0] > 0.0f && params.targets.empty())) {
        throw std::invalid_argument("make_flight: a speed, clearance, body or weights out of range");
    }
    for (int axis = 0; axis < 3; ++axis) {
        const double lo = within(params.volume.min, axis), hi = within(params.volume.max, axis);
        if (!(lo < hi) || std::abs(lo) + body > float_max || std::abs(hi) + body > float_max) {
            throw std::invalid_argument("make_flight: a volume whose min is not below its max, or past float's range");
        }
    }
    const Context c{params, static_cast<double>(body), obstacles};

    // Steps 1 to 4: the episodes in order, each with the transit into it.
    std::vector<Segment> behaviours;
    std::vector<std::vector<Segment>> transits;  // transits[k] leads into behaviours[k], k >= 1
    behaviours.reserve(episodes);
    transits.resize(episodes);
    {
        // Episode 0: a drift about the start.
        const Draws d{params.seed, 0, 0};
        const Segment first = drift_about(params, d, of(start));
        if (!clear(c, first)) {
            throw std::invalid_argument("the flight's start, the shape's center, is not clear of the still shapes "
                                        "and inside the volume by the clearance");
        }
        behaviours.push_back(first);
    }
    const auto build = [&](int k, int round) {
        for (int attempt = round * attempts; attempt < (round + 1) * attempts; ++attempt) {
            const Draws d{params.seed, static_cast<std::uint64_t>(k), static_cast<std::uint64_t>(attempt)};
            const Segment& last = behaviours[static_cast<std::size_t>(k - 1)];
            const Segment next = behaviour_for(c, d, evaluate(last, last.duration));
            if (!clear(c, next)) {
                continue;  // step 3: draw again
            }
            std::vector<Segment> into = transit(c, last, next);
            if (into.empty()) {
                continue;  // step 4: draw again
            }
            if (static_cast<std::size_t>(k) < behaviours.size()) {
                behaviours[static_cast<std::size_t>(k)] = next;
            } else {
                behaviours.push_back(next);
            }
            transits[static_cast<std::size_t>(k)] = std::move(into);
            return true;
        }
        return false;
    };
    // An episode that cannot be drawn may be boxed in by where the one before
    // ended: back up and draw that one again, from its next round of draws
    // (step 4).
    std::array<int, episodes> round{};
    int backtracks = 0;
    for (int k = 1; k < episodes;) {
        if (build(k, round[static_cast<std::size_t>(k)])) {
            if (++k < episodes) {
                round[static_cast<std::size_t>(k)] = 0;
            }
            continue;
        }
        if (k == 1 || round[static_cast<std::size_t>(k - 1)] + 1 >= rounds || backtracks == most_backtracks) {
            throw std::invalid_argument("flight episode " + std::to_string(k) + " could not be drawn clear of the "
                                        "still shapes, after " + std::to_string(backtracks) +
                                        " redraws of the episodes before it: a target with no room to circle it, "
                                        "or a volume too tight");
        }
        ++round[static_cast<std::size_t>(k - 1)];
        ++backtracks;
        --k;
    }
    // Step 5: close the loop, drawing the last episode again until it can.
    std::vector<Segment> closing = transit(c, behaviours.back(), behaviours.front());
    for (int attempt = 0; closing.empty() && attempt < attempts; ++attempt) {
        // Draw episode 63 afresh, from attempt numbers past the ones its own
        // build used, so the draws differ.
        for (int a = attempts * (attempt + 1); a < attempts * (attempt + 2); ++a) {
            const Draws d{params.seed, episodes - 1, static_cast<std::uint64_t>(a)};
            const Segment& last = behaviours[episodes - 2];
            const Segment next = behaviour_for(c, d, evaluate(last, last.duration));
            std::vector<Segment> into;
            if (clear(c, next) && !(into = transit(c, last, next)).empty()) {
                behaviours[episodes - 1] = next;
                transits[episodes - 1] = std::move(into);
                break;
            }
        }
        closing = transit(c, behaviours.back(), behaviours.front());
    }
    if (closing.empty()) {
        throw std::invalid_argument("the flight's loop could not be closed clear of the still shapes");
    }

    // Step 6: the segments in time.
    Flight flight;
    flight.volume = params.volume;
    double t = 0.0;
    const auto lay = [&](Segment s) {
        s.start = t;
        t += s.duration;
        flight.segments.push_back(s);
    };
    lay(behaviours[0]);
    for (int k = 1; k < episodes; ++k) {
        for (const Segment& s : transits[static_cast<std::size_t>(k)]) {
            lay(s);
        }
        lay(behaviours[static_cast<std::size_t>(k)]);
    }
    for (const Segment& s : closing) {
        lay(s);
    }
    flight.loop = t;

    // Step 7: the flashes: each swoop's climb; while circling and drifting,
    // at the firefly's own rates.
    const Draws rates{params.seed, episodes, 0};
    const double circling = rates.between(rate_circle, 0.1, 0.25);
    const double drifting = rates.between(rate_drift, 0.02, 0.08);
    std::vector<double> starts;
    for (std::size_t i = 0; i < flight.segments.size(); ++i) {
        const Segment& s = flight.segments[i];
        if (s.behaviour == Behaviour::swoop) {
            starts.push_back(s.start + 0.55 * s.duration);
            continue;
        }
        const double rate = s.behaviour == Behaviour::circle ? circling : (s.behaviour == Behaviour::drift ? drifting : 0.0);
        const Draws d{params.seed, episodes + 1, i};
        const auto count = static_cast<int>(rate * s.duration + d(flashes));
        for (int n = 0; n < count; ++n) {
            starts.push_back(s.start + s.duration * draw(params.seed, episodes + 2, i, static_cast<std::uint64_t>(n)));
        }
    }
    std::sort(starts.begin(), starts.end());
    flight.flashes.loop = flight.loop;
    for (double s : starts) {
        if (flight.flashes.starts.empty() || s - flight.flashes.starts.back() >= 1.0) {
            flight.flashes.starts.push_back(s);
        }
    }
    // The loop's last and first, a second apart too.
    while (flight.flashes.starts.size() > 1 &&
           flight.flashes.starts.front() + flight.loop - flight.flashes.starts.back() < 1.0) {
        flight.flashes.starts.pop_back();
    }
    return flight;
}

contracts::Float3 position(const Flight& flight, frame::Seconds t) {
    // Step E1: t into the loop.
    double tau = std::fmod(t.count(), flight.loop);
    if (tau < 0.0) {
        tau += flight.loop;
    }
    // Step E2: the segment holding it.
    const auto after = std::upper_bound(flight.segments.begin(), flight.segments.end(), tau,
                                        [](double value, const Segment& s) { return value < s.start; });
    const Segment& s = *(after - 1);
    // Step E3: its closed form.
    const V p = evaluate(s, tau - s.start);
    return {static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z)};
}

std::vector<Flight> make_flights(const std::vector<FlightJob>& jobs, const contracts::Obstacles& obstacles) {
    std::vector<Flight> flights(jobs.size());
    // Each job's failure, whatever it threw, kept to be rethrown here, where
    // it can be reported (E.17): catching it in the worker keeps any
    // exception from ending the program there. Taking it cannot throw.
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
            } catch (const std::invalid_argument& refused) {
                throw FlightsError(k, "flight " + std::to_string(k) + ": " + refused.what());
            }
        }
    }
    return flights;
}

Extent extent(const Flight& flight) {
    return flight.volume;
}

}  // namespace serenity::animation
