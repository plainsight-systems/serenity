#include "core/scene/swarm.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

#include "core/animation/draw.h"

namespace serenity::scene {

namespace {

// Step 2p's numbers (swarm.h), named (ES.45): how near the perch's height
// above a surface the trace must come, and the central difference's step
// either side of the perch, a difference over perch_gap / 2 in all.
constexpr double perch_tolerance = animation::perch_gap / 10.0;
constexpr double facing_step = animation::perch_gap / 4.0;

// Step 2's range on each axis: the volume shrunk by radius + delta +
// first_drift_reach on every side.
struct Range {
    std::array<double, 3> low{};
    std::array<double, 3> high{};
};

Range shrunk(const Swarm& swarm, std::uint32_t i) {
    const double margin = static_cast<double>(swarm.radius) + animation::flight_delta + animation::first_drift_reach;
    Range range;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        range.low[axis] = contracts::component(swarm.flight.volume.min, static_cast<int>(axis)) + margin;
        range.high[axis] = contracts::component(swarm.flight.volume.max, static_cast<int>(axis)) - margin;
        if (!(range.low[axis] < range.high[axis])) {
            throw animation::MotionError("firefly " + std::to_string(i) +
                                         ": the volume is too small for a firefly to drift in");
        }
    }
    return range;
}

// The distance step 2's start must keep from every still surface, for its
// whole first drift to keep the flight's clearance.
double needed_of(const Swarm& swarm) {
    return static_cast<double>(swarm.flight.clearance) + static_cast<double>(swarm.radius) + animation::flight_delta +
           animation::first_drift_reach * std::sqrt(3.0);
}

// A point drawn within the swarm's volume, or a perch box, both inside the
// world, so within float's range (ES.46).
contracts::Float3 to_float(const std::array<double, 3>& p) noexcept {
    return {static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2])};
}

// What every step of one firefly reads: its swarm, its number and seed,
// step 2's range and distance, and the still shapes. One argument for each
// step rather than five or six (I.23); read through references, so never
// copied (C.12).
struct Drawing {
    Drawing(const Swarm& s, std::uint32_t n, const contracts::Obstacles& o)
        : swarm(s), i(n), seed(firefly_seed(s.flight.seed, n)), range(shrunk(s, n)), needed(needed_of(s)),
          obstacles(o) {}
    Drawing(const Drawing&) = delete;
    Drawing& operator=(const Drawing&) = delete;
    Drawing(Drawing&&) = delete;
    Drawing& operator=(Drawing&&) = delete;
    ~Drawing() = default;

    const Swarm& swarm;
    std::uint32_t i;        // which firefly
    std::uint64_t seed;     // its own (step 1)
    Range range;            // step 2's
    double needed;          // step 2's distance from every still surface
    const contracts::Obstacles& obstacles;
};

// Step 2: a start in `range`, the draws keyed (seed_i, attempt, axis), as
// they were before starts had kinds.
contracts::Float3 start_in(const Drawing& d, const Range& range) {
    for (int attempt = 0; attempt < start_attempts; ++attempt) {
        std::array<double, 3> p{};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const double u =
                animation::draw(d.seed, static_cast<std::uint64_t>(attempt), static_cast<std::uint64_t>(axis));
            p[axis] = range.low[axis] + u * (range.high[axis] - range.low[axis]);
        }
        const contracts::Float3 start = to_float(p);
        if (d.obstacles.distance(start) >= d.needed) {
            return start;
        }
    }
    throw animation::MotionError("firefly " + std::to_string(d.i) + ": no start clear of the still shapes in " +
                                 std::to_string(start_attempts) + " draws; the volume is too full");
}

// A perch and the loop start straight above it (step 2p).
struct Perched {
    contracts::Float3 perch{};
    contracts::Float3 start{};
};

// The vertical line a perch is traced down: its x and z, named, not two
// doubles side by side (I.24).
struct Vertical {
    double x = 0.0;
    double z = 0.0;
};

// Step 2p's sphere trace down `line` from the box's top: the first point at
// radius + perch_gap above a surface, within perch_tolerance, or none: the
// box's top inside a shape or within that height of one, no surface above
// the box's floor, or no answer in perch_steps steps. The distance is exact
// (contract 11), so no step passes a surface.
std::optional<contracts::Float3> trace_down(const Drawing& d, const contracts::Box& box, Vertical line) {
    const double height = static_cast<double>(d.swarm.radius) + animation::perch_gap;
    double y = box.max.y;
    for (int step = 0; step < perch_steps; ++step) {
        const contracts::Float3 at = to_float({line.x, y, line.z});
        const double ahead = d.obstacles.distance(at) - height;
        if (std::abs(ahead) <= perch_tolerance) {
            return at;
        }
        if (!(ahead > 0.0)) {
            return std::nullopt;  // the top inside a shape, or nearer one than the perch's height
        }
        y -= ahead;
        if (y < static_cast<double>(box.min.y)) {
            return std::nullopt;  // never below the box's floor
        }
    }
    return std::nullopt;
}

// Whether the surface under `perch` faces up: the distance's rise along y,
// by a central difference over perch_gap / 2, at least cos(perch_steepest).
bool faces_up(const Drawing& d, contracts::Float3 perch) {
    // Narrowed to float within a millimeter of a perch in its box, inside
    // the world: within float's range (ES.46).
    const double y = perch.y;
    const contracts::Float3 above{perch.x, static_cast<float>(y + facing_step), perch.z};
    const contracts::Float3 below{perch.x, static_cast<float>(y - facing_step), perch.z};
    // Over the float points' own span, which rounding may make other than
    // perch_gap / 2.
    const double span = static_cast<double>(above.y) - static_cast<double>(below.y);
    const double rise = (d.obstacles.distance(above) - d.obstacles.distance(below)) / span;
    return rise >= std::cos(perch_steepest);
}

// Step 2p, attempt `attempt`, from draws keyed (seed_i, perch_draws,
// attempt, axis): a perch in the box and its loop start, or none.
std::optional<Perched> perch_from(const Drawing& d, const PerchStart& start, std::uint64_t attempt) {
    const auto pick = [&](std::uint64_t axis) { return animation::draw(d.seed, perch_draws, attempt, axis); };
    const contracts::Box& box = start.box;
    const Vertical line{.x = box.min.x + pick(0) * (static_cast<double>(box.max.x) - box.min.x),
                        .z = box.min.z + pick(2) * (static_cast<double>(box.max.z) - box.min.z)};
    const std::optional<contracts::Float3> perch = trace_down(d, box, line);
    if (!perch || !faces_up(d, *perch)) {
        return std::nullopt;
    }
    // Its loop's start straight above it, within step 2's range, and clear
    // as step 2's start must be.
    const Range& range = d.range;
    const contracts::Float3 above =
        to_float({std::clamp(static_cast<double>(perch->x), range.low[0], range.high[0]),
                  range.low[1] + pick(1) * (range.high[1] - range.low[1]),
                  std::clamp(static_cast<double>(perch->z), range.low[2], range.high[2])});
    if (!(d.obstacles.distance(above) >= d.needed)) {
        return std::nullopt;
    }
    return Perched{.perch = *perch, .start = above};
}

Perched perch_for(const Drawing& d, const PerchStart& start) {
    for (int attempt = 0; attempt < start_attempts; ++attempt) {
        if (const std::optional<Perched> found = perch_from(d, start, static_cast<std::uint64_t>(attempt))) {
            return *found;
        }
    }
    throw animation::MotionError("firefly " + std::to_string(d.i) + ": no perch found in the perch box in " +
                                 std::to_string(start_attempts) + " draws: a box with no upward-facing surface in "
                                 "it, its top inside a shape, or no clear loop start above its perches");
}

// The numbers make_firefly takes as given, which the scene reader checks
// first (I.5): refused as a std::invalid_argument here, never drawn from.
void check_numbers(const Swarm& swarm) {
    const auto at_least_zero = [](double v) { return std::isfinite(v) && v >= 0.0; };
    const bool start_in_range = std::visit(
        animation::Visit{[](const AirStart&) { return true; },
                         [](const AboveStart& a) { return std::isfinite(a.depth) && a.depth > 0.0; },
                         [&](const PerchStart& p) {
                             const bool box = p.box.min.x < p.box.max.x && p.box.min.y < p.box.max.y &&
                                              p.box.min.z < p.box.max.z;
                             return box && at_least_zero(p.linger_least) && std::isfinite(p.linger_most) &&
                                    p.linger_most >= p.linger_least;
                         }},
        swarm.start);
    const auto wake_ok = [&](const SwarmWake& w) {
        return at_least_zero(w.from) && std::isfinite(w.to) && w.to >= w.from && std::isfinite(w.power) &&
               w.power > 0.0 && at_least_zero(w.ramp);
    };
    if (!start_in_range || (swarm.wake && !wake_ok(*swarm.wake))) {
        throw std::invalid_argument("make_firefly: a swarm's start or wake out of range");
    }
}

}  // namespace

std::uint64_t firefly_seed(std::uint64_t swarm_seed, std::uint32_t i) {
    // Step 1.
    return animation::splitmix64(animation::splitmix64(swarm_seed) ^ i);
}

Firefly make_firefly(const Swarm& swarm, std::uint32_t i, const contracts::Obstacles& obstacles) {
    check_numbers(swarm);
    const Drawing d{swarm, i, obstacles};
    Firefly firefly;

    // Step 3: its wake, the share awake by t ((t - from) / (to - from))^power
    // from u's uniform draw; held to `to`, which rounding could pass.
    double wakes = 0.0;
    if (swarm.wake) {
        const SwarmWake& w = *swarm.wake;
        const double u = animation::draw(d.seed, wake_draws);
        wakes = std::min(w.from + (w.to - w.from) * std::pow(u, 1.0 / w.power), w.to);
        firefly.wake = animation::Wake{.at = wakes, .ramp = w.ramp};
    }

    // Steps 2 and 2p, and the prelude each kind's start makes (step 4).
    std::visit(animation::Visit{[&](const AirStart&) {
                                    firefly.start = start_in(d, d.range);
                                    firefly.prelude = animation::NoPrelude{};
                                },
                                [&](const AboveStart& above) {
                                    // Its height within the top `depth` of the
                                    // range, or all of it; the draws otherwise air's.
                                    Range top = d.range;
                                    top.low[1] = std::max(d.range.low[1], d.range.high[1] - above.depth);
                                    firefly.start = start_in(d, top);
                                    firefly.prelude = animation::Hold{.until = wakes};
                                },
                                [&](const PerchStart& perch) {
                                    const Perched found = perch_for(d, perch);
                                    // Its linger, held to the most, which rounding could pass.
                                    const double linger = std::min(
                                        perch.linger_least + (perch.linger_most - perch.linger_least) *
                                                                 animation::draw(d.seed, linger_draws),
                                        perch.linger_most);
                                    firefly.start = found.start;
                                    firefly.prelude = animation::Perch{.at = found.perch, .until = wakes + linger};
                                }},
               swarm.start);
    return firefly;
}

}  // namespace serenity::scene
