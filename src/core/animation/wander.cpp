#include "core/animation/wander.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace serenity::animation {

namespace {

constexpr int terms = 3;

// splitmix64's finalizer (Steele, Lea and Flood 2014): a bijection on 64
// bits whose outputs pass BigCrush for consecutive inputs.
std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

enum class Draw : std::uint64_t { weight = 0, ratio = 1, phase = 2 };

// A number in [0, 1) that is a function of the seed, the axis, the term and
// which of the term's numbers it is, alone: the top 53 bits of the hash, a
// double's mantissa.
double draw(std::uint64_t seed, int axis, int term, Draw which) {
    const std::uint64_t key = (static_cast<std::uint64_t>(axis) << 16) | (static_cast<std::uint64_t>(term) << 8) |
                              static_cast<std::uint64_t>(which);
    const std::uint64_t bits = splitmix64(splitmix64(seed) ^ key);
    return static_cast<double>(bits >> 11) * 0x1.0p-53;
}

double component(contracts::Float3 v, int axis) {
    return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

}  // namespace

Wander make_wander(contracts::Float3 anchor, float reach, float speed, std::uint64_t seed) {
    constexpr double float_max = std::numeric_limits<float>::max();
    if (!(std::isfinite(reach) && reach > 0.0f) || !(std::isfinite(speed) && speed > 0.0f)) {
        throw std::invalid_argument("make_wander: reach and speed must be finite and greater than 0");
    }
    for (int axis = 0; axis < 3; ++axis) {
        const double a = component(anchor, axis);
        if (!std::isfinite(a) || std::abs(a) + static_cast<double>(reach) > float_max) {
            throw std::invalid_argument("make_wander: anchor +/- reach must lie within float's range");
        }
    }
    constexpr double two_pi = 2.0 * std::numbers::pi;

    Wander wander;
    wander.anchor = anchor;
    wander.reach = reach;
    double mean_square_speed = 0.0;  // s0^2, with the ratios g as frequencies
    for (int axis = 0; axis < 3; ++axis) {
        // Step 1: the weights, ratios and phases, from the seed alone.
        double weights[terms];
        double total = 0.0;
        for (int k = 0; k < terms; ++k) {
            weights[k] = 0.5 + 0.5 * draw(seed, axis, k, Draw::weight);
            total += weights[k];
            wander.frequency[axis][k] = 0.4 + 1.2 * draw(seed, axis, k, Draw::ratio);
            wander.phase[axis][k] = two_pi * draw(seed, axis, k, Draw::phase);
        }
        // Step 2: the amplitudes, summing to the reach on each axis.
        for (int k = 0; k < terms; ++k) {
            wander.amplitude[axis][k] = static_cast<double>(reach) * weights[k] / total;
            const double v = two_pi * wander.frequency[axis][k] * wander.amplitude[axis][k];
            mean_square_speed += 0.5 * v * v;
        }
    }
    // Step 3: every frequency scaled so the root-mean-square speed is `speed`.
    const double scale = static_cast<double>(speed) / std::sqrt(mean_square_speed);
    for (auto& axis : wander.frequency) {
        for (double& f : axis) {
            f *= scale;
        }
    }
    return wander;
}

contracts::Float3 position(const Wander& wander, frame::Seconds t) {
    constexpr double two_pi = 2.0 * std::numbers::pi;
    constexpr double float_max = std::numeric_limits<float>::max();
    const double seconds = t.count();
    double p[3];
    for (int axis = 0; axis < 3; ++axis) {
        double sum = component(wander.anchor, axis);
        for (int k = 0; k < terms; ++k) {
            sum += wander.amplitude[axis][k] *
                   std::sin(two_pi * wander.frequency[axis][k] * seconds + wander.phase[axis][k]);
        }
        // The exact path stays within anchor +/- reach, inside float's range
        // (make_wander); the clamp absorbs only the double sums' rounding at
        // that edge, so the narrowing is always defined.
        p[axis] = std::clamp(sum, -float_max, float_max);
    }
    return {static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2])};
}

Extent extent(const Wander& wander) {
    // Rounded outward, so the float box holds every point the path reaches:
    // the scene reader's check against still shapes stays conservative.
    const auto down = [](double x) {
        const float f = static_cast<float>(x);
        return static_cast<double>(f) > x ? std::nextafter(f, -std::numeric_limits<float>::infinity()) : f;
    };
    const auto up = [](double x) {
        const float f = static_cast<float>(x);
        return static_cast<double>(f) < x ? std::nextafter(f, std::numeric_limits<float>::infinity()) : f;
    };
    const double r = wander.reach;
    const contracts::Float3& a = wander.anchor;
    return Extent{
        {down(a.x - r), down(a.y - r), down(a.z - r)},
        {up(a.x + r), up(a.y + r), up(a.z + r)},
    };
}

}  // namespace serenity::animation
