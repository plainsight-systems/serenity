#include "core/animation/wander.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>

#include "core/animation/draw.h"

namespace serenity::animation {

namespace {

constexpr int terms = 3;
constexpr double two_pi = 2.0 * std::numbers::pi;
constexpr double float_max = std::numeric_limits<float>::max();

// Step 1's draws: the weight in [0.5, 1) and the frequency ratio in
// [0.4, 1.6) of a term, each as least + span x a draw.
constexpr double least_weight = 0.5;
constexpr double weight_span = 0.5;
constexpr double least_ratio = 0.4;
constexpr double ratio_span = 1.2;

enum class Draw : std::uint64_t { weight = 0, ratio = 1, phase = 2 };

// A number in [0, 1) that is a function of the seed, the axis, the term and
// which of the term's numbers it is, alone: the top 53 bits of the hash, a
// double's mantissa. The values of Draw and the key's layout are the
// wander's keying, which its tests pin.
double wander_draw(std::uint64_t seed, int axis, int term, Draw which) {
    const std::uint64_t key = (static_cast<std::uint64_t>(axis) << 16) | (static_cast<std::uint64_t>(term) << 8) |
                              static_cast<std::uint64_t>(which);
    const std::uint64_t bits = splitmix64(splitmix64(seed) ^ key);
    return static_cast<double>(bits >> 11) * 0x1.0p-53;
}

// `x` rounded to a float no greater, and no less: so a float box holds every
// point of the double one it rounds (F.10).
float round_down(double x) noexcept {
    const auto f = static_cast<float>(x);
    return static_cast<double>(f) > x ? std::nextafter(f, -std::numeric_limits<float>::infinity()) : f;
}

float round_up(double x) noexcept {
    const auto f = static_cast<float>(x);
    return static_cast<double>(f) < x ? std::nextafter(f, std::numeric_limits<float>::infinity()) : f;
}

// The box `center` +/- `grown` on each axis, in double, rounded outward.
contracts::Box outward(contracts::Float3 center, double grown) {
    return contracts::Box{
        {round_down(center.x - grown), round_down(center.y - grown), round_down(center.z - grown)},
        {round_up(center.x + grown), round_up(center.y + grown), round_up(center.z + grown)},
    };
}

}  // namespace

Wander make_wander(const WanderParams& params, float body, const contracts::Obstacles& obstacles) {
    const float reach = params.reach;
    const float speed = params.speed;
    if (!(std::isfinite(reach) && reach > 0.0f) || !(std::isfinite(speed) && speed > 0.0f) ||
        !(std::isfinite(body) && body >= 0.0f)) {
        throw std::invalid_argument(
            "make_wander: reach and speed must be finite and greater than 0, and body finite and 0 or more");
    }
    for (int axis = 0; axis < 3; ++axis) {
        const double a = contracts::component(params.anchor, axis);
        if (!std::isfinite(a) || std::abs(a) + static_cast<double>(reach) + static_cast<double>(body) > float_max) {
            throw std::invalid_argument("make_wander: anchor +/- (reach + body) must lie within float's range");
        }
    }
    // Wherever it wanders, its body touches no still shape: the reach grown
    // by the body, rounded outward, against every still shape.
    if (obstacles.touches(outward(params.anchor, static_cast<double>(reach) + static_cast<double>(body)))) {
        throw Refusal("its wander could carry it into a still shape");
    }

    Wander wander{.anchor = params.anchor, .reach = reach};
    double mean_square_speed = 0.0;  // s0^2, with the ratios g as frequencies
    for (int axis = 0; axis < 3; ++axis) {
        const auto a = static_cast<std::size_t>(axis);
        // Step 1: the weights, ratios and phases, from the seed alone.
        std::array<double, terms> weights{};
        double total = 0.0;
        for (int k = 0; k < terms; ++k) {
            const auto i = static_cast<std::size_t>(k);
            weights[i] = least_weight + weight_span * wander_draw(params.seed, axis, k, Draw::weight);
            total += weights[i];
            wander.frequency[a][i] = least_ratio + ratio_span * wander_draw(params.seed, axis, k, Draw::ratio);
            wander.phase[a][i] = two_pi * wander_draw(params.seed, axis, k, Draw::phase);
        }
        // Step 2: the amplitudes, summing to the reach on each axis.
        for (std::size_t i = 0; i < terms; ++i) {
            wander.amplitude[a][i] = static_cast<double>(reach) * weights[i] / total;
            const double v = two_pi * wander.frequency[a][i] * wander.amplitude[a][i];
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
    const double seconds = t.count();
    std::array<double, 3> p{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        double sum = contracts::component(wander.anchor, static_cast<int>(axis));
        for (std::size_t k = 0; k < terms; ++k) {
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
    return outward(wander.anchor, static_cast<double>(wander.reach));
}

}  // namespace serenity::animation
