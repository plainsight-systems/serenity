#include "core/animation/glow.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "core/animation/draw.h"

namespace serenity::animation {

namespace {

// How far a rhythm's flash may move from its beat: up to a fifth of the
// period either way, so the jitter spans 0.4 of it (ES.45).
constexpr double jitter_span = 0.4;

// The largest |k| a rhythm's flash is counted to: past it, a double's
// floor(t / period) is no longer converted exactly, nor safely, to int64.
constexpr double most_flash_number = 0x1p62;

// The pulse of a flash starting at `start`, at `t`: 0 outside it.
double pulse(double t, double start, double flash) {
    if (t < start || t >= start + flash) {
        return 0.0;
    }
    const double s = std::sin(std::numbers::pi * (t - start) / flash);
    return s * s;
}

float lit(float dim, double brightness) {
    return static_cast<float>(dim + (1.0 - dim) * brightness);
}

float rhythm_glow(const Rhythm& r, double t) {
    // Flash k starts at k period + jitter(k), the jitter within a fifth of
    // the period: only flashes k - 1 .. k + 1 of k = floor(t / period) can
    // be lit at t. k is compared as a double before it is converted (ES.46).
    const double k_real = std::floor(t / r.period);
    if (!(std::abs(k_real) < most_flash_number)) {
        throw std::invalid_argument("glow: a rhythm's time is past what its period counts to, or not a number");
    }
    const auto k0 = static_cast<std::int64_t>(k_real);
    double p = 0.0;
    for (std::int64_t k = k0 - 1; k <= k0 + 1; ++k) {
        const double jitter = (draw(r.seed, static_cast<std::uint64_t>(k)) - 0.5) * jitter_span * r.period;
        p = std::max(p, pulse(t, static_cast<double>(k) * r.period + jitter, r.flash));
    }
    return lit(r.dim, p);
}

float schedule_glow(const ScheduleGlow& g, double t) {
    const std::vector<double>& starts = g.schedule.starts;
    const double loop = g.schedule.loop;
    if (!(loop > 0.0)) {
        throw std::invalid_argument("glow: a schedule whose loop is not longer than 0");
    }
    if (starts.empty()) {
        return g.dim;
    }
    double tau = std::fmod(t, loop);
    if (tau < 0.0) {
        tau += loop;
    }
    // The last flash starting at or before tau; before the first, the
    // loop's last, a loop earlier.
    const auto after = std::upper_bound(starts.begin(), starts.end(), tau);
    const double start = after == starts.begin() ? starts.back() - loop : *(after - 1);
    return lit(g.dim, pulse(tau, start, g.flash));
}

}  // namespace

float glow(const Glows& glows, GlowRecord record, frame::Seconds t) {
    // No default: a kind without a glow fails to compile (-Wswitch,
    // -Werror). .at() checks the record's index, which the scene reader
    // guarantees.
    switch (record.kind) {
    case GlowKind::rhythm:
        return rhythm_glow(glows.rhythms.at(record.index), t.count());
    case GlowKind::schedule:
        return schedule_glow(glows.schedules.at(record.index), t.count());
    }
    throw std::logic_error("glow: a record whose kind is no GlowKind");
}

}  // namespace serenity::animation
