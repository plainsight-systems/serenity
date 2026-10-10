#include "core/animation/glow.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <optional>
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

// The pulse lit at `t` of a schedule glow with no opening, its loop's starts
// not empty, repeating every loop for any t, negative too: a schedule as it
// was before openings, which one without an opening still is (flashes.h).
// The glow whole, not its loop, flash and t as three doubles side by side
// (I.24).
double loop_pulse(const ScheduleGlow& g, double t) {
    const std::vector<double>& starts = g.schedule.starts;
    const double loop = g.schedule.loop;
    double tau = std::fmod(t, loop);
    if (tau < 0.0) {
        tau += loop;
    }
    // The last flash starting at or before tau; before the first, the
    // loop's last, a loop earlier.
    const auto after = std::upper_bound(starts.begin(), starts.end(), tau);
    const double start = after == starts.begin() ? starts.back() - loop : *(after - 1);
    return pulse(tau, start, g.flash);
}

// The pulse lit at `t` of a schedule glow with an opening (flashes.h): the
// latest start at or before t among the opening's, once each, and begin + k
// loop + each loop start, k = 0, 1, ...; none before the first. Each found
// by binary search, on the opening's starts or the loop's.
double opening_pulse(const ScheduleGlow& g, double t) {
    const FlashSchedule& s = g.schedule;
    const double flash = g.flash;
    const std::vector<double>& opening = s.opening;
    if (t < s.begin) {
        // No loop start is at or before t: k = 0's first is at begin or later.
        const auto after = std::upper_bound(opening.begin(), opening.end(), t);
        return after == opening.begin() ? 0.0 : pulse(t, *(after - 1), flash);
    }
    // t - begin is 0 or more here, so tau is too.
    const double since = t - s.begin;
    const double tau = std::fmod(since, s.loop);
    const auto after = std::upper_bound(s.starts.begin(), s.starts.end(), tau);
    if (after != s.starts.begin()) {
        return pulse(tau, *(after - 1), flash);
    }
    // Before the loop's first: on any pass but the first, the loop's last, a
    // loop earlier; on the first, the opening's last, never a phantom of the
    // loop's from before begin.
    if (since >= s.loop && !s.starts.empty()) {
        return pulse(tau, s.starts.back() - s.loop, flash);
    }
    return opening.empty() ? 0.0 : pulse(t, opening.back(), flash);
}

float schedule_glow(const ScheduleGlow& g, double t) {
    const FlashSchedule& s = g.schedule;
    if (!(s.loop > 0.0)) {
        throw std::invalid_argument("glow: a schedule whose loop is not longer than 0");
    }
    if (!(std::isfinite(s.begin) && s.begin >= 0.0)) {
        throw std::invalid_argument("glow: a schedule whose begin is not finite and 0 or more");
    }
    if (s.opening.empty() && s.begin == 0.0) {
        // No opening: one loop from 0, for every t, as before openings.
        return s.starts.empty() ? g.dim : lit(g.dim, loop_pulse(g, t));
    }
    return lit(g.dim, opening_pulse(g, t));
}

// w(t), the wake's factor (glow.h), the one function both kinds call
// (ES.3). Its tests in the header's order: a woken light first, which also
// takes a ramp of 0 at t = at, with no division; then a light not yet
// awake; then the ramp's smoothstep. x is held to 1 at most: (t - at) / ramp
// is rounded, and the smoothstep past 1 turns back down.
double wake_factor(const Wake& w, double t) {
    if (!(std::isfinite(w.at) && std::isfinite(w.ramp) && w.ramp >= 0.0)) {
        throw std::invalid_argument("glow: a wake whose at is not finite, or whose ramp is not finite and 0 or more");
    }
    if (t >= w.at + w.ramp) {
        return 1.0;
    }
    if (t < w.at) {
        return 0.0;
    }
    const double x = std::min((t - w.at) / w.ramp, 1.0);
    return x * x * (3.0 - 2.0 * x);
}

// The kind's glow `g` at `t`, scaled by its wake if it has one.
float woken(const std::optional<Wake>& wake, float g, double t) {
    if (!wake) {
        return g;  // lit from the start, as before wakes
    }
    return static_cast<float>(wake_factor(*wake, t) * static_cast<double>(g));
}

}  // namespace

float glow(const Glows& glows, GlowRecord record, frame::Seconds t) {
    // No default: a kind without a glow fails to compile (-Wswitch,
    // -Werror). .at() checks the record's index, which the scene reader
    // guarantees. The kind's glow is found first, so its checks of t hold
    // whether the light is awake or not.
    switch (record.kind) {
    case GlowKind::rhythm: {
        const Rhythm& r = glows.rhythms.at(record.index);
        return woken(r.wake, rhythm_glow(r, t.count()), t.count());
    }
    case GlowKind::schedule: {
        const ScheduleGlow& g = glows.schedules.at(record.index);
        return woken(g.wake, schedule_glow(g, t.count()), t.count());
    }
    }
    throw std::logic_error("glow: a record whose kind is no GlowKind");
}

}  // namespace serenity::animation
