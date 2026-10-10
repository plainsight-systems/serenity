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

// A rhythm's beat at t, k = floor(t / period), checked: a t whose k would
// pass most_flash_number, or is not a number, is refused (glow.h). k is
// compared as a double before it is converted (ES.46).
std::int64_t rhythm_beat(const Rhythm& r, double t) {
    const double k_real = std::floor(t / r.period);
    if (!(std::abs(k_real) < most_flash_number)) {
        throw std::invalid_argument("glow: a rhythm's time is past what its period counts to, or not a number");
    }
    return static_cast<std::int64_t>(k_real);
}

// A rhythm's glow at t, from its checked beat k0. Flash k starts at k period
// + jitter(k), the jitter within a fifth of the period: only flashes k0 - 1
// .. k0 + 1 can be lit at t.
float rhythm_glow(const Rhythm& r, double t, std::int64_t k0) {
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

// A schedule's preconditions (glow.h): its loop longer than 0, its begin
// finite and 0 or more.
void check_schedule(const FlashSchedule& s) {
    if (!(s.loop > 0.0)) {
        throw std::invalid_argument("glow: a schedule whose loop is not longer than 0");
    }
    if (!(std::isfinite(s.begin) && s.begin >= 0.0)) {
        throw std::invalid_argument("glow: a schedule whose begin is not finite and 0 or more");
    }
}

// A checked schedule's glow at t.
float schedule_glow(const ScheduleGlow& g, double t) {
    const FlashSchedule& s = g.schedule;
    if (s.opening.empty() && s.begin == 0.0) {
        // No opening: one loop from 0, for every t, as before openings.
        return s.starts.empty() ? g.dim : lit(g.dim, loop_pulse(g, t));
    }
    return lit(g.dim, opening_pulse(g, t));
}

// w(t), the wake's factor (glow.h), the one function both kinds call
// (ES.3); 1 with no wake, lit from the start. Its tests in the header's
// order: not yet awake; then the ramp over, judged from t - at, never from
// at + ramp, which a large at would round to at and so erase the ramp. A
// ramp of 0 is over at once, t - at >= 0, by that same comparison: a
// switch with no division and no test of its own. Then the smoothstep:
// t - at < ramp there, so x = (t - at) / ramp is at most 1, rounded or not.
double wake_factor(const std::optional<Wake>& wake, double t) {
    if (!wake) {
        return 1.0;
    }
    const Wake& w = *wake;
    if (!(std::isfinite(w.at) && std::isfinite(w.ramp) && w.ramp >= 0.0)) {
        throw std::invalid_argument("glow: a wake whose at is not finite, or whose ramp is not finite and 0 or more");
    }
    if (t < w.at) {
        return 0.0;
    }
    const double since = t - w.at;
    if (since >= w.ramp) {
        return 1.0;
    }
    const double x = since / w.ramp;
    return x * x * (3.0 - 2.0 * x);
}

// The kind's glow `g` scaled by its wake's factor `w`: g itself with no
// wake, or once woken, bit for bit.
float woken(const std::optional<Wake>& wake, double w, float g) {
    return wake ? static_cast<float>(w * static_cast<double>(g)) : g;
}

}  // namespace

float glow(const Glows& glows, GlowRecord record, frame::Seconds t) {
    // No default: a kind without a glow fails to compile (-Wswitch,
    // -Werror). .at() checks the record's index, which the scene reader
    // guarantees. Each kind's checks and its wake's come first, on every
    // call, awake or not, so a bad record never hides behind a wake (I.5,
    // E.2); a light not yet awake answers 0 before its glow is evaluated,
    // which is the work the wake saves (COPY.9: not as an argument, which
    // would be evaluated first).
    const double time = t.count();
    switch (record.kind) {
    case GlowKind::rhythm: {
        const Rhythm& r = glows.rhythms.at(record.index);
        const std::int64_t beat = rhythm_beat(r, time);
        const double w = wake_factor(r.wake, time);
        return w == 0.0 ? 0.0f : woken(r.wake, w, rhythm_glow(r, time, beat));
    }
    case GlowKind::schedule: {
        const ScheduleGlow& g = glows.schedules.at(record.index);
        check_schedule(g.schedule);
        const double w = wake_factor(g.wake, time);
        return w == 0.0 ? 0.0f : woken(g.wake, w, schedule_glow(g, time));
    }
    }
    throw std::logic_error("glow: a record whose kind is no GlowKind");
}

}  // namespace serenity::animation
