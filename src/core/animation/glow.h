#pragma once

#include <cstdint>
#include <vector>

#include "core/animation/flashes.h"
#include "core/frame/frame_inputs.h"

namespace serenity::animation {

// Axis: Animation (glow).
//
// How bright a light is at t: a factor on its radiance, from `dim` between
// flashes up to 1 at a flash's peak. A firefly blinks: Photinus pyralis
// flashes for about a third of a second, some five seconds apart, and is
// near dark between. A closed form in t, as motions are (motion.h), so any
// instant renders directly. Which light glows by which kind, and its numbers,
// are the scene's (core/scene/scene.h).
//
// A flash starting at f, of length `flash`, is the smooth pulse
//
//   g(t) = dim + (1 - dim) sin^2(pi (t - f) / flash),   f <= t < f + flash,
//
// and g(t) = dim outside every flash. Flashes never overlap, so at most one
// is lit at any t.
//
// The kinds, a record and an array per kind (Enum.2, C.181), as for motions:
//
//   rhythm  flashes every `period` seconds, each moved by up to a fifth of
//           the period, drawn from the seed per flash: flash k starts at
//           k period + jitter(seed, k). For a light that is not flying, or
//           should not follow its flight. Found for any t from k =
//           floor(t / period) and its neighbors: no state. `flash` at most
//           half the period, so neighbors cannot overlap. The period is at
//           least least_period, which the scene reader holds it to, so k
//           stays an exact integer for any t a run reaches: glow() refuses,
//           by std::invalid_argument, a t whose k would pass 2^62 (some
//           10^8 years at the least period), rather than convert a double
//           past int64's range, which is undefined (ES.46).
//   schedule  flashes at the starts a schedule gives (flashes.h), in its
//           loop. A flying firefly's schedule is its flight's (flight.h,
//           step 7): on each swoop's climb, and now and then while it circles
//           or drifts; the scene reader copies it in, so this kind names no
//           motion. Found by binary search on the starts. Its loop must be
//           longer than 0, which glow() checks (I.5).
//
// The factor reaches the GPU per light per frame (metal/scene/
// light_glows.h); the radiance in the scene is the peak.
//
// Not performance-sensitive per light: a few comparisons and one sine, once
// per glowing light per frame.

// Seconds: the shortest period a rhythm may have, a thousand flashes a
// second, past any firefly's.
inline constexpr double least_period = 1e-3;

enum class GlowKind {
    rhythm,
    schedule,
};

struct GlowRecord {
    GlowKind kind = GlowKind::rhythm;
    std::uint32_t index = 0;  // into that kind's array
};

struct Rhythm {
    double period = 0.0;  // seconds between flashes; at least least_period
    double flash = 0.0;   // a flash's length; in (0, period / 2]
    float dim = 0.0f;     // brightness between flashes; in [0, 1)
    std::uint64_t seed = 0;
};

struct ScheduleGlow {
    FlashSchedule schedule;
    double flash = 0.0;  // a flash's length; in (0, flash_spacing): a schedule's flashes are that far apart
    float dim = 0.0f;    // brightness between flashes; in [0, 1)
};

struct Glows {
    std::vector<Rhythm> rhythms;
    std::vector<ScheduleGlow> schedules;
};

// The factor on its radiance the glow `record` gives at `t`, in [dim, 1].
// The record must index its kind's array; the scene reader makes it so, and
// .at() checks it. A switch with no default; a kind no enumerator names is
// refused by std::logic_error, never answered (P.6). Throws
// std::invalid_argument for a t or a glow outside the kinds' preconditions
// above.
float glow(const Glows& glows, GlowRecord record, frame::Seconds t);

}  // namespace serenity::animation
