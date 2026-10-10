#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "core/animation/flashes.h"
#include "core/frame/frame_inputs.h"

namespace serenity::animation {

// Axis: Animation (glow).
//
// How bright a light is at t: a factor on its radiance, from `dim` between
// flashes up to 1 at a flash's peak, and 0 before it wakes. A firefly
// blinks: Photinus pyralis flashes for about a third of a second, some five
// seconds apart, and is near dark between. A closed form in t, as motions
// are (motion.h), so any instant renders directly. Which light glows by
// which kind, and its numbers, are the scene's (core/scene/scene.h).
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
//   schedule  flashes at the starts a schedule gives (flashes.h): its
//           opening's once, then its loop's. A flying firefly's schedule is
//           its flight's (flight.h, step 7): on each swoop's climb, and now
//           and then while it circles, drifts or waits in its prelude; the
//           scene reader copies it in, so this kind names no motion. Found by
//           binary search on the opening's starts or the loop's. Its loop
//           must be longer than 0 and its begin finite and 0 or more, which
//           glow() checks (I.5).
//
// Either kind may wake: dark until a moment, `at`, then brightening to its
// glow over `ramp` seconds, so a scene can open in the dark and its lights
// come on one by one. The factor is the kind's g(t) times
//
//   w(t) = 0,                    t < at,
//          smoothstep((t - at) / ramp),   at <= t < at + ramp,
//          1,                    at + ramp <= t,
//
// smoothstep(x) = x^2 (3 - 2x), so the light comes up without a jump in its
// rate; a ramp of 0 is a switch at `at`. The wake scales the whole glow,
// dim and flashes alike: a flash during the ramp is as much dimmer. One
// wake, Wake, kept by each kind's record as an optional member: no wake is
// lit from the start, for any t, as a glow was before wakes (an absence that
// means something, scene.h); its arithmetic is one function both kinds call
// (ES.3). `at` finite, `ramp` finite and 0 or more, which glow() checks.
//
// The factor reaches the GPU per light per frame (metal/scene/
// light_glows.h); the radiance in the scene is the peak.
//
// Not performance-sensitive per light: a few comparisons and one sine, once
// per glowing light per frame; a wake adds a comparison, and a polynomial
// for the length of its ramp.

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

// When a glow wakes (above): dark before `at`, its full glow from at + ramp.
struct Wake {
    double at = 0.0;    // seconds; finite
    double ramp = 0.0;  // seconds; finite, 0 or more
};

struct Rhythm {
    double period = 0.0;  // seconds between flashes; at least least_period
    double flash = 0.0;   // a flash's length; in (0, period / 2]
    float dim = 0.0f;     // brightness between flashes; in [0, 1)
    std::uint64_t seed = 0;
    std::optional<Wake> wake;  // none: lit from the start
};

struct ScheduleGlow {
    FlashSchedule schedule;
    double flash = 0.0;  // a flash's length; in (0, flash_spacing): a schedule's flashes are that far apart
    float dim = 0.0f;    // brightness between flashes; in [0, 1)
    std::optional<Wake> wake;  // none: lit from the start
};

struct Glows {
    std::vector<Rhythm> rhythms;
    std::vector<ScheduleGlow> schedules;
};

// The factor on its radiance the glow `record` gives at `t`: in [dim, 1]
// once awake, in [0, 1] while it wakes, 0 before.
// The record must index its kind's array; the scene reader makes it so, and
// .at() checks it. A switch with no default; a kind no enumerator names is
// refused by std::logic_error, never answered (P.6). Throws
// std::invalid_argument for a t or a glow outside the kinds' preconditions
// above.
float glow(const Glows& glows, GlowRecord record, frame::Seconds t);

}  // namespace serenity::animation
