#pragma once

#include <cstdint>
#include <vector>

#include "core/animation/motion.h"
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
//           half the period, so neighbors cannot overlap.
//   flight  flashes when its flight says (flight.h, step 7): on each swoop's
//           climb, and now and then while it circles or drifts. Its light's
//           shape must fly. Found by binary search on the flight's flashes,
//           in its loop.
//
// The factor reaches the GPU per light per frame (metal/scene/
// light_glows.h); the radiance in the scene is the peak.
//
// Not performance-sensitive per light: a few comparisons and one sine, once
// per glowing light per frame.

enum class GlowKind : std::uint32_t {
    rhythm = 0,
    flight = 1,
};

struct GlowRecord {
    GlowKind kind;
    std::uint32_t index;  // into that kind's array
};

struct Rhythm {
    double period = 0.0;  // seconds between flashes; > 0
    double flash = 0.0;   // a flash's length; in (0, period / 2]
    float dim = 0.0f;     // brightness between flashes; in [0, 1)
    std::uint64_t seed = 0;
};

struct FlightGlow {
    std::uint32_t flight = 0;  // the flight it follows, an index into Motions::flights
    double flash = 0.0;        // a flash's length; in (0, 1): a flight's flashes are at least a second apart
    float dim = 0.0f;          // brightness between flashes; in [0, 1)
};

struct Glows {
    std::vector<Rhythm> rhythms;
    std::vector<FlightGlow> flights;
};

// The factor on its radiance the glow `record` gives at `t`, in [dim, 1].
// The record must index its kind's array, and a flight glow a flight; the
// scene reader makes them so. A switch with no default.
float glow(const Glows& glows, const Motions& motions, GlowRecord record, frame::Seconds t);

}  // namespace serenity::animation
