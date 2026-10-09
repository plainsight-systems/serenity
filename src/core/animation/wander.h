#pragma once

#include <array>
#include <cstdint>

#include "core/animation/extent.h"
#include "core/contracts/float3.h"
#include "core/frame/frame_inputs.h"

namespace serenity::animation {

// Axis: Animation (wander).
//
// A drift about a fixed point: a firefly hovering in place, never twice the
// same way. Its position is a closed form in time, so a frame at any t is
// placed directly, with no state carried from the frame before
// (logical-overview.md, principle 1): the headless renderer jumps to any
// instant, and the same t always places it in the same spot.
//
// The scene gives four numbers (core/scene/scene.h): the anchor, the shape's
// own center; the reach, how far it strays along each axis; the speed, its
// root-mean-square speed; and a seed, so no two wander alike. make_wander()
// turns them into the path, by these steps, which the code carries by
// number:
//
//   Step 1  Draw, for each axis a and each of K = 3 terms k, from the seed
//           alone (below): a weight w in [0.5, 1), a frequency ratio g in
//           [0.4, 1.6) and a phase phi in [0, 2 pi).
//   Step 2  Amplitudes: A(a,k) = reach x w(a,k) / (sum over k of w(a,k)),
//           so the amplitudes of each axis sum to the reach, and the path
//           never leaves anchor +/- reach on any axis.
//   Step 3  Frequencies: with g as frequencies, the mean of the squared
//           speed is s0^2 = sum over a and k of (2 pi g A)^2 / 2 (the cross
//           terms average to 0, the frequencies being distinct); scaling
//           every frequency by speed / s0 makes it speed^2. So
//           f(a,k) = g(a,k) x speed / s0.
//
// and position() evaluates
//
//   p(t) = anchor + sum over k of A(a,k) sin(2 pi f(a,k) t + phi(a,k)),
//
// on each axis a. Three terms of unrelated frequency per axis never line up
// into a visible repeat; that is the whole of what makes it read as a
// drift and not an orbit.
//
// The draws are a hash of (seed, axis, term, which number), splitmix64
// (Steele, Lea and Flood 2014), its top 53 bits as a double in [0, 1). Not
// <random>'s distributions: the standard fixes their results but not their
// algorithms, so the same seed would wander differently under another
// standard library (environmental determinism). Evaluated in double: at
// t = one day, 2 pi f t is near 10^5 and a double holds it to 10^-11; the
// result is rounded to float once, at the end.
//
// The extent is the box the center never leaves: anchor +/- reach on each
// axis. The scene reader checks that the shape, wherever in it, touches
// nothing that does not also move (core/scene/scene.h).
//
// Not performance-sensitive per wander: make_wander() runs once, at load;
// position() is 9 sines in double, once per moving shape per frame.

struct Wander {
    contracts::Float3 anchor;
    float reach = 0.0f;  // greater than 0
    // Per axis (x, y, z), per term: the amplitude, the frequency in hertz,
    // and the phase in radians.
    std::array<std::array<double, 3>, 3> amplitude{};
    std::array<std::array<double, 3>, 3> frequency{};
    std::array<std::array<double, 3>, 3> phase{};
};

// The path for these numbers, by steps 1 to 3. `reach` and `speed` must be
// finite and greater than 0; the scene reader makes them so.
Wander make_wander(contracts::Float3 anchor, float reach, float speed, std::uint64_t seed);

// Where it is at `t`.
contracts::Float3 position(const Wander& wander, frame::Seconds t);

// The box its center never leaves: anchor +/- reach.
Extent extent(const Wander& wander);

}  // namespace serenity::animation
