#pragma once

#include <cstdint>
#include <numbers>
#include <optional>
#include <variant>

#include "core/animation/flight.h"
#include "core/animation/glow.h"
#include "core/contracts/float3.h"
#include "core/contracts/obstacles.h"

namespace serenity::scene {

// Axis: Scene content (swarms).
//
// Many fireflies from one entry in a scene file: a [[swarms]] table
// (scene.h) becomes `count` fireflies, each exactly what a firefly written
// out by hand is (architecture/file-mapping.md, Composition): a sphere of
// the swarm's radius wearing its emissive material, so a sphere light,
// flying (core/animation/flight.h) and flashing with its flight
// (core/animation/glow.h, the flight kind). They differ only in their
// seeds, where they start and when they wake, which rules draw, so a scene
// of five hundred fireflies is one entry, not five hundred that differ in a
// few numbers.
//
// A swarm adds no kind to any family: no shape, light, motion or glow kind,
// and nothing past the scene reader knows a firefly came from one. Its
// fireflies are checked as written ones are, by the same rules (scene.h):
// a written firefly may hold or perch, and wake, as a swarm's does.
//
// How its fireflies start, the swarm's start kind (none written: air):
//
//   air     in the air, anywhere in the volume, flying from the first
//           moment: the flight's prelude none (core/animation/flight.h).
//   above   in the top `depth` meters of the volume, holding still there
//           until it wakes (its glow's wake, below), then flying: the
//           prelude hold. Its loop, mostly circling the marbles below where
//           the scene weights it so, brings it down lit.
//   perch   resting on a still surface, a marble's top or the table, at a
//           point found in the swarm's perch box, until it wakes and has
//           lingered there a drawn while, glowing; then it rises to its
//           loop's first point, straight above the perch, and flies: the
//           prelude perch.
//
// When its fireflies wake (none written: lit from the start): each its own
// moment from `from` to `to` seconds, then brightening over `ramp` seconds
// (glow.h, Wake). The moments are drawn so that the share awake by t grows
// as ((t - from) / (to - from))^power: power 1 an even rate, larger a slow
// start, one by one, and then many.
//
// The fireflies of a swarm, by these steps, which the code carries by
// number:
//
//   Step 1  Firefly i, from 0 to count - 1, has its own seed:
//           splitmix64(splitmix64(seed) ^ i) (core/animation/draw.h), the
//           swarm's seed and i alone, so adding a swarm, or fireflies at the
//           end of one, moves no other firefly: numbers from (seed, stream,
//           counter), never from a shared sequence (GDSA.3), and each step a
//           pure function of them (F.8).
//   Step 2  Its start, where its flight's first drift hovers (flight.h,
//           step 2), which may carry it first_drift_reach from the start on
//           each axis: a point drawn uniformly within the swarm's volume
//           shrunk by radius + delta + first_drift_reach on every side
//           (delta, the flight's sampling bound, flight.h step 3), from
//           draw(seed_i, attempt, axis) (core/animation/draw.h); drawn again
//           while it is nearer than clearance + radius + delta +
//           first_drift_reach sqrt 3 to a still surface (contract 11), up to
//           start_attempts times. So the whole first drift keeps the
//           flight's clearance and stays in its volume, as the flight
//           requires of its start. Failing all, the swarm is refused, naming
//           the firefly: a volume the still shapes fill. The flight's
//           rounding allowance (flight.h, some 2 x 10^-7 m at the marbles'
//           scale) is not added here, so an air swarm draws the starts it
//           drew before the allowance; a start within it of the threshold
//           is refused by the flight, naming the firefly, never flown
//           closer. For above, the
//           height is drawn within the top `depth` of that shrunk range
//           (all of it, if it is shallower than `depth`); the draws are
//           otherwise air's, keyed alike.
//   Step 2p For perch, in place of step 2, each attempt up to
//           start_attempts, from draws keyed (seed_i, perch_draws, attempt,
//           axis), apart from step 2's. The loop's first point, where the
//           rise ends, is not the start but the start plus
//           first_offset(seed_i) (flight.h, step 2), up to
//           first_drift_reach on each axis; the start is placed so that
//           the first point is straight above the perch:
//             - x and z drawn uniformly within the perch box's; the start's
//               x and z, those less the offset's, rounded to float; the
//               vertical line traced is the first point's, the start's x
//               and z plus the offset's. The start's x and z must lie in
//               step 2's shrunk range: one outside it fails the attempt,
//               never moved into it;
//             - from the box's top, sphere tracing down that line (Hart
//               1996): step down by d - (radius + perch_gap), d the
//               distance to the still surfaces there (contract 11), until
//               within perch_gap / 10 of it, at most perch_steps steps, and
//               never below the box's floor. The distance is exact, so no
//               step passes a surface: the perch is the first point down
//               the line at the perch's height above a surface, its x and z
//               the line's rounded to float, and within the box;
//             - that surface must face up: the distance's rise along y, by
//               a central difference over perch_gap / 2, at least
//               cos(perch_steepest), so the firefly sits on a marble's top
//               or the table, not on a marble's flank;
//             - the start's height drawn uniformly in step 2's shrunk
//               range, as step 2 draws it; the first point, that plus the
//               offset's, above the perch; the start clear as step 2's must
//               be.
//           So the first point placed (position() at the loop's begin) has
//           the perch's x and z exactly, and the rise's chord is vertical.
//           A box with no surface in it, its top inside a shape, a perch
//           whose start would leave step 2's range, or one whose start is
//           never clear, fails the attempt; failing all, the swarm is
//           refused, naming the firefly: a box wholly outside the range a
//           first point can be above is refused so. The flight checks the
//           perch and the rise again, as it checks any (flight.h, P1 and
//           P2).
//   Step 3  Its wake, with the swarm's: at = from + (to - from) u^(1/power),
//           u = draw(seed_i, wake_draws), so the share of the swarm awake by
//           t, from u's uniform draw, is ((t - from) / (to - from))^power;
//           its ramp the swarm's. For perch, its linger, uniform from
//           draw(seed_i, linger_draws) between the swarm's least and most.
//   Step 4  It becomes a sphere at its start, of the swarm's radius, wearing
//           the swarm's material; a sphere light; a flight with the swarm's
//           volume, targets, speed, clearance and weights, its own seed, and
//           its prelude: for air none; for above, hold until its wake's at
//           (0 with no wake); for perch, perch at its perch until its wake's
//           at (0 with no wake) plus its linger; and a flight glow with the
//           swarm's flash and dim, and its wake. Shapes of swarms follow the
//           file's [[shapes]], swarm by swarm in file order, firefly by
//           firefly: a firefly's index, and so its primitive and its light,
//           is fixed by the file.
//
// A firefly's draws are keyed apart by purpose: step 2's by attempt, under
// start_attempts; the others' first counters, perch_draws, wake_draws and
// linger_draws, at 2^32 and up, past any attempt. So a swarm of air
// fireflies with no wake starts each where it started before starts and
// wakes had kinds (GDSA.3).
//
// The flights themselves are made with every other flight in the scene,
// in parallel (flight.h, make_flights).
//
// Cost, at load: step 2 asks contract 11 for a distance a few times per
// firefly; step 2p some tens, a trace of a few steps and its checks, per
// attempt; the flights are flight.h's.

// How a swarm's fireflies start (above): one of three kinds, each with only
// its own numbers, a std::variant as the flight's Prelude is (C.181, C.182).
struct AirStart {};

struct AboveStart {
    double depth = 0.0;  // meters of the volume's top it starts in; > 0
};

struct PerchStart {
    contracts::Box box{};       // where perches are found; finite, min below max
    double linger_least = 0.0;  // seconds it stays after waking; 0 or more
    double linger_most = 0.0;   // at least linger_least; with the wake's to, at most most_wait (flight.h)
};

using SwarmStart = std::variant<AirStart, AboveStart, PerchStart>;

// When a swarm's fireflies wake (above, step 3).
struct SwarmWake {
    double from = 0.0;   // seconds; finite, 0 or more
    double to = 0.0;     // seconds; at least from, at most most_wait (flight.h)
    double power = 1.0;  // finite, > 0
    double ramp = 0.0;   // seconds; finite, 0 or more
};

// A swarm as read from its table (scene.h): everything but the material,
// which the scene reader keeps.
struct Swarm {
    std::uint32_t count = 0;             // 1 to max_swarm
    float radius = 0.0f;                 // each firefly's; > 0
    animation::FlightParams flight;      // the swarm's seed in flight.seed
    float flash = 0.0f;                  // the flight glow's (glow.h)
    float dim = 0.0f;
    SwarmStart start;                    // none written: air
    std::optional<SwarmWake> wake;       // none written: lit from the start
};

// One firefly as steps 2 to 3 draw it: where its loop starts, its prelude
// (flight.h) and its wake (glow.h).
struct Firefly {
    contracts::Float3 start{};
    animation::Prelude prelude;
    std::optional<animation::Wake> wake;
};

// The most fireflies one swarm makes: some 25 KB of flight each (flight.h),
// 100 MB at this count, and a load of seconds.
inline constexpr std::uint32_t max_swarm = 4096;

// Step 2's and step 2p's draws before the swarm is refused.
inline constexpr int start_attempts = 64;

// Step 2p's sphere tracing: its most steps, and the steepest a surface may
// be from level for a firefly to perch on it, in radians (30 degrees).
inline constexpr int perch_steps = 64;
inline constexpr double perch_steepest = 30.0 * std::numbers::pi / 180.0;

// The first counters of a firefly's draws other than step 2's (above).
inline constexpr std::uint64_t perch_draws = std::uint64_t{1} << 32;
inline constexpr std::uint64_t wake_draws = perch_draws + 1;
inline constexpr std::uint64_t linger_draws = perch_draws + 2;

// Step 1: firefly i's seed.
std::uint64_t firefly_seed(std::uint64_t swarm_seed, std::uint32_t i);

// Steps 2 to 3: firefly i, clear of `obstacles`. Throws an
// animation::MotionError (core/animation/motion_error.h) naming the firefly
// if its volume has no room for a drift, or none of its draws is clear or
// finds a perch; and std::invalid_argument for a start's or a wake's
// numbers outside the ranges above, which the scene reader checks first
// (I.5): a perch box's coordinates finite, min below max; a depth above 0;
// a linger's least 0 or more, its most at least that and at most most_wait;
// a wake's from 0 or more, its to from that to most_wait, its power above
// 0, its ramp 0 or more; and a wake's to (0 with no wake) plus a linger's
// most at most most_wait, compared without adding, so no sum overflows.
Firefly make_firefly(const Swarm& swarm, std::uint32_t i, const contracts::Obstacles& obstacles);

}  // namespace serenity::scene
