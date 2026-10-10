#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "core/animation/extent.h"
#include "core/animation/flashes.h"
#include "core/animation/motion_error.h"
#include "core/contracts/float3.h"
#include "core/contracts/obstacles.h"
#include "core/frame/frame_inputs.h"

namespace serenity::animation {

// Axis: Animation (flight).
//
// A firefly flying free: circling the marbles and the brass sphere, swooping
// in J-strokes, drifting, and travelling between them, never touching a
// still shape and, once its loop begins, never leaving its volume. Made
// once, at load, as an opening, its prelude, and a loop of episodes, then
// evaluated in closed form: where it is at t is a lookup and a formula, with
// no state carried from the frame before (logical-overview.md, principle 1),
// so any instant renders directly and the same t always places it in the
// same spot.
//
// Why a loop made at load, and not a formula of t alone, as the wander is:
// each episode must start where the last one ended, and every stretch of the
// path must be checked clear of the still shapes. A path drawn afresh for
// any t could promise neither without walking every episode before it. Made
// at load, the whole loop is built in order and checked once, and the
// guarantee holds for all time; the loop then repeats. At 64 episodes of a
// few seconds each, with the transits between, it lasts some ten minutes,
// and among many fireflies, each with its own loop, a repeat is not seen.
//
// The behaviours, each a closed form over its own time tau from 0 to its
// duration, every number drawn from the seed:
//
//   circle   loops about a target, a still sphere the scene names: an orbit
//            of radius r0 in a plane tilted up to 25 degrees from level, or
//            less where the volume's floor leaves no room for that tilt (a
//            marble on a table, its orbit skimming the top), its center
//            raised a drawn lift of 0 to the target's radius above the
//            target's, and higher where that leaves its bob no room above
//            the volume's floor: at height max(center.y + lift, min.y + body
//            + delta + 0.05), the bob's 5 cm, a named constant beside the
//            orbit's other numbers in flight.cpp (ES.45), so a level orbit
//            about a marble a centimeter across, on the table, keeps its
//            body inside the volume at the bob's lowest; the tilt's room is
//            then measured from that height. It flies either way round, at the
//            cruising speed; its radius breathes by up to 10%
//            and it bobs by up to 5 cm, each at a frequency of its own. Four
//            to nine seconds: part of a loop about a wide orbit, more than
//            one about a tight one.
//   swoop    a J-stroke, the flight of Photinus pyralis: from a point in the
//            air, a dip of 10 to 25 cm along a level heading, then a climb of
//            30 to 60 cm, as one cubic Bezier curve. It flashes on the climb.
//            It starts low enough to leave 35 cm above its climb: it ends
//            climbing, and the transit out rises on before it turns.
//   drift    hovering about a point, as the wander does (wander.h), with a
//            reach of 10 cm at a third of the cruising speed. Two to four
//            seconds.
//   transit  the way from where one behaviour ends to where the next
//            begins: a cubic Hermite curve matching both ends' positions and
//            velocities, so the path is smooth through every join; over a
//            waypoint at the volume's top when the direct curve would touch
//            something.
//   still    holding one point: a prelude's wait (below).
//
// The prelude, what it does before its loop begins, a Prelude given with
// each flight (FlightJob), its kind one of:
//
//   NoPrelude  the loop begins at once, at t = 0: the flight as it was
//              before preludes, the same loop for the same seed.
//   Hold       it holds still at the loop's first point until `until`, then
//              the loop begins: a firefly that waits, dark (glow.h, its
//              wake), at the top of its volume, and comes down lit.
//   Perch      it rests at `at`, a point on a still surface, until `until`,
//              then rises to the loop's first point and the loop begins: a
//              firefly that wakes on a marble or the table, glows and
//              flashes there, and takes off. The perch is outside the volume
//              as a rule (the volume's floor is above the marbles), and so
//              is the rise's start: the volume bounds the loop, not the
//              prelude.
//
// A perched firefly sits closer to the surface below it than the clearance
// allows the loop, by design: `perch_gap` from it. Its perch and its rise are held instead to keeping the body at
// least perch_gap / 2 from every still surface, at every point of the path,
// not only at samples (steps P1 and P2). So the guarantee that a flight
// never touches a still shape holds through the prelude; the clearance is
// the loop's.
//
// make_flight(), by these steps, which the code carries by number:
//
//   Step 1  For episode k of 64, choose its behaviour by the weights: circle,
//           swoop or drift (a circle only if the scene names targets), and
//           a circle's target uniformly among them.
//   Step 2  Draw the behaviour's numbers. Episode 0 is a drift about the
//           shape's center in the scene file, the flight's start. Its point
//           at time 0, the loop's first point, is the start plus an offset
//           that its seed alone fixes, first_offset(seed), below: episode
//           0's amplitudes and phases are drawn from (seed, 0, 0) whatever
//           the start, and its frequencies, the speed's, vanish at time 0.
//   Step 3  Check it: samples along its path, close enough that the path
//           between two lies within delta = 1 cm of them, must each be at
//           least clearance + body radius + delta from every still surface
//           (contracts::Obstacles::distance, contract 11) and within the
//           volume shrunk by body radius + delta on every side. Every point
//           of the path is within delta of a sample, so the whole path, not
//           only the samples, keeps the clearance and keeps its body inside
//           the volume. Each threshold also carries the rounding allowance
//           rho, below, for the volume, so the guarantee holds of the float
//           points that are asked about and rendered, not only of the
//           double curve. If any sample is not, draw again (step 2), up to
//           16 times. A segment that would take more than the params'
//           most_steps samples is not clear either: its count is compared
//           as a double, before it is made an integer (ES.46), so no number
//           the scene reader accepts, a speed near FLT_MAX included, can
//           overflow the conversion.
//   Step 4  The transit into it from episode k - 1's end: the direct curve,
//           checked as in step 3; failing that, over the waypoint; failing
//           both, draw episode k again (step 2). An episode's 16 draws are a
//           round. If episode k fails a whole round, where episode k - 1
//           ended may leave no way on (beside a marble, flying at it): draw
//           episode k - 1 again from its next round, then k afresh. At most
//           4 rounds an episode and 64 such redraws a flight; past them the
//           flight is refused. On open ground no episode needs a second
//           round, and the draws are those of the first.
//   Step 5  Close the loop: the transit from episode 63's end to episode 0's
//           start, checked as in step 4; failing it, draw episode 63 again.
//   Step 6  Lay the segments out in time, transit 0, episode 0, transit 1,
//           ..., each starting where the one before ends; the loop's length
//           is their sum.
//   Step 7  The flashes, the starts of the firefly's own blinks, as a
//           schedule (flashes.h, read by a glow, glow.h): over the loop, one
//           at each swoop's climb, and, at a rate drawn per firefly, a few
//           while circling and fewer while drifting; a flash that would
//           start within flash_spacing (a second) of the one before is left
//           out, so no two are closer, the loop's last and first included.
//           In time order. The loop's flashes are drawn as they were before
//           preludes, from the same keys, so a flight's loop flashes the
//           same with any prelude. Over the opening, once each: while it
//           waits (hold or perch), at its drifting rate, as seldom as a
//           drifting firefly, at most some 300 over a wait of most_wait
//           (step P1), counted as a double before it is made an integer
//           (ES.46); none on the rise. Those are drawn from keys of
//           their own, and an opening flash within flash_spacing of the one
//           before, or of the loop's first, is left out. A held firefly is
//           dark while it waits, as the scene gives it (its glow's wake,
//           scene.h), so its opening flashes show only where a wake comes
//           before the hold ends: the schedule is the flight's, the wake the
//           glow's, and neither knows the other.
//
// Then the prelude, by these steps, made after the loop, since it ends where
// the loop starts:
//
//   Step P1  Check the prelude's numbers: `until` finite, 0 or more and at
//            most most_wait, so an opening's flashes are bounded (step 7);
//            a perch's distance to the still surfaces (contract 11) at least
//            body + 3 perch_gap / 4 + rho, so the rise starts with room
//            (P2), rho the rounding allowance of the rise's hull; and that
//            rho under perch_gap / 8, so float can place a body at its
//            perch's scale there at all (it can within 512 m of the
//            origin, where float's spacing is at most 3 x 10^-5 m). A perch nearer, or too far out for
//            its gap, is refused, as an episode is.
//   Step P2  The opening's segments: for hold, a still segment at the
//            loop's first point lasting `until`; for perch, a still segment
//            at the perch lasting `until`, then the rise, a transit from the
//            perch, leaving from rest, its velocity 0 as the perch's is, to
//            the loop's first point, arriving at the loop's own velocity
//            there, so the path is smooth through both of its joins; its
//            duration a transit's (from its chord at the cruising speed). A
//            cubic from rest sets off toward its second control point, the
//            loop's first point less a third of its arrival: up, for a
//            loop whose first point is straight above the perch, as a
//            swarm's perched firefly's is (core/scene/swarm.h, step 2p,
//            which places the start by first_offset() so that it is). A
//            still segment of no length is left out. The rise is checked by conservative advancement
//            (Mirtich 1996; Hart 1996, sphere tracing): from u = 0, at the
//            point p(u) the distance d to the still surfaces is exact for
//            the kinds there are (contract 11), so no surface lies within
//            d - r of any point within r of p; the slack s = d - body -
//            perch_gap / 2 - rho is how far the path may go before it could come
//            too near, and the next sample is at u + s / V, V bounding the
//            curve's speed in u (its derivative's Bezier hull, as step 3's
//            bound). A slack under perch_gap / 4 refuses the rise, which
//            keeps every step at least perch_gap / (4 V) and the samples at
//            most 4 V / perch_gap and the last at u = 1: compared with
//            most_steps as a double before the walk (ES.46). The rise must also keep within the
//            world, which its hull, below, shows the reader.
//
//            The one join that is not smooth is a hold's end: the firefly
//            holds still, then its loop starts at the drift's own speed, a
//            third of the cruising speed. It is the moment it wakes, dark or
//            barely lit (scene.h: a swarm holds until its wake), and is
//            stated here rather than smoothed.
//            A rise that is not clear is refused (a MotionError): a perch
//            under an overhang, or with something between it and the loop.
//   Step P3  Lay the opening out from 0; the loop begins at its end,
//            `begin`. No prelude, or a hold of no length: no opening, begin
//            0.
//
// The rounding allowance, rho, of a box (rounding_allowance(), below): sqrt
// 3 times float's spacing (one ulp) at the largest |coordinate| in it. A point of the double curve is
// rounded to float when its distance is asked (contract 11 takes a Float3)
// and when it is placed (position() returns a Float3): each moves it at
// most half an ulp an axis, so the two together at most rho. At the
// marbles' meter scale rho is some 2 x 10^-7 m; at the world's edge, 10^6
// m, some 0.1 m, which no gap of a millimeter survives, and which P1
// refuses for a perch.
//
// If an episode cannot be drawn clear within those redraws, make_flight throws
// a MotionError (motion_error.h) naming it: a target with no room to circle it, a
// volume too tight for the clearance. The scene reader reports it against
// the motion's line (core/scene/scene.h). Numbers out of range, which the
// reader checks first, are a std::invalid_argument.
//
// position(), by these steps, of a flight make_flight made: one with
// segments and a loop longer than 0, which it checks, throwing
// std::invalid_argument for one that is not (I.5, E.2); a default Flight is
// not one.
//
//   Step E0  A flight with an opening, at t before `begin`: the opening's
//            segment holding t, or its first at t before 0, where it is at
//            0; its closed form there. A flight without one keeps E1's
//            repeat before 0, as it did before preludes.
//   Step E1  t into the loop: tau = (t - begin) mod the loop's length.
//   Step E2  The segment holding tau, by binary search on the starts.
//   Step E3  Its closed form at tau - its start.
//
// The draws are splitmix64 hashes of (seed, episode, attempt, which number),
// as the wander's are (wander.h): integers, the same under any standard
// library. The loop they make is evaluated in double, rounded to float once,
// with libm's sin, cos and asin, which are not correctly rounded and differ
// between libms and with floating-point contraction; step 3 accepts or
// rejects on a threshold, so one ulp can choose another attempt. The level
// held is therefore: the same seed flies the same loop on the same
// toolchain, libm and flags (GDSA.2). A test pins one seed's loop
// (tests/flight_test.cpp), so a change of any of them that moves it is seen.
//
// Cost. At load, per firefly: 64 episodes and 64 transits, each some tens to
// a few hundred samples, each sample one distance (contract 11) over every
// still shape: some 10^4 to 10^5 distance tests per firefly per still
// shape, the load's largest cost. A rise adds tens to hundreds (step P2). The load's measurements, and when a
// spatial index behind the obstacles would pay, are in
// docs/research/2026-10-10-flight-load.md. Memory: some 140 segments of 22
// doubles, some 25 KB per firefly. Per frame: one binary search over some
// 140 starts and one closed form of a few sines or a cubic, nothing
// allocated (MEM.9); a flight with an opening, one comparison with begin
// more, and before it a search of at most two segments.

// Step 3's sampling bound, delta, in meters.
inline constexpr double flight_delta = 0.01;

// How far episode 0, the drift about the start, may carry the firefly from
// its start on each axis, in meters (step 2): its reach. A start clear by
// this much more than step 3 asks keeps the whole first drift clear
// (core/scene/swarm.h draws its starts so).
inline constexpr double first_drift_reach = 0.1;

// Where on a swoop its flash starts, a fraction of the swoop's duration: on
// its climb (step 7).
inline constexpr double swoop_flash_at = 0.55;

// The most samples step 3 checks one segment in, by default: a bound on the
// work one segment can ask for, each sample one distance over every still
// shape, not a tuning number. A firefly's segment needs some 10^2 to 10^3
// (at most some 0.5 m/s for at most some 9 s, a sample a centimeter); a
// million is a path of 10 km in one segment, which no volume a scene can
// hold a firefly in asks for, so only numbers that are not a firefly's
// reach it (ES.45, CDSA.21).
inline constexpr std::int64_t flight_most_steps = 1'000'000;

// The longest a prelude may wait, in seconds: an hour, past any opening a
// scene shows, and a bound on its flashes (step 7) and their memory, some
// 300 of them, 2.4 KB, at the fastest drifting rate.
inline constexpr double most_wait = 3600.0;

// How far a perched firefly's body sits from the surface it rests on, in
// meters: half a millimeter, a firefly's legs (step P1). Its perch and rise
// keep at least half of it from every still surface (step P2).
inline constexpr double perch_gap = 0.0005;

// A still sphere a flight circles: where it is and how big.
struct Target {
    contracts::Float3 center{};
    float radius = 0.0f;
};

struct FlightParams {
    Extent volume;                 // where it flies; its body stays inside
    std::vector<Target> targets;   // what it circles; none, and it does not circle
    float speed = 0.0f;            // cruising, in meters a second; > 0
    float clearance = 0.0f;        // from every still surface to its own, in meters; >= 0
    std::array<float, 3> weights{};  // circle, swoop, drift: >= 0, summing > 0; circle 0 with no targets
    std::uint64_t seed = 0;
    std::int64_t most_steps = flight_most_steps;  // step 3's bound on one segment's samples; > 0
};

enum class Behaviour {
    transit,
    circle,
    swoop,
    drift,
    still,
};

// What a flight does before its loop begins (above): one of three kinds,
// each with only its own numbers, as a std::variant, a tagged union the
// library keeps type safe (C.181, C.182): a hold's numbers cannot be read as
// a perch's. Made at load and read there, never on a frame's path. Visited
// with a case per kind, so a kind without one fails the build.
struct NoPrelude {};

struct Hold {
    double until = 0.0;  // seconds it waits; 0 to most_wait
};

struct Perch {
    contracts::Float3 at{};  // where its center rests (step P1)
    double until = 0.0;      // seconds it rests; 0 to most_wait
};

using Prelude = std::variant<NoPrelude, Hold, Perch>;

// The cases of a std::visit, one per kind of a variant, as one overload set:
// a kind with no case fails the build. Written once (ES.3), for the Prelude
// here and the swarm's start (core/scene/swarm.h). Each case a class, a
// lambda's closure, which is all an overload set can inherit (T.10).
template <typename... Cases>
    requires(std::is_class_v<Cases> && ...)
struct Visit : Cases... {
    using Cases::operator()...;
};

// One stretch of the path: its behaviour, when it starts within the loop,
// how long it lasts, and the numbers of its closed form, whose meaning is
// its behaviour's: flight.cpp names where each number is, per behaviour
// (P.1), and reads them only by those names.
struct Segment {
    Behaviour behaviour = Behaviour::transit;
    double start = 0.0;
    double duration = 0.0;
    std::array<double, 19> numbers{};
};

struct Flight {
    Extent volume;                  // the loop's
    Extent reach;                   // the box its center never leaves: the volume, grown to hold the opening
    std::vector<Segment> opening;   // the prelude, in time order, covering [0, begin); none without one
    double begin = 0.0;             // when the loop begins, in seconds
    std::vector<Segment> segments;  // the loop, in time order, covering [0, loop) from begin
    double loop = 0.0;              // the loop's length, in seconds
    FlashSchedule flashes;          // its flashes, over the opening and the loop (step 7)
};

// One flight's inputs: the shared numbers, where its loop starts (episode
// 0's drift is about it), its body's radius and its prelude. One argument
// for make_flight() rather than five (I.23), the same record make_flights()
// takes a vector of.
struct FlightJob {
    FlightParams params;
    contracts::Float3 start{};
    float body = 0.0f;
    Prelude prelude;
};

// The flight for `job`, of a body of radius job.body, kept clear of
// `obstacles`, by steps 1 to 7 and P1 to P3. Throws std::invalid_argument
// for numbers out of range (above, and a prelude's until, or a perch not
// finite), and a MotionError for a start, a perch or a rise not clear, a
// perch too far out for its gap, an opening whose reach passes float's
// range, or an episode that cannot be drawn clear.
Flight make_flight(const FlightJob& job, const contracts::Obstacles& obstacles);

// Many flights, each as make_flight() makes it, made in parallel: flight k
// of the result is make_flight(jobs[k], obstacles), whatever the number of
// threads, since each is a function of its own numbers alone and
// `obstacles` answers the same from any thread (contract 11). `workers`
// threads (at least one, and no more than there are jobs), each taking the
// next unmade flight: an
// input, not read from the machine here (I.1), so a test can show the
// result is the same for any count; the scene reader passes
// flight_workers(). If any cannot be made, throws, once every thread
// has finished, the failure of the lowest k that failed: tasks, not
// threads (CP.4), the threads made once a load (CP.41) and joined however
// the call ends (CP.25). Each failure is caught in the worker that met it,
// kept as a std::exception_ptr and rethrown here, in the caller, which can
// report it: an exception that left a worker's function would end the
// program. A refusal, make_flight's MotionError, is rethrown as a FlightsError
// that carries k and the refusal's reason, its message "flight k: " and the
// reason; anything else as itself. So the error does not depend on the
// threads either.
class FlightsError : public MotionError {
public:
    FlightsError(std::size_t job, const std::string& reason)
        : MotionError("flight " + std::to_string(job) + ": " + reason), job(job), reason(reason) {}
    std::size_t job;     // which job was refused
    std::string reason;  // why, as make_flight said it (I.4: not parsed back out of what())
};

std::vector<Flight> make_flights(const std::vector<FlightJob>& jobs, const contracts::Obstacles& obstacles,
                                 std::size_t workers);

// Many flights, made as make_flights() makes them, each job's outcome kept
// rather than the first refusal thrown: outcome k is make_flight(jobs[k],
// obstacles), or, where that refused with a MotionError, the refusal's
// reason. For a caller that can draw a refused job again (a swarm's
// firefly, core/scene/swarm.h, step 5). A std::variant, the library's
// tagged union (C.181, C.182): a refused job holds no flight that looks
// made. Anything thrown that is not a MotionError is rethrown, the lowest
// k's, as make_flights() rethrows it. make_flights() is this, throwing a
// FlightsError for the lowest k refused (ES.3: one way to make many).
struct FlightRefusal {
    std::string reason;  // as make_flight said it
};
using FlightOutcome = std::variant<Flight, FlightRefusal>;
std::vector<FlightOutcome> try_flights(const std::vector<FlightJob>& jobs, const contracts::Obstacles& obstacles,
                                       std::size_t workers);

// As many workers as the machine has cores (std::thread::hardware_concurrency),
// at least one: the one place the machine is asked.
std::size_t flight_workers();

// Where the loop's first point is from the flight's start, in meters, for
// a flight of seed `seed` (step 2): episode 0's drift at time 0, less its
// center. A function of the seed alone, so a start can be placed for a
// first point chosen first: make_flight()'s first loop point, in double, is
// start + first_offset(seed), each axis bit for bit, and position() at the
// loop's begin is that rounded to float. Each axis at most
// first_drift_reach.
std::array<double, 3> first_offset(std::uint64_t seed);

// Where it is at `t`, by steps E0 to E3.
contracts::Float3 position(const Flight& flight, frame::Seconds t);

// The rounding allowance rho (above) of a box whose largest |coordinate| is
// `largest`: sqrt 3 times float's spacing just above the float at or above
// `largest`, which is at least the spacing anywhere in the box. Infinite
// past float's range, where no float holds a point: every threshold that
// carries it then refuses. Throws std::invalid_argument for a `largest`
// below 0 or not a number.
double rounding_allowance(double largest);

// The box its center never leaves: its reach, the volume grown to hold the
// opening: the perch, and the rise's Bezier control points, whose box holds
// the curve (the convex hull property), rounded outward to float.
Extent extent(const Flight& flight);

}  // namespace serenity::animation
