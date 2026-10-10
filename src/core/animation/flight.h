#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
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
// still shape and never leaving its volume. Made once, at load, as a loop of
// episodes, then evaluated in closed form: where it is at t is a lookup and
// a formula, with no state carried from the frame before (logical-
// overview.md, principle 1), so any instant renders directly and the same t
// always places it in the same spot.
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
//
// make_flight(), by these steps, which the code carries by number:
//
//   Step 1  For episode k of 64, choose its behaviour by the weights: circle,
//           swoop or drift (a circle only if the scene names targets), and
//           a circle's target uniformly among them.
//   Step 2  Draw the behaviour's numbers. Episode 0 is a drift about the
//           shape's center in the scene file: where the loop starts.
//   Step 3  Check it: samples along its path, close enough that the path
//           between two lies within delta = 1 cm of them, must each be at
//           least clearance + body radius + delta from every still surface
//           (contracts::Obstacles::distance, contract 11) and within the
//           volume shrunk by body radius + delta on every side. Every point
//           of the path is within delta of a sample, so the whole path, not
//           only the samples, keeps the clearance and keeps its body inside
//           the volume. If any sample is not, draw again (step 2), up to 16
//           times. A segment that would take more than the params'
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
//           schedule over the loop (flashes.h, read by a glow, glow.h): one
//           at each swoop's climb, and, at a rate drawn per firefly, a few
//           while circling and fewer while drifting; a flash that would
//           start within flash_spacing (a second) of the one before is left
//           out, so no two are closer, the loop's last and first included.
//           In time order.
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
//   Step E1  t into the loop: tau = t mod the loop's length.
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
// shape, the load's largest cost. The load's measurements, and when a
// spatial index behind the obstacles would pay, are in
// docs/research/2026-10-10-flight-load.md. Memory: some 140 segments of 22
// doubles, some 25 KB per firefly. Per frame: one binary search over some
// 140 starts and one closed form of a few sines or a cubic, nothing
// allocated (MEM.9).

// Step 3's sampling bound, delta, in meters.
inline constexpr double flight_delta = 0.01;

// How far episode 0, the drift about the start, may carry the firefly from
// its start on each axis, in meters (step 2): its reach. A start clear by
// this much more than step 3 asks keeps the whole first drift clear
// (core/scene/swarm.h draws its starts so).
inline constexpr double first_drift_reach = 0.1;

// The most samples step 3 checks one segment in, by default: a bound on the
// work one segment can ask for, each sample one distance over every still
// shape, not a tuning number. A firefly's segment needs some 10^2 to 10^3
// (at most some 0.5 m/s for at most some 9 s, a sample a centimeter); a
// million is a path of 10 km in one segment, which no volume a scene can
// hold a firefly in asks for, so only numbers that are not a firefly's
// reach it (ES.45, CDSA.21).
inline constexpr std::int64_t flight_most_steps = 1'000'000;

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
    Extent volume;
    std::vector<Segment> segments;  // in time order, covering [0, loop)
    double loop = 0.0;              // the loop's length, in seconds
    FlashSchedule flashes;          // its flashes, over the same loop (step 7)
};

// The flight for these numbers, starting at `start`, of a body of radius
// `body`, kept clear of `obstacles`, by steps 1 to 7. Throws
// std::invalid_argument for numbers out of range (above), a start not clear,
// or an episode that cannot be drawn clear.
Flight make_flight(const FlightParams& params, contracts::Float3 start, float body,
                   const contracts::Obstacles& obstacles);

// Many flights, each as make_flight() makes it, made in parallel: flight k
// of the result is make_flight(jobs[k]...), whatever the number of threads,
// since each is a function of its own numbers alone and `obstacles` answers
// the same from any thread (contract 11). As many threads as the machine
// has cores (std::thread::hardware_concurrency, at least one), each taking
// the next unmade flight. If any cannot be made, throws, once every thread
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

struct FlightJob {
    FlightParams params;
    contracts::Float3 start{};
    float body = 0.0f;
};
std::vector<Flight> make_flights(const std::vector<FlightJob>& jobs, const contracts::Obstacles& obstacles);

// Where it is at `t`, by steps E1 to E3.
contracts::Float3 position(const Flight& flight, frame::Seconds t);

// The box its center never leaves: its volume.
Extent extent(const Flight& flight);

}  // namespace serenity::animation
