#pragma once

// The flight jobs the flight and swarm tests share (ES.3): a firefly about
// the ball of ball_on_floor.h, in a volume over its floor, as a swarm of
// swarm_test.cpp's numbers flies it. The batch tests of flight_test.cpp
// make many of them at once (core/animation/flight.h, make_flights and
// try_flights); swarm_test.cpp reads its swarm's numbers from them.

#include <cstdint>

#include "core/animation/flight.h"
#include "core/contracts/float3.h"
#include "core/contracts/obstacles.h"

namespace serenity::tests {

// The job's firefly and its volume: a body of 2 cm keeping 5 cm, in a box
// 4 m across from just over the floor to 2.2 m.
inline constexpr float job_body = 0.02f;
inline constexpr float job_clearance = 0.05f;
inline constexpr contracts::Box job_volume{{-2.0f, 0.05f, -2.0f}, {2.0f, 2.2f, 2.0f}};

// A start clear of the ball and the floor, and one inside the ball.
inline constexpr contracts::Float3 clear_start{1.2f, 1.0f, 0.8f};
inline constexpr contracts::Float3 inside_ball{0.0f, 0.5f, 0.0f};

// A flight job about the ball, in the job's volume, from `start`.
inline animation::FlightJob job(std::uint64_t seed, contracts::Float3 start) {
    return {.params = {.volume = job_volume,
                       .targets = {{{0.0f, 0.5f, 0.0f}, 0.5f}},
                       .speed = 0.4f,
                       .clearance = job_clearance,
                       .weights = {2.0f, 1.0f, 2.0f},
                       .seed = seed},
            .start = start,
            .body = job_body};
}

}  // namespace serenity::tests
