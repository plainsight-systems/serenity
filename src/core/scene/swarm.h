#pragma once

#include <cstdint>
#include <stdexcept>

#include "core/animation/flight.h"
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
// seeds and where they start, which a rule draws, so a scene of five hundred
// fireflies is one entry, not five hundred that differ in two numbers.
//
// A swarm adds no kind to any family: no shape, light, motion or glow kind,
// and nothing past the scene reader knows a firefly came from one. Its
// fireflies are checked as written ones are, by the same rules (scene.h).
//
// The fireflies of a swarm, by these steps, which the code carries by
// number:
//
//   Step 1  Firefly i, from 0 to count - 1, has its own seed:
//           splitmix64(splitmix64(seed) ^ i) (core/animation/draw.h), the
//           swarm's seed and i alone, so adding a swarm, or fireflies at the
//           end of one, moves no other firefly.
//   Step 2  Its start, where its flight's first drift hovers (flight.h,
//           step 2): a point drawn uniformly within the swarm's volume
//           shrunk by radius + clearance + delta on every side (delta, the
//           flight's 1 cm, flight.h step 3), from draw(seed_i, attempt,
//           axis); drawn again while it is nearer than clearance + radius +
//           delta to a still surface (contract 11), up to start_attempts
//           times. Failing all, the swarm is refused, naming the firefly: a
//           volume the still shapes fill.
//   Step 3  It becomes a sphere at its start, of the swarm's radius, wearing
//           the swarm's material; a sphere light; a flight with the swarm's
//           volume, targets, speed, clearance and weights and its own seed;
//           and a flight glow with the swarm's flash and dim. Shapes of
//           swarms follow the file's [[shapes]], swarm by swarm in file
//           order, firefly by firefly: a firefly's index, and so its
//           primitive and its light, is fixed by the file.
//
// The flights themselves are made with every other flight in the scene,
// in parallel (flight.h, make_flights).
//
// Cost, at load: step 2 asks contract 11 for a distance a few times per
// firefly; the flights are flight.h's.

// A swarm as read from its table (scene.h): everything but the material,
// which the scene reader keeps.
struct Swarm {
    std::uint32_t count = 0;             // 1 to max_swarm
    float radius = 0.0f;                 // each firefly's; > 0
    animation::FlightParams flight;      // the swarm's seed in flight.seed
    float flash = 0.0f;                  // the flight glow's (glow.h)
    float dim = 0.0f;
};

// The most fireflies one swarm makes: some 25 KB of flight each (flight.h),
// 100 MB at this count, and a load of seconds.
inline constexpr std::uint32_t max_swarm = 4096;

// Step 2's draws before the swarm is refused.
inline constexpr int start_attempts = 64;

// Step 1: firefly i's seed.
std::uint64_t firefly_seed(std::uint64_t swarm_seed, std::uint32_t i);

// Step 2: firefly i's start, clear of `obstacles`. Throws
// std::invalid_argument naming the firefly if none of its draws is clear.
contracts::Float3 firefly_start(const Swarm& swarm, std::uint32_t i, const contracts::Obstacles& obstacles);

}  // namespace serenity::scene
