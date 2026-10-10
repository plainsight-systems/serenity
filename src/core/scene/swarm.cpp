#include "core/scene/swarm.h"

#include <array>
#include <cmath>
#include <string>

#include "core/animation/draw.h"

namespace serenity::scene {

std::uint64_t firefly_seed(std::uint64_t swarm_seed, std::uint32_t i) {
    // Step 1.
    return animation::splitmix64(animation::splitmix64(swarm_seed) ^ i);
}

contracts::Float3 firefly_start(const Swarm& swarm, std::uint32_t i, const contracts::Obstacles& obstacles) {
    // Step 2: the volume its first drift may fill, and the distance its
    // start must keep for that drift to keep the clearance.
    const double body = swarm.radius;
    const double margin = body + animation::flight_delta + animation::first_drift_reach;
    const double needed = static_cast<double>(swarm.flight.clearance) + body + animation::flight_delta +
                          animation::first_drift_reach * std::sqrt(3.0);
    std::array<double, 3> low{};
    std::array<double, 3> high{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        low[axis] = contracts::component(swarm.flight.volume.min, static_cast<int>(axis)) + margin;
        high[axis] = contracts::component(swarm.flight.volume.max, static_cast<int>(axis)) - margin;
        if (!(low[axis] < high[axis])) {
            throw animation::MotionError("firefly " + std::to_string(i) +
                                     ": the volume is too small for a firefly to drift in");
        }
    }
    const std::uint64_t seed = firefly_seed(swarm.flight.seed, i);
    for (int attempt = 0; attempt < start_attempts; ++attempt) {
        std::array<double, 3> p{};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const double u =
                animation::draw(seed, static_cast<std::uint64_t>(attempt), static_cast<std::uint64_t>(axis));
            p[axis] = low[axis] + u * (high[axis] - low[axis]);
        }
        const contracts::Float3 start{static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2])};
        if (obstacles.distance(start) >= needed) {
            return start;
        }
    }
    throw animation::MotionError("firefly " + std::to_string(i) + ": no start clear of the still shapes in " +
                             std::to_string(start_attempts) + " draws; the volume is too full");
}

}  // namespace serenity::scene
