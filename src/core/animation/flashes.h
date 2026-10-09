#pragma once

#include <vector>

namespace serenity::animation {

// Axis: Animation.
//
// When a light flashes, as plain data: the starts of its flashes within a
// loop that repeats. A flight makes one, from its own episodes (flight.h,
// step 7); a glow of the schedule kind reads one (glow.h). The data passes
// between them through the scene reader, which copies a flight's schedule
// into its light's glow, so a glow depends on no motion kind and a flight on
// no glow kind.
struct FlashSchedule {
    double loop = 0.0;          // seconds, > 0: the schedule repeats every loop
    std::vector<double> starts;  // within [0, loop), in order, at least a second apart, the last and the
                                 // loop's next first included
};

}  // namespace serenity::animation
