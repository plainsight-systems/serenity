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
//
// Its flashes are at least flash_spacing apart, the loop's last and its
// next first included: one number, named once (ES.45), which a flight keeps
// its flashes apart by and a schedule glow's flash length stays under, so
// no two flashes overlap.

// Seconds: the least time from one flash's start to the next's.
inline constexpr double flash_spacing = 1.0;

struct FlashSchedule {
    double loop = 0.0;          // seconds, > 0: the schedule repeats every loop
    std::vector<double> starts;  // within [0, loop), in order, at least flash_spacing apart, the last and
                                 // the loop's next first included
};

}  // namespace serenity::animation
