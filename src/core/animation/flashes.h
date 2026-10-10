#pragma once

#include <vector>

namespace serenity::animation {

// Axis: Animation.
//
// When a light flashes, as plain data: the starts of its flashes, first
// those of its opening, once each, then those of a loop that repeats from
// the opening's end. A flight makes one, from its own episodes and its
// prelude (flight.h, step 7); a glow of the schedule kind reads one
// (glow.h). The data passes between them through the scene reader, which
// copies a flight's schedule into its light's glow, so a glow depends on no
// motion kind and a flight on no glow kind.
//
// Its flashes are at least flash_spacing apart, wherever they fall: within
// the opening, within the loop, from the opening's last to the loop's first,
// and from the loop's last to its next first. One number, named once
// (ES.45), which a flight keeps its flashes apart by and a schedule glow's
// flash length stays under, so no two flashes overlap.
//
// The flash lit at t, if any, is the one with the latest start at or before
// t among the opening's starts and begin + k loop + each loop start, k = 0,
// 1, ...: none before the first. A schedule with no opening (begin 0, as a
// flight without a prelude makes) is one loop from 0, as before there were
// openings.

// Seconds: the least time from one flash's start to the next's.
inline constexpr double flash_spacing = 1.0;

struct FlashSchedule {
    double begin = 0.0;            // seconds, >= 0: when the loop starts, the opening's end
    std::vector<double> opening;   // once each, within [0, begin), in order, flash_spacing apart, the last
                                   // at least flash_spacing before begin + the loop's first
    double loop = 0.0;             // seconds, > 0: the loop repeats every loop from begin
    std::vector<double> starts;    // within [0, loop) from begin, in order, at least flash_spacing apart,
                                   // the last and the loop's next first included
};

}  // namespace serenity::animation
