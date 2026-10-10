#pragma once

#include <chrono>

#include "core/frame/frame_inputs.h"

namespace serenity::app {

// Axis: Presentation.
//
// The measured clock: the one clock in the program, and it lives in the app.
// The window's loop reads it once a frame and passes the result in as the
// frame's time (frame/frame_inputs.h); the renderer never reads it
// (principle 1, I.1). The headless renderer has no clock at all: its time is
// the frame's index times a fixed step.
//
// It counts from its construction, on the standard library's monotonic clock
// (std::chrono::steady_clock; SL.2: no window library needed for a clock),
// so time never runs backwards when the system clock is set. The GPU's
// instants (metal/device/submission.h) are host time too, but from another
// origin, so nothing subtracts one from the other (TLM.11).
class Clock {
public:
    Clock();
    frame::Seconds elapsed() const;

private:
    std::chrono::steady_clock::time_point start_;
};

}  // namespace serenity::app
