#pragma once

#include <cstdint>

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
// It counts from its construction, on a monotonic clock (SDL's nanosecond
// ticks), so time never runs backwards when the system clock is set. It is
// CPU time: GPU timestamps run on a different clock, and nothing here
// compares the two (TLM.11).
class Clock {
public:
    Clock();
    frame::Seconds elapsed() const;

private:
    std::uint64_t start_ns_;
};

}  // namespace serenity::app
