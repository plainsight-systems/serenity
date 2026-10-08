#pragma once

#include <cstdint>
#include <optional>

#include "core/frame/frame_inputs.h"

namespace serenity::measurement {

// Axis: Measurement.
//
// How long frames took on the GPU, summarized over a reporting period: the
// number of frames, and the mean, the shortest and the longest. The durations
// come from the GPU's own timeline, the start and end Metal 4 reports for
// each submission (metal/device/submission.h), not from the CPU's clock
// around a frame, which measures waiting as well as work (GPU.10). They are
// durations on one clock, so nothing is compared across the CPU's and the
// GPU's clocks (TLM.11).
//
// A frame time is a measurement of this machine and this build; whoever
// shows it says which (the window's title names the resolution).
//
// Pure: add() and take() are given everything they use, including the time
// a period is reported at, so a test drives it with numbers (F.8).
struct Summary {
    std::uint64_t frames = 0;
    frame::Seconds mean{0.0};
    frame::Seconds shortest{0.0};
    frame::Seconds longest{0.0};
};

class FrameTimes {
public:
    explicit FrameTimes(frame::Seconds period) : period_(period) {}

    // Adds one frame's GPU duration.
    void add(frame::Seconds gpu);

    // At `now`, if a period has passed since the last summary and at least
    // one frame was added, the summary of the frames since then, which then
    // starts a new period. Otherwise none. The first call starts the first
    // period.
    std::optional<Summary> take(frame::Seconds now);

private:
    frame::Seconds period_;
    std::optional<frame::Seconds> started_;
    Summary current_;
    double total_ = 0.0;
};

}  // namespace serenity::measurement
