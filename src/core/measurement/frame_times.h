#pragma once

#include <cstdint>
#include <optional>

#include "core/frame/frame_inputs.h"

namespace serenity::measurement {

// Axis: Measurement.
//
// How long frames took on the GPU, summarized over a reporting period: the
// number of frames, and the mean, the shortest and the longest. A frame's
// GPU time is the span between the two instants the backend reports for it,
// when the GPU began its work and when it finished (Metal 4's commit
// feedback, metal/device/submission.h), not the CPU's clock around the
// frame, which measures waiting as well as work (GPU.10). Both instants are
// host time, on one clock, so their difference is a duration on that clock
// (TLM.11).
//
// The caller adds each frame once, and only frames: it knows which
// submissions were frames by their sequences, and the backend hands each
// submission's instants back exactly once.
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

    // Adds one frame, from when the GPU began it to when it finished, both
    // in host time. `gpu_end` is not before `gpu_start`.
    void add(frame::Seconds gpu_start, frame::Seconds gpu_end);

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
