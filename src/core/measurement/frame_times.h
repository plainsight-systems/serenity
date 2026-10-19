#pragma once

#include <cstdint>
#include <optional>

#include "core/frame/frame_inputs.h"

namespace serenity::measurement {

// Axis: Measurement.
//
// How long frames took on the GPU, summarized over a reporting period: the
// number of frames, and the mean, the shortest and the longest. A frame's
// GPU time is the time it added to the GPU's timeline: from when the GPU
// began its work, or finished the frame before it if that is later, to when
// it finished; both instants the backend reports (Metal 4's commit
// feedback, metal/device/submission.h), not the CPU's clock around the
// frame, which measures waiting as well as work (GPU.10). When frames run
// one after another, the frame before ends before the next begins, and
// this is the span between its two instants, as it always was. Frames that
// overlap (a denoised frame's image half beside the next frame's light
// half, submission.h, "Two parts") would each count the overlap again by
// their spans, and a frame's span is then its latency, not its cost;
// counted from the frame before's end, the mean is the interval between
// frames, what the frame rate is, while the GPU is the frame's limit. When
// the display asks for frames less often than the GPU could render them
// (the window's display link, metal/device/display_link.h), the GPU waits
// between frames, which no frame counts: the mean is then the GPU's time
// a frame, and the interval between frames is the display's. A frame that
// ends before the frame before it added no time: 0. Both instants are host
// time, on one clock, so their difference is a duration on that clock (TLM.11).
//
// The caller adds each frame once, and only frames: it knows which
// submissions were frames by their sequences, and the backend hands each
// submission's instants back exactly once.
//
// A frame time is a measurement of this machine, this build and this run,
// not a benchmark (TLM.6): whoever shows one names the build type and the
// resolution beside it, and that it was measured in a running window, which
// shares the GPU with whatever else draws, not on an idle GPU. A figure
// quoted as the renderer's speed is measured as docs/research's notes are,
// on the release build, with the GPU otherwise idle.
//
// Deterministic, with no ambient input: add() and take() change the
// summary they keep, and are given everything they use, including the time
// a period is reported at, so a test drives them with numbers (I.1). Not
// pure functions (F.8): they are the summary's update.
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
  // in host time, counted from the end of the frame added before it if
  // that is later (above). Frames are added in the order they were
  // submitted. Precondition, checked (I.5, E.2): `gpu_end` is not before
  // `gpu_start`, or the frame is refused by std::invalid_argument and
  // nothing is added.
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
  frame::Seconds total_{0.0};  // the period's frames' durations, summed (I.4)
  std::optional<frame::Seconds>
      last_end_;  // the end of the frame added last, across periods
};

}  // namespace serenity::measurement
