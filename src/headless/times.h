#pragma once

#include <string>

#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/measurement/frame_times.h"

namespace serenity::headless {

// Axis: Presentation (headless): what --times prints (options.h).
//
// The period --times reports over, in GPU time, and the line it prints for
// each, apart from the run so a test reads the line without a backend
// (F.8: a pure function of its arguments).

// A second of GPU time, as the window's title reports (app/window_frames.h).
inline constexpr frame::Seconds times_period{1.0};

// The line for a period's summary at `size`: its mean, shortest and
// longest frame in milliseconds to two places, and its frames, beside the
// size and that the headless renderer measured it, so a figure is never
// quoted without them (TLM.6).
std::string times_line(frame::Extent size, const measurement::Summary& summary);

}  // namespace serenity::headless
