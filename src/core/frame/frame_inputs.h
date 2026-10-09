#pragma once

#include <chrono>
#include <cstdint>
#include <optional>

#include "core/contracts/camera.h"

namespace serenity::frame {

// Axis: Frame graph.
//
// What a frame is a function of, beyond the scene and the resources it
// renders into (logical-overview.md, principle 1). The window and the
// headless renderer each build these and hand them to the frame; nothing that
// renders reads a clock or a counter of its own (I.1, F.8). So the same inputs
// render the same image, whichever of the two produced them.
//
// The camera is one of them: the caller decides which camera a frame is seen
// through, and the backend frames and binds what it is given. It holds no
// camera of its own (logical-overview.md, principle 10).

// Time, in seconds, typed so a count of milliseconds or frames cannot be
// passed where seconds are meant (I.4).
using Seconds = std::chrono::duration<double>;

struct FrameInputs {
    // Since the run began. The window measures it from a monotonic clock, the
    // headless renderer computes it as index times a fixed step. Double here;
    // shaders receive it as float (contracts/frame_constants.h), whose spacing
    // is 0.24 ms after an hour and 7.8 ms after a day: animation steps
    // visibly after about a day of running.
    Seconds time{0.0};

    // The frame's position in the run, from 0. Seeds per-frame randomness
    // (metal/sampler/sampler.metal.h).
    std::uint64_t index = 0;

    // The frame from which the image accumulates: a pass that converges over
    // frames shows the mean of frames accumulated_since .. index, so the
    // frame's image is a function of this as well (principle 1). Equal to
    // index to start over. The window starts over whenever what it shows
    // would otherwise change: its size, and later the camera and anything
    // that moves; and when the image would hold 2^24 frames, the most it
    // holds (metal/frame/accumulation.h). The headless renderer accumulates
    // from its first frame, so its frame first + k is the mean of k + 1
    // frames, and refuses more than 2^24 of them. Never after index.
    std::uint64_t accumulated_since = 0;

    // The camera the frame is seen through, at the frame's time
    // (contracts/camera.h). None when the frame has no scene; a frame graph
    // that reads a scene refuses to render without one
    // (metal/frame/renderer.h).
    std::optional<contracts::Camera> camera;
};

}  // namespace serenity::frame
