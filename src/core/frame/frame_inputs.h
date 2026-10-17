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

// The most frames an accumulated image holds before a frame joins it: a
// pixel's count is a float, exact to 2^24 (metal/film/accumulate.metal.h).
// A caller starts over (accumulated_since = index) before index -
// accumulated_since would pass it.
inline constexpr std::uint64_t max_accumulated_frames = (std::uint64_t{1} << 24) - 1;

// The frame before a frame, as the frame's motion is measured from it
// (contract 17, the features' motion): its time and the camera it was seen
// through. A struct, so the two are named where they are given (I.24).
struct PreviousFrame {
    Seconds time{0.0};
    contracts::Camera camera;
};

struct FrameInputs {
    // Since the run began. The window measures it from a monotonic clock, the
    // headless renderer computes it as index times a fixed step. Double here;
    // shaders receive it as float (contracts/frame_constants.h), whose spacing
    // is 0.24 ms after an hour and 7.8 ms after a day: what a shader
    // animates by it steps visibly after about a day of running. The scene's
    // motions are evaluated from it on the CPU, in double
    // (core/animation/motion.h), and do not step.
    Seconds time{0.0};

    // The frame's position in the run, from 0. Seeds per-frame randomness
    // (metal/sampler/sampler.metal.h).
    std::uint64_t index = 0;

    // The frame from which the image accumulates: a pass that converges over
    // frames shows the mean of frames accumulated_since .. index, so the
    // frame's image is a function of this as well (principle 1). Equal to
    // index to start over. Which frames may share an image, and the window's
    // and the headless renderer's plans for when to start over, are the core's
    // (frame/history.h): the window starts over when its size changes, every
    // frame while the scene's shapes move, and when the image is full; the
    // headless renderer renders each frame as samples of one instant,
    // accumulating across frames only while the scene looks the same at each.
    // Never after index.
    std::uint64_t accumulated_since = 0;

    // The camera the frame is seen through, at the frame's time
    // (contracts/camera.h). None when the frame has no scene; a frame graph
    // that reads a scene refuses to render without one
    // (metal/frame/renderer.h).
    std::optional<contracts::Camera> camera;

    // The frame from which a graph that reuses history (frame::
    // reuses_history: ReSTIR PT's temporal reuse, the denoiser) carries its
    // reservoirs and primary hits: frame index's image is a function of the frames
    // reused_since .. index (principle 1), as an accumulated image is of
    // accumulated_since's. Equal to index to start over. A history apart
    // from the accumulated image's: a changing scene starts its accumulated
    // image over every frame and keeps this one, since following a moving
    // light is what temporal reuse is for. Which frames may share it, and
    // the window's and the headless renderer's plans, are the core's
    // (frame/history.h, ReuseHistory). Unread by a graph that does not
    // reuse history. Never after index. Last, after the members callers
    // name in order.
    std::uint64_t reused_since = 0;

    // The frame before this one, which the features' motion vectors point
    // back to (core/contracts/features.h): where the camera and the moving
    // shapes were then. An input, not the renderer's memory of the last
    // frame it recorded, because a frame of N samples is N recordings at one
    // instant, and a headless run that starts at frame f has recorded none
    // before it: the caller knows which frame came before, the renderer
    // does not (principle 1). None at the first frame of a window or of a
    // timeline, where the motion is 0 (as DLSS's reset). Read only by a
    // graph with the features pass; none needed by any other. Last, after
    // the members callers name in order.
    std::optional<PreviousFrame> previous;
};

}  // namespace serenity::frame
