#pragma once

#include <optional>

#include "core/frame/frame_inputs.h"

namespace serenity::app {

// Axis: Presentation.
//
// The window's time: when the display will show each frame, counted from
// the first. The display link hands each frame the instant the display is
// expected to show it (metal/device/display_link.h, DisplayUpdate), and
// the window passes that instant, from this clock, as the frame's time
// (frame/frame_inputs.h), as Apple's display link sample and documentation
// animate by targetPresentationTimestamp: what moves, moves by the time
// between showings, not by when the CPU happened to record the frame
// (logical-overview.md, "The frame", "Pacing"). The renderer never reads a
// clock (principle 1, I.1). The headless renderer has no clock at all: its
// time is the frame's index times a fixed step.
//
// The first instant given is time 0; each later one is its distance from
// the first. Instants are given as the display link gives them, in order;
// the clock passes on whatever it is given and checks nothing, since a
// frame is a function of any time (principle 1). The GPU's instants
// (metal/device/submission.h) are host time on the same clock as the
// display link's, but the frame times subtract only GPU instants from GPU
// instants (core/measurement/frame_times.h), so nothing mixes the two
// (TLM.11).
//
// Deterministic, with no ambient input (I.1): at() is given the instant,
// so a test drives it with numbers. Not a pure function (F.8): the first
// call sets the origin.
class PresentationClock {
public:
    frame::Seconds at(frame::Seconds presentation);

private:
    std::optional<frame::Seconds> first_;
};

}  // namespace serenity::app
