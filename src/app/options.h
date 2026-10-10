#pragma once

#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>

#include "core/frame/extent.h"

namespace serenity::app {

// Axis: Presentation.
//
// What the window is asked to show, from its command line:
//
//   serenity --graph FILE [--scene FILE] [--scale S]
//
// The frame graph is required: there is no default and no path baked into
// the program, so what the window shows is always named where it is run.
// The scene is required only by a graph whose passes read one, and the
// renderer says so if it is missing (metal/frame/renderer.h).
//
// The scale is the size frames are rendered at, as a fraction of the
// window's pixels, in (0, 1]; 1 by default. Below 1 the window's layer
// stretches each frame to fill the window, with its own linear filtering:
// fewer pixels to render, for interactivity, and nothing else. It is not
// upscaling in the frame graph's sense: no pass runs and no history is
// kept. The GPU time the title shows is the frame's, at the size rendered;
// the stretch is the layer's, outside it.
//
// parse() throws OptionsError naming the argument for an unknown option, a
// missing or malformed value, a scale outside (0, 1], or a missing --graph
// (E.2, E.14). Numbers are read whole, by std::from_chars: no sign, space
// or locale (E.28). It reads neither file; they are loaded after parsing
// succeeds (core/frame/graph_file.h, core/scene/scene.h).
class OptionsError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// What parse() read: plain values, which parse() alone makes and checks
// (C.2: a struct, since nothing here keeps an invariant after it).
struct Options {
    std::filesystem::path graph;
    std::filesystem::path scene;  // empty when none was given
    double scale = 1.0;           // in (0, 1]
};

// The size frames are rendered at for a window of `window` pixels at
// `scale`: each side scaled and rounded to the nearest pixel, and at least 1.
// Throws OptionsError for a scale outside (0, 1], as parse() refuses it
// (I.5, I.6).
frame::Extent render_size(frame::Extent window, double scale);

// `args` are the arguments after the program's name.
Options parse(std::span<const char* const> args);

}  // namespace serenity::app
