#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>

#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"

namespace serenity::headless {

// Axis: Presentation (headless).
//
// What the headless renderer is asked to render, from its command line:
//
//   serenity-headless --scene FILE --out DIRECTORY
//                     [--frames N] [--first I] [--step SECONDS]
//                     [--size WIDTHxHEIGHT]
//
// It renders the scene in FILE (core/scene/scene.h), frames first ..
// first + frames - 1, frame i at time i x step,
// each to DIRECTORY/frame-NNNNNN.png (i, zero-padded to six digits). Time is
// computed, never measured: the same command writes the same files, on any
// run (principle 1). Defaults: one frame, from frame 0, a step of 1/60 s,
// 1920 x 1080.
//
// parse() checks everything before anything renders and throws Error naming
// the argument (E.2, E.14): an unknown option, a missing or malformed value,
// zero frames, a size with a zero side, a step that is not positive and
// finite, a missing --scene or --out. It does not touch the file system: the
// scene is read, and the directory created, after parsing succeeds.
class Error : public std::runtime_error {
public:
    explicit Error(const std::string& what) : std::runtime_error(what) {}
};

struct Options {
    std::uint64_t frames = 1;
    std::uint64_t first = 0;
    frame::Seconds step{1.0 / 60.0};
    frame::Extent size{1920, 1080};
    std::filesystem::path scene;
    std::filesystem::path out;
};

// `args` are the arguments after the program's name.
Options parse(std::span<const char* const> args);

}  // namespace serenity::headless
