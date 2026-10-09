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
//   serenity-headless --graph FILE [--scene FILE] --out DIRECTORY
//                     [--frames N] [--first I] [--step SECONDS]
//                     [--size WIDTHxHEIGHT] [--write all|last|doubling]
//
// It renders the frame graph in FILE (core/frame/graph_file.h), over the
// scene in --scene's FILE if the graph reads one (core/scene/scene.h), frames
// first .. first + frames - 1, frame i at time i x step, accumulating from
// frame first (core/frame/frame_inputs.h): a graph that converges over
// frames writes its frame first + k as the mean of k + 1 frames,
// each to DIRECTORY/frame-NNNNNN.png (i, zero-padded to six digits). Time is
// computed, never measured: the same command writes the same files, on any
// run (principle 1). Defaults: one frame, from frame 0, a step of 1/60 s,
// 1920 x 1080, every frame written.
//
// Which frames are written, of those rendered:
//   all       every one: a movie's frames;
//   last      the last only: a converged image, a reference;
//   doubling  the 1st, 2nd, 4th, 8th ... and the last: the points of a
//             convergence curve, error against samples on doubling axes.
// A frame not written is not read back either, so `last` costs little more
// than the GPU's time.
//
// parse() checks everything before anything renders and throws Error naming
// the argument (E.2, E.14): an unknown option, a missing or malformed value
// (a --write other than all, last or doubling),
// zero frames, a size with a zero side, a step that is not positive and
// finite, a range past the last frame there can be, a missing --graph or
// --out. It does not touch the file system: the graph is read, and the
// directory created, after parsing succeeds.
class Error : public std::runtime_error {
public:
    explicit Error(const std::string& what) : std::runtime_error(what) {}
};

enum class Write {
    all,
    last,
    doubling,
};

struct Options {
    std::uint64_t frames = 1;
    std::uint64_t first = 0;
    frame::Seconds step{1.0 / 60.0};
    frame::Extent size{1920, 1080};
    std::filesystem::path graph;
    std::filesystem::path scene;  // empty when none was given
    std::filesystem::path out;
    Write write = Write::all;
};

// Whether the frame `n` frames after the first, of `frames`, is written.
bool written(Write write, std::uint64_t n, std::uint64_t frames);

// `args` are the arguments after the program's name.
Options parse(std::span<const char* const> args);

}  // namespace serenity::headless
