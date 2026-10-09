#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
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
//                     [--frames N] [--first I]
//                     [--step SECONDS | --time SECONDS]
//                     [--size WIDTHxHEIGHT] [--write all|last|doubling]
//                     [--samples N]
//
// It renders the frame graph in FILE (core/frame/graph_file.h), over the
// scene in --scene's FILE if the graph reads one (core/scene/scene.h), frames
// first .. first + frames - 1, frame i at time i x step, each to
// DIRECTORY/frame-NNNNNN.png (i, zero-padded to six digits). Time is
// computed, never measured: the same command writes the same files, on any
// run (principle 1). Defaults: one frame, from frame 0, a step of 1/60 s,
// 1920 x 1080, every frame written, one sample.
//
// --samples: each frame is rendered N times at its instant, as the
// renderer's frames i x N .. i x N + N - 1 (core/frame/frame_inputs.h: the
// index, which seeds every random number, so no two samples of a run share
// one), and written after the last. What a graph that converges over frames
// writes:
//
//   - a still scene, or time frozen by --time: the mean of every sample from
//     frame first's first on, so frame first + k is the mean of (k + 1) x N.
//     It looks the same at every instant, so the whole run converges;
//   - a scene whose shapes move (core/animation/animate.h), time
//     advancing: the mean of frame i's own N samples, started over at each
//     frame, since frames at two instants are two scenes
//     (metal/frame/accumulation.h). This is how a movie of moving fireflies
//     is made clean: N samples of each instant.
//
// A graph that converges over nothing renders each of a frame's samples
// alike, and writes the last.
//
// --time freezes it: every frame is at time SECONDS, while the frame index,
// and with it every random number, still advances. That is how a reference
// is rendered (logical-overview.md: time frozen, thousands of samples): a
// converging graph's frames are then samples of one instant, not a moving
// scene averaged into a blur. --time and --step are exclusive.
//
// Which frames are written, of those rendered:
//   all       every one: a movie's frames;
//   last      the last only: a converged image, a reference;
//   doubling  the 1st, 2nd, 4th, 8th ... and the last: the points of a
//             convergence curve, error against samples on doubling axes.
// counted in frames, not samples. A frame not written is not read back
// either, so `last` costs little more than the GPU's time.
//
// parse() checks everything before anything renders and throws Error naming
// the argument (E.2, E.14): an unknown option, a missing or malformed value (a
// --write other than all, last or doubling), zero frames, zero samples or
// more than an image holds (2^24 - 1, metal/frame/accumulation.h), a size
// with a zero side, a step that is not positive and finite, a time that is
// not finite or is negative, --time with --step, a range whose last sample's
// index, (first + frames) x N - 1, is past the last there can be, a missing
// --graph or --out. A run that accumulates more samples across its frames
// than an image holds is refused by the renderer at the first sample past
// it: whether it accumulates across frames depends on the scene, which
// parse() does not read. It does not touch the file system: the graph is
// read, and the directory created, after parsing succeeds.
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
    std::optional<frame::Seconds> time;  // frozen at this, when given
    frame::Extent size{1920, 1080};
    std::filesystem::path graph;
    std::filesystem::path scene;  // empty when none was given
    std::filesystem::path out;
    Write write = Write::all;
    std::uint64_t samples = 1;  // rendered per frame, at its instant
};

// Whether the frame `n` frames after the first, of `frames`, is written.
bool written(Write write, std::uint64_t n, std::uint64_t frames);

// `args` are the arguments after the program's name.
Options parse(std::span<const char* const> args);

}  // namespace serenity::headless
