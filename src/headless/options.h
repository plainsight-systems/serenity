#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>

#include "core/frame/extent.h"
#include "core/output/image_format.h"
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
//                     [--samples N] [--format KIND]
//
// DIRECTORY must be new or empty (prepare_output, below). It renders the
// frame graph in FILE (core/frame/graph_file.h), over the scene in
// --scene's FILE if the graph reads one (core/scene/scene.h), frames
// first .. first + frames - 1, frame i at time i x step, each to
// DIRECTORY/frame-NNNNNN and its kind's extension (i, zero-padded to six
// digits; --format, below). Time is computed, never measured: the same
// command writes the same files, on any run (principle 1). Defaults: one
// frame, from frame 0, a step of 1/60 s, 1920 x 1080, every frame written,
// one sample, Output's default kind.
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
//     (core/frame/history.h). This is how a movie of moving, blinking
//     fireflies is made clean: N samples of each instant.
//
// A graph that converges over nothing renders each frame once, whatever N:
// its samples would be the same image again (core/frame/history.h).
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
// What a written frame is, --format KIND: one of Output's kinds, by the
// name Output gives it (core/output/image_format.h, which lists them, what
// each is written from, and its extension; png by default). This program
// names no kind: it reads back what the kind's source is, the frame as
// displayed or the graph's accumulated image, and hands it to Output's
// write_image(), each frame to frame-NNNNNN and the kind's extension. A
// kind written from the accumulated image needs a graph that accumulates
// (core/frame/history.h, accumulates): asked of one that does not, the run
// is refused by OptionsError naming the graph, after the graph is read and
// before anything renders or the directory is made, rather than at the
// first frame written.
//
// parse() checks everything it can before anything renders and throws
// OptionsError naming the argument (E.2, E.14): an unknown option, a
// missing or malformed value (a --write other than all, last or doubling,
// a --format that names none of Output's kinds; a number with a sign, a
// space or anything after it, read by std::from_chars, E.28), zero frames,
// zero samples or more than an image holds (2^24 - 1,
// frame::max_accumulated_frames, core/frame/frame_inputs.h), a size with a
// zero side, a step that is not positive and finite, a time that is not
// finite or is negative, --time with --step, a missing --graph or --out,
// and a range whose last frame, first + frames - 1, is past 2^32 - 1, for
// every frame renders a sample at least.
//
// The run's last sample index, (first + frames) x N - 1, must be under
// 2^32 too, N being the samples the plan renders a frame
// (core/frame/history.h, plan_headless): --samples for a graph that
// accumulates, 1 for one that does not, whatever --samples says. parse()
// cannot check it alone, for it does not read the graph; check_samples()
// does, once the plan is made, before anything renders or the directory is
// made, and throws OptionsError naming the range. The shaders key every
// random number by an index's low 32 bits (contracts/frame_constants.h,
// frame_index), so a sample at 2^32 + k would draw sample k's numbers
// again: two samples of a run, or a reference's batch and an image judged
// against it, that look independent and are one
// (core/measurement/reference.h). The window's frames wrap there, after
// two years; a headless run is refused before it renders one (E.2, ES.46).
// A run that accumulates more samples across its frames than an image
// holds is refused by the renderer at the first sample past it: whether it
// accumulates across frames depends on the scene. parse() does not touch
// the file system: the graph is read, and the directory made ready
// (prepare_output), after parsing succeeds.
//
// The directory holds this run's frames and nothing else: prepare_output()
// creates it, or takes it if it is empty, and refuses by OptionsError one
// that holds anything, so no frame of an earlier run sits beside this run's
// looking like one of them (principle 1: the same command writes the same
// files). The Makefile's targets clear their own directories first.
class OptionsError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class Write {
    all,
    last,
    doubling,
};


// What parse() read: plain values, which parse() alone makes and checks
// (C.2: a struct, since nothing here keeps an invariant after it).
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
    output::ImageFormat format = output::default_format;
};

// Which of a run's frames: the one `after_first` frames after its first, of
// `frames` in all. A struct, so the two counts are named where they are
// given (I.24).
struct RunFrame {
    std::uint64_t after_first = 0;
    std::uint64_t frames = 0;
};

// Whether `frame` is written. Throws std::logic_error for a Write with no
// rule, rather than writing (P.6).
bool written(Write write, RunFrame frame);

// `args` are the arguments after the program's name.
Options parse(std::span<const char* const> args);

// Throws OptionsError if the run `options` asks for, at `samples_per_frame`
// (the plan's, core/frame/history.h), has a last sample index,
// (first + frames) x samples_per_frame - 1, past 2^32 - 1; see above.
// `options` as parse() made it; `samples_per_frame` from 1 to
// frame::max_accumulated_frames, else std::invalid_argument (I.5).
void check_samples(const Options& options, std::uint64_t samples_per_frame);

// Makes `out` ready for a run's frames: creates it if it does not exist;
// throws OptionsError if it is not an empty directory (see above), or
// std::filesystem::filesystem_error if it cannot be created or read.
void prepare_output(const std::filesystem::path& out);

}  // namespace serenity::headless
