#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"

namespace serenity::frame {

// Axis: Frame graph (history across frames: contract 6's accumulated image).
//
// Which frames an accumulated image holds: the rule, and the two plans that
// keep it. Decided here, in the core, so every backend holds the same
// history for the same inputs, and the window and the headless renderer only
// run the frames the plans give them (logical-overview.md, principle 10).
// A backend keeps the image itself, on its GPU, and asks History whether a
// frame may join it (metal/frame/accumulation.h).
//
// The rule. An image holds the frames accumulated_since .. index - 1 of one
// size, and a frame joins it, holding the count of frames before it, when:
//
//   - it starts over (accumulated_since == index): it holds none, and the
//     image is made anew if it has no size yet or another one;
//   - or it is the next frame (index is the one after the last held), of
//     the same accumulated_since and the same size, and, when the scene
//     changes (core/animation/animate.h: anything moves or glows), at the
//     same time exactly: frames at two instants are two scenes, and their
//     mean a blur no camera sees. A still scene looks the same at every t,
//     so its frames join whatever their times;
//   - and the image would hold at most 2^24 - 1 frames
//     (max_accumulated_frames), past which a pixel's count, a float, is no
//     longer exact.
//
// Anything else is refused by HistoryError: a frame skipped, a size changed
// or an accumulated_since moved without starting over, an accumulated_since
// after the frame, another instant of a changing scene, or one frame too
// many.
// The image would then hold something other than what the frame claims to
// show, and a frame that lies about its image is refused, not shown
// (principle 1).
//
// The window's plan (LiveHistory): its time always advances, so it starts
// over whenever what it shows would otherwise change: its size or what it
// views (the caller says), every frame while the scene changes, and when the
// image is full.
//
// The headless renderer's plan (headless_sample): each of its frames is N
// samples of its instant, the renderer's frames i x N .. i x N + N - 1
// (the index, which seeds every random number, so no two samples of a run
// share one). Its image holds:
//
//   - a still scene's, or frozen time's, every sample from its first
//     frame's first on: it looks the same at every frame's time, so the
//     whole run converges;
//   - a changing scene's, time advancing, frame i's own N samples, started
//     over at each frame.
//
// A graph that accumulates nothing (frame::accumulates) has one sample a
// frame, whatever N: its frames are the same function of the same inputs, so
// a second would render the same image again.
//
// Not performance-sensitive: a few comparisons a frame.

class HistoryError : public std::runtime_error {
public:
    explicit HistoryError(const std::string& what) : std::runtime_error(what) {}
};

// What a frame joining the image is told.
struct Joined {
    std::uint32_t held = 0;  // frames the image holds before this one
    bool remade = false;     // the image must be made anew, at the frame's size
};

class History {
public:
    // Frame `inputs`, at `size`, of a scene that changes or not, joins the
    // image; see above. Throws HistoryError, changing nothing, if it may not.
    Joined join(const FrameInputs& inputs, Extent size, bool scene_changes);

private:
    bool made_ = false;  // whether the image has a size yet
    Extent size_;
    std::uint64_t since_ = 0;  // accumulated_since of the frames it holds
    std::uint64_t next_ = 0;   // the index of the frame it takes next
    Seconds time_{0.0};        // the time of the frames it holds
};

// A running window's accumulated_since, frame by frame.
class LiveHistory {
public:
    explicit LiveHistory(bool scene_changes) : changes_(scene_changes) {}

    // accumulated_since for frame `index`, the frames asked for in order;
    // `view_changed` when what the window shows changed since the last
    // frame (its size). Throws HistoryError for an index before the one the
    // image started at: frames out of order (I.5).
    std::uint64_t since(std::uint64_t index, bool view_changed);

private:
    bool changes_;
    std::uint64_t since_ = 0;
};

// One of the headless renderer's samples: the renderer's frame for it.
struct Sample {
    std::uint64_t index = 0;
    std::uint64_t accumulated_since = 0;
};

// How the headless renderer renders: `samples` asked for per frame, from
// frame `first`, for a graph that accumulates or not, a scene that changes or
// not, time frozen or not.
struct HeadlessPlan {
    std::uint64_t first = 0;
    std::uint64_t samples = 1;  // rendered per frame: 1 for a graph that accumulates nothing
    bool instants = false;      // each frame's image its own: the scene changes and time advances

    // Sample `s` of frame `frame`. Preconditions, checked (I.5, E.2): s <
    // samples, else std::invalid_argument; and the frame's indices within
    // 64 bits, else std::overflow_error rather than a wrapped index, which
    // would seed a sample with numbers another already used.
    Sample sample(std::uint64_t frame, std::uint64_t s) const;
};

HeadlessPlan plan_headless(std::uint64_t first, std::uint64_t samples, bool accumulates, bool scene_changes,
                           bool time_frozen);

}  // namespace serenity::frame
