#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"

namespace serenity::frame {

// Axis: Frame graph (history across frames: contract 6's accumulated image,
// and ReSTIR's reservoirs and surface records).
//
// Which frames an accumulated image holds: the rule, and the two plans that
// keep it. Decided here, in the core, so every backend holds the same
// history for the same inputs, and the window and the headless renderer only
// run the frames the plans give them (logical-overview.md, principle 10).
// A backend keeps the image itself, on its GPU, and asks History whether a
// frame may join it (the Metal backend asks in metal/frame/accumulation.h;
// the limit itself is the core's, max_accumulated_frames, frame_inputs.h).
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
// The reuse history: ReSTIR DI's reservoirs and surface records (contracts
// 14 and 5), which its temporal merge carries from one frame to the next
// (frame::reuses_history; metal/reuse/temporal.metal.h). History as the
// accumulated image is: a frame with temporal reuse is a function of the
// frames since its history started, reused_since (frame_inputs.h), so the
// same rule-and-plans shape keeps it, here, for every backend. A backend
// keeps the records and reservoirs on its GPU, and asks ReuseHistory
// whether a frame may join them (metal/frame/frame_images.h, ReuseImages).
//
// The rule. The reservoirs and records hold the frames reused_since ..
// index - 1, of one size, and a frame joins them when:
//
//   - it starts over (reused_since == index): it holds none, the temporal
//     merge merges nothing, and the images are made anew if they have no
//     size yet or another one;
//   - or it is the next frame (index is the one after the last held), of
//     the same reused_since and the same size.
//
// No condition on time: frames at two instants are what temporal reuse
// reuses across, and each merge evaluates last frame's samples as last
// frame lit them and this frame's as this one does (logical-overview.md,
// principle 6), so moving fireflies, a blinking one, and a camera that
// moves never start it over; the camera's motion is the reprojection's to
// follow. And no cap on the frames held: what bounds an old sample's weight
// is the temporal merge's cap on its count, H x M (core/passes/restir_di.h).
// Anything else is refused by HistoryError, as for the accumulated image: a
// frame skipped, a size changed or a reused_since moved without starting
// over, or a reused_since after the frame. Merged in, last frame's
// reservoirs would then be some other frame's, and the frame would claim a
// history it does not have (principle 1).
//
// What starts it over: a new size (the plans below), and a new graph,
// whose renderer makes its images anew. A camera cut would, and is not yet
// a frame input: a camera the caller replaces with another is, to the
// history, a camera that moved far in one frame, and the merges' similarity
// tests and Z then keep the frame unbiased while its reuse finds little
// (metal/reuse/unbiased.metal.h). The input arrives with the first caller
// that cuts.
//
// The postcondition both plans keep (I.7): the same frames since the same
// start give the same reservoirs, and so the same image, in the window, in
// a headless run, and rendered alone.
//
// The window's plan (LiveReuse): starts over when its size changes (the
// caller says) and when a frame's index is not the one after the last it
// planned; never while the scene changes.
//
// The headless renderer's plan. Frame k rendered alone must be frame k of
// a run from the start (principle 2), and a reusing frame is a function of
// every frame since its history started. So a graph that reuses history
// starts its history at the timeline's start, the renderer's frame 0 (the
// first sample of frame 0), and a run asked to start at frame f renders
// frames 0 .. f - 1 first, by the same plan, writing none of them: it
// replays them (HeadlessPlan::first, first_written). Each replayed frame is
// the plan's own sample of it, at its own time, so a still scene's
// accumulated image holds the replayed samples too, exactly as a run from
// frame 0's would. A run from frame 0, as every movie is, replays nothing;
// one from frame f pays f x N renderer frames before its first written one,
// the price of an exact frame f (principle 2), stated rather than avoided
// by starting the history late and rendering another image than the run's.
//
// Not performance-sensitive: a few comparisons a frame.

class HistoryError : public std::runtime_error {
public:
    explicit HistoryError(const std::string& what) : std::runtime_error(what) {}
};

// Whether the scene changes from one instant to another (anything moves or
// glows): two named values, not a bool, so a call says which it means (I.4).
enum class SceneMotion { still, changing };

// Whether what a window shows changed since its last frame (its size).
enum class View { same, changed };

// What a frame joining the image is told.
struct Joined {
    std::uint32_t held = 0;  // frames the image holds before this one
    bool remade = false;     // the image must be made anew, at the frame's size
};

class History {
public:
    // Frame `inputs`, at `size`, of a scene that changes or not, joins the
    // image; see above. Throws HistoryError, changing nothing, if it may not.
    Joined join(const FrameInputs& inputs, Extent size, SceneMotion motion);

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
    explicit LiveHistory(SceneMotion motion) : changes_(motion == SceneMotion::changing) {}

    // accumulated_since for frame `index`, the frames asked for in order;
    // `view`, whether what the window shows changed since the last frame
    // (its size). Throws HistoryError for an index before the one the
    // image started at: frames out of order (I.5).
    std::uint64_t since(std::uint64_t index, View view);

private:
    bool changes_;
    std::uint64_t since_ = 0;
};

// What a frame joining the reuse history is told.
struct ReuseJoined {
    std::uint64_t held = 0;  // frames the history holds before this one: 0, the temporal merge merges nothing
    bool remade = false;     // the records and reservoirs must be made anew, at the frame's size
};

// The reuse history's rule (above), as History is the accumulated image's.
class ReuseHistory {
public:
    // Frame `inputs`, at `size`, joins the reservoirs and records; see
    // above. Throws HistoryError, changing nothing, if it may not.
    ReuseJoined join(const FrameInputs& inputs, Extent size);

private:
    bool made_ = false;  // whether the images have a size yet
    Extent size_;
    std::uint64_t since_ = 0;  // reused_since of the frames held
    std::uint64_t next_ = 0;   // the index of the frame they take next
};

// A running window's reused_since, frame by frame.
class LiveReuse {
public:
    // reused_since for frame `index`; `view`, whether what the window shows
    // changed since the last frame (its size). Starts over at the first
    // frame, when the view changed, and when `index` is not the one after
    // the last asked for. Throws HistoryError for an index at or before the
    // last asked for: frames out of order (I.5).
    std::uint64_t since(std::uint64_t index, View view);

private:
    bool started_ = false;
    std::uint64_t since_ = 0;
    std::uint64_t last_ = 0;
};

// One of the headless renderer's samples: the renderer's frame for it.
struct Sample {
    std::uint64_t index = 0;
    std::uint64_t accumulated_since = 0;
    std::uint64_t reused_since = 0;  // 0 for a graph that reuses history (above); unread otherwise
};

// What the headless renderer is asked to run, each by name, so two counts
// and four flags cannot be passed in each other's places (I.24).
struct HeadlessRun {
    std::uint64_t first = 0;     // the first frame's index
    std::uint64_t samples = 1;   // asked for per frame
    bool accumulates = false;    // the graph accumulates (frame::accumulates)
    bool reuses = false;         // the graph reuses history (frame::reuses_history)
    bool scene_changes = false;  // anything in the scene moves or glows
    bool time_frozen = false;    // every frame at one time
};

// Which sample of which frame, by name (I.24).
struct SampleOf {
    std::uint64_t frame = 0;
    std::uint64_t sample = 0;
};

// How the headless renderer renders a run.
struct HeadlessPlan {
    // The first frame rendered: the run's first, or 0 for a graph that
    // reuses history, which replays the frames before the run's first
    // (above). The plan's samples, accumulated image included, start here.
    std::uint64_t first = 0;
    // The first frame written: the run's first, always. Frames first ..
    // first_written - 1 are rendered and not written.
    std::uint64_t first_written = 0;
    std::uint64_t samples = 1;  // rendered per frame: 1 for a graph that accumulates nothing
    bool instants = false;      // each frame's image its own: the scene changes and time advances

    // Sample `of.sample` of frame `of.frame`. Preconditions, checked (I.5,
    // E.2): of.sample < samples, else std::invalid_argument; and the frame's
    // indices within 64 bits, else std::overflow_error rather than a wrapped
    // index, which would seed a sample with numbers another already used.
    Sample sample(SampleOf of) const;
};

HeadlessPlan plan_headless(const HeadlessRun& run);

}  // namespace serenity::frame
