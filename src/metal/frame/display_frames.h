#pragma once

#include <cstdint>
#include <exception>

#include "core/frame/frame_inputs.h"
#include "core/presentation/frame_source.h"
#include "metal/device/display_link.h"
#include "metal/device/presenter.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"

namespace serenity::metal {

// Axis: Frame graph.
//
// Metal's frame source (core/presentation/frame_source.h): the window's
// frames paced by the display, through a display link on the presenter's
// layer (metal/device/display_link.h), as Apple's display link sample
// renders on the main thread. What it hands the window is the seam's: an
// instant, a size, a report; the drawable stays here, the target this
// source hands its own renderer (render_to_window, renderer.h).
//
// run(), on the main thread:
//   - makes the display link, at the rate asked for, for the run's length
//     (R.1, E.6: it is destroyed, and calls no frame after, however run()
//     ends);
//   - loops on the client's events(), each wait at most event_wait, while
//     the main run loop, which the client's event wait runs (SDL's does,
//     app/window.h), calls the display link's frames in between;
//   - ends when events() returns false, or a frame failed (below).
// Which submissions are frames is this source's to know: those from the
// run's start on, start-up work having been submitted before it; only
// they are reported as settled (FrameReport).
//
// Each frame the display link calls, in order:
//   1. the size the client wants; none if it has a side of 0 (a minimized
//      window: nothing is owed one that shows nothing);
//   2. the presenter follows it (presenter.h, resize), which also restores
//      the layer's size if SDL reset it; a change is told to the next frame
//      rendered as View::changed;
//   3. a drawable not the presenter's size, made before the size changed,
//      is left to Core Animation, no frame rendered into it (presenter.h,
//      renders_into): the next one is the new size;
//   4. the client's inputs() for the drawable's presentation instant;
//   5. the frame rendered into the drawable and presented
//      (render_to_window);
//   6. the client's shown(), with the size, the frame its submission
//      settled if one, and the renderer's bug counters.
// One indirect call to the client per step, one per frame each; nothing
// allocated here per frame (MEM.9).
//
// A frame's failure. The display link calls frames from Core Animation's
// run loop, through a callback that must not throw (display_link.h,
// Client): so the callback catches what a frame throws, the client's or
// the renderer's, keeps the first, and calls no frame after it; run()
// notices it when the client's wait returns, at most event_wait later,
// and rethrows it once the link is gone (E.2, E.17: caught at the boundary
// it cannot cross, thrown where the run can end). A failure in events()
// itself goes straight through run().
//
// Threads (CP.1): the main thread only, as the display link. The
// presenter, the submission, the renderer and, during run(), the client
// must outlive what uses them; the client is held only while run() runs.
// Copying and moving are deleted (C.21): the display link holds this
// object as its client while run() runs.
class DisplayFrames final : public presentation::FrameSource, private DisplayLink::Client {
public:
    // Frames at `rate`, refused by MetalError when run() makes the display
    // link if it is not one the link takes (display_link.h).
    DisplayFrames(Submission& submission, Presenter& presenter, Renderer& renderer, presentation::FrameRate rate);

    DisplayFrames(const DisplayFrames&) = delete;
    DisplayFrames& operator=(const DisplayFrames&) = delete;
    DisplayFrames(DisplayFrames&&) = delete;
    DisplayFrames& operator=(DisplayFrames&&) = delete;
    ~DisplayFrames() = default;

    void run(presentation::FrameClient& client) override;

private:
    // The display link's callback; steps 1 to 6 above, failures kept.
    void frame(const DisplayUpdate& update) noexcept override;
    void render(const DisplayUpdate& update, presentation::FrameClient& client);

    Submission& submission_;
    Presenter& presenter_;
    Renderer& renderer_;
    presentation::FrameRate rate_;

    // During run() only.
    presentation::FrameClient* client_ = nullptr;
    std::uint64_t first_frame_ = 0;  // the sequence of the run's first frame's submission
    bool view_changed_ = false;      // the size changed since the last frame rendered
    std::exception_ptr failure_;     // the first a frame threw; frames stop at it
};

}  // namespace serenity::metal
