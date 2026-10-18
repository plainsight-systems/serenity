#pragma once

#include <memory>

#include <QuartzCore/QuartzCore.hpp>

#include "core/frame/frame_inputs.h"
#include "metal/device/presenter.h"

namespace serenity::metal {

// Axis: GPU backend.
//
// The window's frames, paced by the display: a Metal display link
// (Core Animation's CAMetalDisplayLink, macOS 14) on the presenter's layer,
// as Apple paces Metal apps ("Achieving smooth frame rates with a Metal
// display link", its sample's GameView; logical-overview.md, "The frame",
// "Pacing"). The display link is the layer's: Core Animation makes it for
// a CAMetalLayer, and it hands over that layer's drawables, so it is the
// backend's, beside the presenter (change-axes.md). What a frame renders is
// the caller's, behind Client.
//
// What it does, as Apple's sample does on the main thread
// (RENDER_ON_MAIN_THREAD):
//   - made for the presenter's layer, with the frame rate asked for and a
//     latency of two frames, and added to the main thread's run loop in
//     its common modes, so it calls the frame whenever that run loop runs,
//     in its default mode and while a window is resized or a menu is
//     tracked alike;
//   - each time the display wants a frame, it calls Client::frame on the
//     main thread with a DisplayUpdate: the drawable to render into and
//     the instant the display is expected to show it (CAMetalDisplayLink
//     Update's targetPresentationTimestamp, which Apple's sample and its
//     documentation animate by);
//   - destroyed, it invalidates the link, which takes it off the run loop
//     and the layer (Apple: "When your app finishes with a display link,
//     call invalidate()"), so no frame is called after.
//
// The run loop, not this object, decides when frames are called: only while
// the main thread runs its run loop, which SDL does while it waits for
// events (src/app/window.h, wait). Nothing here blocks or waits for the
// display; the display link's drawable is ready when the frame is called.
// So the window's pacing is the display's, and nothing in the frame waits
// on a drawable as Presenter used to (nextDrawable).
//
// The rate. CAMetalDisplayLink calls the frame at a rate within the
// FrameRate asked for (preferredFrameRateRange), one the display can show
// and the system allows (Low Power Mode and thermal state lower it); the
// caller asks for one its frames can keep up with (Apple: "Choose a frame
// rate range that your app can consistently maintain"). A FrameRate whose
// rates are not positive and in order (minimum <= preferred <= maximum) is
// refused by MetalError (I.5, E.2).
//
// The latency is two frames (preferredFrameLatency, whose only values are 1
// and 2), as Apple's sample asks, and as many as the submission keeps in
// flight (submission.h, frames_in_flight; checked at compile time): a
// frame is presented before the display's deadline (targetTimestamp) and
// the GPU has another frame's time to finish it. Only present() is used to
// present: a presentation at a set time asserts with a display link
// (Apple, metalDisplayLink(_:needsUpdate:)); Submission::present calls
// present().
//
// The drawable a frame is handed is Core Animation's, valid for the
// call: the frame renders into it and presents it there (Submission::
// present), or leaves it, and Core Animation takes it back. Its size is the
// layer's drawable size when Core Animation made it, which may be one the
// presenter has since changed (presenter.h, resize); the frame checks it
// (metal/frame/renderer.h, render_to_window).
//
// Client is an interface of one function (I.25, C.121), its destructor
// protected and not virtual, since nothing is destroyed through it (C.35).
// frame() is noexcept (F.6, E.12): it is called from Core Animation's run
// loop, through Objective-C and AppKit frames that an exception must not
// unwind; a client that can fail catches there, where it can stop its run
// and say why (E.17: the boundary is where the handling is), as the window
// does (src/app/main.cpp). One indirect call a frame; nothing is allocated
// for it by this object (MEM.9): the update and its drawable are Core
// Animation's.
//
// Threads (CP.1): made, called and destroyed on the main thread only. The
// constructor refuses another thread by MetalError, since the frames would
// be called on a run loop nobody runs. The client and the presenter must
// outlive this object; declared after both, it is destroyed before them
// (R.1).
//
// Copying and moving are deleted (C.21): an object is one link, made once
// for the window's life and invalidated once, as the Window is (src/app/
// window.h). Its Objective-C objects, the link and its delegate, are held
// behind the pointer to State, defined in display_link.mm: I.27's form,
// here to keep a language out of the header rather than for an ABI, so
// this header is C++ and whoever includes it (the app) sees no
// Objective-C.
//
// Not performance-sensitive in itself: one callback a frame, whose cost is
// the frame's.

// A rate range in frames a second (Hz), as CAFrameRateRange takes it.
struct FrameRate {
    float minimum = 0.0F;
    float maximum = 0.0F;
    float preferred = 0.0F;
};

// One frame the display asks for.
struct DisplayUpdate {
    // Core Animation's, valid for the call (above); never null.
    CA::MetalDrawable* drawable = nullptr;
    // When the display is expected to show the frame, host time (the clock
    // CACurrentMediaTime reads), as the display link targets it.
    frame::Seconds presentation{0.0};
};

class DisplayLink {
public:
    class Client {
    public:
        virtual void frame(const DisplayUpdate& update) noexcept = 0;

    protected:
        Client() = default;
        Client(const Client&) = default;
        Client& operator=(const Client&) = default;
        Client(Client&&) = default;
        Client& operator=(Client&&) = default;
        ~Client() = default;
    };

    // Makes the link on `presenter`'s layer at `rate`, adds it to the main
    // run loop, and calls `client` for each frame from then on. Throws
    // MetalError off the main thread, for a rate refused above, or if Core
    // Animation makes no link.
    DisplayLink(const Presenter& presenter, FrameRate rate, Client& client);

    DisplayLink(const DisplayLink&) = delete;
    DisplayLink& operator=(const DisplayLink&) = delete;
    DisplayLink(DisplayLink&&) = delete;
    DisplayLink& operator=(DisplayLink&&) = delete;
    // Invalidates the link: no frame is called after.
    ~DisplayLink();

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace serenity::metal
