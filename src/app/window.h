#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "core/frame/extent.h"

struct SDL_Window;

namespace serenity::app {

// Axis: Presentation.
//
// The window, through SDL3: the only code that uses SDL, and it keeps SDL's
// headers in its .cpp so nothing that includes this sees them. Plain C++: SDL
// builds the Metal-backed view itself, so no Objective-C is written here.
//
// A Window owns SDL's video subsystem, the window, and the Metal view inside
// it, each released by its own handle in reverse order (R.1, R.20). Copying
// and moving are deleted: SDL's video subsystem is one per process, started
// and stopped by the one Window, which a program opens once and keeps for
// its life, so a moved-from Window, owning nothing, would serve nothing.
//
// The window's surface reaches the renderer as metal_layer(): the view's
// CAMetalLayer, as SDL returns it, an opaque pointer. The app wraps it in
// metal::LayerHandle; this file names no Metal type (file-mapping.md: the
// app depends on the backend, and only the backend uses Metal).
//
// Two units of size, kept apart by type (I.4): the window is opened at a
// size in points, the unit window systems lay windows out in (Points), and
// everything rendered is in pixels (frame::Extent, size_in_pixels()): on the
// development machine's display one point is two pixels, and the drawable
// must match the pixels.
//
// SDL sets the view's layer's drawable size to the window's full pixels on
// every resize (its Metal view's event watch); the renderer's presenter sets
// it back to the size it renders at on its next frame
// (metal/device/presenter.h, resize).
//
// Events and frames share the main thread's run loop (logical-overview.md,
// "The frame", "Pacing"). wait() blocks in SDL until an event comes, and
// SDL waits by running the main run loop in its default mode
// (nextEventMatchingMask), so the display link on the layer calls the
// window's frames during the wait (metal/device/display_link.h): the
// arrangement of Apple's display link sample on the main thread, with SDL's
// wait as the run loop. A wait is bounded, at a tenth of a second: SDL
// turns an interrupt (Control-C, SIGINT) into its quit event only when it
// next pumps its events, which a wait with no bound would not do until
// another event came. Ten empty returns a second, each a pump. wake() ends a wait from inside it, from a frame,
// by an event of the Window's own (SDL_RegisterEvents), which SDL wakes its
// wait for (SDL_PushEvent, its wake-up event): how a frame that failed ends
// the run from the loop, not from the run loop's callout.
//
// Failure throws WindowError with SDL's description (E.2, E.5, E.14): a window
// opened off the main thread, which SDL requires, a size SDL cannot take,
// or SDL unable to start its video subsystem, make the window, make the
// view, or give the Window its event. wake() cannot report a failure to the
// frame it is called from, which must not throw (display_link.h): if SDL
// cannot queue its event, wake() ends the program (std::terminate), rather
// than leave the loop waiting for a wake that never comes (E.26).
//
// Not performance-sensitive: wait() returns once an event comes, and
// drains the events that came with it.

class WindowError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// A window's size in points.
struct Points {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

class Window {
public:
    // Opens a resizable, high-density window titled `title`, `size` across,
    // on the main display. Must be called on the main thread; throws WindowError
    // if it is not.
    Window(const char* title, Points size);

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;
    Window(Window&&) = delete;
    Window& operator=(Window&&) = delete;
    ~Window();

    // What happened since the last wait.
    struct Events {
        bool quit = false;   // the window was closed, or Escape pressed
        bool woken = false;  // wake() was called
    };
    // Blocks until an event comes or the bound passes, the display link
    // calling frames meanwhile (above), and returns what came; nothing, if
    // nothing did.
    Events wait();
    // Ends the wait under way, or the next one, with `woken`. For a frame,
    // so noexcept (above).
    void wake() noexcept;

    frame::Extent size_in_pixels() const;

    // Sets the window's title: where the frame time is shown.
    void set_title(const std::string& title);

    // The Metal view's CAMetalLayer, valid for the Window's lifetime.
    void* metal_layer() const;

private:
    // SDL_Init(SDL_INIT_VIDEO), and at the end SDL_Quit, which shuts down
    // every subsystem: the program starts no other. Held by value (R.5); it
    // is a call and its undoing, with nothing to hand on.
    struct SdlVideo {
        SdlVideo();
        ~SdlVideo();
        SdlVideo(const SdlVideo&) = delete;
        SdlVideo& operator=(const SdlVideo&) = delete;
        SdlVideo(SdlVideo&&) = delete;
        SdlVideo& operator=(SdlVideo&&) = delete;
    };
    struct DestroyWindow {
        void operator()(SDL_Window* window) const;
    };
    // SDL_Metal_DestroyView, for the view SDL_Metal_CreateView made (an
    // SDL_MetalView, which is a void*).
    struct DestroyView {
        void operator()(void* view) const;
    };

    // Declaration order is the reverse of release order.
    SdlVideo video_;
    std::unique_ptr<SDL_Window, DestroyWindow> window_;
    std::unique_ptr<void, DestroyView> view_;
    std::uint32_t wake_event_;  // the Window's own event type, for wake()
};

}  // namespace serenity::app
