#pragma once

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
// it, each released by its own handle in reverse order (R.1, R.20); copying
// and moving are deleted, since SDL's video subsystem is one per process.
//
// The window's surface reaches the renderer as metal_layer(): the view's
// CAMetalLayer, as SDL returns it, an opaque pointer. The app wraps it in
// metal::LayerHandle; this file names no Metal type (file-mapping.md: the
// app depends on the backend, and only the backend uses Metal).
//
// Sizes are in pixels, not points: on the development machine's display one
// point is two pixels, and the drawable must match the pixels.
//
// Failure throws Error with SDL's description (E.2, E.5, E.14): SDL cannot
// start its video subsystem, make the window, or make the view.
//
// Not performance-sensitive: poll() drains events once a frame, which costs
// nothing measurable against a frame.

class Error : public std::runtime_error {
public:
    explicit Error(const std::string& what) : std::runtime_error(what) {}
};

class Window {
public:
    // Opens a resizable, high-density window titled `title`, `size` points
    // across, on the main display. Must be called on the main thread.
    Window(const char* title, frame::Extent size);

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;
    Window(Window&&) = delete;
    Window& operator=(Window&&) = delete;
    ~Window();

    // What happened since the last poll.
    struct Events {
        bool quit = false;     // the window was closed, or Escape pressed
        bool resized = false;  // the size in pixels changed
    };
    Events poll();

    frame::Extent size_in_pixels() const;

    // Sets the window's title: where the frame time is shown.
    void set_title(const std::string& title);

    // The Metal view's CAMetalLayer, valid for the Window's lifetime.
    void* metal_layer() const;

private:
    struct SdlVideo;   // SDL_Init and SDL_Quit for the video subsystem
    struct MetalView;  // SDL_Metal_CreateView and SDL_Metal_DestroyView
    struct DestroyWindow {
        void operator()(SDL_Window* window) const;
    };

    // Declaration order is the reverse of release order.
    std::unique_ptr<SdlVideo> video_;
    std::unique_ptr<SDL_Window, DestroyWindow> window_;
    std::unique_ptr<MetalView> view_;
};

}  // namespace serenity::app
