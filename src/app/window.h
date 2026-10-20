#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"

struct SDL_Window;

namespace serenity::app {

// Axis: Presentation.
//
// The window, through SDL3: the only code that uses SDL, and it keeps SDL's
// headers in its .cpp so nothing that includes this sees them. Plain C++: SDL
// builds the Metal-backed view itself, so no Objective-C is written here.
// One window on each platform, its surface the platform's backend's: on
// macOS a Metal view, on Linux (the RTX 3090 Ti, X11) a window made for
// Vulkan, which the CUDA backend presents through
// (docs/research/2026-10-20-cuda-window.md). Only the surface differs, so
// only its members are the platform's (below).
//
// A Window owns SDL's video subsystem, the window, and on macOS the Metal
// view inside it, each released by its own handle in reverse order (R.1,
// R.20). Copying
// and moving are deleted: SDL's video subsystem is one per process, started
// and stopped by the one Window, which a program opens once and keeps for
// its life, so a moved-from Window, owning nothing, would serve nothing.
//
// The window's surface reaches the renderer as metal_layer() on macOS: the
// view's CAMetalLayer, as SDL returns it, an opaque pointer, which the app
// wraps in metal::LayerHandle; on Linux as vulkan_instance_extensions(),
// what an instance needs for the surface, and make_vulkan_surface(), the
// surface made for an instance, a VkInstance as a pointer and the
// VkSurfaceKHR as its 64 bits, which the caller then owns and destroys
// before the instance. This file names no Metal or Vulkan type
// (file-mapping.md: the app depends on the backend, and only the backend
// uses its API).
//
// Two units of size, kept apart by type (I.4): the window is opened at a
// size in points, the unit window systems lay windows out in (Points), and
// everything rendered is in pixels (frame::Extent, size_in_pixels()): on the
// development machine's display one point is two pixels, and the drawable
// must match the pixels.
//
// SDL sets the view's layer's drawable size to the window's full pixels on
// every resize (its Metal view's event watch); Metal's presenter sets it
// back to the size it renders at on its next frame
// (metal/device/presenter.h, resize).
//
// wait() handles the events, waiting at most as long as it is asked, for
// the frame source (core/presentation/frame_source.h, FrameClient::events),
// which decides how long that is: Metal's waits here while its display
// link calls frames from the main run loop, which SDL runs while it waits
// (nextEventMatchingMask, in the run loop's default mode); a swapchain's
// would ask for no wait at all. The Window knows neither. Its own bound,
// a tenth of a second, holds whatever it is asked: SDL turns an interrupt
// (Control-C, SIGINT) into its quit event only when it next pumps its
// events, which a longer wait would not do until another event came.
//
// Failure throws WindowError with SDL's description (E.2, E.5, E.14): a window
// opened off the main thread, which SDL requires, a size SDL cannot take,
// or SDL unable to start its video subsystem, make the window, or make the
// view.
//
// Not performance-sensitive: wait() returns once an event comes or its
// time passes, and drains the events that came with it.

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
    bool quit = false;  // the window was closed, Escape pressed, or the program
                        // interrupted
  };
  // Blocks until an event comes, or `longest` passes (at most the
  // Window's own bound, above), and returns what came; nothing, if
  // nothing did. A `longest` of 0 does not wait.
  Events wait(frame::Seconds longest);

  frame::Extent size_in_pixels() const;

  // Sets the window's title: where the frame time is shown.
  void set_title(const std::string& title);

#if defined(__APPLE__)
  // The Metal view's CAMetalLayer, valid for the Window's lifetime.
  void* metal_layer() const;
#else
  // The instance extensions a Vulkan surface for the window needs, SDL's
  // strings, valid for the Window's lifetime.
  std::vector<const char*> vulkan_instance_extensions() const;
  // A Vulkan surface for the window on `instance` (a VkInstance), as its 64
  // bits; throws WindowError if SDL cannot make one.
  std::uint64_t make_vulkan_surface(void* instance) const;
#endif

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
#if defined(__APPLE__)
  std::unique_ptr<void, DestroyView> view_;
#endif
};

}  // namespace serenity::app
