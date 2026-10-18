#include "app/window.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>

#include <SDL3/SDL.h>

namespace serenity::app {

namespace {

[[noreturn]] void fail(const char* what) {
    throw WindowError(std::string{what} + ": " + SDL_GetError());
}

// SDL takes a window's size as int.
int points(std::uint32_t side) {
    if (side == 0 || side > INT_MAX) {
        throw WindowError("a window side of " + std::to_string(side) + " points is not one SDL takes");
    }
    return static_cast<int>(side);
}

SDL_Window* open(const char* title, Points size) {
    SDL_Window* window = SDL_CreateWindow(title, points(size.width), points(size.height),
                                          SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY | SDL_WINDOW_METAL);
    if (!window) {
        fail("SDL_CreateWindow");
    }
    return window;
}

void* make_view(SDL_Window* window) {
    SDL_MetalView view = SDL_Metal_CreateView(window);
    if (!view) {
        fail("SDL_Metal_CreateView");
    }
    return view;
}

// The longest a wait waits, whatever it is asked (window.h): a tenth of a
// second, so an interrupt from the terminal ends the run in that time.
constexpr std::chrono::milliseconds wait_bound{100};

}  // namespace

Window::SdlVideo::SdlVideo() {
    // SDL's video is the main thread's alone (window.h).
    if (!SDL_IsMainThread()) {
        throw WindowError("a window must be opened on the main thread");
    }
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        fail("SDL_Init");
    }
}

Window::SdlVideo::~SdlVideo() {
    SDL_Quit();
}

void Window::DestroyWindow::operator()(SDL_Window* window) const {
    SDL_DestroyWindow(window);
}

void Window::DestroyView::operator()(void* view) const {
    SDL_Metal_DestroyView(view);
}

Window::Window(const char* title, Points size) : window_(open(title, size)), view_(make_view(window_.get())) {}

Window::~Window() = default;

Window::Events Window::wait(frame::Seconds longest) {
    Events events;
    const auto take = [&](const SDL_Event& event) {
        switch (event.type) {
        case SDL_EVENT_QUIT:
        case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
            events.quit = true;
            break;
        case SDL_EVENT_KEY_DOWN:
            if (event.key.key == SDLK_ESCAPE) {
                events.quit = true;
            }
            break;
        default:
            break;
        }
    };
    // One event waited for, at most `longest` and at most wait_bound, then
    // those that came with it, or the signals SDL turns into events when it
    // pumps them (window.h). Whole milliseconds, as SDL takes them; a
    // `longest` under one (or not a number) does not wait.
    const double asked = std::chrono::duration<double, std::milli>{longest}.count();
    const double bound = std::chrono::duration<double, std::milli>{wait_bound}.count();
    const Sint32 wait_ms = asked >= 1.0 ? static_cast<Sint32>(std::min(asked, bound)) : 0;
    SDL_Event event{};
    if (wait_ms > 0 && SDL_WaitEventTimeout(&event, wait_ms)) {
        take(event);
    }
    while (SDL_PollEvent(&event)) {
        take(event);
    }
    return events;
}

frame::Extent Window::size_in_pixels() const {
    int width = 0;
    int height = 0;
    if (!SDL_GetWindowSizeInPixels(window_.get(), &width, &height)) {
        fail("SDL_GetWindowSizeInPixels");
    }
    // A minimized window may report nothing: 0, never negative.
    return frame::Extent{static_cast<std::uint32_t>(width > 0 ? width : 0),
                         static_cast<std::uint32_t>(height > 0 ? height : 0)};
}

void Window::set_title(const std::string& title) {
    if (!SDL_SetWindowTitle(window_.get(), title.c_str())) {
        fail("SDL_SetWindowTitle");
    }
}

void* Window::metal_layer() const {
    return SDL_Metal_GetLayer(view_.get());
}

}  // namespace serenity::app
