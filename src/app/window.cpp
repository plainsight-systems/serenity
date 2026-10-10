#include "app/window.h"

#include <climits>
#include <cstdint>

#include <SDL3/SDL.h>

namespace serenity::app {

namespace {

[[noreturn]] void fail(const char* what) {
    throw Error(std::string{what} + ": " + SDL_GetError());
}

// SDL takes a window's size as int.
int points(std::uint32_t side) {
    if (side == 0 || side > INT_MAX) {
        throw Error("a window side of " + std::to_string(side) + " points is not one SDL takes");
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

}  // namespace

Window::SdlVideo::SdlVideo() {
    // SDL's video is the main thread's alone (window.h).
    if (!SDL_IsMainThread()) {
        throw Error("a window must be opened on the main thread");
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

Window::Events Window::poll() {
    Events events;
    SDL_Event event{};
    while (SDL_PollEvent(&event)) {
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
        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
            events.resized = true;
            break;
        default:
            break;
        }
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
