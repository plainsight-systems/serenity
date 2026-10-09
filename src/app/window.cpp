#include "app/window.h"

#include <SDL3/SDL.h>

namespace serenity::app {

namespace {

[[noreturn]] void fail(const char* what) {
    throw Error(std::string(what) + ": " + SDL_GetError());
}

}  // namespace

struct Window::SdlVideo {
    SdlVideo() {
        if (!SDL_Init(SDL_INIT_VIDEO)) {
            fail("SDL_Init");
        }
    }
    ~SdlVideo() { SDL_Quit(); }
    SdlVideo(const SdlVideo&) = delete;
    SdlVideo& operator=(const SdlVideo&) = delete;
};

struct Window::MetalView {
    explicit MetalView(SDL_Window* window) : view(SDL_Metal_CreateView(window)) {
        if (view == nullptr) {
            fail("SDL_Metal_CreateView");
        }
    }
    ~MetalView() { SDL_Metal_DestroyView(view); }
    MetalView(const MetalView&) = delete;
    MetalView& operator=(const MetalView&) = delete;

    SDL_MetalView view;
};

void Window::DestroyWindow::operator()(SDL_Window* window) const {
    SDL_DestroyWindow(window);
}

Window::Window(const char* title, frame::Extent size) : video_(std::make_unique<SdlVideo>()) {
    window_.reset(SDL_CreateWindow(title, static_cast<int>(size.width), static_cast<int>(size.height),
                                   SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY | SDL_WINDOW_METAL));
    if (!window_) {
        fail("SDL_CreateWindow");
    }
    view_ = std::make_unique<MetalView>(window_.get());
}

Window::~Window() = default;

Window::Events Window::poll() {
    Events events;
    SDL_Event event;
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
    return frame::Extent{static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height)};
}

void Window::set_title(const std::string& title) {
    if (!SDL_SetWindowTitle(window_.get(), title.c_str())) {
        fail("SDL_SetWindowTitle");
    }
}

void* Window::metal_layer() const {
    return SDL_Metal_GetLayer(view_->view);
}

}  // namespace serenity::app
