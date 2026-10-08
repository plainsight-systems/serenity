#include "app/clock.h"

#include <SDL3/SDL_timer.h>

namespace serenity::app {

Clock::Clock() : start_ns_(SDL_GetTicksNS()) {}

frame::Seconds Clock::elapsed() const {
    const std::uint64_t now = SDL_GetTicksNS();
    return frame::Seconds(static_cast<double>(now - start_ns_) * 1e-9);
}

}  // namespace serenity::app
