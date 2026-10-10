#include "app/clock.h"

namespace serenity::app {

Clock::Clock() : start_(std::chrono::steady_clock::now()) {}

frame::Seconds Clock::elapsed() const {
    return std::chrono::steady_clock::now() - start_;
}

}  // namespace serenity::app
