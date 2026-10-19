#include "app/clock.h"

namespace serenity::app {

frame::Seconds PresentationClock::at(frame::Seconds presentation) {
  if (!first_) {
    first_ = presentation;
  }
  return presentation - *first_;
}

}  // namespace serenity::app
