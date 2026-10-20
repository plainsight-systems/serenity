#include "headless/times.h"

#include <chrono>
#include <format>
#include <string>

namespace serenity::headless {

std::string times_line(frame::Extent size,
                       const measurement::Summary& summary) {
  using Milliseconds = std::chrono::duration<double, std::milli>;
  return std::format(
      "{} x {}: GPU {:.2f} ms mean, {:.2f} to {:.2f}, over {} frames "
      "(headless)",
      size.width, size.height, Milliseconds{summary.mean}.count(),
      Milliseconds{summary.shortest}.count(),
      Milliseconds{summary.longest}.count(), summary.frames);
}

}  // namespace serenity::headless
