#pragma once

#include <cstdint>
#include <string>

namespace serenity::frame {

// Axis: Frame graph.
//
// The size of an image in pixels: the window's drawable, a headless target,
// any image between passes. Width and height are both at least 1 wherever an
// Extent describes a real image; an Extent of zero is only ever "no image
// yet", and nothing renders into it.
struct Extent {
  std::uint32_t width = 0;
  std::uint32_t height = 0;

  friend bool operator==(const Extent&, const Extent&) = default;
};

// "W x H": a size as every message names one, written once (ES.3).
inline std::string to_string(Extent extent) {
  return std::to_string(extent.width) + " x " + std::to_string(extent.height);
}

}  // namespace serenity::frame
