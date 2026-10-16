#pragma once

#include <cmath>
#include <limits>

namespace serenity::animation {

// Axis: Animation.
//
// A double rounded to a float outward, for the kinds that work a box out in
// double and hand it on in float (wander.cpp's outward(), flight.cpp's
// reach_of()): so the float box holds every point of the double one it
// rounds. Written once for the family (ES.3, F.10); shapes.cpp keeps its own,
// as families depend on contracts, not on each other (file-mapping.md; the
// 2026-10-10 sweep's verdict). `x` within float's range, which each caller
// checks before it asks (ES.46).

// `x` rounded to a float no greater than it: a box's min.
inline float round_down(double x) noexcept {
    const auto f = static_cast<float>(x);
    return static_cast<double>(f) > x ? std::nextafter(f, -std::numeric_limits<float>::infinity()) : f;
}

// `x` rounded to a float no less than it: a box's max.
inline float round_up(double x) noexcept {
    const auto f = static_cast<float>(x);
    return static_cast<double>(f) < x ? std::nextafter(f, std::numeric_limits<float>::infinity()) : f;
}

}  // namespace serenity::animation
