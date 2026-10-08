// The C++ half of core/portable_math.h's contract: this build does not
// contract a * b + c, and portable::fma does fuse it. The GPU half, and the
// two halves against each other, are tests/gpu/portable_math_test.cpp.

#include <bit>
#include <cstdint>

#include <doctest/doctest.h>

#include "core/portable_math.h"

namespace {

// a * b is 1 + 2^-11 + 2^-24 exactly. Rounded to float, the 2^-24 is half an
// ulp and rounds to even, away: the product is 1 + 2^-11, and adding c gives
// 0. Fused, nothing is rounded until the end, and the result is 2^-24.
// Volatile, so the compiler cannot fold the expression at compile time.
volatile float a = 1.0f + 0x1p-12f;
volatile float b = 1.0f + 0x1p-12f;
volatile float c = -(1.0f + 0x1p-11f);

}  // namespace

TEST_CASE("this build does not contract a * b + c into a fused multiply-add") {
    const float x = a;
    const float y = b;
    const float z = c;
    const float unfused = x * y + z;
    CHECK(unfused == 0.0f);
}

TEST_CASE("portable::fma fuses, rounding once") {
    CHECK(serenity::portable::fma(a, b, c) == 0x1p-24f);
}

TEST_CASE("portable::sqrt is correctly rounded") {
    CHECK(serenity::portable::sqrt(4.0f) == 2.0f);
    // sqrt(2) rounded to nearest float.
    CHECK(std::bit_cast<std::uint32_t>(serenity::portable::sqrt(2.0f)) == 0x3fb504f3u);
}
