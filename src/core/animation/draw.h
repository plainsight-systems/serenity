#pragma once

#include <cstdint>

namespace serenity::animation {

// Axis: Animation.
//
// Numbers drawn from a seed, for the kinds that draw them (flight.h,
// glow.h): a function of the seed and up to three counters alone, so the
// same seed gives the same numbers under any standard library, where
// <random>'s distributions are free to differ (environmental determinism).
// splitmix64's finalizer (Steele, Lea and Flood 2014), applied to the seed
// and then to each counter in turn; the top 53 bits as a double in [0, 1).
// The wander shares splitmix64 but keeps its own, older keying of it
// (wander.cpp), whose bits its tests pin. Both are constexpr (F.4): pure
// integer arithmetic, the same at compile time as at run time.

constexpr std::uint64_t splitmix64(std::uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

constexpr double draw(std::uint64_t seed, std::uint64_t a, std::uint64_t b = 0, std::uint64_t c = 0) noexcept {
    std::uint64_t h = splitmix64(seed);
    h = splitmix64(h ^ a);
    h = splitmix64(h ^ b);
    h = splitmix64(h ^ c);
    return static_cast<double>(h >> 11) * 0x1.0p-53;
}

}  // namespace serenity::animation
