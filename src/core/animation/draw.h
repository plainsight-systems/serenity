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

// splitmix64's increment (2^64 over the golden ratio) and its finalizer's
// shifts and multipliers, as its authors give them (ES.45).
inline constexpr std::uint64_t splitmix64_gamma = 0x9e3779b97f4a7c15ull;
inline constexpr int splitmix64_shift_1 = 30;
inline constexpr std::uint64_t splitmix64_multiplier_1 = 0xbf58476d1ce4e5b9ull;
inline constexpr int splitmix64_shift_2 = 27;
inline constexpr std::uint64_t splitmix64_multiplier_2 = 0x94d049bb133111ebull;
inline constexpr int splitmix64_shift_3 = 31;

// A double in [0, 1) from 64 bits: the top 53, a double's mantissa, times
// 2^-53.
inline constexpr int unit_dropped_bits = 64 - 53;
inline constexpr double unit_scale = 0x1.0p-53;

constexpr std::uint64_t splitmix64(std::uint64_t x) noexcept {
  x += splitmix64_gamma;
  x = (x ^ (x >> splitmix64_shift_1)) * splitmix64_multiplier_1;
  x = (x ^ (x >> splitmix64_shift_2)) * splitmix64_multiplier_2;
  return x ^ (x >> splitmix64_shift_3);
}

// The double in [0, 1) the top 53 bits of `bits` are.
constexpr double unit_of(std::uint64_t bits) noexcept {
  return static_cast<double>(bits >> unit_dropped_bits) * unit_scale;
}

constexpr double draw(std::uint64_t seed, std::uint64_t a, std::uint64_t b = 0,
                      std::uint64_t c = 0) noexcept {
  std::uint64_t h = splitmix64(seed);
  h = splitmix64(h ^ a);
  h = splitmix64(h ^ b);
  h = splitmix64(h ^ c);
  return unit_of(h);
}

}  // namespace serenity::animation
