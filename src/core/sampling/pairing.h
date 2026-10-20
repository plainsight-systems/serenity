#pragma once

// Axis: Sampler (ReSTIR PT's paired spatial reuse, its static data).
//
// The reuse textures of ReSTIR PT Enhanced's paired spatial reuse (Lin,
// Kettunen and Wyman 2026, §3): square textures in which every texel is
// linked to exactly one other, each storing the offset to its partner, so
// that a pixel A reading its offset finds B, and B reading its own finds A.
// One shift from A to B and one from B to A then serve both pixels' merges,
// half the shifts of drawing neighbors independently (metal/reuse/
// pairing.metal.h reads them; metal/passes/restir_pt/restir_pt.metal.h
// uses them).
//
// How a texture is built (§3.1): a width x width image, width even, holds
// link indices, texel i the index i / 2, so each link joins two texels side
// by side; then n_sigma tiled 2 x 2 shuffles, each permuting every 2 x 2
// block's four indices at random, every other shuffle's blocks offset
// diagonally by one texel and wrapping at the edges, move each index by a
// random walk; finally each texel is paired with the other texel of its
// link, and the offset to it, wrapped into [-width / 2, width / 2] so the
// texture tiles, is stored. A shuffle moves an index by sigma / sqrt(2)
// over n_sigma of them, so the offset between a link's two texels, the
// difference of two such walks, has a standard deviation of sigma.
//
//   n_sigma = floor(sigma^2 / 2 + 1.46 / sigma - 1.76 / sigma^2
//                   + 0.656 / sigma^3 + 0.5)                      (eq. 3)
//
// the negative powers the paper's fit for small sigma: 128 for sigma = 16,
// and 1 for 0.8, the least sigma, which the paper's footnote gives one
// shuffle (one shuffle of side-by-side links leaves offsets of deviation
// sqrt(2 / 3) = 0.82 a side). The paper's text as extracted loses the
// fit's signs; this reading is the one that meets its footnote (with every
// sign +, 0.8 would take 6 shuffles).
//
// Each offset is stored as two signed 8-bit numbers, x in the low byte and y
// in the high, in a 16-bit word: every width here is at most 254, so a
// wrapped offset is within [-127, 127] (the paper's footnote 2). Pairing is
// an involution by construction: the offset stored at A's partner is the
// negation of A's (both wrapped the same way), which a test holds, with the
// offsets' standard deviation (tests/restir_pt_core_test.cpp).
//
// Static data turned once, at the renderer's construction, into the layout
// the shader reads (CDSA.32): the three textures one after another, in
// contract 15's order (core/contracts/path_reuse_settings.h, pairing_widths
// and pairing_starts). Deterministic: a seed, and a counter-based generator
// of this file's own (splitmix64, whose output is specified to the bit), not
// a standard distribution, whose results the standard leaves to each
// library. Cost: n_sigma passes over width^2 texels, some 1.7 x 10^7 texel
// moves for the three textures at sigma = 16, once.

#include <cstdint>
#include <limits>
#include <vector>

namespace serenity::sampling {

// The widest texture an offset's signed 8-bit coordinates allow: wrapped
// into [-width / 2, width / 2], an offset of a texture this wide or less is
// within [-127, 127] (above). Every one of contract 15's widths is within
// it (pairing.cpp asserts it).
inline constexpr std::uint32_t largest_pairing_width =
    2u * std::numeric_limits<std::int8_t>::max();

// eq. 3's shuffle count for `sigma`. Precondition: sigma at least
// contracts::min_pairing_sigma (0.8), finite.
std::uint32_t shuffle_count(double sigma);

// One reuse texture, width x width offsets row by row, each packed as
// above. Precondition: width even, in [2, largest_pairing_width]; sigma at
// least contracts::min_pairing_sigma, finite.
std::vector<std::uint16_t> reuse_texture(std::uint32_t width, double sigma,
                                         std::uint64_t seed);

// The three textures of contract 15's widths, one after another, each from
// its own seed derived from `seed`.
std::vector<std::uint16_t> reuse_textures(double sigma, std::uint64_t seed);

// The renderers' seed for reuse_textures(): fixed, so every run, on every
// backend, pairs the same pixels.
inline constexpr std::uint64_t reuse_texture_seed = 0x5e2e417f;

// The sigma the reuse textures are built at, for a graph's `sigma` in
// pixels: the same, or, when the light is traced checkerboarded (contract
// 20), sigma over the traced lattice's step, so its neighbors are drawn
// sigma pixels along each axis on the lattice (contracts/checkerboard.h).
// The core's, so every backend builds the same textures (A.1; F.8).
double reuse_texture_sigma(float sigma, bool checkerboard);

// An offset's two coordinates, from its packed word.
struct Offset {
  int x;
  int y;
};
// A packed offset's layout: x in the low byte, y in the high, each a signed
// byte.
inline constexpr unsigned offset_x_mask = 0xffu;
inline constexpr unsigned offset_y_shift = 8u;

constexpr Offset unpack_offset(std::uint16_t packed) {
  return Offset{
      static_cast<int>(static_cast<std::int8_t>(packed & offset_x_mask)),
      static_cast<int>(static_cast<std::int8_t>(packed >> offset_y_shift))};
}

}  // namespace serenity::sampling
