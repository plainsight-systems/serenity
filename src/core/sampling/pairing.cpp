#include "core/sampling/pairing.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/contracts/path_reuse_settings.h"

namespace serenity::sampling {

namespace {

// splitmix64 (Steele, Lea and Flood 2014): a counter-based generator whose
// output is a fixed function of its state, the same on every platform. Its
// increment (2^64 over the golden ratio) and finalizer, as its authors give
// them (ES.45; animation/draw.h finalizes counters with the same).
class Generator {
  static constexpr std::uint64_t gamma = 0x9e3779b97f4a7c15ull;
  static constexpr unsigned shift_1 = 30u;
  static constexpr std::uint64_t multiplier_1 = 0xbf58476d1ce4e5b9ull;
  static constexpr unsigned shift_2 = 27u;
  static constexpr std::uint64_t multiplier_2 = 0x94d049bb133111ebull;
  static constexpr unsigned shift_3 = 31u;

 public:
  explicit Generator(std::uint64_t seed) : state_(seed) {}

  std::uint64_t next() {
    std::uint64_t z = (state_ += gamma);
    z = (z ^ (z >> shift_1)) * multiplier_1;
    z = (z ^ (z >> shift_2)) * multiplier_2;
    return z ^ (z >> shift_3);
  }

  // A number in [0, n), n at most 24: the bias of taking the remainder of
  // 64 bits is under 2^-59.
  std::uint32_t below(std::uint32_t n) {
    return static_cast<std::uint32_t>(next() % n);
  }

 private:
  std::uint64_t state_;
};

// The 24 orders of four things, each a block's texels' new order.
constexpr std::array<std::array<std::uint8_t, 4>, 24> orders = {{
    {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2},
    {0, 3, 2, 1}, {1, 0, 2, 3}, {1, 0, 3, 2}, {1, 2, 0, 3}, {1, 2, 3, 0},
    {1, 3, 0, 2}, {1, 3, 2, 0}, {2, 0, 1, 3}, {2, 0, 3, 1}, {2, 1, 0, 3},
    {2, 1, 3, 0}, {2, 3, 0, 1}, {2, 3, 1, 0}, {3, 0, 1, 2}, {3, 0, 2, 1},
    {3, 1, 0, 2}, {3, 1, 2, 0}, {3, 2, 0, 1}, {3, 2, 1, 0},
}};

// Each texture's seed, from the one given: times xorshift64*'s multiplier
// (Vigna 2016), an odd constant that sends consecutive seeds far apart, plus
// the texture's index (reuse_textures).
constexpr std::uint64_t texture_seed_step = 0x2545f4914f6cdd1dull;

// Every width contract 15 lays out is one reuse_texture() takes.
static_assert(
    [] {
      for (const std::uint32_t width : contracts::pairing_widths) {
        if (width < 2u || width > largest_pairing_width || width % 2u != 0u) {
          return false;
        }
      }
      return true;
    }(),
    "contract 15's pairing widths are even, within [2, largest_pairing_width]");

// `d` wrapped into [-width / 2, width / 2] (above, "wrapped"): a link longer
// than half the texture is the shorter one across its edge.
int wrapped(int d, int width) {
  if (d > width / 2) {
    return d - width;
  }
  if (d < -width / 2) {
    return d + width;
  }
  return d;
}

std::uint16_t packed(int x, int y) {
  const auto low = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(x)));
  const auto high = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(y)));
  return static_cast<std::uint16_t>(low | (high << offset_y_shift));
}

}  // namespace

std::uint32_t shuffle_count(double sigma) {
  if (!std::isfinite(sigma) ||
      sigma < static_cast<double>(contracts::min_pairing_sigma)) {
    throw std::invalid_argument(
        "shuffle_count: sigma " + std::to_string(sigma) + " is not at least " +
        std::format("{}", contracts::min_pairing_sigma));
  }
  const double n = sigma * sigma / 2.0 + 1.46 / sigma - 1.76 / (sigma * sigma) +
                   0.656 / (sigma * sigma * sigma) + 0.5;
  return static_cast<std::uint32_t>(std::floor(n));
}

std::vector<std::uint16_t> reuse_texture(std::uint32_t width, double sigma,
                                         std::uint64_t seed) {
  if (width < 2 || width > largest_pairing_width || width % 2 != 0) {
    throw std::invalid_argument("reuse_texture: width " +
                                std::to_string(width) +
                                " is not even within [2, " +
                                std::to_string(largest_pairing_width) + "]");
  }
  const std::uint32_t shuffles = shuffle_count(sigma);
  const std::size_t texels = std::size_t{width} * width;
  std::vector<std::uint32_t> links(texels);
  for (std::size_t i = 0; i < texels; ++i) {
    links[i] = static_cast<std::uint32_t>(i / 2);
  }
  Generator random(seed);
  const std::uint32_t blocks = width / 2;
  for (std::uint32_t pass = 0; pass < shuffles; ++pass) {
    // every other pass's blocks offset diagonally by one
    const std::uint32_t shift = pass % 2;
    for (std::uint32_t by = 0; by < blocks; ++by) {
      for (std::uint32_t bx = 0; bx < blocks; ++bx) {
        const std::uint32_t x0 = 2 * bx + shift;
        const std::uint32_t y0 = 2 * by + shift;
        const std::uint32_t x1 = (x0 + 1) % width;
        const std::uint32_t y1 = (y0 + 1) % width;
        const std::array<std::size_t, 4> at = {
            std::size_t{y0 % width} * width + x0 % width,
            std::size_t{y0 % width} * width + x1,
            std::size_t{y1} * width + x0 % width, std::size_t{y1} * width + x1};
        const std::array<std::uint32_t, 4> held = {links[at[0]], links[at[1]],
                                                   links[at[2]], links[at[3]]};
        const std::array<std::uint8_t, 4>& order =
            orders.at(random.below(static_cast<std::uint32_t>(orders.size())));
        for (std::size_t i = 0; i < 4; ++i) {
          links[at.at(i)] = held.at(order.at(i));
        }
      }
    }
  }
  // Each link's two texels, then each one's offset to the other.
  constexpr std::uint32_t none = 0xffffffffu;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> ends(texels / 2,
                                                            {none, none});
  for (std::size_t i = 0; i < texels; ++i) {
    auto& end = ends[links[i]];
    (end.first == none ? end.first : end.second) =
        static_cast<std::uint32_t>(i);
  }
  const int w = static_cast<int>(width);
  std::vector<std::uint16_t> offsets(texels);
  for (const auto& [a, b] : ends) {
    const int dx =
        wrapped(static_cast<int>(b % width) - static_cast<int>(a % width), w);
    const int dy =
        wrapped(static_cast<int>(b / width) - static_cast<int>(a / width), w);
    offsets[a] = packed(dx, dy);
    offsets[b] = packed(-dx, -dy);
  }
  return offsets;
}

std::vector<std::uint16_t> reuse_textures(double sigma, std::uint64_t seed) {
  std::vector<std::uint16_t> all;
  all.reserve(contracts::pairing_entries);
  // The contract's widths, a C array for the shaders, as a std::array here,
  // to read with bounds checked (Bounds.2).
  constexpr auto widths = std::to_array(contracts::pairing_widths);
  for (std::uint32_t t = 0; t < contracts::max_path_neighbors; ++t) {
    const std::vector<std::uint16_t> texture =
        reuse_texture(widths.at(t), sigma, seed * texture_seed_step + t);
    all.insert(all.end(), texture.begin(), texture.end());
  }
  return all;
}

}  // namespace serenity::sampling
