// Checkerboard rendering of the light (contract 20, core/contracts/
// checkerboard.h; core/frame/schedule.h), the invariants the GPU's kernels
// and the host's dispatches read alike from the contract's one definition:
//
//   - every pixel is traced in exactly one of any two frames in a row, so
//     at least every other frame, across the frame index's wrap too;
//   - the packed grid's threads take every pixel a frame traces, each
//     once, and none it does not, each at its own slot;
//   - the frame before's pixel beside a pixel is one that frame traced;
//   - the arrays' block-linear slots take every place of their grid, each
//     at its own slot, within the grid's size in whole blocks;
//
// and a denoised graph's one rule for it: a sigma the traced lattice can
// pair at.

#include "core/contracts/checkerboard.h"

#include <doctest/doctest.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

#include "core/frame/schedule.h"
#include "support/schedules.h"

namespace contracts = serenity::contracts;
namespace frame = serenity::frame;
using serenity::frame::PassKind;
using serenity::frame::Schedule;

namespace {

// Odd and even sizes, and one a frame's light at 1920 x 1080 is.
constexpr std::array<std::uint32_t, 5> widths = {1, 2, 7, 8, 960};
constexpr std::array<std::uint32_t, 3> heights = {1, 3, 540};
// Frames about the index's wrap at 2^32, and two plain ones.
constexpr std::array<std::uint32_t, 4> frames = {
    0u, 1u, 41u, std::numeric_limits<std::uint32_t>::max()};

}  // namespace

TEST_CASE("every pixel is traced in exactly one of any two frames in a row") {
  for (const std::uint32_t frame : frames) {
    for (std::uint32_t y = 0; y < 5; ++y) {
      for (std::uint32_t x = 0; x < 5; ++x) {
        INFO("frame " << frame << ", pixel " << x << ", " << y);
        CHECK(contracts::checkerboard_traced(x, y, frame) !=
              contracts::checkerboard_traced(x, y, frame + 1u));
      }
    }
  }
}

TEST_CASE(
    "the packed grid takes each pixel a frame traces once, at its own slot, "
    "and no other") {
  for (const std::uint32_t width : widths) {
    for (const std::uint32_t height : heights) {
      for (const std::uint32_t frame : frames) {
        INFO(width << " x " << height << ", frame " << frame);
        const std::uint32_t columns = contracts::checkerboard_width(width);
        std::vector<int> taken(std::size_t{width} * height, 0);
        std::vector<int> slots(contracts::reservoir_slots(columns, height), 0);
        bool beyond = true;  // every thread's pixel inside the image, or past
                             // an odd width's edge
        std::size_t misplaced = 0;
        for (std::uint32_t y = 0; y < height; ++y) {
          for (std::uint32_t k = 0; k < columns; ++k) {
            const std::uint32_t x = contracts::checkerboard_x(k, y, frame);
            if (x >= width) {
              beyond = beyond && width % 2u == 1u && x == width;
              continue;
            }
            ++taken[std::size_t{y} * width + x];
            // The backends' packing (Traced::slot()): half the pixel's x,
            // at its block-linear slot.
            const std::uint32_t slot =
                contracts::reservoir_slot(x / 2u, y, columns);
            misplaced +=
                slot != contracts::reservoir_slot(k, y, columns) ? 1u : 0u;
            ++slots[slot];
          }
        }
        CHECK(beyond);
        CHECK(misplaced == 0u);
        std::size_t wrong = 0;
        for (std::uint32_t y = 0; y < height; ++y) {
          for (std::uint32_t x = 0; x < width; ++x) {
            const int expected =
                contracts::checkerboard_traced(x, y, frame) ? 1 : 0;
            wrong += taken[std::size_t{y} * width + x] != expected ? 1u : 0u;
          }
        }
        CHECK(wrong == 0u);
        std::size_t shared = 0;
        for (const int used : slots) {
          shared += used > 1 ? 1u : 0u;
        }
        CHECK(shared == 0u);
      }
    }
  }
}

TEST_CASE(
    "the arrays' block-linear slots take each place of their grid once, within "
    "the grid's size in whole blocks") {
  for (const std::uint32_t width : widths) {
    for (const std::uint32_t height : heights) {
      INFO(width << " x " << height);
      const std::uint32_t size = contracts::reservoir_slots(width, height);
      CHECK(size % (contracts::reservoir_block * contracts::reservoir_block) ==
            0u);
      CHECK(size >= width * height);
      std::vector<int> slots(size, 0);
      std::size_t outside = 0;
      for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
          const std::uint32_t slot = contracts::reservoir_slot(x, y, width);
          if (slot >= size) {
            ++outside;
          } else {
            ++slots[slot];
          }
        }
      }
      CHECK(outside == 0u);
      std::size_t shared = 0;
      for (const int used : slots) {
        shared += used > 1 ? 1u : 0u;
      }
      CHECK(shared == 0u);
    }
  }
}

TEST_CASE(
    "the frame before's pixel beside a pixel is one that frame traced, at most "
    "one step along the row") {
  for (const std::uint32_t frame : frames) {
    for (std::int32_t y = 0; y < 4; ++y) {
      for (std::int32_t x = 1; x < 6; ++x) {
        const std::int32_t before =
            contracts::checkerboard_previous_x(x, y, frame);
        INFO("frame " << frame << ", pixel " << x << ", " << y << ": "
                      << before);
        CHECK(contracts::checkerboard_traced(static_cast<std::uint32_t>(before),
                                             static_cast<std::uint32_t>(y),
                                             frame - 1u));
        CHECK(std::abs(before - x) <= 1);
      }
    }
  }
}

TEST_CASE(
    "a graph whose light is the network's input, traced checkerboarded, "
    "refuses a sigma the lattice cannot "
    "pair at") {
  const std::vector<PassKind> light = {PassKind::restir_pt_initial,
                                       PassKind::restir_pt_spatial,
                                       PassKind::restir_pt_shade};
  enum class Kind : std::uint8_t { light, input, denoised };
  const auto graph = [&](Kind kind, float sigma) {
    std::vector<PassKind> kinds = light;
    const bool denoised = kind == Kind::denoised;
    if (kind != Kind::light) {
      kinds.push_back(PassKind::features);
    }
    if (denoised) {
      kinds.push_back(PassKind::denoise);
      kinds.push_back(PassKind::depth_of_field);
    }
    kinds.push_back(PassKind::tone_map);
    Schedule s{kinds, serenity::tests::look};
    s.restir_pt = serenity::tests::written_restir_pt(kinds);
    s.restir_pt->sigma = sigma;
    if (denoised) {
      s.denoise = serenity::passes::Denoise{"network"};
    }
    return s;
  };
  // 0.8 pixels, the least a full image pairs at, is a lattice step's 0.57.
  CHECK(serenity::tests::refusal(graph(Kind::light, 0.8f)).empty());
  CHECK_FALSE(frame::feeds_network(graph(Kind::light, 0.8f)));
  for (const Kind kind : {Kind::input, Kind::denoised}) {
    CHECK(frame::feeds_network(graph(kind, 1.2f)));
    CHECK(serenity::tests::refused_for(
        graph(kind, 0.8f),
        "must be at least 1.13137 pixels in a graph whose light "
        "is the network's input"));
    CHECK(serenity::tests::refusal(graph(kind, 1.2f)).empty());
  }
}
