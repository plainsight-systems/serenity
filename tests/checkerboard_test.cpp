// Checkerboard rendering of the light (contract 20, core/contracts/
// checkerboard.h; core/frame/schedule.h), the invariants the GPU's kernels
// and the host's dispatches read alike from the contract's one definition:
//
//   - every pixel is traced in exactly one of any two frames in a row, so
//     at least every other frame, across the frame index's wrap too;
//   - the packed grid's threads take every pixel a frame traces, each
//     once, and none it does not, each at its own slot;
//   - the frame before's pixel beside a pixel is one that frame traced;
//
// and the setting's refusals, which keep the fill's inputs there: only
// with ReSTIR PT's passes and the denoise pass, a boolean, and a sigma the
// traced lattice can pair at.

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "core/contracts/checkerboard.h"
#include "core/frame/graph_file.h"
#include "core/frame/schedule.h"
#include "support/text.h"
#include "test_paths.h"

namespace contracts = serenity::contracts;
using serenity::frame::GraphFileError;
using serenity::frame::parse_schedule;
using serenity::tests::contains;

namespace {

// Odd and even sizes, and one a frame's light at 1920 x 1080 is.
constexpr std::uint32_t widths[] = {1, 2, 7, 8, 960};
constexpr std::uint32_t heights[] = {1, 3, 540};
// Frames about the index's wrap at 2^32, and two plain ones.
constexpr std::uint32_t frames[] = {0u, 1u, 41u, std::numeric_limits<std::uint32_t>::max()};

}  // namespace

TEST_CASE("every pixel is traced in exactly one of any two frames in a row") {
    for (const std::uint32_t frame : frames) {
        for (std::uint32_t y = 0; y < 5; ++y) {
            for (std::uint32_t x = 0; x < 5; ++x) {
                INFO("frame " << frame << ", pixel " << x << ", " << y);
                CHECK(contracts::checkerboard_traced(x, y, frame) != contracts::checkerboard_traced(x, y, frame + 1u));
            }
        }
    }
}

TEST_CASE("the packed grid takes each pixel a frame traces once, at its own slot, and no other") {
    for (const std::uint32_t width : widths) {
        for (const std::uint32_t height : heights) {
            for (const std::uint32_t frame : frames) {
                INFO(width << " x " << height << ", frame " << frame);
                const std::uint32_t columns = contracts::checkerboard_width(width);
                std::vector<int> taken(std::size_t{width} * height, 0);
                std::vector<int> slots(std::size_t{columns} * height, 0);
                bool beyond = true;  // every thread's pixel inside the image, or past an odd width's edge
                std::size_t misplaced = 0;
                for (std::uint32_t y = 0; y < height; ++y) {
                    for (std::uint32_t k = 0; k < columns; ++k) {
                        const std::uint32_t x = contracts::checkerboard_x(k, y, frame);
                        if (x >= width) {
                            beyond = beyond && width % 2u == 1u && x == width;
                            continue;
                        }
                        ++taken[std::size_t{y} * width + x];
                        const std::uint32_t slot = contracts::checkerboard_slot(x, y, width);
                        misplaced += slot != y * columns + k ? 1u : 0u;
                        ++slots[slot];
                    }
                }
                CHECK(beyond);
                CHECK(misplaced == 0u);
                std::size_t wrong = 0;
                for (std::uint32_t y = 0; y < height; ++y) {
                    for (std::uint32_t x = 0; x < width; ++x) {
                        const int expected = contracts::checkerboard_traced(x, y, frame) ? 1 : 0;
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

TEST_CASE("the frame before's pixel beside a pixel is one that frame traced, at most one step along the row") {
    for (const std::uint32_t frame : frames) {
        for (std::int32_t y = 0; y < 4; ++y) {
            for (std::int32_t x = 1; x < 6; ++x) {
                const std::int32_t before = contracts::checkerboard_previous_x(x, y, frame);
                INFO("frame " << frame << ", pixel " << x << ", " << y << ": " << before);
                CHECK(contracts::checkerboard_traced(static_cast<std::uint32_t>(before), static_cast<std::uint32_t>(y),
                                                     frame - 1u));
                CHECK(std::abs(before - x) <= 1);
            }
        }
    }
}

TEST_CASE("checkerboard is read only beside ReSTIR PT's passes and the denoise pass, off unless written") {
    std::ifstream file(serenity::tests::graphs_dir / "restir_pt_denoised.toml");
    std::stringstream read;
    read << file.rdbuf();
    const std::string denoised = read.str();
    // A top-level key goes before the first table.
    const auto with = [&](const std::string& line) {
        std::string text = denoised;
        text.insert(text.find("\n["), "\n" + line);
        return text;
    };
    const auto error_for = [](const std::string& text) {
        return serenity::tests::error_of<GraphFileError>([&] { return parse_schedule(text, "graph.toml"); });
    };
    CHECK_FALSE(parse_schedule(denoised, "graph.toml").checkerboard);
    CHECK(parse_schedule(with("checkerboard = true"), "graph.toml").checkerboard);
    CHECK_FALSE(parse_schedule(with("checkerboard = false"), "graph.toml").checkerboard);
    CHECK(contains(error_for(with("checkerboard = 1")), "'checkerboard' must be true or false"));
    // Without the denoise pass, whose hand-off the fill writes.
    std::ifstream light_file(serenity::tests::graphs_dir / "restir_pt.toml");
    std::stringstream light_read;
    light_read << light_file.rdbuf();
    std::string light = light_read.str();
    light.insert(light.find("\n["), "\ncheckerboard = true");
    CHECK(contains(error_for(light), "sets checkerboard and lacks ReSTIR PT's passes"));
    // A sigma the traced lattice cannot pair at: 0.8 pixels, a lattice's 0.57.
    std::string narrow = with("checkerboard = true");
    const std::string sigma = "sigma = 16.0";
    narrow.replace(narrow.find(sigma), sigma.size(), "sigma = 0.8");
    CHECK(contains(error_for(narrow), "must be at least 1.13137 pixels with checkerboard"));
}
