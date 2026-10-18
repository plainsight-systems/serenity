// The headless renderer's PFM frames (headless/options.h, --format pfm) on
// the GPU: each is the accumulated image's radiance, bit for bit, as the
// renderer reads it back (metal/frame/renderer.h, read_accumulated) after
// the same samples rendered here, in this process, by the core's plan
// (core/frame/history.h). The program is run as its user runs it; what it
// refuses, and the names it writes, are tests/options_test.cpp's.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <optional>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "core/contracts/linear_image.h"
#include "core/frame/graph_file.h"
#include "core/frame/history.h"
#include "core/frame/schedule.h"
#include "core/output/pfm.h"
#include "core/scene/scene.h"
#include "metal/device/device.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"
#include "programs.h"
#include "support/files.h"
#include "support/run_program.h"
#include "support/same_bits.h"
#include "support/scene_text.h"

using namespace serenity;

TEST_CASE("headless: a PFM frame is the accumulated image's radiance, bit for bit") {
    // Frames 3 and 4 of the path graph over a brass ball under still
    // fireflies, time frozen, three samples a frame: frame 4's file holds
    // frames 3 and 4's six samples (a still scene, or frozen time, converges
    // over the run). The graph and the scene are this test's, written for
    // the program to read.
    const tests::ScratchDirectory scratch("gpu-headless-pfm");
    const std::filesystem::path graph = scratch / "path.toml";
    const std::filesystem::path scene_file = scratch / "scene.toml";
    tests::write_bytes(graph, "passes = [\"path\", \"tone_map\"]\n[tone_map]\nexposure = 0.0\nbloom = 0.0\n");
    tests::write_bytes(scene_file, tests::still_fireflies());
    const std::filesystem::path out = scratch / "frames";
    constexpr frame::Extent size{24, 16};
    const tests::Ran ran = tests::run_program(
        tests::headless_program,
        {"--graph", graph.string(), "--scene", scene_file.string(), "--out", out.string(), "--size", "24x16",
         "--first", "3", "--frames", "2", "--samples", "3", "--time", "0", "--format", "pfm"},
        scratch.path());
    INFO("stderr: " << ran.err);
    REQUIRE(ran.status == 0);
    CHECK(ran.out == (out / "frame-000003.pfm").string() + "\n" + (out / "frame-000004.pfm").string() + "\n");

    // The same samples, rendered here as the program renders them.
    const frame::Schedule schedule = frame::load_schedule(graph);
    const scene::SceneDescription description = scene::load(scene_file);
    metal::Device device;
    metal::Submission submission(device);
    metal::Offscreen target(device, submission, size);
    metal::Renderer renderer(device, submission, schedule, &description);
    const frame::HeadlessPlan plan = frame::plan_headless({
        .first = 3,
        .samples = 3,
        .accumulates = frame::accumulates(schedule),
        .scene_changes = scene::changes(description),
        .time_frozen = true,
    });
    REQUIRE(plan.samples == 3);
    std::vector<float> rgba(std::size_t{size.width} * size.height * 4);
    for (std::uint64_t index = 3; index <= 4; ++index) {
        INFO("frame " << index);
        std::uint64_t last = 0;
        for (std::uint64_t s = 0; s < plan.samples; ++s) {
            const frame::Sample sample = plan.sample({.frame = index, .sample = s});
            last = metal::render_to_offscreen(
                submission, target, renderer,
                frame::FrameInputs{.time = frame::Seconds{0.0},
                                   .index = sample.index,
                                   .accumulated_since = sample.accumulated_since,
                                   .camera = std::optional{description.camera}});
        }
        (void)submission.wait_until_complete(last);
        CHECK(renderer.non_finite_samples() == 0);
        renderer.read_accumulated(rgba);
        const contracts::LinearImage expected = contracts::from_rgba(size, rgba);

        const contracts::LinearImage written = output::read_pfm(out / std::format("frame-{:06}.pfm", index));
        CHECK(written.extent == size);
        CHECK(tests::same_bits(written, expected));
        // Not an empty image compared with another: the ball and the checks
        // are lit.
        REQUIRE_FALSE(written.rgb.empty());
        CHECK(std::ranges::max(written.rgb) > 0.0f);
    }
    submission.finish();
}
