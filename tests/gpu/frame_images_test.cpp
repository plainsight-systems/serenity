// The images between passes (metal/frame/frame_images.h) as the renderer
// keeps them: shared by the frames in flight, guarded by the barrier before
// the pass that writes them (metal/frame/renderer.h), remade when the frame's
// size changes; and the path tracer's graph in graphs/, ending in the tone
// map, running.

#include <cstdint>
#include <memory>
#include <vector>

#include <doctest/doctest.h>

#include "core/frame/graph_file.h"
#include "core/scene/scene.h"
#include "gpu/support/rendering.h"
#include "metal/device/device.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"
#include "test_paths.h"

using namespace serenity;

TEST_CASE("frames in flight share the images between passes, and a resize remakes them") {
    // Each frame, rendered back to back with the frame before still in
    // flight, is the frame rendered alone: the radiance image they share is
    // never written while the frame before still reads it.
    constexpr std::uint64_t frames = 4;
    const scene::SceneDescription description = scene::load(tests::scenes_dir / "brass_sphere_flight.toml");
    const frame::Schedule graph =
        frame::parse_schedule("passes = [\"preview\", \"tone_map\"]\n[tone_map]\nexposure = 0.5\nbloom = 0.1\n", "t");
    constexpr frame::Extent size{480, 270};
    // One instant per image: the scene moves (core/frame/history.h).
    const auto inputs = [&](std::uint64_t i) {
        return tests::frame_at(i, i, 0.1 * static_cast<double>(i), description.camera);
    };

    std::vector<std::vector<std::uint8_t>> alone;
    for (std::uint64_t i = 0; i < frames; ++i) {
        alone.push_back(tests::render_once(description, graph, size, inputs(i)));
    }

    metal::Device device;
    metal::Submission submission(device);
    metal::Renderer renderer(device, submission, graph, &description);
    std::vector<std::unique_ptr<metal::Offscreen>> targets;
    std::uint64_t last = 0;
    for (std::uint64_t i = 0; i < frames; ++i) {
        targets.push_back(std::make_unique<metal::Offscreen>(device, submission, size));
        last = metal::render_to_offscreen(submission, *targets.back(), renderer, inputs(i));
    }
    (void)submission.wait_until_complete(last);
    for (std::uint64_t i = 0; i < frames; ++i) {
        INFO("frame " << i);
        CHECK(tests::differing(tests::read_back(*targets[i]), alone[i]) == 0);
    }

    // A resize remakes the images, and the next frame is whole: byte for
    // byte the frame a renderer that never saw the larger size renders.
    constexpr frame::Extent resized{97, 61};
    metal::Offscreen small(device, submission, resized);
    (void)submission.wait_until_complete(metal::render_to_offscreen(submission, small, renderer, inputs(0)));
    CHECK(tests::differing(tests::read_back(small), tests::render_once(description, graph, resized, inputs(0))) == 0);

    // And the path tracer's own graph, in graphs/, runs.
    const frame::Schedule path = frame::load_schedule(tests::graphs_dir / "path.toml");
    metal::Renderer tracer(device, submission, path, &description);
    metal::Offscreen shown(device, submission, size);
    for (std::uint64_t i = 0; i < 3; ++i) {
        last = metal::render_to_offscreen(submission, shown, tracer, inputs(i));
    }
    (void)submission.wait_until_complete(last);
    CHECK(renderer.non_finite_samples() == 0);
    CHECK(tracer.non_finite_samples() == 0);
}
