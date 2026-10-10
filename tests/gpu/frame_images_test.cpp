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
#include "metal/device/device.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"

using namespace serenity;

TEST_CASE("frames in flight share the images between passes, and a resize remakes them") {
    // Each frame, rendered back to back with the frame before still in
    // flight, is the frame rendered alone: the radiance image they share is
    // never written while the frame before still reads it.
    const scene::SceneDescription scene = scene::load(SERENITY_SCENES_DIR "/brass_sphere_flight.toml");
    const frame::Schedule graph =
        frame::parse_schedule("passes = [\"preview\", \"tone_map\"]\n[tone_map]\nexposure = 0.5\nbloom = 0.1\n", "t");
    const frame::Extent size{480, 270};
    const auto inputs = [&](std::uint64_t i) {
        // One instant per image: the scene moves (core/frame/history.h).
        return frame::FrameInputs{
            .time = frame::Seconds(0.1 * i), .index = i, .accumulated_since = i, .camera = scene.camera};
    };
    const auto read = [](const metal::Offscreen& target) {
        std::vector<std::uint8_t> rgba(target.rgba_size());
        target.read_rgba(rgba);
        return rgba;
    };

    std::vector<std::vector<std::uint8_t>> alone;
    for (std::uint64_t i = 0; i < 4; ++i) {
        metal::Device device;
        metal::Submission submission(device);
        metal::Offscreen target(device, submission, size);
        metal::Renderer renderer(device, submission, graph, &scene);
        (void)submission.wait_until_complete(metal::render_to_offscreen(submission, target, renderer, inputs(i)));
        alone.push_back(read(target));
    }

    metal::Device device;
    metal::Submission submission(device);
    metal::Renderer renderer(device, submission, graph, &scene);
    std::vector<std::unique_ptr<metal::Offscreen>> targets;
    std::vector<std::uint64_t> sequences;
    for (std::uint64_t i = 0; i < 4; ++i) {
        targets.push_back(std::make_unique<metal::Offscreen>(device, submission, size));
        sequences.push_back(metal::render_to_offscreen(submission, *targets.back(), renderer, inputs(i)));
    }
    (void)submission.wait_until_complete(sequences.back());
    for (std::uint64_t i = 0; i < 4; ++i) {
        INFO("frame " << i);
        CHECK(read(*targets[i]) == alone[i]);
    }

    // A resize remakes the images, and the next frame is whole: byte for
    // byte the frame a renderer that never saw the larger size renders.
    const frame::Extent resized{97, 61};
    metal::Offscreen small(device, submission, resized);
    (void)submission.wait_until_complete(metal::render_to_offscreen(submission, small, renderer, inputs(0)));
    {
        metal::Device fresh_device;
        metal::Submission fresh_submission(fresh_device);
        metal::Offscreen fresh_target(fresh_device, fresh_submission, resized);
        metal::Renderer fresh(fresh_device, fresh_submission, graph, &scene);
        (void)fresh_submission.wait_until_complete(
            metal::render_to_offscreen(fresh_submission, fresh_target, fresh, inputs(0)));
        CHECK(read(small) == read(fresh_target));
    }

    // And the path tracer's own graph, in graphs/, runs.
    const frame::Schedule path = frame::load_schedule(SERENITY_GRAPHS_DIR "/path.toml");
    metal::Renderer tracer(device, submission, path, &scene);
    metal::Offscreen shown(device, submission, size);
    std::uint64_t last = 0;
    for (std::uint64_t i = 0; i < 3; ++i) {
        last = metal::render_to_offscreen(submission, shown, tracer, inputs(i));
    }
    (void)submission.wait_until_complete(last);
    CHECK(renderer.non_finite_samples() == 0);
    CHECK(tracer.non_finite_samples() == 0);
}
