#pragma once

// Frames rendered the way the GPU tests render them (ES.3, F.10): the frame
// graphs they run, a frame's inputs, a frame rendered whole on a renderer
// of its own and read back, and how two images differ.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <vector>

#include <doctest/doctest.h>

#include "core/frame/frame_inputs.h"
#include "core/frame/graph_file.h"
#include "core/frame/schedule.h"
#include "core/scene/scene.h"
#include "metal/device/device.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"

namespace serenity::tests {

// The preview, shown as it is (graphs/preview.toml's passes).
inline frame::Schedule preview_graph() {
    return frame::parse_schedule("passes = [\"preview\", \"display\"]\n", "preview graph");
}

// The path tracer, shown as it is: its accumulated image's mean, unexposed.
inline frame::Schedule path_graph() {
    return frame::parse_schedule("passes = [\"path\", \"display\"]\n", "path graph");
}

// Frame `index` at `seconds`, its image accumulated since frame `since`,
// seen by `camera`.
inline frame::FrameInputs frame_at(std::uint64_t index, std::uint64_t since, double seconds,
                                   std::optional<contracts::Camera> camera) {
    return frame::FrameInputs{.time = frame::Seconds(seconds), .index = index, .accumulated_since = since,
                              .camera = camera};
}

// What `target` holds, read back.
inline std::vector<std::uint8_t> read_back(const metal::Offscreen& target) {
    std::vector<std::uint8_t> rgba(target.rgba_size());
    target.read_rgba(rgba);
    return rgba;
}

// Frame `inputs` of `description` through `schedule`, at `size`, on a device,
// queue and renderer of its own, read back.
inline std::vector<std::uint8_t> render_once(const scene::SceneDescription& description,
                                             const frame::Schedule& schedule, frame::Extent size,
                                             const frame::FrameInputs& inputs) {
    metal::Device device;
    metal::Submission submission(device);
    metal::Offscreen target(device, submission, size);
    metal::Renderer renderer(device, submission, schedule, &description);
    (void)submission.wait_until_complete(metal::render_to_offscreen(submission, target, renderer, inputs));
    CHECK(renderer.non_finite_samples() == 0);
    return read_back(target);
}

// How many bytes two images of one size differ in: 0 for the same image.
// A count, not the images, is what a failing check prints.
inline std::size_t differing(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
    REQUIRE(a.size() == b.size());
    return std::transform_reduce(a.begin(), a.end(), b.begin(), std::size_t{0}, std::plus<>(),
                                 [](std::uint8_t x, std::uint8_t y) { return std::size_t{x != y ? 1u : 0u}; });
}

}  // namespace serenity::tests
