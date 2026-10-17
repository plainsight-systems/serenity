#pragma once

// Frames rendered the way the GPU tests render them (ES.3, F.10): the frame
// graphs they run, a frame's inputs, a frame rendered whole on a renderer
// of its own and read back, a renderer whose frames accumulate, a patch's
// mean, and how two images differ.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
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
#include "support/files.h"
#include "support/references.h"
#include "support/text.h"
#include "test_paths.h"

namespace serenity::tests {

// The preview, shown as it is (graphs/preview.toml's passes).
inline frame::Schedule preview_graph() {
    return frame::parse_schedule("passes = [\"preview\", \"display\"]\n", "preview graph");
}

// The path tracer, shown as it is: its accumulated image's mean, unexposed.
inline frame::Schedule path_graph() {
    return frame::parse_schedule("passes = [\"path\", \"display\"]\n", "path graph");
}

// graphs/restir_pt_denoised.toml with the test fixture's network
// (tests/data/denoise/network) in place of the one beside the checkout.
inline frame::Schedule fixture_denoised_graph() {
    const std::string text = replaced(read_bytes(graphs_dir / "restir_pt_denoised.toml"),
                                      "\"../../serenity-data/denoiser-v1-model/wide\"",
                                      "\"" + (data_dir / "denoise" / "network").string() + "\"");
    return frame::parse_schedule(text, "restir_pt_denoised.toml, the fixture's network");
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

// A renderer of `schedule` over a scene, at a size, rendering frames that
// accumulate from frame 0, each at time 0, its reuse history kept from
// frame 0 too. A fixture: the renderer reads the scene it holds, so it is
// neither copied nor moved (C.8).
class AccumulatingRig {
public:
    AccumulatingRig(std::string_view text, const frame::Schedule& schedule, frame::Extent size)
        : description_(scene::parse(text, "test scene")), size_(size),
          renderer_(device_, submission_, schedule, &description_) {}
    AccumulatingRig(const AccumulatingRig&) = delete;
    AccumulatingRig& operator=(const AccumulatingRig&) = delete;
    AccumulatingRig(AccumulatingRig&&) = delete;
    AccumulatingRig& operator=(AccumulatingRig&&) = delete;
    ~AccumulatingRig() = default;

    // Renders `frames` more frames and returns the last, as displayed, with
    // every sample of them finite.
    std::vector<std::uint8_t> render(std::uint64_t frames) {
        std::uint64_t sequence = 0;
        for (std::uint64_t i = 0; i < frames; ++i, ++next_) {
            sequence = metal::render_to_offscreen(submission_, target_, renderer_,
                                                  frame_at(next_, 0, 0.0, description_.camera));
        }
        (void)submission_.wait_until_complete(sequence);
        CHECK(renderer_.non_finite_samples() == 0);
        return read_back(target_);
    }

    // The accumulated image of the frames rendered so far, RGBA floats
    // (Renderer::read_accumulated).
    std::vector<float> accumulated() {
        std::vector<float> floats(std::size_t{size_.width} * size_.height * 4);
        renderer_.read_accumulated(floats);
        return floats;
    }

    // Renders `frames` more frames and returns the accumulated image.
    std::vector<float> render_accumulated(std::uint64_t frames) {
        (void)render(frames);
        return accumulated();
    }

    const scene::SceneDescription& description() const noexcept { return description_; }
    frame::Extent size() const noexcept { return size_; }
    metal::Renderer& renderer() noexcept { return renderer_; }

private:
    scene::SceneDescription description_;
    frame::Extent size_;
    metal::Device device_;
    metal::Submission submission_{device_};
    metal::Offscreen target_{device_, submission_, size_};
    metal::Renderer renderer_;
    std::uint64_t next_ = 0;
};

// A square of pixels: its middle, and how far it reaches either way.
struct Patch {
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    std::uint32_t half = 0;
};

// The square of half-width `half` in the middle of an image of `size`.
constexpr Patch middle_of(frame::Extent size, std::uint32_t half) noexcept {
    return {size.width / 2, size.height / 2, half};
}

// The mean of `channel` over `patch` of an RGBA image of `size`, each
// value read by `value`. The patch must lie within the image: one at an
// edge is a failed test, not an index wrapped past it (ES.106, SL.con.3).
template <typename T, typename Value>
double patch_mean(std::span<const T> rgba, frame::Extent size, Patch patch, std::size_t channel, Value value) {
    REQUIRE(rgba.size() == std::size_t{size.width} * size.height * 4);
    REQUIRE(channel < 4);
    REQUIRE(patch.x >= patch.half);
    REQUIRE(patch.y >= patch.half);
    REQUIRE(patch.x + patch.half < size.width);
    REQUIRE(patch.y + patch.half < size.height);
    double sum = 0.0;
    for (std::uint32_t y = patch.y - patch.half; y <= patch.y + patch.half; ++y) {
        for (std::uint32_t x = patch.x - patch.half; x <= patch.x + patch.half; ++x) {
            sum += value(rgba[(std::size_t{y} * size.width + x) * 4 + channel]);
        }
    }
    const std::uint32_t side = 2 * patch.half + 1;
    return sum / (side * side);
}

// A displayed image's: its 8-bit sRGB values decoded to linear radiance.
inline double patch_mean(std::span<const std::uint8_t> rgba, frame::Extent size, Patch patch,
                         std::size_t channel = 0) {
    return patch_mean(rgba, size, patch, channel, [](std::uint8_t v) { return linear_of(v); });
}

// An accumulated image's: its linear radiance as it is.
inline double patch_mean(std::span<const float> rgba, frame::Extent size, Patch patch, std::size_t channel = 0) {
    return patch_mean(rgba, size, patch, channel, [](float v) { return static_cast<double>(v); });
}

// How many bytes two images of one size differ in: 0 for the same image.
// A count, not the images, is what a failing check prints.
inline std::size_t differing(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
    REQUIRE(a.size() == b.size());
    return std::transform_reduce(a.begin(), a.end(), b.begin(), std::size_t{0}, std::plus<>(),
                                 [](std::uint8_t x, std::uint8_t y) { return std::size_t{x != y ? 1u : 0u}; });
}

}  // namespace serenity::tests
