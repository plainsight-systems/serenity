// The denoised pipeline, three integration tests. The denoise pass
// (passes/denoise/denoise.h) against PyTorch: the
// committed fixture, tests/data/denoise (training/denoise_check.py made it:
// a network of training/model.py's structure at small widths, randomly
// initialized, head included; three consecutive frames of a training
// sequence's renderer images, a 40 x 22 crop where fireflies move, so the
// U-Net's padding, the masked taps, the depth median, the motion vectors
// and the history all take part; and model.Denoiser.frame's float32
// outputs). Run through the GPU as the renderer runs it, frame after frame
// with its own history:
//
//   - each output is PyTorch's to within `tolerance`, relative: half
//     floats in the U-Net (denoise.metal, "Precision"), float sums. On this
//     fixture the largest is 2.2e-4 (docs/research/2026-10-15-denoiser-
//     live.md), so 1e-3 leaves a margin of four;
//   - a network one weight off is not: one of the last convolution's
//     weights a tenth off gives 4.0e-3, four times the tolerance.
//
// The pipeline in the renderer, ReSTIR PT's light traced checkerboarded,
// filled, denoised and tone mapped (gpu/support/rendering.h,
// fixture_denoised_graph), over moving fireflies: it runs, every value
// finite, and the same frames give the same bits; and frame N + 1's light
// half never writes what frame N's image half reads (below).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <doctest/doctest.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/contracts/linear_image.h"
#include "core/output/pfm.h"
#include "core/scene/scene.h"
#include "gpu/support/denoise_fixture.h"
#include "gpu/support/probe_runner.h"
#include "gpu/support/rendering.h"
#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/device/support.h"
#include "metal/frame/renderer.h"
#include "serenity/metallib/probes.h"
#include "support/same_bits.h"
#include "support/scene_text.h"
#include "test_paths.h"

using namespace serenity;

namespace {

constexpr double tolerance = 1e-3;
// Below this, a value counts as this: the floor of the relative test, so
// near-black pixels are held to an absolute 1e-6.
constexpr double floor_value = 1e-3;

// The largest |gpu - expected| / (|expected| + floor_value), over every
// frame, pixel and channel.
double worst(const tests::DenoiseFixture& fixture, const std::vector<contracts::LinearImage>& outputs) {
    REQUIRE(outputs.size() == fixture.frames);
    double largest = 0.0;
    for (std::size_t k = 0; k < outputs.size(); ++k) {
        const contracts::LinearImage expected = output::read_pfm(fixture.file(k, "expected"));
        REQUIRE(expected.extent == outputs[k].extent);
        for (std::size_t i = 0; i < expected.rgb.size(); ++i) {
            const double e = expected.rgb[i];
            const double g = outputs[k].rgb[i];
            REQUIRE(std::isfinite(g));
            largest = std::max(largest, std::abs(g - e) / (std::abs(e) + floor_value));
        }
    }
    return largest;
}

}  // namespace

TEST_CASE("the denoise pass computes what PyTorch computes, frame after frame") {
    const tests::DenoiseFixture fixture = tests::read_denoise_fixture(tests::data_dir / "denoise");
    REQUIRE(fixture.frames == 3);
    const double error = worst(fixture, tests::run_denoise_fixture(fixture));
    INFO("largest relative difference: " << error);
    CHECK(error < tolerance);
}

TEST_CASE("a network one weight off does not pass the denoise check") {
    const tests::DenoiseFixture fixture = tests::read_denoise_fixture(tests::data_dir / "denoise");
    // One weight of the last convolution, by a tenth: about the size of
    // its weights.
    const double error = worst(fixture, tests::run_denoise_fixture(fixture, [](denoise::Network& network) {
        network.dec_final.weight.at(0) += 0.1f;
    }));
    INFO("largest relative difference: " << error);
    CHECK(error > tolerance);
}

// The denoised pipeline in the renderer, the fixture's network over
// wandering fireflies at 64 x 48 (a 32 x 24 light). No glass: MNEE's sum
// of three or more requests in a pixel is in the order the GPU solves them
// (metal/passes/mnee/mnee.h, "Determinism"), which this test does not hold.
namespace {

frame::FrameInputs frame_of(std::uint64_t index, std::uint64_t since, const scene::SceneDescription& scene) {
    // A changing scene: the accumulated image starts over every frame.
    frame::FrameInputs inputs = tests::frame_at(index, index, 30.0 + static_cast<double>(index) / 60.0, scene.camera);
    inputs.reused_since = since;
    if (index > since) {
        inputs.previous = frame::PreviousFrame{frame::Seconds{30.0 + static_cast<double>(index - 1) / 60.0}, scene.camera};
    }
    return inputs;
}

// The frames `inputs`, each at its size, on one renderer; the last one's
// denoised radiance (Renderer::read_accumulated).
std::vector<float> rendered(const std::vector<std::pair<frame::FrameInputs, frame::Extent>>& frames,
                            const scene::SceneDescription& scene) {
    metal::Device device;
    metal::Submission submission(device);
    metal::Renderer renderer(device, submission, tests::fixture_denoised_graph(), &scene);
    std::vector<float> out;
    for (const auto& [inputs, size] : frames) {
        metal::Offscreen target(device, submission, size);
        (void)submission.wait_until_complete(metal::render_to_offscreen(submission, target, renderer, inputs));
        out.assign(std::size_t{size.width} * size.height * 4, 0.0f);
        renderer.read_accumulated(out);
    }
    return out;
}

}  // namespace

TEST_CASE("the denoised pipeline runs, every value finite, and the same frames give the same bits") {
    const scene::SceneDescription scene = scene::parse(tests::wandering_fireflies(), "wandering fireflies");
    constexpr frame::Extent size{64, 48};
    const std::vector<std::pair<frame::FrameInputs, frame::Extent>> frames = {
        {frame_of(0, 0, scene), size}, {frame_of(1, 0, scene), size}, {frame_of(2, 0, scene), size}};
    const std::vector<float> first = rendered(frames, scene);
    const std::vector<float> again = rendered(frames, scene);
    CHECK(tests::same_bits(first, again));
    CHECK(std::ranges::all_of(first, [](float v) { return std::isfinite(v); }));
    // And it shows the scene: light, not a black or an empty image.
    CHECK(std::ranges::any_of(first, [](float v) { return v > 0.0f; }));
}

// The hand-off (metal/frame/renderer.h, "Two halves"): frame N + 1's light
// half never writes what frame N's image half reads. Frames with their
// halves on two queues, two in flight and none waited for, end in the same
// bits as the same frames each recorded into one part and waited for: every
// frame as displayed, and the last frame's radiance. A hand-off frame N + 1
// overwrote under frame N's image half would change frame N's image (the
// full-size albedo, which the denoiser multiplies back after its history)
// or reach the denoiser's history and the frames after it (the radiance and
// the features).
//
// Left to the GPU, frame N's image half reads the hand-off at its start,
// before frame N + 1's light half has reached its end and written it, so a
// shared hand-off would pass too. So each frame's image half is held back:
// its second part begins with the tests' delay kernel (kernels/
// delay.metal, about 30 ms on the M3 Max), which the image half's queue
// barrier waits for, while frame N + 1's light half, a few milliseconds
// here, runs to its end beside it. The test shows it was so: each frame's
// light half began before the frame before it had finished (submission.h,
// the records), and every frame after the first was held back for longer
// than the frame before it took to trace its light (by the records too: the
// image half's start, which the delay sets, is after the next frame's light
// half ended, within the GPU's own times).
namespace {

// The delay, its count of steps (kernels/delay.metal) and its buffers,
// recorded at the start of a second part.
class Delay {
public:
    Delay(const metal::Device& device, metal::Submission& submission)
        : library_(device, metallib::probes), pipeline_(library_.compute_pipeline("delay")) {
        const auto pool = metal::scoped_pool();
        MTL::Device* mtl = device.handle();
        out_ = NS::TransferPtr(mtl->newBuffer(sizeof(std::uint32_t), MTL::ResourceStorageModeShared));
        count_ = NS::TransferPtr(mtl->newBuffer(sizeof(std::uint32_t), MTL::ResourceStorageModeShared));
        REQUIRE(out_);
        REQUIRE(count_);
        constexpr std::uint32_t steps = 10'000'000;
        std::memcpy(count_->contents(), &steps, sizeof steps);
        submission.make_resident(out_.get());
        submission.make_resident(count_.get());
        auto descriptor = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
        descriptor->setMaxBufferBindCount(2);
        NS::Error* error = nullptr;
        table_ = NS::TransferPtr(mtl->newArgumentTable(descriptor.get(), &error));
        INFO("argument table: " << tests::reason(error));
        REQUIRE(table_);
        table_->setAddress(out_->gpuAddress(), 0);
        table_->setAddress(count_->gpuAddress(), 1);
    }

    void record(MTL4::CommandBuffer* commands) const {
        MTL4::ComputeCommandEncoder* encoder = commands->computeCommandEncoder();
        REQUIRE(encoder != nullptr);
        encoder->setArgumentTable(table_.get());
        encoder->setComputePipelineState(pipeline_.get());
        encoder->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
        encoder->endEncoding();
    }

private:
    metal::Library library_;
    NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
    NS::SharedPtr<MTL::Buffer> out_;
    NS::SharedPtr<MTL::Buffer> count_;
    NS::SharedPtr<MTL4::ArgumentTable> table_;
};

// Frame `index` of the hand-off's run, 50 ms after the one before, so the
// fireflies move a pixel or so between frames: every image the halves hand
// over, the full-size albedo's among them, differs from frame to frame.
frame::FrameInputs stepped_frame(std::uint64_t index, const scene::SceneDescription& scene) {
    const auto at = [](std::uint64_t i) { return frame::Seconds{30.0 + 0.05 * static_cast<double>(i)}; };
    frame::FrameInputs inputs = tests::frame_at(index, index, at(index).count(), scene.camera);
    inputs.reused_since = 0;
    if (index > 0) {
        inputs.previous = frame::PreviousFrame{at(index - 1), scene.camera};
    }
    return inputs;
}

struct HandOffRun {
    std::vector<std::vector<std::uint8_t>> shown;  // each frame, as displayed
    std::vector<float> last;                       // the last frame's denoised radiance
    std::vector<metal::Completed> records;         // the frames', in sequence order
};

// `frames` frames of the wandering fireflies through the fixture's denoised graph
// at `size`, each into a target of its own, so every frame's image is read
// back, not only the last's: of two parts, each image half held back by the
// delay, none waited for; or of one part, each waited for.
HandOffRun run_hand_off(const scene::SceneDescription& scene, frame::Extent size, std::uint64_t frames,
                        metal::Parts parts) {
    metal::Device device;
    metal::Submission submission(device);
    const Delay delay(device, submission);
    metal::Renderer renderer(device, submission, tests::fixture_denoised_graph(), &scene);
    std::vector<std::unique_ptr<metal::Offscreen>> targets;
    for (std::uint64_t i = 0; i < frames; ++i) {
        targets.push_back(std::make_unique<metal::Offscreen>(device, submission, size));
    }
    const std::uint64_t first = submission.next_sequence();
    HandOffRun run;
    for (std::uint64_t i = 0; i < frames; ++i) {
        const auto pool = metal::scoped_pool();
        const frame::FrameInputs inputs = stepped_frame(i, scene);
        renderer.prepare(inputs, size);
        const metal::FrameSlot begun = submission.begin(parts);
        REQUIRE((begun.second != nullptr) == (parts == metal::Parts::two));
        if (begun.second != nullptr) {
            delay.record(begun.second);
        }
        renderer.record(begun, inputs, targets[i]->texture(), size);
        submission.commit();
        if (begun.settled && begun.settled->sequence >= first) {
            run.records.push_back(*begun.settled);
        }
        if (parts == metal::Parts::one) {
            if (const auto settled = submission.wait_until_complete(begun.sequence)) {
                run.records.push_back(*settled);
            }
        }
    }
    for (const metal::Completed& settled : submission.drain()) {
        run.records.push_back(settled);
    }
    std::ranges::sort(run.records, {}, &metal::Completed::sequence);
    for (const std::unique_ptr<metal::Offscreen>& target : targets) {
        run.shown.push_back(tests::read_back(*target));
    }
    run.last.assign(std::size_t{size.width} * size.height * 4, 0.0f);
    renderer.read_accumulated(run.last);
    return run;
}

}  // namespace

TEST_CASE("frame N + 1's light half never writes what frame N's image half reads") {
    const scene::SceneDescription scene = scene::parse(tests::wandering_fireflies(), "wandering fireflies");
    constexpr frame::Extent size{256, 192};
    constexpr std::uint64_t frames = 6;
    const HandOffRun overlapped = run_hand_off(scene, size, frames, metal::Parts::two);
    const HandOffRun serial = run_hand_off(scene, size, frames, metal::Parts::one);
    REQUIRE(overlapped.shown.size() == frames);
    for (std::size_t k = 0; k < frames; ++k) {
        INFO("frame " << k);
        CHECK(overlapped.shown[k] == serial.shown[k]);
    }
    CHECK(tests::same_bits(overlapped.last, serial.last));

    // It exercised the hazard: each frame's light half (its record's start)
    // began, and ended, while the frame before was held back. A serial
    // frame's span is its light half and image half alone, no delay: the
    // longest of them bounds a light half, and each overlapped frame's own
    // span is far longer, the delay's.
    REQUIRE(overlapped.records.size() == frames);
    REQUIRE(serial.records.size() == frames);
    double longest_serial = 0.0;
    for (const metal::Completed& record : serial.records) {
        longest_serial = std::max(longest_serial, (record.gpu_end - record.gpu_start).count());
    }
    for (std::size_t k = 1; k < overlapped.records.size(); ++k) {
        const metal::Completed& before = overlapped.records[k - 1];
        const metal::Completed& next = overlapped.records[k];
        INFO("frame " << k << ": began " << (before.gpu_end - next.gpu_start).count() * 1e3
                      << " ms before the frame before ended; a serial frame takes at most " << longest_serial * 1e3
                      << " ms");
        CHECK(next.gpu_start + frame::Seconds{longest_serial} < before.gpu_end);
    }
}
