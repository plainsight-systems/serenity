// serenity: the window. Reads a frame graph, and a scene if one is given,
// opens a window, and renders frames into it until the window is closed or
// Escape is pressed.
//
// Frames are rendered at --scale of the window's pixels, and the window's
// layer stretches them to fill it (app/options.h).
//
// Each frame's inputs are the measured clock (app/clock.h), the count of
// frames rendered, and the scene's camera, still until the camera moves; all
// three are handed to the renderer (principle 1). The loop is paced by the
// display: render_to_window() waits for a drawable (metal/device/presenter.h).
//
// The title shows the frames' GPU time once a second (core/measurement/
// frame_times.h): each frame is counted once, when the submission that
// frees its slot settles it, and only frames are counted, by sequence:
// everything from the first frame's on is a frame, since start-up work is
// submitted, and settled, before it. The frames settled at the end by
// finish() come after the last title and are not shown. It is a live
// window's figure, with the window's own work beside it, not a benchmark,
// and the title says so (TLM.6); figures that are recorded come from an idle
// GPU (docs/research/).
//
// A graph that converges accumulates from the first frame and starts over
// as the core's plan says (core/frame/history.h, LiveHistory): whenever the
// window's size changes; every frame, while the scene's shapes move, since
// its time always advances; and when the image would hold more frames than
// it can. Samples the pass left out for not being finite, a bug, are shown
// in the title.
//
// At the end the submission is finished, so a GPU failure in the last frames
// is reported too. Any failure ends the run with its message and a non-zero
// status; the renderer and the window's surface wait for the frames in
// flight before they are released (metal/device/submission.h, Lifetime).

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <iostream>
#include <optional>
#include <span>
#include <string>

#include "app/clock.h"
#include "app/options.h"
#include "app/window.h"
#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/graph_file.h"
#include "core/frame/history.h"
#include "core/measurement/frame_times.h"
#include "core/scene/scene.h"
#include "metal/device/device.h"
#include "metal/device/presenter.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"

namespace {

using namespace serenity;

// The window's size when it opens, in points.
constexpr app::Points initial_window{1280, 800};

// How often the title's frame times are refreshed.
constexpr frame::Seconds title_period{1.0};

// The arguments after the program's name; none if the system gave not even
// that (POSIX allows argc == 0).
std::span<const char* const> arguments(int argc, char** argv) {
    if (argc < 1) {
        return {};
    }
    return {argv + 1, static_cast<std::size_t>(argc - 1)};
}

std::string title(frame::Extent size, const measurement::Summary& summary, std::uint64_t non_finite) {
    using Milliseconds = std::chrono::duration<double, std::milli>;
    std::string result = std::format(
        "Serenity  |  {} x {}  |  GPU {:.2f} ms mean, {:.2f} to {:.2f}, over {} frames (live window, not a benchmark)",
        size.width, size.height, Milliseconds{summary.mean}.count(), Milliseconds{summary.shortest}.count(),
        Milliseconds{summary.longest}.count(), summary.frames);
    if (non_finite != 0) {
        result += std::format("  |  {} SAMPLES NOT FINITE (a bug)", non_finite);
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const app::Options options = app::parse(arguments(argc, argv));
        const frame::Schedule schedule = frame::load_schedule(options.graph);
        const std::optional<scene::SceneDescription> loaded =
            options.scene.empty() ? std::nullopt : std::optional{scene::load(options.scene)};

        app::Window window("Serenity", initial_window);
        metal::Device device;
        metal::Submission submission(device);
        metal::Presenter presenter(device, submission, metal::LayerHandle{window.metal_layer()},
                                   app::render_size(window.size_in_pixels(), options.scale));
        metal::Renderer renderer(device, submission, schedule, loaded ? &*loaded : nullptr);
        const std::uint64_t first_frame = submission.next_sequence();

        const app::Clock app_clock;
        measurement::FrameTimes frame_times(title_period);
        std::uint64_t index = 0;
        // When the image starts over is the core's plan (core/frame/history.h).
        frame::LiveHistory history(loaded && scene::changes(*loaded) ? frame::SceneMotion::changing
                                                                   : frame::SceneMotion::still);
        for (app::Window::Events events = window.poll(); !events.quit; events = window.poll()) {
            if (events.resized) {
                presenter.resize(app::render_size(window.size_in_pixels(), options.scale));
            }
            const frame::FrameInputs inputs{
                .time = app_clock.elapsed(),
                .index = index,
                .accumulated_since = history.since(index, events.resized ? frame::View::changed : frame::View::same),
                .camera = loaded ? std::optional{loaded->camera} : std::nullopt,
            };
            if (const auto rendered = metal::render_to_window(submission, presenter, renderer, inputs)) {
                ++index;
                if (rendered->settled && rendered->settled->sequence >= first_frame) {
                    frame_times.add(rendered->settled->gpu_start, rendered->settled->gpu_end);
                }
            }
            if (const auto summary = frame_times.take(app_clock.elapsed())) {
                window.set_title(title(presenter.size(), *summary, renderer.non_finite_samples()));
            }
        }
        // What finish() settles comes after the last title: the run reports
        // only whether they failed.
        submission.finish();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "serenity: " << error.what() << '\n';
        return 1;
    }
}
