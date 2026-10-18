// serenity: the window. Reads a frame graph, and a scene if one is given,
// opens a window, and renders frames into it until the window is closed or
// Escape is pressed.
//
// Frames are rendered at --scale of the window's pixels, and the window's
// layer stretches them to fill it (app/options.h). A denoised graph's light
// is traced at half that size each way and its network, not the layer,
// upscales it (core/frame/schedule.h, light_size).
//
// The display paces the frames (logical-overview.md, "The frame",
// "Pacing"): a display link on the window's layer calls WindowFrames::frame
// whenever the display wants a frame, handing it the drawable to render
// into and when the display will show it (metal/device/display_link.h), at
// 60 frames a second, the project's target (logical-overview.md, "The
// frame"), the rate asked of it. The loop below only waits for the
// window's events; SDL waits by running the main run loop, which calls the
// frames meanwhile (app/window.h, wait), as Apple's display link sample
// renders on the main thread. A graph that shows no image (the features
// pass alone, core/frame/schedule.h) is refused before the window opens.
//
// Each frame's inputs are the time the display will show it, counted from
// the first frame's (app/clock.h), the count of frames rendered, and the
// scene's camera, still until the camera moves; all three are handed to
// the renderer (principle 1). Each frame is also told the frame before it,
// its time and camera, which a graph with the denoiser's features measures
// motion from (core/frame/frame_inputs.h, previous); the first has none.
//
// The window's size is followed by the frame, not by the resize event:
// each frame asks the presenter for the size its window's pixels and the
// scale give, which also restores the size SDL sets on a resize
// (metal/device/presenter.h, resize); a frame whose drawable was made at
// the size before is skipped (metal/frame/renderer.h, render_to_window),
// and the next frame rendered is told its view changed. A minimized
// window, of no pixels, is rendered nothing.
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
// Failure. A frame must not throw (display_link.h, Client): it is called
// from Core Animation's run loop. So WindowFrames::frame catches what
// rendering throws, keeps the first failure, renders no frame after it, and
// wakes the loop (app/window.h, wake), which stops the display link and
// rethrows it (E.17: caught where the run can be ended; E.2). At the end the
// submission is finished, so a GPU failure in the last frames is reported
// too. Any failure ends the run with its message and a non-zero status; the
// renderer and the window's surface wait for the frames in flight before
// they are released (metal/device/submission.h, Lifetime).

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
#include "metal/device/display_link.h"
#include "metal/device/presenter.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"

namespace {

using namespace serenity;

// The window's size when it opens, in points.
constexpr app::Points initial_window{1280, 800};

// The frame rate asked of the display link: the project's target, 60
// frames a second, and only it (logical-overview.md, "The frame").
constexpr metal::FrameRate frame_rate{.minimum = 60.0F, .maximum = 60.0F, .preferred = 60.0F};

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

std::string title(frame::Extent size, const measurement::Summary& summary, std::uint64_t non_finite,
                  std::uint64_t mnee_lost) {
    using Milliseconds = std::chrono::duration<double, std::milli>;
    std::string result = std::format(
        "Serenity  |  {} x {}  |  GPU {:.2f} ms mean, {:.2f} to {:.2f}, over {} frames (live window, not a benchmark)",
        size.width, size.height, Milliseconds{summary.mean}.count(), Milliseconds{summary.shortest}.count(),
        Milliseconds{summary.longest}.count(), summary.frames);
    if (non_finite != 0) {
        result += std::format("  |  {} SAMPLES NOT FINITE (a bug)", non_finite);
    }
    if (mnee_lost != 0) {
        result += std::format("  |  {} MNEE REQUESTS LOST (a bug)", mnee_lost);
    }
    return result;
}

// The window's frames, as the display link asks for them (above). Holds
// what passes from frame to frame: the clock, the count, the histories,
// the frame before, the frame times. Everything it is given must outlive
// it; the display link calling it is destroyed first.
class WindowFrames final : public metal::DisplayLink::Client {
public:
    WindowFrames(app::Window& window, metal::Submission& submission, metal::Presenter& presenter,
                 metal::Renderer& renderer, const scene::SceneDescription* scene, double scale,
                 std::uint64_t first_frame)
        : window_(window),
          submission_(submission),
          presenter_(presenter),
          renderer_(renderer),
          scene_(scene),
          scale_(scale),
          first_frame_(first_frame),
          // When the image starts over is the core's plan (core/frame/history.h).
          history_(scene && scene::changes(*scene) ? frame::SceneMotion::changing : frame::SceneMotion::still) {}

    // A frame the display asked for; on a failure, none after (above).
    void frame(const metal::DisplayUpdate& update) noexcept override {
        if (failure_) {
            return;
        }
        try {
            render(update);
        } catch (...) {
            failure_ = std::current_exception();
            window_.wake();
        }
    }

    // Rethrows the failure a frame caught, if one did.
    void rethrow_failure() const {
        if (failure_) {
            std::rethrow_exception(failure_);
        }
    }

private:
    void render(const metal::DisplayUpdate& update) {
        const frame::Extent pixels = window_.size_in_pixels();
        if (pixels.width == 0 || pixels.height == 0) {
            return;  // minimized: nothing owed to a window that shows nothing
        }
        const frame::Extent wanted = app::render_size(pixels, scale_);
        if (wanted != presenter_.size()) {
            view_changed_ = true;  // told to the next frame rendered
        }
        presenter_.resize(wanted);

        const frame::Seconds time = clock_.at(update.presentation);
        const frame::View view = view_changed_ ? frame::View::changed : frame::View::same;
        const frame::FrameInputs inputs{
            .time = time,
            .index = index_,
            .accumulated_since = history_.since(index_, view),
            .camera = scene_ ? std::optional{scene_->camera} : std::nullopt,
            .reused_since = reuse_.since(index_, view),
            .previous = previous_,
        };
        if (const auto rendered = metal::render_to_window(submission_, presenter_, renderer_, inputs, update.drawable)) {
            ++index_;
            view_changed_ = false;
            if (inputs.camera) {
                previous_ = frame::PreviousFrame{.time = inputs.time, .camera = *inputs.camera};
            }
            if (rendered->settled && rendered->settled->sequence >= first_frame_) {
                frame_times_.add(rendered->settled->gpu_start, rendered->settled->gpu_end);
            }
        }
        if (const auto summary = frame_times_.take(time)) {
            window_.set_title(
                title(presenter_.size(), *summary, renderer_.non_finite_samples(), renderer_.mnee_lost_requests()));
        }
    }

    app::Window& window_;
    metal::Submission& submission_;
    metal::Presenter& presenter_;
    metal::Renderer& renderer_;
    const scene::SceneDescription* scene_;  // none for a graph that reads no scene
    double scale_;
    std::uint64_t first_frame_;  // the sequence of the first frame's submission

    app::PresentationClock clock_;
    measurement::FrameTimes frame_times_{title_period};
    std::uint64_t index_ = 0;
    frame::LiveHistory history_;
    frame::LiveReuse reuse_;
    std::optional<frame::PreviousFrame> previous_;  // the last frame rendered, none before the first
    bool view_changed_ = false;  // the size changed since the last frame rendered
    std::exception_ptr failure_;  // the first a frame caught; frames stop at it
};

}  // namespace

int main(int argc, char** argv) {
    try {
        const app::Options options = app::parse(arguments(argc, argv));
        const frame::Schedule schedule = frame::load_schedule(options.graph);
        if (!frame::presents(schedule)) {
            // The program's own refusal of its arguments, as serenity-headless
            // refuses a graph that shows nothing (E.14).
            throw app::OptionsError("the graph " + options.graph.string() +
                                    " shows no image (it renders features alone, for serenity-headless "
                                    "--features); a window needs one that does");
        }
        const std::optional<scene::SceneDescription> loaded =
            options.scene.empty() ? std::nullopt : std::optional{scene::load(options.scene)};

        app::Window window("Serenity", initial_window);
        metal::Device device;
        metal::Submission submission(device);
        metal::Presenter presenter(device, submission, metal::LayerHandle{window.metal_layer()},
                                   app::render_size(window.size_in_pixels(), options.scale));
        metal::Renderer renderer(device, submission, schedule, loaded ? &*loaded : nullptr);
        WindowFrames frames(window, submission, presenter, renderer, loaded ? &*loaded : nullptr, options.scale,
                            submission.next_sequence());
        {
            // Frames are called from here on, while the loop waits; none
            // after the link is destroyed, at the end of this block.
            const metal::DisplayLink link(presenter, frame_rate, frames);
            for (app::Window::Events events = window.wait(); !events.quit && !events.woken;
                 events = window.wait()) {
                // Nothing between waits: the frames run inside them.
            }
        }
        frames.rethrow_failure();
        // What finish() settles comes after the last title: the run reports
        // only whether they failed.
        submission.finish();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "serenity: " << error.what() << '\n';
        return 1;
    }
}
