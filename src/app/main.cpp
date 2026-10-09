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
// finish() come after the last title and are not shown.
//
// At the end the submission is finished, so a GPU failure in the last frames
// is reported too. Any failure ends the run with its message and a non-zero
// status.

#include <cstdio>
#include <exception>
#include <optional>
#include <string>

#include "app/clock.h"
#include "app/options.h"
#include "app/window.h"
#include "core/frame/graph_file.h"
#include "core/measurement/frame_times.h"
#include "core/scene/scene.h"
#include "metal/device/device.h"
#include "metal/device/presenter.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"

namespace {

std::string title(serenity::frame::Extent size, const serenity::measurement::Summary& summary) {
    char text[160];
    std::snprintf(text, sizeof(text), "Serenity  |  %u x %u  |  GPU %.2f ms mean, %.2f to %.2f, over %llu frames",
                  size.width, size.height, summary.mean.count() * 1e3, summary.shortest.count() * 1e3,
                  summary.longest.count() * 1e3, static_cast<unsigned long long>(summary.frames));
    return text;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace serenity;
    try {
        const app::Options options = app::parse({argv + 1, static_cast<std::size_t>(argc - 1)});
        const frame::Schedule schedule = frame::load_schedule(options.graph);
        std::optional<scene::SceneDescription> scene;
        if (!options.scene.empty()) {
            scene = scene::load(options.scene);
        }

        app::Window window("Serenity", frame::Extent{1280, 800});
        metal::Device device;
        metal::Submission submission(device);
        metal::Presenter presenter(device, submission, metal::LayerHandle{window.metal_layer()},
                                   app::render_size(window.size_in_pixels(), options.scale));
        metal::Renderer renderer(device, submission, schedule, scene ? &*scene : nullptr);
        const std::uint64_t first_frame = submission.next_sequence();

        const app::Clock clock;
        measurement::FrameTimes frame_times(frame::Seconds(1.0));
        std::uint64_t index = 0;
        for (;;) {
            const app::Window::Events events = window.poll();
            if (events.quit) {
                break;
            }
            if (events.resized) {
                presenter.resize(app::render_size(window.size_in_pixels(), options.scale));
            }
            frame::FrameInputs inputs{clock.elapsed(), index, std::nullopt};
            if (scene) {
                inputs.camera = scene->camera;
            }
            if (const auto rendered = metal::render_to_window(submission, presenter, renderer, inputs)) {
                ++index;
                if (rendered->settled && rendered->settled->sequence >= first_frame) {
                    frame_times.add(rendered->settled->gpu_start, rendered->settled->gpu_end);
                }
            }
            if (const auto summary = frame_times.take(clock.elapsed())) {
                window.set_title(title(presenter.size(), *summary));
            }
        }
        (void)submission.finish();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "serenity: %s\n", error.what());
        return 1;
    }
}
