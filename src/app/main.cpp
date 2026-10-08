// serenity: the window. Reads a frame graph, opens a window, and renders
// frames into it until the window is closed or Escape is pressed.
//
// Each frame's time is the measured clock (app/clock.h) and its index counts
// the frames rendered; both are inputs, handed to the renderer (principle 1).
// The loop is paced by the display: render_to_window() waits for a drawable
// (metal/device/presenter.h). At the end the submission is finished, so a GPU
// failure in the last frames is reported too. Any failure ends the run with
// its message and a non-zero status.

#include <cstdio>
#include <exception>

#include "app/clock.h"
#include "app/options.h"
#include "app/window.h"
#include "core/frame/graph_file.h"
#include "metal/device/device.h"
#include "metal/device/presenter.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"

int main(int argc, char** argv) {
    using namespace serenity;
    try {
        const app::Options options = app::parse({argv + 1, static_cast<std::size_t>(argc - 1)});
        const frame::Schedule schedule = frame::load_schedule(options.graph);

        app::Window window("Serenity", frame::Extent{1280, 800});
        metal::Device device;
        metal::Submission submission(device);
        metal::Presenter presenter(device, submission, metal::LayerHandle{window.metal_layer()},
                                   window.size_in_pixels());
        metal::Renderer renderer(device, submission, schedule);

        const app::Clock clock;
        std::uint64_t index = 0;
        for (;;) {
            const app::Window::Events events = window.poll();
            if (events.quit) {
                break;
            }
            if (events.resized) {
                presenter.resize(window.size_in_pixels());
            }
            if (metal::render_to_window(submission, presenter, renderer,
                                        frame::FrameInputs{clock.elapsed(), index})) {
                ++index;
            }
        }
        submission.finish();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "serenity: %s\n", error.what());
        return 1;
    }
}
