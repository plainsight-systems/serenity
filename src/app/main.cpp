// serenity: the window. Reads a frame graph, and a scene if one is given,
// opens a window, and renders frames into it until the window is closed,
// Escape is pressed or the program is interrupted.
//
// Frames are rendered at --scale of the window's pixels, and the window's
// layer stretches them to fill it (app/options.h). A denoised graph's light
// is traced at half that size each way and its network, not the layer,
// upscales it (core/frame/schedule.h, light_size).
//
// This file composes the window with a backend, and holds nothing of how
// frames are paced (logical-overview.md, "The frame", "Pacing";
// change-axes.md): the backend's frame source owns the loop and calls the
// window's side of it, app::WindowFrames (app/window_frames.h), for its
// events and for each frame's inputs (core/presentation/frame_source.h).
// The backend is Metal's: its source is metal::DisplayFrames
// (metal/frame/display_frames.h). A graph that shows no image (the
// features pass alone, core/frame/schedule.h) is refused before the window
// opens.
//
// The frame rate asked of the source is the user's (2026-10-18): 60 frames
// a second, the project's target (logical-overview.md, "The frame"), down
// to 30 where a frame takes longer, so a variable-refresh display shows a
// slow frame late rather than a repeated one.
//
// At the end the submission is finished, so a GPU failure in the last
// frames is reported too; what it settles comes after the last title.
// Any failure ends the run with its message and a non-zero status; the
// renderer and the window's surface wait for the frames in flight before
// they are released (metal/device/submission.h, Lifetime).

#include <cstddef>
#include <exception>
#include <iostream>
#include <optional>
#include <span>

#include "app/options.h"
#include "app/window.h"
#include "app/window_frames.h"
#include "core/frame/graph_file.h"
#include "core/presentation/frame_source.h"
#include "core/scene/scene.h"
#include "metal/device/device.h"
#include "metal/device/presenter.h"
#include "metal/device/submission.h"
#include "metal/frame/display_frames.h"
#include "metal/frame/renderer.h"

namespace {

namespace app = serenity::app;
namespace frame = serenity::frame;
namespace metal = serenity::metal;
namespace presentation = serenity::presentation;
namespace scene = serenity::scene;

// The window's size when it opens, in points.
constexpr app::Points initial_window{1280, 800};

// The frame rate asked of the frame source (above).
constexpr presentation::FrameRate frame_rate{
    .minimum = 30.0F, .maximum = 60.0F, .preferred = 60.0F};

// The arguments after the program's name; none if the system gave not even
// that (POSIX allows argc == 0).
std::span<const char* const> arguments(int argc, char** argv) {
  if (argc < 1) {
    return {};
  }
  return {argv + 1, static_cast<std::size_t>(argc - 1)};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const app::Options options = app::parse(arguments(argc, argv));
    const frame::Schedule schedule = frame::load_schedule(options.graph);
    if (!frame::presents(schedule)) {
      // The program's own refusal of its arguments, as serenity-headless
      // refuses a graph that shows nothing (E.14).
      throw app::OptionsError(
          "the graph " + options.graph.string() +
          " shows no image (it renders features alone, for serenity-headless "
          "--features); a window needs one that does");
    }
    const std::optional<scene::SceneDescription> loaded =
        options.scene.empty() ? std::nullopt
                              : std::optional{scene::load(options.scene)};
    const scene::SceneDescription* described = loaded ? &*loaded : nullptr;

    app::Window window("Serenity", initial_window);
    metal::Device device;
    metal::Submission submission(device);
    metal::Presenter presenter(
        device, submission, metal::LayerHandle{window.metal_layer()},
        app::render_size(window.size_in_pixels(), options.scale));
    metal::Renderer renderer(device, submission, schedule, described);
    metal::DisplayFrames metal_frames(submission, presenter, renderer,
                                      frame_rate);

    app::WindowFrames frames(window, described, options.scale);
    presentation::FrameSource& source = metal_frames;
    source.run(frames);

    submission.finish();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "serenity: " << error.what() << '\n';
    return 1;
  }
}
