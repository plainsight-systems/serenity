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
// The backend is the platform's: on macOS Metal's, its source
// metal::DisplayFrames (metal/frame/display_frames.h); on Linux CUDA's, its
// source cuda::SwapchainFrames (cuda/frame/swapchain_frames.h), presenting
// through Vulkan (cuda/present/presenter.h; docs/research/
// 2026-10-20-cuda-window.md). The window opens at --size points (app/
// options.h). A graph that shows no image (the features pass alone,
// core/frame/schedule.h) is refused before the window opens.
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
#if defined(__APPLE__)
#include "metal/device/device.h"
#include "metal/device/presenter.h"
#include "metal/device/submission.h"
#include "metal/frame/display_frames.h"
#include "metal/frame/renderer.h"
#else
#include "cuda/device/device.h"
#include "cuda/device/submission.h"
#include "cuda/frame/renderer.h"
#include "cuda/frame/swapchain_frames.h"
#include "cuda/present/presenter.h"
#endif

namespace {

namespace app = serenity::app;
namespace frame = serenity::frame;
#if defined(__APPLE__)
namespace metal = serenity::metal;
#else
namespace cuda = serenity::cuda;
#endif
namespace presentation = serenity::presentation;
namespace scene = serenity::scene;

// The frame rate asked of the frame source (above).
constexpr presentation::FrameRate frame_rate{
    .minimum = 30.0F, .maximum = 60.0F, .preferred = 60.0F};

// The arguments after the program's name; none if the system gave not even
// that (POSIX allows argc == 0).
std::span<const char* const> arguments(int argc, char** argv) {
  if (argc < 1) {
    return {};
  }
  return std::span<const char* const>(argv, static_cast<std::size_t>(argc))
      .subspan(1);
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

    app::Window window(
        "Serenity", app::Points{options.window.width, options.window.height});
#if defined(__APPLE__)
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
#else
    cuda::Device device;
    cuda::Submission submission(device);
    // The window's surface, by opaque handles (app/window.h).
    const cuda::SurfaceSource surface{
        .instance_extensions = window.vulkan_instance_extensions(),
        .make_surface =
            [&window](void* instance) {
              return window.make_vulkan_surface(instance);
            },
    };
    cuda::Presenter presenter(
        device, submission, surface,
        app::render_size(window.size_in_pixels(), options.scale));
    cuda::Renderer renderer(device, submission, schedule, described);
    cuda::SwapchainFrames cuda_frames(submission, presenter, renderer,
                                      frame_rate);
    app::WindowFrames frames(window, described, options.scale);
    presentation::FrameSource& source = cuda_frames;
    source.run(frames);
    submission.finish();
#endif
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "serenity: " << error.what() << '\n';
    return 1;
  }
}
