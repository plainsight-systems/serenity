// serenity-headless: renders a frame graph's frames, over a scene if one is
// given, to PNG files (headless/options.h). Each frame's time is computed
// from its index, or frozen by --time, never measured; its camera is the
// scene's; each frame is --samples renders of its instant; and a graph that
// converges accumulates from the first frame while the scene looks the same
// at every frame's time, and from each frame's first sample when it moves.
// So the same command writes the same files (principle 1).
//
// Frames are rendered one at a time: each frame's samples are committed, the
// last waited for, and those --write selects are read back and written,
// before the next frame begins.
// Readback is the point of this program, and it is not held to the frame
// budget (GPU.1). A frame that left out samples for not being finite
// (metal/film/non_finite.h) fails the run once it has completed, naming the
// count. Any failure ends the run with its message and a non-zero status.

#include <cstdio>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/animation/animate.h"
#include "core/frame/graph_file.h"
#include "core/output/png.h"
#include "core/scene/scene.h"
#include "headless/options.h"
#include "metal/device/device.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"

int main(int argc, char** argv) {
    using namespace serenity;
    try {
        const headless::Options options = headless::parse({argv + 1, static_cast<std::size_t>(argc - 1)});
        const frame::Schedule schedule = frame::load_schedule(options.graph);
        std::optional<scene::SceneDescription> scene;
        if (!options.scene.empty()) {
            scene = scene::load(options.scene);
        }

        metal::Device device;
        metal::Submission submission(device);
        metal::Offscreen target(device, submission, options.size);
        metal::Renderer renderer(device, submission, schedule, scene ? &*scene : nullptr);

        std::filesystem::create_directories(options.out);
        std::vector<std::uint8_t> rgba(std::size_t{options.size.width} * options.size.height * 4);

        // A moving scene's frames are each their own instant; a still
        // scene, or frozen time, looks the same at every frame, so the run
        // converges as a whole (headless/options.h).
        const bool instants = scene && animation::moves(scene->animation) && !options.time;
        const std::uint64_t samples = options.samples;

        // Counted, so the last frame there can be is reachable without its
        // successor (headless/options.h).
        for (std::uint64_t n = 0; n < options.frames; ++n) {
            const std::uint64_t index = options.first + n;
            const frame::Seconds time = options.time ? *options.time : options.step * static_cast<double>(index);
            std::uint64_t sequence = 0;
            for (std::uint64_t s = 0; s < samples; ++s) {
                const frame::FrameInputs inputs{
                    .time = time,
                    .index = index * samples + s,
                    .accumulated_since = (instants ? index : options.first) * samples,
                    .camera = scene ? std::optional(scene->camera) : std::nullopt,
                };
                sequence = metal::render_to_offscreen(submission, target, renderer, inputs);
            }
            (void)submission.wait_until_complete(sequence);
            if (const std::uint64_t failed = renderer.non_finite_samples(); failed != 0) {
                throw std::runtime_error("frame " + std::to_string(index) + ": " + std::to_string(failed) +
                                         " samples were not finite, and were left out (a bug)");
            }
            if (!headless::written(options.write, n, options.frames)) {
                continue;
            }
            target.read_rgba(rgba);

            char name[32];
            std::snprintf(name, sizeof(name), "frame-%06llu.png", static_cast<unsigned long long>(index));
            const std::filesystem::path path = options.out / name;
            output::write_png(path, options.size, rgba);
            std::printf("%s\n", path.c_str());
        }
        (void)submission.finish();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "serenity-headless: %s\n", error.what());
        return 1;
    }
}
