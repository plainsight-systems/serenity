// serenity-headless: renders a frame graph's frames, over a scene if one is
// given, to PNG files as displayed, or to PFM files of the accumulated
// image's radiance (headless/options.h, --format). Each frame's time is
// computed from its index, or frozen by --time, never measured; its camera
// is the scene's; and which samples each frame is, and what a converging graph's
// image holds, is the core's plan (core/frame/history.h, plan_headless). So
// the same command writes the same files (principle 1), into a directory
// that holds nothing else (headless/options.h, prepare_output).
//
// Frames are rendered one at a time: each frame's samples are committed, the
// last waited for, and those --write selects are read back and written,
// before the next frame begins.
// Readback is the point of this program, and it is not held to the frame
// budget (GPU.1). A frame that left out samples for not being finite
// (metal/film/non_finite.h) fails the run once it has completed, naming the
// count. Any failure ends the run with its message and a non-zero status;
// the renderer and its target wait for the frames in flight before they are
// released (metal/device/submission.h, Lifetime).

#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "core/contracts/linear_image.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/graph_file.h"
#include "core/frame/history.h"
#include "core/frame/schedule.h"
#include "core/output/image_format.h"
#include "core/output/pfm.h"
#include "core/output/png.h"
#include "core/scene/scene.h"
#include "headless/options.h"
#include "metal/device/device.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"

namespace {

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
    using namespace serenity;
    try {
        const headless::Options options = headless::parse(arguments(argc, argv));
        const frame::Schedule schedule = frame::load_schedule(options.graph);
        // A PFM is the accumulated image, which only a graph that accumulates
        // has: refused before anything renders or is made (headless/options.h).
        if (options.format == output::ImageFormat::pfm && !frame::accumulates(schedule)) {
            throw headless::OptionsError("--format pfm writes the accumulated image, and the graph " +
                                         options.graph.string() + " accumulates nothing");
        }
        const std::optional<scene::SceneDescription> loaded =
            options.scene.empty() ? std::nullopt : std::optional{scene::load(options.scene)};

        metal::Device device;
        metal::Submission submission(device);
        metal::Offscreen target(device, submission, options.size);
        metal::Renderer renderer(device, submission, schedule, loaded ? &*loaded : nullptr);

        headless::prepare_output(options.out);
        // Throws for a value no kind names, before any frame renders, so the
        // switches below need no default (P.6); and with none, a kind with no
        // readback fails to compile (ES.79).
        const std::string_view extension = output::extension(options.format);
        // The one buffer a written frame is read back into, of its format's
        // kind, allocated once (MEM.9).
        std::vector<std::uint8_t> rgba;
        std::vector<float> accumulated;
        switch (options.format) {
        case output::ImageFormat::png:
            rgba.resize(target.rgba_size());
            break;
        case output::ImageFormat::pfm: {
            // RGBA32Float a pixel (metal/frame/renderer.h, read_accumulated).
            constexpr std::size_t accumulated_channels = 4;
            accumulated.resize(std::size_t{options.size.width} * options.size.height * accumulated_channels);
            break;
        }
        }
        // Frame `index`, read back and written; its path.
        const auto write_frame = [&](std::uint64_t index) {
            const std::filesystem::path path = options.out / std::format("frame-{:06}.{}", index, extension);
            switch (options.format) {
            case output::ImageFormat::png:
                target.read_rgba(rgba);
                output::write_png(path, options.size, rgba);
                break;
            case output::ImageFormat::pfm:
                renderer.read_accumulated(accumulated);
                output::write_pfm(path, contracts::from_rgba(options.size, accumulated));
                break;
            }
            return path;
        };

        // Which samples each frame is, and what its image holds, is the
        // core's plan (core/frame/history.h).
        const frame::HeadlessPlan plan = frame::plan_headless({
            .first = options.first,
            .samples = options.samples,
            .accumulates = frame::accumulates(schedule),
            .scene_changes = loaded && scene::changes(*loaded),
            .time_frozen = options.time.has_value(),
        });

        // Frame `index`'s samples, each committed; the last one's sequence.
        // The plan has at least one sample a frame (core/frame/history.h).
        const auto render_samples = [&](std::uint64_t index) {
            const frame::Seconds time = options.time ? *options.time : options.step * static_cast<double>(index);
            std::optional<std::uint64_t> last;
            for (std::uint64_t s = 0; s < plan.samples; ++s) {
                const frame::Sample sample = plan.sample({.frame = index, .sample = s});
                const frame::FrameInputs inputs{
                    .time = time,
                    .index = sample.index,
                    .accumulated_since = sample.accumulated_since,
                    .camera = loaded ? std::optional{loaded->camera} : std::nullopt,
                };
                last = metal::render_to_offscreen(submission, target, renderer, inputs);
            }
            if (!last) {
                throw std::logic_error("frame " + std::to_string(index) + ": the plan gave it no sample");
            }
            return *last;
        };

        // Counted, so the last frame there can be is reachable without its
        // successor (headless/options.h).
        for (std::uint64_t n = 0; n < options.frames; ++n) {
            const std::uint64_t index = options.first + n;
            submission.wait_until_complete(render_samples(index));
            if (const std::uint64_t failed = renderer.non_finite_samples(); failed != 0) {
                throw std::runtime_error(std::format("frame {}: {} samples were not finite, and were left out (a bug)",
                                                     index, failed));
            }
            if (headless::written(options.write, {.after_first = n, .frames = options.frames})) {
                std::cout << write_frame(index).string() << '\n';
            }
        }
        submission.finish();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "serenity-headless: " << error.what() << '\n';
        return 1;
    }
}
