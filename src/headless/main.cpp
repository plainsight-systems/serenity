// serenity-headless: renders a scene's frames to PNG files
// (headless/options.h). Each frame's time is computed from its index, never
// measured, so the same command writes the same files (principle 1).
//
// Frames are rendered one at a time: each is committed, waited for, read back
// and written before the next begins. Readback is the point of this program,
// and it is not held to the frame budget (GPU.1). Any failure ends the run
// with its message and a non-zero status.

#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "core/output/png.h"
#include "core/frame/graph_file.h"
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

        metal::Device device;
        metal::Submission submission(device);
        metal::Offscreen target(device, submission, options.size);
        metal::Renderer renderer(device, submission, schedule);

        std::filesystem::create_directories(options.out);
        std::vector<std::uint8_t> rgba(std::size_t{options.size.width} * options.size.height * 4);

        // Counted, so the last frame there can be is reachable without its
        // successor (headless/options.h).
        for (std::uint64_t n = 0; n < options.frames; ++n) {
            const std::uint64_t index = options.first + n;
            const frame::FrameInputs inputs{options.step * static_cast<double>(index), index};
            const std::uint64_t sequence = metal::render_to_offscreen(submission, target, renderer, inputs);
            submission.wait_until_complete(sequence);
            target.read_rgba(rgba);

            char name[32];
            std::snprintf(name, sizeof(name), "frame-%06llu.png", static_cast<unsigned long long>(index));
            const std::filesystem::path path = options.out / name;
            output::write_png(path, options.size, rgba);
            std::printf("%s\n", path.c_str());
        }
        submission.finish();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "serenity-headless: %s\n", error.what());
        return 1;
    }
}
