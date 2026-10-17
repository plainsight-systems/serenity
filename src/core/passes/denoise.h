#pragma once

#include <filesystem>

namespace serenity::passes {

// Axis: Pass (the denoiser: its graph setting).
//
// The denoise pass runs Milestone 3's trained network on the frame
// (core/contracts/denoise.h; docs/research/2026-10-15-denoiser-live.md):
// the light at half the output's size each way, its features beside it,
// in; the denoised radiance at the output's size, out. Its one setting:
//
//   weights  the directory training/export.py wrote the network into
//            (weights.bin, layout.json: core/denoise/network.h). A relative
//            path is relative to the frame graph file's directory
//            (core/frame/graph_file.h, load_schedule), so a graph names the
//            network beside the checkout wherever it is run from.
struct Denoise {
    std::filesystem::path weights;
};

}  // namespace serenity::passes
