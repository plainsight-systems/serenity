#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace serenity::frame {

// Axis: Frame graph.
//
// What a frame computes: the passes it runs, in order. Written in a frame
// graph file (core/frame/graph_file.h), carried in the core, and only carried
// out by a backend (logical-overview.md, principle 10), so every backend
// renders the same frame and none can reorder it. The core holds no schedules
// of its own: which passes a frame runs is data (change-axes.md: mechanism is
// code, values are data), kept apart from the scene, so one scene runs under
// any graph.
//
// A pass is named by its kind. Each backend implements each kind once
// (change-axes.md: every shader-side axis exists once per backend), mapping
// the kinds with a switch that has no default, so adding a kind here fails
// every backend's build until that backend implements it or refuses it by
// name. A backend never silently skips a pass.
//
// Today every pass writes the frame's target. The images passed between
// passes (contract 6) join the schedule with the first pass that reads what
// another wrote, and with them the barriers between passes.
enum class PassKind : std::uint8_t {
    test_pattern,  // a diagnostic image, a function of pixel and time
    preview,       // deterministic ray tracing, direct light only; needs a scene
    path,          // path tracing, naive, accumulating while the image holds still; needs a scene
};

// Whether a pass of `kind` reads the scene. A frame graph with such a pass
// cannot run without one (metal/frame/renderer.h).
bool needs_scene(PassKind kind);

// Whether a pass of `kind` averages its frames into an accumulated image
// (frame/frame_inputs.h, accumulated_since), so that its frames are a
// function of the frames before them since the image started over.
bool accumulates(PassKind kind);

struct Schedule {
    std::vector<PassKind> passes;  // run in this order; at least one
};

// The name each kind is written as in a frame graph file, and back. One table, so
// a kind and its name cannot disagree in two places.
std::string_view name(PassKind kind);
std::optional<PassKind> pass_kind(std::string_view name);

// Every kind, in declaration order: for naming the known ones when a graph
// file names an unknown one.
std::span<const PassKind> all_pass_kinds();

}  // namespace serenity::frame
