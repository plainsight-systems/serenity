#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "core/frame/frame_inputs.h"
#include "core/frame/schedule.h"
#include "core/scene/scene.h"

namespace serenity::frame {

// Axis: Frame graph (what a scene or a frame may not be, for every
// backend).
//
// What a backend refuses before a frame's work begins (E.4: an error after
// it begins leaves the backend's submission open): each a limit of a
// contract the core defines, a float the shaders read, a 16-bit index a
// layout packs, so the same scene and frame are refused on every backend,
// with the same words. Each gives the reason, or none, as frame::invalid
// does (schedule.h); the backend throws its own error with it (Metal's:
// "Renderer: " and the reason, metal/frame/renderer.h). Refused rather than
// wrapped or clamped (I.5).
//
// The core's (A.1): the limits are the contracts', not a backend's. Each a
// pure function of its arguments (F.8).

// A time the shaders' float cannot hold (contracts/frame_constants.h): not
// finite, or past float's range, which would reach them as infinity.
// `which` names it, frame `index`'s own time or the frame before's. The
// reason begins "frame N: ".
std::optional<std::string> time_refusal(Seconds time, std::uint64_t index,
                                        std::string_view which);

// A graph whose passes read a scene (needs_scene) given none.
std::optional<std::string> scene_refusal(const Schedule& schedule,
                                         const scene::SceneDescription* scene);

// A frame of a graph that reads a scene with no camera. The reason begins
// "frame N: ".
std::optional<std::string> camera_refusal(const FrameInputs& inputs,
                                          bool needs_scene);

// A frame before whose camera cannot be framed (camera/thin_lens.h,
// invalid), which a graph with the features pass frames to measure motion
// from; its time is time_refusal()'s to refuse. None without a frame
// before. The reason begins "frame N: ".
std::optional<std::string> previous_camera_refusal(const FrameInputs& inputs);

// What ReSTIR PT's G-buffer and light tiles pack of `scene`
// (contracts/primary_hit.h, light_tile.h; scene/packed_range.h): fewer
// media than a medium's 16-bit index holds with its "none", every material
// value within the packed BSDF's range, and at most 65536 shapes and
// lights, each tile entry's 16-bit index.
std::optional<std::string> restir_pt_refusal(
    const scene::SceneDescription& scene);

// What MNEE's requests pack of `scene` (contract 16): at most 65536
// shapes, a request's 16-bit index.
std::optional<std::string> mnee_refusal(const scene::SceneDescription& scene);

// A glowing light whose target is not one of the scene's sphere lights
// (core/animation/animate.h), which animating it would write past.
std::optional<std::string> glower_refusal(const scene::SceneDescription& scene);

}  // namespace serenity::frame
