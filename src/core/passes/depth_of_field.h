#pragma once

#include "core/contracts/camera.h"
#include "core/contracts/depth_of_field.h"
#include "core/frame/extent.h"

namespace serenity::passes {

// Axis: Pass (depth of field: what a frame's constants are, for every
// backend).
//
// The depth-of-field pass applies the scene's lens to a denoised frame
// after the network (contract 21, core/contracts/depth_of_field.h;
// logical-overview.md, principle 12). It has no settings of its own: the
// lens is the camera's (contract 8), the same one the frame's other graphs
// trace, and the pass's one constant, the largest radius, is Unreal's
// default (the contract). So a graph names the pass and nothing else
// (core/frame/schedule.h).
//
// depth_of_field_constants(): the pass's constants for a frame seen
// through `camera` (its lens on, as the scene describes it) at the
// output's `size`, a pure function of the two (F.8), so every backend
// blurs by the same numbers:
//
//   coc_scale       camera::circle_of_confusion_scale(camera, size);
//   focus_distance  the camera's;
//   max_radius      contracts::dof_max_radius x size.width;
//   half_*          size halved each way, rounded up (contracts::dof_scale);
//   tiles_*         the half grid in tiles of contracts::dof_tile, rounded up;
//   dilation        the tiles within which a near radius of max_radius,
//                   in half-resolution pixels, can reach a tile: ceil of
//                   max_radius / dof_scale / dof_tile, at least 1.
//
// Precondition: `camera` can be framed and `size` is not empty
// (camera/thin_lens.h); throws std::invalid_argument otherwise (I.5).
contracts::DepthOfFieldConstants depth_of_field_constants(
    const contracts::Camera& camera, frame::Extent size);

// Whether a frame with `constants` blurs at all: false for a pinhole (a
// scale of 0), whose depth of field is the image itself, so the pass
// records nothing and the image is the network's to the bit
// (metal/passes/depth_of_field/depth_of_field.h).
constexpr bool blurs(
    const contracts::DepthOfFieldConstants& constants) noexcept {
  return constants.coc_scale > 0.0f;
}

}  // namespace serenity::passes
