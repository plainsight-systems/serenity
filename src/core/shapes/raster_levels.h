#pragma once

#include <optional>

#include "core/contracts/camera.h"
#include "core/contracts/float3.h"

namespace serenity::shapes {

// Axis: Acceleration (the rasterizer's meshes:
// docs/research/2026-10-23-apple-silicon-design.md, section 6, change 2).
//
// What the features' raster pass draws of an instanced sphere each frame,
// as raster engines choose a level of detail and cull to the view: its
// icosphere at the coarsest level whose gap, seen from where the sphere is
// this frame, is within the bound the tracer's levels keep from the
// nearest the sphere ever comes (mesh_levels.h, screen_bound), so the
// rasterized surface is within a thirtieth of a pixel of the sphere as the
// traced one is; and nothing where the sphere is out of the camera's view.
// The tracer keeps its levels: a ray from anywhere may meet a sphere from
// anywhere, the rasterizer's rays only from the camera this frame.
//
// Pure functions (F.8), on the host each frame, unit tested.

// The level for a sphere of `radius` centered at `center` this frame, seen
// by `camera`: sphere_level() of a sphere that stays where it is; none
// where none up to the finest serves (the sphere at the lens), whose
// caller draws the tracer's own level.
std::optional<unsigned> raster_level(double radius, contracts::Float3 center,
                                     const contracts::Camera& camera);

// Whether any of the sphere of `radius` at `center` is within the pinhole
// view of `camera` (a frame's: contracts/camera.h, CameraData, its right
// and up the image's half extents a unit ahead): in front of the camera
// and inside each of the view's four sides, the sphere's radius allowed.
bool in_view(const contracts::CameraData& camera, contracts::Float3 center,
             double radius);

}  // namespace serenity::shapes
