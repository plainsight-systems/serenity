#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "core/contracts/camera.h"
#include "core/contracts/obstacles.h"
#include "core/scene/scene.h"

namespace serenity::cuda {

// Axis: Acceleration (docs/research/2026-10-20-nvidia-hardware-design.md,
// "The spheres as triangles").
//
// Which icosphere each sphere is an instance of (shape_meshes.h): the
// coarsest level whose gap, seen from the nearest the camera comes to the
// sphere, is within screen_bound of a pixel at the largest size the
// project renders (largest_rows): a tessellation set by its error on
// screen, not chosen to fit a time. The nearest: from the lens's edge (the
// camera's place, grown by its lens radius) to the sphere's surface,
// wherever in its reach its center is, from outside it or from within; a
// moving sphere's reach is its motion's extent (core/animation/extent.h),
// a still one's its center. The camera does not move (core/scene/scene.h).
//
// A ray leaves a mesh from the point it traced to, the triangle's
// (shapes/shapes.cuh), so the level bounds only what is seen, not how a
// ray leaves.
//
// Pure functions of the scene description (F.8), on the host, unit tested
// with spheres built for each case. A sphere no level up to the finest
// serves (one whose surface can reach the lens, or so near it no mesh is
// fine enough) is refused by name, never given a level that misses the
// bound (P.6, I.6).

// 3456 x 2234, the M3 Max's window: the most rows a frame of the project
// has. A larger size renders the same meshes with a larger error.
inline constexpr double largest_rows = 2234.0;
// The gap on screen at most a thirtieth of a pixel.
inline constexpr double screen_bound = 1.0 / 30.0;
// A sphere as the bound sees it: its radius and the box its center keeps
// inside.
struct SphereSight {
  double radius = 0.0;
  contracts::Box reach{};
};

// From the lens's edge to the nearest point of `sphere`'s surface the
// sphere's reach allows, from outside it or from inside (a camera within a
// cavity); 0 when its surface can pass through the lens.
double nearest_distance(const SphereSight& sphere,
                        const contracts::Camera& camera);

// The coarsest level meeting the bound, or none if no level up to the
// finest does.
std::optional<unsigned> sphere_level(const SphereSight& sphere,
                                     const contracts::Camera& camera);

// What a shape that is not a sphere is given in sphere_levels(): it is no
// instance of an icosphere.
inline constexpr std::uint8_t no_level = 0xffu;

// A level for each of `scene`'s shapes, in shape order: a sphere's by
// sphere_level(), from its transform at rest (its radius the scale), the
// extent of the motion that moves it if one does (the scene's movers);
// no_level for any other shape.
// Throws CudaError naming the first sphere no level serves.
std::vector<std::uint8_t> sphere_levels(const scene::SceneDescription& scene);

}  // namespace serenity::cuda
