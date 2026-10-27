#pragma once

// Axis: Pass (every pass): where each kernel's arguments are bound.
//
// The argument-table index of every buffer and texture each pass's kernels
// take, named once and read by both sides: a kernel's parameter list in its
// .metal file ([[buffer(n)]], [[texture(n)]]) and its pass's record() in
// its .cpp (MTL4::ArgumentTable::setAddress, setResource, setTexture). A
// binding moved here moves on both sides, so they cannot drift apart
// (ES.45, P.5); before this file each side wrote its numbers and a comment
// said they matched.
//
// Plain enumerations, one namespace per kernel, of a fixed underlying type:
// the shading language's attributes and Metal's setters both take an index
// as an integer, which a scoped enumeration would need a cast for at every
// use; the values are the binding layout (Enum.8). Buffers and textures are
// numbered apart, as Metal numbers them. Each is a byte, which every
// table's indices fit (Enum.7: the underlying type saves space).

#include "core/contracts/shared_layout.h"

// NOLINTBEGIN(cppcoreguidelines-use-enum-class): [[buffer(n)]], [[texture(n)]]
// and Metal's setters take an integer index (Enum.3 deviation, AGENTS.md).
namespace serenity {
namespace bindings {

// The scene as every pass that reads it binds it (metal/passes/
// scene_binding.h, bind_scene()): the frame's constants and camera, the
// structure, the scene's block whole (core/scene/scene_block.h), and what
// changes per frame beside it, the sphere lights' glows and the shapes'
// transforms. Each such pass's namespace below repeats these indices, for
// its kernels' parameter lists to name, and is asserted to (below, at the
// end); a pass that reads no camera leaves its index unbound.
namespace scene_view {
enum Buffer : uint8_t {
  constants = 0,
  camera = 1,
  structure = 2,
  scene = 3,
  glows = 4,
  transforms = 5
};
}  // namespace scene_view

namespace test_pattern {
enum Buffer : uint8_t { constants = 0 };
enum Texture : uint8_t { target = 0 };
}  // namespace test_pattern

namespace path {
enum Buffer : uint8_t {
  constants = 0,
  camera = 1,
  structure = 2,
  scene = 3,
  glows = 4,
  transforms = 5,
  non_finite = 6,  // the frame's count of samples not finite
                   // (metal/film/failure_count.h)
};
enum Texture : uint8_t {
  accumulated = 0
};  // also the frame's radiance (path/path.h)
}  // namespace path

// The MNEE pass's two kernels (passes/mnee/mnee.h), recorded by ReSTIR PT's
// initial pass after its own: the scene at the path tracer's indices, then
// the work list, its header, its indirect grid, the light buffer each
// request adds to its pixel (a float4 a pixel), and the frame's count of
// requests lost to a full list (metal/film/failure_count.h).
namespace mnee {
enum Buffer : uint8_t {
  constants = 0,
  structure = 2,
  scene = 3,
  glows = 4,
  transforms = 5,
  non_finite = 6,
  requests = 7,
  header = 8,
  grid = 9,
  light = 10,
  lost = 11,
  // The walks between rounds (contract 16's MneeWalk; mnee.h), rebound
  // between dispatches: the list a dispatch reads, the list it appends the
  // walks still going to and that list's count, the count a grid kernel
  // reads, the converged list and its count.
  walks_in = 12,
  walks_out = 13,
  walk_count_out = 14,
  walk_count = 15,
  converged = 16,
  converged_count = 17,
};
}  // namespace mnee

// The tone map's four kernels (passes/tone_map/tone_map.h): each reads its
// input at texture 0 and writes at 1; the last reads B_0 at 1 and writes
// the target at 2.
namespace tone_map {
enum Buffer : uint8_t { settings = 0 };
enum Texture : uint8_t { input = 0, level = 1, bloom = 1, target = 2 };
}  // namespace tone_map

// ReSTIR PT's passes (metal/passes/restir_pt/restir_pt.metal.h) share one
// layout, each kernel binding the part it reads or writes.
namespace restir_pt {
enum Buffer : uint8_t {
  constants = 0,
  camera = 1,
  structure = 2,
  scene = 3,
  glows = 4,
  transforms = 5,
  seed_functions = 6,  // MNEE's seed's intersection function table
  reservoirs = 7,      // the path reservoirs, a pixel's each (contract 15)
  primaries = 8,    // each pixel's primary hit, for the merge's similarity test
  light = 9,        // the frame's light buffer, a float4 a pixel
  pairing = 10,     // the reuse textures (core/sampling/pairing.h)
  shifts = 11,      // the shifted paths, 3 a pixel
  pairs = 12,       // a list of pairs (contracts::pair_lists): the replays
                    // the pairing appends to; the list a shift dispatch
                    // reads, rebound to each list in turn
  counts = 13,      // the two lists' counts, a word each
  dispatch = 14,    // the lists' indirect grids: the first, which the grid
                    // kernel writes both from; a shift dispatch's list's
  non_finite = 15,  // the frame's count of samples not finite
                    // (metal/film/failure_count.h)
  light_tiles = 16,     // the frame's light tiles (contracts/light_tile.h)
  path_constants = 17,  // contracts::PathReuseConstants: last frame's camera,
                        // whether it is history
  last_primaries = 18,  // last frame's primary hits
  finals = 19,  // this frame's final path reservoirs, which the next frame's
                // temporal reuse reads
  last_finals = 20,  // last frame's
  previous_positions =
      21,  // where each pixel's primary hit was last frame, a Float3
  previous_pixels = 22,  // the temporal input's pixel, a uint a pixel
  duplication = 23,      // the duplication map, D a pixel
  last_transforms = 24,  // last frame's shape transforms
  last_glows = 25,       // and glows
  mnee_requests = 26,    // MNEE's work list (contract 16), the initial pass's
  mnee_header = 27,      // its count and capacity
  light_power = 28,      // the frame's power pyramid (contracts/light_tile.h)
  last_light_power =
      29,  // last frame's, which a shift to last frame's pixel weighs by
  reconnects = 30,  // the pairs that reconnect at x2, which the pairing
                    // appends to
};
enum Texture : uint8_t {
  accumulated = 0
};  // the shade's: also the frame's radiance, as the path pass's
}  // namespace restir_pt

// The features pass (passes/features/features.h): the scene at the path
// tracer's indices, its constants (contract 17's FeatureConstants) and last
// frame's transforms; the four feature images it writes.
namespace features {
enum Buffer : uint8_t {
  constants = 0,
  camera = 1,
  structure = 2,
  scene = 3,
  glows = 4,
  transforms = 5,
  feature_constants = 6,
  last_transforms = 7,
  mesh_block = 8,  // the meshes a visibility texel's hit is rebuilt through
};
enum Texture : uint8_t {
  albedo = 0,
  normal = 1,
  depth = 2,
  motion = 3,
  visibility = 4,  // the primary hits the raster pass drew (visibility.h)
};
}  // namespace features

// The features' raster pass (passes/features/raster.metal), its vertex and
// fragment functions one layout: the camera and the frame's constants at
// the pass's size (its size and jitter); a mesh's vertices and, for a still
// structure's fragments, its triangles; for an instanced mesh, the shapes
// grouped by mesh, and the scene's block and the shapes' transforms, which
// place each.
namespace raster {
enum Buffer : uint8_t {
  camera = 0,
  constants = 1,
  vertices = 2,
  triangles = 3,
  instance_shapes = 4,
  scene = 5,
  transforms = 6,
};
}  // namespace raster

// The denoise pass's kernels (passes/denoise/denoise.h), one layout for all
// of them, each taking what it reads: its constants (contract 18's
// DenoiseConstants) and a dispatch's ConvParams or ResampleParams; the
// buffer it reads, the weights and biases, and the buffer it writes; the
// depth median's histogram and state; the demodulated light; last frame's
// output and this frame's; a decoder's level below. Its textures: the light
// pass's accumulated image and the four features at the light size, the
// full-size albedo, and the radiance image it writes.
namespace denoise {
enum Buffer : uint8_t {
  constants = 0,
  params = 1,
  input = 2,
  weights = 3,
  bias = 4,
  output = 5,
  histogram = 6,
  median = 7,
  demod = 8,
  history = 9,
  next_history = 10,
  coarse = 11,  // a decoder convolution's level below, upsampled as read
};
enum Texture : uint8_t {
  light = 0,
  albedo = 1,
  normal = 2,
  depth = 3,
  motion = 4,
  full_albedo = 5,
  radiance = 6
};
}  // namespace denoise

// The checkerboard's fill (passes/checkerboard_fill/checkerboard_fill.h):
// the light-size frame's constants and its own (contract 20's
// CheckerboardConstants); this frame's accumulated image, the frame
// before's history (metal/frame/checkerboard_images.h: its light and
// depth) and this frame's depth and motion, read; this frame's hand-off
// and history, written.
namespace checkerboard_fill {
enum Buffer : uint8_t { constants = 0, fill_constants = 1 };
enum Texture : uint8_t {
  accumulated = 0,
  previous_light = 1,
  depth = 2,
  motion = 3,
  light = 4,
  history = 5
};
}  // namespace checkerboard_fill

// The depth of field's six kernels (passes/depth_of_field/depth_of_field.h),
// one layout for all of them, each taking what it reads and writes: its
// constants (contract 21's DepthOfFieldConstants) and the tiles' two
// buffers; the radiance image it reads and, at the composite, writes, the
// full-size depth, and its half-resolution images
// (metal/frame/depth_of_field_images.h).
namespace depth_of_field {
enum Buffer : uint8_t { constants = 0, tiles = 1, dilated = 2 };
enum Texture : uint8_t {
  radiance = 0,
  depth = 1,
  presort = 2,
  far = 3,
  near = 4,
  far_filtered = 5,
  near_filtered = 6,
};
}  // namespace depth_of_field

// Every pass that reads the scene binds it at scene_view's indices (above):
// each index as an unsigned, since two enumerations are not compared.
constexpr bool scene_at(unsigned constants, unsigned structure, unsigned scene,
                        unsigned glows, unsigned transforms) {
  return constants == scene_view::constants &&
         structure == scene_view::structure && scene == scene_view::scene &&
         glows == scene_view::glows && transforms == scene_view::transforms;
}
constexpr bool camera_at(unsigned camera) {
  return camera == scene_view::camera;
}
static_assert(scene_at(path::constants, path::structure, path::scene,
                       path::glows, path::transforms));
static_assert(scene_at(features::constants, features::structure,
                       features::scene, features::glows, features::transforms));
static_assert(scene_at(restir_pt::constants, restir_pt::structure,
                       restir_pt::scene, restir_pt::glows,
                       restir_pt::transforms));
static_assert(scene_at(mnee::constants, mnee::structure, mnee::scene,
                       mnee::glows, mnee::transforms));
static_assert(camera_at(path::camera));
static_assert(camera_at(features::camera));
static_assert(camera_at(restir_pt::camera));

}  // namespace bindings
}  // namespace serenity
// NOLINTEND(cppcoreguidelines-use-enum-class)
