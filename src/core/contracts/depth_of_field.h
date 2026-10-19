#pragma once

// Contract 21: depth of field after the denoiser (docs/architecture/
// logical-overview.md, principle 12). Owned by the Pass family; read by
// the Metal depth-of-field pass (metal/passes/depth_of_field/
// depth_of_field.h), host and shaders alike, and filled each frame by the
// frame graph from the camera's lens (metal/frame/renderer.cpp).
//
// The circle of confusion. A thin lens of radius r focused at F
// (contract 8) blurs a point at view depth z (contract 17's depth) to a
// disc, in an image of height H through a vertical field of view fov, of
// radius
//
//   coc(z) = scale (z - F) / z  pixels,  scale = r H / (2 F tan(fov / 2))
//
// (camera/thin_lens.h, circle_of_confusion_scale, which derives it):
// signed, negative in front of the plane of focus (the near field),
// positive behind it (the far field), so one number says both how large a
// pixel's blur is and on which side of the focus it lies. The sky, depth
// 0 (no surface), is at infinity: coc = scale. Clamped to +-max_radius:
// Unreal's largest foreground and background radii by default, 2.5% of the
// view's width each (r.DOF.Kernel.MaxForegroundRadius and
// MaxBackgroundRadius, 0.025), dof_max_radius below. The marbles' lens
// reaches 11 to 13 pixels at 1080 lines, so the clamp bounds the work and
// does not shape the image.
//
// The grid. The gather runs at half the output's size each way, rounded
// up (dof_scale), as Jimenez's (SIGGRAPH 2014) runs at half resolution:
// one half-resolution pixel holds a 2 x 2 block of the output's, its color
// their mean and its circle of confusion the nearest's, so a near edge
// keeps its extent. Its tiles are dof_tile half-resolution pixels on a
// side, each holding the largest near-field radius in it; a tile's
// dilation is the largest of the tiles within `dilation` tiles of it whose
// radius reaches it, so the gather at a sharp pixel beside a defocused
// near object reaches as far as that object spreads (Jimenez's tile max
// and its neighborhood, there 3 x 3 tiles; here as many as the largest
// radius needs at the output's size).
//
// The gather. dof_taps taps a pixel: its own and dof_rings rings, ring k of
// dof_ring_taps x k taps at k / dof_rings of the kernel's radius, Jimenez's
// 49-tap disc; the radius is the larger of the pixel's own and its tile's
// dilated near radius. Its alpha is the near field's coverage, by
// Jimenez's normalization with its factor of dof_alpha_gain.
//
// Layout rules as for every shared contract (contracts/frame_constants.h).

#include "core/contracts/shared_layout.h"

namespace serenity {
namespace contracts {

// of the output's width, as a radius
SERENITY_CONSTANT float dof_max_radius = 0.025f;
// the gather's grid: the output's, halved each way
SERENITY_CONSTANT uint32_t dof_scale = 2u;
// half-resolution pixels on a tile's side
SERENITY_CONSTANT uint32_t dof_tile = 8u;
SERENITY_CONSTANT uint32_t dof_rings = 3u;
// ring k's taps: dof_ring_taps x k
SERENITY_CONSTANT uint32_t dof_ring_taps = 8u;
SERENITY_CONSTANT uint32_t dof_taps = 49u;  // the pixel's own and the rings'
static_assert(dof_taps ==
                  1u + dof_ring_taps * dof_rings * (dof_rings + 1u) / 2u,
              "dof_taps is the pixel and its rings' taps");
SERENITY_CONSTANT float dof_alpha_gain = 2.0f;

// What the pass's kernels read beyond the frame's constants, one per frame
// in flight. Sizes in pixels.
struct DepthOfFieldConstants {
  // output pixels: coc(z) = coc_scale (z - focus_distance) / z
  float coc_scale;
  float focus_distance;  // meters along the camera's forward axis, > 0
  float max_radius;      // output pixels: |coc| at most
  uint32_t width;        // the output's
  uint32_t height;
  uint32_t half_width;  // the gather's grid: (width + 1) / 2
  uint32_t half_height;
  uint32_t tiles_x;  // the grid's tiles: ceil(half_width / dof_tile)
  uint32_t tiles_y;
  uint32_t dilation;  // tiles a tile's dilation reaches each way
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays): MSL has no std::array
  uint32_t padding[2];
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers): Metal's layout
static_assert(sizeof(DepthOfFieldConstants) == 48,
              "DepthOfFieldConstants must be the same 48 bytes on the host and "
              "in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<DepthOfFieldConstants>,
              "DepthOfFieldConstants is written to the GPU as bytes");
#endif

}  // namespace contracts
}  // namespace serenity
