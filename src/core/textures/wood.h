#pragma once

// Axis: Texture (wood).
//
// A plank tabletop: boards side by side, each cut from its own log, showing
// its growth rings as the long arcs a flat-sawn board shows, wavering with
// the wood's grain, fine pores streaking along it, and a dark seam between
// boards. The marbles' table (scenes/marbles.toml). Decided here, in the
// core, so every backend shows the same table (principle 10); a backend
// computes it in its shaders (metal/textures/wood.metal.h), and its tests
// check its properties there (noise.h says why not against a CPU copy).
//
// Laid on the world's x-z plane, as the checker is (checker.h): the boards
// run along x, side by side along z, and a point's height does not change
// its color, so a box's top shows the boards and its sides their ends drawn
// down. Solid wood would color the sides by the rings' cross-section; the
// table's sides are 3 cm, seen edge-on, and do not need it.
//
// wood(p), for a point p on a surface, by these steps, which a backend's code
// carries by number:
//
//   Step 1  The board: b = floor(p.z / board), and u = p.z / board - b,
//           where p lies across it, in [0, 1).
//   Step 2  The board's log, from pcg3d (noise.h, step 2) of (b as 32-bit
//           unsigned, seed, wood_salt): three numbers h0, h1, h2 in [0, 1),
//           the top 24 bits of each component over 2^24. The log's axis runs
//           along x, offset across the board by (h0 - 0.5) board from its
//           middle, at a depth below the face of d = (wood_depth_least + h1
//           (wood_depth_most - wood_depth_least)) board. The board's own
//           shade is s = 1 + wood_board_shade (2 h2 - 1). So neighboring
//           boards differ, as boards from different logs do.
//   Step 3  The ring radius: the face lies d from the axis, so the point is
//           r = sqrt(a^2 + d^2) from it, a = (u - h0) board being how far
//           across from the axis it is. On the face, rings of radius r are
//           the long arcs, tightest over the axis.
//   Step 4  The grain's waver: r += wood_waver ring fbm(q, wood_waver_octaves,
//           seed) (noise.h),
//           at q = (p.x / wood_waver_along, 0, p.z / wood_waver_across): the
//           noise stretched along the board, so the rings wander slowly
//           along it and quickly across, and taken at height 0 whatever p's,
//           as everything here is.
//   Step 5  The growth ring: t = fract(r / ring), where in its ring the point
//           is, from the ring's inside (earlywood, grown in spring, light)
//           to its outside (latewood, grown in summer, dark):
//           w = smoothstep(wood_latewood, 1, t), and the color
//           light (1 - w) + dark w. The next ring starts light again, so
//           each ring is a gradual darkening and a sharp edge, as wood is.
//   Step 6  The pores: the color times 1 - wood_pores saturate(0.5 + 0.5
//           noise(g, seed + wood_pores_seed)), the noise's bound being past
//           1, g = (p.x / wood_pores_along, 0, p.z / wood_pores_across):
//           fine streaks along the board.
//   Step 7  The board: the color times s (step 2); and within wood_seam of
//           the board's edge, min(u, 1 - u) board < wood_seam, times
//           wood_seam_shade: the gap between boards.
//
// Every channel of the result lies within [min(light, dark) (1 -
// wood_pores) (1 - wood_board_shade) wood_seam_shade, max(light, dark) (1 +
// wood_board_shade)], which the tests hold it to; no channel is negative.
//
// The scene's woods are one array in this layout (core/scene/scene.h,
// SceneDescription::woods), which every pass that reads the scene binds at
// buffer 19 (passes/path/path.metal, passes/preview/preview.metal), the
// first free in both, and hands to the textures (metal/textures/
// textures.metal.h).
//
// Finite everywhere a ray can hit: the world is within 10^6 m of the origin
// (core/scene/scene.h, world_extent), and the scene reader holds ring and
// board within the bounds below. So p.z / board is at most 10^9; r, within
// about 2 boards of the axis plus the waver's 3 rings, over ring is at most
// some 2 x 10^5; and the noise's lattice coordinates, the largest p.z /
// wood_pores_across, about 6.7 x 10^8, fit a 32-bit integer, which noise.h's
// step 1 takes them as. Far from the origin a float's spacing outgrows the
// rings, and the wood there is plain; the table is at the origin.
//
// No filtering: rings and pores are millimetres wide, and where a pixel
// covers several, the path tracer's point drawn anew within the pixel each
// frame (passes/path/path.h) averages them over frames, as it does the
// checker's edges.
//
// Cost, per evaluation: four noises (three octaves and the pores), one
// pcg3d for the board, a square root and a few dozen flops; one branch,
// the seam's, which neighboring pixels take alike but along a seam's line
// (GPU.4). Every tuning number, the octave count and the pores' seed offset
// included, is a named constant below (ES.45), for the shader half to read
// rather than repeat.

#include "core/contracts/shared_layout.h"
#include "core/contracts/float3.h"

namespace serenity {
namespace textures {

SERENITY_CONSTANT uint32_t wood_salt = 0x77D0u;           // step 2: boards' hashes apart from noise's
SERENITY_CONSTANT float wood_depth_least = 0.3f;          // step 2: the face's depth below the log's axis,
SERENITY_CONSTANT float wood_depth_most = 1.5f;           //   in board widths
SERENITY_CONSTANT float wood_board_shade = 0.12f;         // step 2: boards up to 12% lighter or darker
SERENITY_CONSTANT float wood_waver = 1.5f;                // step 4: in rings
SERENITY_CONSTANT float wood_waver_along = 0.5f;          // step 4: meters
SERENITY_CONSTANT float wood_waver_across = 0.05f;        // step 4: meters
SERENITY_CONSTANT uint32_t wood_waver_octaves = 3u;       // step 4: fbm's octaves
SERENITY_CONSTANT float wood_latewood = 0.7f;             // step 5: where in a ring it darkens from
SERENITY_CONSTANT float wood_pores = 0.25f;               // step 6: how much the pores darken
SERENITY_CONSTANT float wood_pores_along = 0.08f;         // step 6: meters
SERENITY_CONSTANT float wood_pores_across = 0.0015f;      // step 6: meters
SERENITY_CONSTANT uint32_t wood_pores_seed = 1u;          // step 6: added to the seed
SERENITY_CONSTANT float wood_seam = 0.0015f;              // step 7: meters
SERENITY_CONSTANT float wood_seam_shade = 0.25f;          // step 7
SERENITY_CONSTANT float wood_least_ring = 1.0e-4f;        // meters: the finest rings read
SERENITY_CONSTANT float wood_least_board = 1.0e-3f;       // meters
SERENITY_CONSTANT float wood_most_board = 10.0f;          // meters

struct WoodData {
    contracts::Float3 light;  // earlywood, linear RGB in [0, 1]
    float ring;               // meters between growth rings; at least wood_least_ring
    contracts::Float3 dark;   // latewood, linear RGB in [0, 1]
    float board;              // meters across a board; wood_least_board to wood_most_board
    uint32_t seed;
    uint32_t padding[3];
};

static_assert(sizeof(WoodData) == 48, "WoodData must be the same 48 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<WoodData>, "WoodData is written to the GPU as bytes");
#endif

}  // namespace textures
}  // namespace serenity
