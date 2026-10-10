#pragma once

// Axis: Pass (tone map: what the pass computes, for every backend).
//
// How a frame's linear radiance becomes what the display shows when the look
// wants it shown: the tone-map pass's computation, decided here, in the core,
// so every backend shows the same image for the same radiance
// (logical-overview.md, principle 10). A backend carries it out in its own
// shaders, with its own textures, dispatches and barriers
// (metal/passes/tone_map/tone_map.h), and takes the numbers below from this
// header, which its shading language includes, as layouts are shared
// (contracts/frame_constants.h gives the rules): one definition of every
// constant, none to drift.
//
// Its settings are the graph's (core/frame/graph_file.h), the same every
// frame; the graph's rules check them (core/frame/schedule.h):
//
//   exposure  in stops, within [-10, 10]: the radiance is scaled by
//             2^exposure before anything else. 0 shows it as it is.
//   bloom     the fraction of the light spread into glare, in [0, 1).
//
// No automatic exposure: a frame stays a function of its inputs (principle
// 1), and a flash shows as a flash rather than being adapted away.
//
// Why it exists: a firefly bright enough to light the scene is far brighter
// than the display shows, at its base glow and its flash alike, so its body
// clips to one brightness either way. An eye or a camera shows such a light
// as glare, a halo that grows with its brightness; bloom is that glare, and
// with it a flash shows on the firefly itself.
//
// The computation, by these steps, which a backend's code carries by number,
// over the radiance image L, at the frame's size W x H, and a bloom pyramid
// B_0 .. B_5 of bloom_levels levels:
//
//   Step 1  Exposure: E = 2^exposure L, and each read of E below clamped
//           to bloom_ceiling per channel: each of step 2's bilinear reads,
//           after it averages its texels, and each of step 4's texels. The
//           ceiling, 65504, the largest half float, bounds what the
//           pyramid stores (steps 2 and 3): light brighter than it after
//           exposure glares as if it were that bright, and its core is
//           white either way (step 5). Clamped after the average rather than texel by texel, so a
//           backend reads E through its hardware's filter, 13 reads a texel
//           of B_0 rather than 52; the two differ only for light past the
//           ceiling.
//   Step 2  Down: level k's size is max(1, ceil(previous / 2)) on each axis,
//           from W x H, so a frame of any size, 1 x 1 included, has every
//           level. B_0 is E filtered to its size and divided by
//           bloom_levels, B_k is B_(k-1) filtered to its size, k = 1 .. 5,
//           by Jimenez's 13-tap filter (SIGGRAPH 2014,
//           "Next Generation Post Processing in Call of Duty: Advanced
//           Warfare"): five overlapping 2 x 2 box averages about the output
//           texel's center in the source, the middle one weighted
//           down_middle and the four corner ones down_corner each, read as
//           13 bilinear samples at offsets of 0, 1 and 2 source texels.
//           Coordinates are normalized, the texel centers of a level at
//           (i + 0.5) / size, so a level whose size was rounded up samples
//           its source a fraction past the edge; edges clamp, so a uniform
//           image stays uniform at every level. Not the Karis average Jimenez
//           applies to the first level: it weights a pixel by 1 / (1 + its
//           luminance) to keep isolated over-bright pixels from blooming, and
//           the isolated over-bright pixels here are the fireflies, whose
//           glare is the point. The path tracer's own bright noise blooms with
//           them, until a denoiser takes the noise out first.
//   Step 3  Up: for k = 4 down to 0, B_k += tent(B_(k+1)), the 3 x 3 tent
//           (weights 1, 2, 1 by 1, 2, 1, over 16) of the level below, at a
//           radius of one of its texels, read bilinearly at B_k's texel
//           centers, edges clamped. B_0 is then the mean of six blurs of E,
//           from narrow to wide. Divided first rather than summed and
//           divided last because the pyramid is half floats: six levels at
//           the ceiling would sum past it, to infinity, which step 5 turns
//           to NaN. Divided, each level holds at most bloom_ceiling /
//           bloom_levels (step 1, and each filter's weights sum to 1, so no
//           filter exceeds its largest input), and every partial sum, stored
//           rounded to the nearest half float or toward zero, stays within
//           the ceiling: the sums are worked through, with the rounding the
//           M3 Max's texture writes do, in
//           docs/research/2026-10-10-bloom-half-float.md.
//   Step 4  Composite: C = (1 - bloom) E + bloom tent(B_0), at the frame's
//           size. Each blur keeps E's mean (each filter's weights sum to 1,
//           and edges clamp), so their mean does, and C is a mean of E and
//           it: bloom moves light and makes none.
//   Step 5  Roll-off: Khronos' PBR Neutral tone mapper (KhronosGroup/
//           ToneMapping, PBR_Neutral, 2024), from its published equations:
//           x the smallest channel of C, C -= (x < 0.08 ? x - 6.25 x^2 :
//           0.04); p its largest channel; if p >= neutral_start (0.8 - F90,
//           F90 = 0.04), with d = 1 - neutral_start, p_n = 1 - d^2 / (p + d -
//           neutral_start), C *= p_n / p, and C mixed toward (p_n, p_n, p_n)
//           by 1 - 1 / (neutral_desaturation (p - p_n) + 1). Below the start
//           colors pass as they are; above, they roll off toward 1 and toward
//           white, so a firefly's core goes white-hot while its glare keeps
//           its yellow. Chosen over ACES and AgX because it keeps hues where
//           they are, and the brass and the fireflies are the scene's colors
//           (the note above gives AgX's measured shift of brass). Every
//           input is finite (step 1), so every output is.
//   Step 6  Encode: sRGB's transfer function, into the target.

#include "core/contracts/shared_layout.h"

namespace serenity {
namespace passes {

SERENITY_CONSTANT unsigned int bloom_levels = 6;
SERENITY_CONSTANT float bloom_ceiling = 65504.0f;  // the largest half float (step 1)
SERENITY_CONSTANT float down_middle = 0.5f;        // step 2's weights: the middle box,
SERENITY_CONSTANT float down_corner = 0.125f;      // and each corner box
SERENITY_CONSTANT float neutral_start = 0.76f;     // step 5: 0.8 - F90, F90 = 0.04
SERENITY_CONSTANT float neutral_desaturation = 0.15f;
SERENITY_CONSTANT float max_exposure = 10.0f;      // stops, either way

// The settings, as the graph gives them and a shader reads them.
struct ToneMap {
    float exposure;  // stops; finite, within [-max_exposure, max_exposure]
    float bloom;     // in [0, 1)
    float padding[2];
};

static_assert(sizeof(ToneMap) == 16, "ToneMap must be the same 16 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<ToneMap>, "ToneMap is written to the GPU as bytes");
#endif

}  // namespace passes
}  // namespace serenity
