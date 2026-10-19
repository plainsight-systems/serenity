#pragma once

// Contract 17: the denoiser's features. Owned by Film; written by the
// features pass (metal/passes/features/features.h), read by the headless
// renderer's --features (headless/options.h) and, in Milestone 3's Phase 4,
// by the denoiser's network on the GPU every frame
// (docs/research/2026-10-14-denoiser-training-design.md, "What the network
// sees"; docs/research/2026-10-14-denoiser-features.md).
//
// Four images at the frame's size, one value a pixel each, all from the
// camera's ray the light passes trace at that pixel and frame (the path
// pass's point in the pixel and on the lens, metal/passes/path/path.metal;
// in a ReSTIR PT graph the frame's camera jitter, contracts/
// frame_constants.h, the same for both),
// so at the light graph's size each feature is of the very ray its noisy
// sample took, as production's guide buffers are (Open Image Denoise: "All
// auxiliary images should use the same pixel reconstruction filter as the
// beauty image"):
//
//   albedo  RGBA16Float: the surface's color, independent of light and
//           view, in [0, 1] (a = 1). Open Image Denoise's rules (its
//           README, "Albedos"): a matte or coated surface's base color, its
//           clear coat ignored ("Non-absorbing clear coat layers can be
//           simply ignored"); a metal's reflectivity at normal incidence,
//           f0. Through a delta surface, primary surface replacement:
//           - smooth glass (dielectric, two delta lobes): the Fresnel blend
//             of the reflected and transmitted albedos, F a + (1 - F) b,
//             the path split in two at the first hit as the README advises;
//             each half followed on through further glass by one branch,
//             the one Fresnel weighs more (refraction unless F > 1/2, as
//             NVIDIA's "Rendering Perfect Reflections and Refractions in
//             Path-Traced Games" chooses), to the first surface that is not
//             delta; the medium's absorption on the way multiplied in;
//           - a mirror (a conductor of GGX alpha at most
//             feature_mirror_alpha, which the renderer has in place of a
//             delta mirror: the marbles' brass, steel and copper, alpha
//             0.001 to 0.004): followed by reflection, its albedo f0 times
//             the reflected surface's (the reflectance on the way, as the
//             glass's Fresnel weights are);
//           - a light (a firefly) or the sky: 1, nothing to divide the
//             light by, as Open Image Denoise's "the albedo can be simply
//             1" for what has none known;
//           - a chain of more than feature_delta_hops delta hits: 1, as the
//             sky.
//   normal  RGBA16Float: the shading normal, world space, unit, at the
//           same surface as the albedo (Open Image Denoise, "Normals":
//           world or view space, and through a delta surface the next
//           hit's, glass's the Fresnel blend of the two halves', not
//           renormalized, within [-1, 1]); (0, 0, 0) where the ray (or a
//           half) reaches the sky, which has none; a = 0.
//   depth   R32Float: the primary surface's view-space depth, its distance
//           along the camera's forward axis (NRD's IN_VIEWZ, "view-space Z
//           coordinate of primary hits"; the camera's own camera_depth(),
//           metal/camera/thin_lens.metal.h). Of the first surface hit,
//           glass and mirrors included, not the replaced one, so that
//           depth and motion describe one point. 0 where the ray reaches
//           the sky: no surface, as the G-buffer's distance 0
//           (contracts/primary_hit.h), finite where NRD's sky value would
//           be infinite.
//   motion  RGBA16Float: (dx, dy, dz, 0), NRD's 2.5D screen-space motion
//           (its README, IN_MV: "old = new + MV", non-jittered;
//           "`.z = viewZprev - viewZ`"): dx, dy, in pixels of this image,
//           from the primary surface's point's pinhole projection now to
//           its projection last frame, through last frame's camera and
//           with the shape where last frame placed it; dz its view depth
//           last frame less now. The same sign and units as DLSS's
//           (pixels, the current position plus the vector is the previous
//           one). The pinhole projection, not the jittered sample's
//           position, so a still scene under a still camera has motion 0,
//           to float rounding (some 10^-6 pixels), whatever the jitter. The sky's: of the direction, at
//           infinity, so only the camera's turn moves it; dz 0. Where the
//           frame has no frame before (frame::FrameInputs::previous, none
//           at a history's first frame), 0: as DLSS's reset, the consumer
//           told so by the same input. Glass and mirrors show what is
//           behind or before them moving differently from their own
//           surface (production's specular motion vectors); not given
//           here (docs/process/QUEUE.md).
//
// Half floats for three of the four, as production's G-buffers keep them
// (NRD's normals, DLSS's motion vectors RG16F): an albedo or normal to
// 2^-11, a motion of 10 pixels to 0.005; depth at full precision, the
// network's edge test on it.
//
// Layout rules as for every shared contract (contracts/frame_constants.h).

#include "core/contracts/camera.h"
#include "core/contracts/shared_layout.h"

namespace serenity {
namespace contracts {

// A conductor this smooth or smoother is a mirror to the features (above):
// GGX alpha 0.01, roughness 0.1, a lobe about half a degree wide. The
// renderer resolves no conductor below alpha 0.001 (metal/materials/
// resolve.metal.h, least_alpha), so without this it would have no mirror.
SERENITY_CONSTANT float feature_mirror_alpha = 0.01f;

// The most delta hits a feature ray follows past the first: a marble's two
// faces and a cat's-eye core's are four; the rest is margin for marbles
// seen in one another.
SERENITY_CONSTANT uint32_t feature_delta_hops = 8u;

// What the features pass reads beyond the frame's constants and camera:
// whether there is a frame before, and its camera framed for this frame's
// size (camera/thin_lens.h), for the motion.
struct FeatureConstants {
    CameraData previous_camera;
    uint32_t previous;  // 1 when there is a frame before, else 0
    uint32_t padding[3];
};

static_assert(sizeof(FeatureConstants) == 80, "FeatureConstants must be the same 80 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<FeatureConstants>, "FeatureConstants is written to the GPU as bytes");
#endif

}  // namespace contracts
}  // namespace serenity
