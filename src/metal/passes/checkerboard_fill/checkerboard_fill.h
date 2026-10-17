#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (the checkerboard's fill).
//
// With checkerboard rendering (contract 20, core/contracts/checkerboard.h;
// architecture/logical-overview.md, "The frame"), ReSTIR PT traces half
// the light size's pixels a frame. The fill makes the light the denoiser
// reads whole: it writes the frame slot's hand-off of the light's radiance
// (denoise_images.h, light_radiance), in place of the copy a denoised
// frame's light half otherwise ends with (metal/frame/renderer.h, "Two
// halves"), every pixel of it:
//
//   - a pixel the frame traced: the accumulated image's, as the copy wrote
//     it;
//   - a pixel it did not, reprojected (console checkerboard rendering's
//     resolve: Frostbite's and Rainbow Six Siege's, which take a missing
//     pixel from the previous frame's samples where its motion and its
//     depth or ID say they are the same surface, and from the current
//     frame's neighbors where not; NRD's checkerboard mode, which RTXDI
//     hands its half-filled lighting to, rebuilds the missing half the same
//     two ways): where its features (contract 17, this frame's, traced at
//     every pixel) say its surface was last frame, its pixel's center plus
//     its motion, the light of the pixels the frame before traced there:
//     of the four around that point, the two of the frame before's field
//     (so samples, never fills of samples), bilinearly weighed, each kept
//     where the depth that frame saw there is the depth the motion
//     predicts, the kept ones' weights renormalized (same_surface(), in the
//     .metal: within depth_tolerance of it and what the surface's depth
//     changes over a pixel's jitter). Every pixel is traced every other
//     frame, so where nothing moved the one kept is the pixel itself and
//     its value the light it traced a frame ago;
//   - where neither is kept (the surface was hidden last frame, behind a
//     firefly that moved or off the image; or no frame before: a
//     history's first frame, a resize), the four traced pixels beside it,
//     left, right, above and below, this frame's light, each kept where
//     its depth is this pixel's surface's by the same test, their mean;
//     where none is, the mean of all four (at the image's edge, those
//     inside it). Never the frame before's light under another surface.
//
// The test is the depth's alone. Its depth is the first surface the
// camera's ray hits, glass included, so it tells a revealed surface from
// a moved one without regard to the light; the features' normal is of the
// surface seen through glass and mirrors, and a sample whose ray reached a
// firefly through a marble is both bright and has a firefly's normal, so a
// normal test keeps dark samples over bright ones. Considered and not
// taken, each tried on the marbles at t = 20, 40 frames: a normal test
// beside the depth's (within 60 degrees), the untraced half 3.5% darker
// than the traced; the bilinear reprojection of the whole of the frame
// before's handed-off image, as SVGF's and TAA's, which takes fills back
// in, a tap of almost no weight carrying a neighbor's fill wherever the
// pixel's own failed the test and fills of fills compounding, 8% darker.
// As built, 2.8% darker, and the light handed to the denoiser 0.9% darker
// than with every pixel traced: what is left is the fireflies seen
// directly, a pixel or two across and moving, whose pixels the frame
// before's field often did not hold, filled then from the table beside
// them (docs/research/2026-10-17-milestone-4-plan.md, Step 4).
//
// The fill writes only the hand-off, never the accumulated image, so a
// pixel's accumulated mean in a still scene is of the frames that traced it
// alone (restir_pt_shade.metal), and the image half reads what it read:
// one RGBA32Float image at the light size, alpha the accumulated count at
// a traced pixel and 0 at a filled one, which nothing reads.
//
// What the frame before traced is the fill's own history
// (metal/frame/checkerboard_images.h): each frame's fill writes, beside the
// hand-off, every pixel's light with its depth into the image of its
// frame's parity, and reads the other (MEM.4: what a frame leaves the next
// is kept one frame, double-buffered). By parity, not by frame slot, so a
// submission between two frames (a readback) does not take the frame
// before away; on the light's queue alone, so nothing on the image half's
// reads what this writes (CP.3). The renderer tells the fill, through
// contract 20's CheckerboardConstants, whether the other image holds this
// history's frame before at this size (renderer.h, "Checkerboard
// rendering"); without one it reads none of it.
//
// One thread a slot of the packed grid (contract 20), the slot's two
// pixels: the one the frame traced, copied, and the one it did not, filled.
// Every lane of a SIMD group does the same two things, where one thread a
// pixel would split each group between the two (GPU.4); adjacent lanes
// take adjacent slots, so adjacent pairs of pixels (GPU.2).
//
// Reads: the frame's constants (contract 7) and the fill's; the
// accumulated image; this frame's depth and motion; the frame before's
// history. Writes: this frame slot's hand-off and this frame's history.
// Barriers, the renderer's (GPU.8): after the dispatches that wrote the
// accumulated image and the features (the shade, the features pass), and,
// with one part, between it and the image half that reads its output.
//
// Cost, at P light pixels: every pixel 4 bytes of its depth read and 32
// written (the hand-off and the history); P / 2 copied, 16 bytes read; P / 2
// filled, 16 bytes of depth and motion, 4 x 4 of its neighbors' depths and
// 2 x 16 of the frame before's history, and where that fails 4 x 16 of this
// frame's light: some 75 bytes a pixel, 145 MB at 1728 x 1117, against the
// copy's 62 MB; no ray. Measured in docs/research/2026-10-17-milestone-4-
// plan.md, Step 4.
class CheckerboardFillPass {
public:
    explicit CheckerboardFillPass(const Library& library);

    // Records the fill. Throws MetalError if `resources` lacks the fill's
    // constants, the accumulated image, this frame's features or hand-off,
    // or either frame's history.
    void record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const;

private:
    NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
};

}  // namespace serenity::metal
