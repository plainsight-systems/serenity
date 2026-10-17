#pragma once

#include <array>
#include <cstdint>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/frame/extent.h"
#include "core/passes/tone_map.h"

namespace serenity::metal {

// Axis: Frame graph.
//
// What a pass is given to record itself: the frame's constants, the image it
// writes, the argument table it binds through, and the scene, the framed
// camera, the accumulated image and ReSTIR's records and reservoirs, if the
// frame has them. One struct for
// every pass, so the renderer records any pass the same way (renderer.h); each
// pass takes what it needs and refuses, by MetalError, to run without what it
// needs.

// Where a ReSTIR pass that produces reservoirs writes them: the working
// array, when a merge after it reads them, or the frame's final array, when
// it is the last before the shade (frame_images.h). Decided by the graph,
// once, when the renderer builds the pass; a type, so the call says which
// (I.4).
enum class ReservoirOutput { working, final };

class DenoiseImages;  // denoise_images.h

struct FrameResources {
    MTL4::ArgumentTable* arguments = nullptr;
    MTL::GPUAddress constants = 0;  // the frame's contracts::FrameConstants
    MTL::Texture* target = nullptr;
    frame::Extent size;

    // The frame's contracts::CameraData, framed for `size`; 0 when the frame
    // has no camera.
    MTL::GPUAddress camera = 0;

    // The images between the frame's passes (frame_images.h): the radiance
    // image and the bloom pyramid's levels; null when the schedule uses
    // none.
    MTL::Texture* radiance = nullptr;
    std::array<MTL::Texture*, passes::bloom_levels> bloom{};

    // The accumulated image (accumulation.h), the frames it holds, which
    // this one joins, and its counter of samples left out for not being
    // finite; null and 0 when no pass in the graph accumulates.
    MTL::Texture* accumulation = nullptr;
    std::uint32_t accumulated_frames = 0;
    MTL::GPUAddress non_finite_counter = 0;

    // The scene's block, the address of each still array
    // (metal/scene/scene_block.h); the shapes' transforms as this frame places
    // them (metal/scene/shape_transforms.h); the sphere lights' glows as it
    // lights them (metal/scene/light_glows.h); and the structures this frame
    // traces, a block of them by ray mask (metal/acceleration/
    // structure_block.h). All 0
    // when the frame has no scene.
    MTL::GPUAddress scene = 0;
    MTL::GPUAddress transforms = 0;
    MTL::GPUAddress glows = 0;
    MTL::GPUAddress acceleration = 0;

    // ReSTIR DI's (frame_images.h, ReuseImages; core/frame/schedule.h): all
    // 0 when the schedule has none of its passes, and each 0 when its
    // passes do not use it. The reuse constants
    // (core/contracts/reuse_constants.h); this frame's surface records,
    // path states and previous positions (contract 5) and last frame's
    // records; the working reservoirs, and this frame's and last frame's
    // final ones (contract 14); and, for the temporal merge, last frame's
    // transforms and glows (metal/scene/shape_transforms.h, light_glows.h).
    MTL::GPUAddress reuse = 0;
    MTL::GPUAddress records = 0;
    MTL::GPUAddress path_states = 0;
    MTL::GPUAddress previous_positions = 0;
    MTL::GPUAddress last_records = 0;
    MTL::GPUAddress working_reservoirs = 0;
    MTL::GPUAddress final_reservoirs = 0;
    MTL::GPUAddress last_final_reservoirs = 0;
    MTL::GPUAddress last_transforms = 0;
    MTL::GPUAddress last_glows = 0;

    // ReSTIR PT's (frame_images.h, PathReuseImages; renderer.h): its
    // settings and reuse textures, static; its path reservoirs, primary
    // hits, light buffer and light tiles; with either reuse, the shifted
    // paths, the replaying shifts, their count and the replay's indirect
    // grid; and with temporal reuse its reuse constants, last frame's
    // primary hits, this frame's and last frame's final reservoirs, the
    // previous pixels and the duplication map, beside the previous
    // positions and last frame's transforms and glows above, which the two
    // families, never in one graph, share. 0 without its passes.
    MTL::GPUAddress path_settings = 0;
    MTL::GPUAddress pairing = 0;
    MTL::GPUAddress path_reservoirs = 0;
    MTL::GPUAddress primaries = 0;
    MTL::GPUAddress light = 0;
    MTL::GPUAddress shifts = 0;
    MTL::GPUAddress replays = 0;
    MTL::GPUAddress replay_counter = 0;
    MTL::GPUAddress replay_grid = 0;
    MTL::GPUAddress light_tiles = 0;
    MTL::GPUAddress path_constants = 0;
    MTL::GPUAddress last_primaries = 0;
    MTL::GPUAddress path_finals = 0;
    MTL::GPUAddress last_path_finals = 0;
    MTL::GPUAddress previous_pixels = 0;
    MTL::GPUAddress duplication = 0;
    // The grid ReSTIR PT's one-thread-a-pixel kernels dispatch over: `size`,
    // or with checkerboard rendering the packed grid of the frame's traced
    // field (contract 20, core/contracts/checkerboard.h), which the kernels
    // map back to pixels by the same contract. Empty without its passes.
    frame::Extent traced;

    // With mnee (core/frame/schedule.h): MNEE's work list (contract 16),
    // its header and the solve's indirect grid in it, and, for the path
    // pass, each pixel's MNEE light before it is folded in (mnee_list.h);
    // the frame's counter of requests the list had no room for
    // (metal/film/failure_count.h); 0 otherwise.
    MTL::GPUAddress mnee_requests = 0;
    MTL::GPUAddress mnee_header = 0;
    MTL::GPUAddress mnee_grid = 0;
    MTL::GPUAddress mnee_light = 0;
    MTL::GPUAddress mnee_lost_counter = 0;

    // With the features pass (contract 17, feature_images.h): its constants
    // (the previous camera), the shapes' transforms as the frame before
    // placed them (this frame's own where nothing moves), and the four
    // feature images it writes; 0 and null otherwise.
    MTL::GPUAddress feature_constants = 0;
    MTL::GPUAddress feature_last_transforms = 0;
    MTL::Texture* albedo = nullptr;
    MTL::Texture* normal = nullptr;
    MTL::Texture* depth = nullptr;
    MTL::Texture* motion = nullptr;

    // With the denoise pass (core/frame/schedule.h; passes/denoise/
    // denoise.h), given to it alone: the light pass's accumulated image, at
    // the light size, beside the feature images above, at the same size;
    // the full-size albedo it remodulates by; its constants (contract 18's
    // DenoiseConstants) and its buffers (denoise_images.h). The radiance
    // image above is then the one it writes, at the output size. Null and 0
    // otherwise.
    MTL::Texture* light_radiance = nullptr;
    MTL::Texture* full_albedo = nullptr;
    MTL::GPUAddress denoise_constants = 0;
    const DenoiseImages* denoise = nullptr;

    // With checkerboard rendering, given to the fill alone (passes/
    // checkerboard_fill/checkerboard_fill.h), beside the accumulated image,
    // this frame's features and its hand-off, light_radiance above: its
    // constants (contract 20's CheckerboardConstants), and its history
    // (checkerboard_images.h), the frame before's light and depth it reads
    // and this frame's it writes. Null and 0 otherwise.
    MTL::GPUAddress checkerboard_constants = 0;
    MTL::Texture* previous_light = nullptr;
    MTL::Texture* history_light = nullptr;
};

}  // namespace serenity::metal
