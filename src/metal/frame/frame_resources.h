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
    // lights them (metal/scene/light_glows.h); and the structure this frame
    // traces (metal/acceleration/scene_acceleration.h). 0 and no structure
    // when the frame has no scene.
    MTL::GPUAddress scene = 0;
    MTL::GPUAddress transforms = 0;
    MTL::GPUAddress glows = 0;
    MTL::ResourceID acceleration{};

    // ReSTIR DI's (frame_images.h, ReuseImages; core/frame/schedule.h): all
    // 0 when the schedule has none of its passes, and each 0 when its
    // passes do not use it. The reuse constants
    // (core/contracts/reuse_constants.h); this frame's surface records,
    // path states and previous positions (contract 5) and last frame's
    // records; the working reservoirs, and this frame's and last frame's
    // final ones (contract 14); and, for the temporal merge, last frame's
    // transforms and glows, and in raytraced mode its structure
    // (metal/scene/shape_transforms.h, light_glows.h,
    // metal/acceleration/scene_acceleration.h; empty in basic mode).
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
    MTL::ResourceID last_acceleration{};
};

}  // namespace serenity::metal
