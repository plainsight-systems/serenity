#pragma once

#include <array>
#include <cstdint>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/frame/extent.h"
#include "core/passes/tone_map.h"
#include "metal/scene/scene_buffers.h"

namespace serenity::metal {

// Axis: Frame graph.
//
// What a pass is given to record itself: the frame's constants, the image it
// writes, the argument table it binds through, and the scene, the framed
// camera and the accumulated image, if the frame has them. One struct for
// every pass, so the renderer records any pass the same way (renderer.h); each
// pass takes what it needs and refuses, by Error, to run without what it
// needs.
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
    // them (metal/scene/shape_transforms.h); and the structure this frame
    // traces (metal/acceleration/scene_acceleration.h). Null, 0 and no
    // structure when the frame has no scene.
    MTL::GPUAddress scene = 0;  // the scene's block (scene_block.h)
    MTL::GPUAddress transforms = 0;
    MTL::GPUAddress glows = 0;  // the sphere lights' glows this frame (metal/scene/light_glows.h)
    MTL::ResourceID acceleration{};
};

}  // namespace serenity::metal
