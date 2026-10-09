#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/frame/extent.h"
#include "metal/scene/scene_buffers.h"

namespace serenity::metal {

class PrimitiveAcceleration;

// Axis: Frame graph.
//
// What a pass is given to record itself: the frame's constants, the image it
// writes, the argument table it binds through, and the scene and the framed
// camera, if the frame has them. One struct for every pass, so the renderer records any pass the
// same way (renderer.h); each pass takes what it needs and refuses, by Error,
// to run without what it needs.
struct FrameResources {
    MTL4::ArgumentTable* arguments = nullptr;
    MTL::GPUAddress constants = 0;  // the frame's contracts::FrameConstants
    MTL::Texture* target = nullptr;
    frame::Extent size;

    // The frame's contracts::CameraData, framed for `size`; 0 when the frame
    // has no camera.
    MTL::GPUAddress camera = 0;

    // Null when the frame has no scene.
    const SceneBuffers::Addresses* scene = nullptr;
    const PrimitiveAcceleration* acceleration = nullptr;
};

}  // namespace serenity::metal
