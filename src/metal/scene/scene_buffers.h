#pragma once

#include <Metal/Metal.hpp>

#include "core/scene/scene.h"
#include "metal/device/device.h"
#include "metal/device/static_arrays.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: Scene content (on the GPU).
//
// Which of the scene description's arrays (core/scene/scene.h) the GPU
// holds, and the address of each, which the passes bind. This file is the
// list of them and nothing more: how they are allocated, aligned and made
// resident is the GPU backend's (metal/device/static_arrays.h). A new kind's
// array is added here, as it is to the description; a change to how arrays
// reach the GPU is made there. Falcor's Scene keeps the same list, in its
// scene's parameter block.
//
// The layouts are the core's shared ones, so the bytes are copied as they
// are: nothing is converted, and the shaders read exactly what the scene
// reader wrote. An array with no elements has address 0: a shader indexes
// it only through a record that names its kind, and no record names a kind
// the scene has none of.
//
// Static: written once, at construction. Moving fireflies, later, rewrite
// their own ring of data each frame (logical-overview.md, Animate); that is
// not this.
//
// Throws Error if the arrays cannot be put on the GPU (static_arrays.h).
class SceneBuffers {
public:
    SceneBuffers(const Device& device, Submission& submission, const scene::SceneDescription& scene);

    SceneBuffers(const SceneBuffers&) = delete;
    SceneBuffers& operator=(const SceneBuffers&) = delete;
    SceneBuffers(SceneBuffers&&) = delete;
    SceneBuffers& operator=(SceneBuffers&&) = delete;
    ~SceneBuffers() = default;

    // The address of each array, as a shader binds it.
    struct Addresses {
        MTL::GPUAddress environment = 0;
        MTL::GPUAddress textures = 0;
        MTL::GPUAddress checkers = 0;
        MTL::GPUAddress materials = 0;
        MTL::GPUAddress rough = 0;
        MTL::GPUAddress dielectrics = 0;
        MTL::GPUAddress shapes = 0;
        MTL::GPUAddress spheres = 0;
        MTL::GPUAddress boxes = 0;
    };
    const Addresses& addresses() const { return addresses_; }

private:
    StaticArrays arrays_;
    Addresses addresses_;
};

}  // namespace serenity::metal
