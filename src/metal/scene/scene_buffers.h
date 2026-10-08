#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/scene/scene.h"
#include "metal/device/device.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: Scene content (on the GPU).
//
// The scene description's arrays (core/scene/scene.h), copied once into GPU
// memory and made resident, and their addresses, which the passes bind. The
// layouts are the core's shared ones, so the bytes are copied as they are:
// nothing is converted, and the shaders read exactly what the scene reader
// wrote.
//
// One buffer holds every array, each at an offset aligned to 256 bytes, so
// the scene is one allocation and one residency entry, not one per kind
// (GPU.9). An array with no elements has no bytes and its address is 0,
// with a count of 0: a shader indexes it only through a record that names
// its kind, and no record names a kind the scene has none of.
//
// Static: written once, at construction. Moving fireflies, later, rewrite
// their own ring of data each frame (logical-overview.md, Animate); that is
// not this buffer.
//
// Not performance-sensitive: one copy, at start-up.
class SceneBuffers {
public:
    // Copies `scene` into a buffer on `device`, made resident through
    // `submission`. Throws Error if the device cannot make the buffer.
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
    NS::SharedPtr<MTL::Buffer> buffer_;
    Addresses addresses_;
};

}  // namespace serenity::metal
