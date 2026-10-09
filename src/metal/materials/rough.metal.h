#pragma once

// Axis: Material (shader half of materials/rough.h).
//
// A rough surface's color at a point: its texture's, through the texture
// reference, or its constant color. Its BSDF joins this file with light.

#include <metal_stdlib>

#include "core/materials/rough.h"
#include "metal/device/layout.metal.h"
#include "metal/textures/textures.metal.h"

namespace serenity {
namespace shaders {

inline float3 rough_color(serenity::materials::RoughData rough, Textures textures, float3 point) {
    if (rough.texture.index == serenity::contracts::no_texture) {
        return to_float3(rough.color);
    }
    return evaluate_texture(textures, rough.texture, point);
}

}  // namespace shaders
}  // namespace serenity
