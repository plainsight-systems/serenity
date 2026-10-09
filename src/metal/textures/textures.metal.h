#pragma once

// Axis: Texture (shader half of textures/texture.h).
//
// Every texture, as a shader reads it: the records and one array per kind.
// Given a texture reference (contracts/texture_reference.h), its color at a
// point, by its kind: the one place a shader dispatches on texture kind. The
// switch has no default, so a kind added to TextureKind and not here fails
// to compile (-Werror).

#include <metal_stdlib>

#include "core/contracts/texture_reference.h"
#include "core/textures/texture.h"
#include "metal/textures/checker.metal.h"

namespace serenity {
namespace shaders {

struct Textures {
    device const serenity::textures::TextureRecord* records;
    device const serenity::textures::CheckerData* checkers;
};

// `reference` names a texture: its index is not no_texture.
inline float3 evaluate_texture(Textures textures, serenity::contracts::TextureReference reference, float3 point) {
    const serenity::textures::TextureRecord record = textures.records[reference.index];
    switch (record.kind) {
    case serenity::textures::TextureKind::checker:
        return checker(textures.checkers[record.index], point);
    }
    return float3(0.0f);
}

}  // namespace shaders
}  // namespace serenity
