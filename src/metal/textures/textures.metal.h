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
#include "metal/textures/swirl.metal.h"
#include "metal/textures/wood.metal.h"

namespace serenity {
namespace shaders {

struct Textures {
    device const serenity::textures::TextureRecord* records;
    device const serenity::textures::CheckerData* checkers;
    device const serenity::textures::WoodData* woods;
    device const serenity::textures::SwirlData* swirls;
};

// `reference` names a texture: its index is not no_texture. `world` is the
// point in the world, `object` the same point in its shape's own
// coordinates (textures/texture.h): a kind laid on the world reads the
// first, one laid on its shape the second.
inline float3 evaluate_texture(Textures textures, serenity::contracts::TextureReference reference, float3 world,
                               float3 object) {
    const serenity::textures::TextureRecord record = textures.records[reference.index];
    switch (record.kind) {
    case serenity::textures::TextureKind::checker:
        return checker(textures.checkers[record.index], world);
    case serenity::textures::TextureKind::wood:
        return wood(textures.woods[record.index], world);
    case serenity::textures::TextureKind::swirl:
        return swirl(textures.swirls[record.index], object);
    }
    return float3(0.0f);
}

}  // namespace shaders
}  // namespace serenity
