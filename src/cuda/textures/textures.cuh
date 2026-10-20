#pragma once

// Axis: Texture (shader half of textures/texture.h).
//
// Every texture, as a shader reads it: the records and one array per kind.
// Given a texture reference (contracts/texture_reference.h), its color at a
// point, by its kind: the one place a shader dispatches on texture kind. The
// switch has no default, so a kind added to TextureKind and not here
// fails Metal's compile (-Werror), and here, where nvcc does not warn of
// an enumerator a switch leaves out, traps (device/layout.cuh,
// unhandled_kind).
//
// Translated line for line from Metal's metal/textures/textures.metal.h
// (docs/research/2026-10-19-cuda-backend.md, "Phase 3's design"): its
// prose is Metal's, and a cost or a measurement it gives is the M3 Max's.

#include "core/contracts/texture_reference.h"
#include "core/textures/texture.h"
#include "cuda/device/layout.cuh"
#include "cuda/math/vector.cuh"
#include "cuda/textures/checker.cuh"
#include "cuda/textures/swirl.cuh"
#include "cuda/textures/wood.cuh"

namespace serenity {
namespace shaders {

struct Textures {
  const serenity::textures::TextureRecord* records;
  const serenity::textures::CheckerData* checkers;
  const serenity::textures::WoodData* woods;
  const serenity::textures::SwirlData* swirls;
};

// `reference` names a texture: its index is not no_texture. `world` is the
// point in the world, `object` the same point in its shape's own
// coordinates (textures/texture.h): a kind laid on the world reads the
// first, one laid on its shape the second.
__device__ inline float3 evaluate_texture(
    Textures textures, serenity::contracts::TextureReference reference,
    float3 world, float3 object) {
  const serenity::textures::TextureRecord record =
      textures.records[reference.index];
  switch (record.kind) {
    case serenity::textures::TextureKind::checker:
      return checker(textures.checkers[record.index], world);
    case serenity::textures::TextureKind::wood:
      return wood(textures.woods[record.index], world);
    case serenity::textures::TextureKind::swirl:
      return swirl(textures.swirls[record.index], object);
  }
  unhandled_kind();  // a kind no case handles (layout.cuh)
  return make_float3(0.0f);
}

}  // namespace shaders
}  // namespace serenity
