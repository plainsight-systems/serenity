#pragma once

// Axis: Scene content (on the GPU: the scene's block).
//
// Where each of the scene's arrays is on the GPU, as one block a pass binds
// whole: the address of every array SceneBuffers puts there
// (scene_buffers.h), in one shared layout, read by the host, which writes it
// once, at load, and by the shaders, which read their scene through it
// (scene_block.metal.h). Falcor's scene parameter block, in Metal 4's terms:
// a buffer of GPU addresses, the arrays it names resident through the
// submission.
//
// Why a block: a pass binds the scene, not its kinds. A new material,
// texture, medium or light kind adds its array here and to the views the
// shaders build from it, both Scene content's, and changes no pass and no
// renderer (change-axes.md: a new kind touches nothing outside its axis but
// the registry). Before it, every pass bound each array at a slot of its
// own, and each new kind changed every pass.
//
// One layout for both sides: on the host each field is a GPU address, 64
// bits; in a shader, a pointer to device memory of the array's type, also
// 64 bits. Array<T> picks which, a template alias rather than a macro
// (ES.30). Static: the arrays never move once made, so the block is written
// once (CDSA.32: static data put once into the layout its consumer reads).
// What changes per frame, the shapes' transforms and the lights' glows, and
// the acceleration structure, a pass binds beside it.
//
// Cost: the block is 19 addresses, 152 bytes, and reading the arrays
// through it costs time a bound pointer did not: on the M3 Max at 3456 x
// 2234, path traced, nothing else on the GPU, five spheres take 15.6 ms a
// frame against 14.8 with each array at a slot of its own, the marbles
// 11.6 against 11.0, some 5%. Half the cost first measured is recovered by
// building each view where it is used (scene_block.metal.h); the rest is
// the price of passes that bind the scene, not its kinds (GPU.10).

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#else
#include <cstdint>
#endif

#include "core/lights/gradient_sky.h"
#include "core/lights/light.h"
#include "core/lights/sphere_light.h"
#include "core/materials/coated.h"
#include "core/materials/conductor.h"
#include "core/materials/dielectric.h"
#include "core/materials/emissive.h"
#include "core/materials/material.h"
#include "core/materials/rough.h"
#include "core/media/absorbing.h"
#include "core/media/medium.h"
#include "core/shapes/box.h"
#include "core/shapes/primitive.h"
#include "core/textures/checker.h"
#include "core/textures/swirl.h"
#include "core/textures/texture.h"
#include "core/textures/wood.h"

namespace serenity {
namespace gpu {

#if defined(__METAL_VERSION__)
template <typename T>
using Array = device const T*;
#else
template <typename T>
using Array = std::uint64_t;  // the array's GPU address
#endif

struct SceneBlock {
    Array<lights::GradientSkyData> environment;  // one
    Array<textures::TextureRecord> textures;
    Array<textures::CheckerData> checkers;
    Array<textures::WoodData> woods;
    Array<textures::SwirlData> swirls;
    Array<materials::MaterialRecord> materials;
    Array<materials::RoughData> rough;
    Array<materials::DielectricData> dielectrics;
    Array<materials::ConductorData> conductors;
    Array<materials::EmissiveData> emissives;
    Array<materials::CoatedData> coated;
    Array<media::MediumRecord> media;
    Array<media::AbsorbingData> absorbing;
    Array<shapes::ShapeRecord> shapes;
    Array<shapes::BoxData> boxes;
    Array<lights::LightRecord> light_records;
    Array<uint32_t> shape_lights;
    Array<lights::SphereLightData> sphere_lights;
    Array<lights::LightCounts> light_counts;  // one
};

static_assert(sizeof(SceneBlock) == 19 * 8, "SceneBlock must be the same 19 addresses on the host and in shaders");

}  // namespace gpu
}  // namespace serenity
