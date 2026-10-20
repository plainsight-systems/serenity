#pragma once

// Axis: Scene content (the scene's block, a shared layout).
//
// Where each of the scene's arrays is on the GPU, as one block a pass binds
// whole: the address of every array the core lists (scene_arrays.h), in one
// shared layout, written by a backend's host once, at load, and read by its
// shaders, which read their scene through it. A struct of GPU addresses
// the host writes as 64-bit integers, as Apple's bindless scene (WWDC22
// 10101) and Falcor's scene parameter block.
//
// Why a block: a pass binds the scene, not its kinds. A new material,
// texture, medium or light kind adds its array here, to the list
// (scene_arrays.cpp) and to the views each backend's shaders build from
// it, and changes no pass and no renderer (change-axes.md: a new kind
// touches nothing outside its axis but the registry).
//
// One layout for every side: on the host each field is a GPU address, 64
// bits; in a shader, a pointer of the array's type, also 64 bits: Metal's
// in its constant address space, CUDA's device code's a plain pointer
// (nvcc compiles its host half too, which reads the block as device code
// does). Array<T> picks which, a template alias rather than a macro
// (ES.30). Static: the
// arrays never move once made, so the block is written once (CDSA.32: static
// data put once into the layout its consumer reads). What changes per frame,
// the shapes' transforms and the lights' glows, and the acceleration
// structure, a pass binds beside it.
//
// The core's, not a backend's (A.1: stable code apart from what changes
// with the backend): which arrays the scene has and in what order is the
// scene's, the same on every backend; how a backend's shaders read the
// block (Metal's in its constant address space: metal/scene/
// scene_block.metal.h) is the backend's.

#include "core/contracts/shared_layout.h"

#if !defined(__METAL_VERSION__)
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
using Array = constant T*;
#elif defined(__CUDACC__)
template <typename T>
using Array = const T*;
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
  Array<materials::RoughData> roughs;
  Array<materials::DielectricData> dielectrics;
  Array<materials::ConductorData> conductors;
  Array<materials::EmissiveData> emissives;
  Array<materials::CoatedData> coateds;
  Array<media::MediumRecord> media;
  Array<media::AbsorbingData> absorbings;
  Array<shapes::ShapeRecord> shapes;
  Array<shapes::BoxData> boxes;
  Array<lights::LightRecord> light_records;
  Array<uint32_t> shape_lights;
  Array<lights::SphereLightData> sphere_lights;
  Array<lights::LightCounts> light_counts;  // one
};

static_assert(
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers): Metal's layout
    sizeof(SceneBlock) == 19 * sizeof(uint64_t),
    "SceneBlock must be the same 19 addresses on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<SceneBlock>,
              "the block is copied to the GPU as its bytes");
#endif

}  // namespace gpu
}  // namespace serenity
