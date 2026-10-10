#pragma once

// Axis: Scene content (on the GPU: the scene's block).
//
// Where each of the scene's arrays is on the GPU, as one block a pass binds
// whole: the address of every array SceneBuffers puts there
// (scene_buffers.h), in one shared layout, read by the host, which writes it
// once, at load, and by the shaders, which read their scene through it
// (scene_block.metal.h). Apple's bindless scene in Metal's terms (WWDC22
// "Go bindless with Metal 3"): a struct of GPU addresses the host writes as
// 64-bit integers, bound once as `constant Scene&`, the arrays it names
// resident through the submission. Falcor's scene parameter block
// (Scene.slang, gScene) is the same idea on D3D12.
//
// Why a block: a pass binds the scene, not its kinds. A new material,
// texture, medium or light kind adds its array here and to the views the
// shaders build from it, both Scene content's, and changes no pass and no
// renderer (change-axes.md: a new kind touches nothing outside its axis but
// the registry). Before it, every pass bound each array at a slot of its
// own, and each new kind changed every pass.
//
// One layout for both sides: on the host each field is a GPU address, 64
// bits; in a shader, a pointer of the array's type, also 64 bits. Array<T>
// picks which, a template alias rather than a macro (ES.30). Static: the
// arrays never move once made, so the block is written once (CDSA.32: static
// data put once into the layout its consumer reads). What changes per frame,
// the shapes' transforms and the lights' glows, and the acceleration
// structure, a pass binds beside it.
//
// The constant address space, as in Apple's example: every array here is
// small, fixed once loaded, and read by every thread at every bounce, the
// reuse Apple's guidance gives that address space (WWDC16 606). Each array
// starts 256-byte aligned (static_arrays.h), what any Apple GPU asks of a
// constant buffer's offset.
//
// Cost, measured on the M3 Max at 3456 x 2234, path traced, nothing else on
// the GPU, as frame times (GPU.10; Instruments' counters were not available
// to attribute them further):
//
//   five spheres  marbles
//     14.8 ms     11.0 ms   each array at a slot of its own, device or
//                           constant alike (b3977c6)
//     16.2          -       the block, device pointers, views built once
//                           per thread and carried through the path
//     15.6        11.6      views built where they are used
//     15.1        11.2      the arrays in the constant address space
//
// So a device pointer loaded from the block cost what a bound one did not,
// and in the constant address space it costs what a bound one does. Of the
// 0.3 ms left, 0.07 is the medium's (integrator/path.metal.h) and the rest
// the block's, under 2%. Measured and not kept, each within 0.02 ms: the
// sky and the light counts held in the block by value, where Metal could
// preload them; and the shape records and boxes, which the intersection
// loop reads at every candidate, bound directly beside the block, as
// Falcor binds its hottest arrays ([root], a root descriptor on D3D12,
// which skips a descriptor table; Metal has no table to skip).

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
using Array = constant T*;
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
