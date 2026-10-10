#pragma once

// Axis: Scene content (shader half of scene/scene_block.h).
//
// The scene as an estimator reads it: the scene's block (scene_block.h), the
// frame's transforms and glows, and the acceleration structure; and from
// them each family's view, the shapes, materials, textures, lights and media
// (shapes.metal.h, resolve.metal.h, textures.metal.h, emitter.metal.h,
// media.metal.h). A new kind's array joins its family's view here, as it
// joins the block, so no pass changes for it. `Selection` is how the
// estimator reading it chooses among the lights (light_selection/), made from
// the light records and their count: UniformLight for the path tracer
// (path.metal.h), EveryLight for the preview (direct.metal.h).
//
// Each view is built where it is used, not once per thread and carried
// through the path: a view carried is pointers held in registers across
// every ray query and BSDF (GPU.3); built at its use, it is loaded then from
// the block (docs/research/2026-10-10-scene-block.md). The frame's
// transforms and glows are in the constant address space, as the block's
// arrays are, for the same reason (scene_block.h).

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/transform.h"
#include "core/lights/gradient_sky.h"
#include "metal/lights/emitter.metal.h"
#include "metal/materials/resolve.metal.h"
#include "metal/media/media.metal.h"
#include "metal/scene/scene_block.h"
#include "metal/shapes/shapes.metal.h"
#include "metal/textures/textures.metal.h"

namespace serenity {
namespace shaders {

template <typename Selection>
struct SceneView {
    metal::raytracing::primitive_acceleration_structure structure;
    constant serenity::gpu::SceneBlock* block;
    constant serenity::contracts::Transform* transforms;  // as this frame places the shapes
    constant float* glows;                                // as bright as this frame lights them

    Shapes shapes() const { return Shapes{block->shapes, transforms, block->boxes}; }

    Materials materials() const {
        return Materials{block->materials, block->roughs, block->dielectrics, block->conductors, block->coateds};
    }

    Textures textures() const { return Textures{block->textures, block->checkers, block->woods, block->swirls}; }

    Lights lights() const {
        return Lights{block->light_records, block->shape_lights, block->sphere_lights, transforms, glows};
    }

    Media media() const { return Media{block->media, block->absorbings}; }

    serenity::lights::GradientSkyData sky() const { return *block->environment; }

    Selection selection() const { return Selection{block->light_records, block->light_counts->lights}; }
};

}  // namespace shaders
}  // namespace serenity
