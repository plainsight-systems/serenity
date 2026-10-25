#pragma once

// Axis: Pass (the features' visibility buffer: the primary hit as the
// rasterizer leaves it).
//
// Each pixel's primary hit as the raster pass draws it
// (docs/research/2026-10-23-apple-silicon-design.md, section 6, change 2;
// raster.metal) and the features kernel reads it (features.metal), one
// RG32Uint texel: the shape the pixel's first surface belongs to, and the
// triangle, its index in its mesh's per-primitive data (the draw's
// primitive id, each mesh's index buffer drawn as built:
// acceleration/scene_acceleration.h); for an instanced shape, drawn at its
// level this frame, with the mesh drawn above it (acceleration/
// structure_block.h, instance_mesh_shift). The sky, where nothing is drawn, the
// shape sky_shape, the value the pass clears the image to, the emitters'
// no_primitive (contracts/emitter.h). Falcor's VBufferRaster's
// record, the hit rebuilt from it (acceleration/trace.metal.h, hit_of()).

#include "core/contracts/emitter.h"
#include "core/contracts/shared_layout.h"

namespace serenity {
namespace bindings {
namespace visibility {

SERENITY_CONSTANT uint32_t sky_shape = contracts::no_primitive;

}  // namespace visibility
}  // namespace bindings
}  // namespace serenity
