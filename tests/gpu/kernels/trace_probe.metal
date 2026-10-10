// Runs the tracing loop (metal/acceleration/trace.metal.h) over the scene's
// structure, for tests/gpu/acceleration_test.cpp: thread i traces ray i and
// writes the hit's t and its shape, or found 0. What it checks: a world ray
// carried into a shape's object space by its transform keeps the world's t,
// at any scale, and a hit names its shape by its primitive index
// (scene_acceleration.h). Bindings: the output at 0, then the rays, their
// count, the structure, and the shapes' records, transforms and boxes
// (probes.h).

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/transform.h"
#include "core/shapes/box.h"
#include "core/shapes/primitive.h"
#include "metal/acceleration/trace.metal.h"
#include "metal/device/layout.metal.h"
#include "probes.h"

using namespace serenity::shaders;
using serenity::tests::TraceHit;
using serenity::tests::TraceRay;

kernel void trace_probe(device TraceHit* out [[buffer(0)]],
                        device const TraceRay* rays [[buffer(1)]],
                        constant uint& count [[buffer(2)]],
                        metal::raytracing::primitive_acceleration_structure structure [[buffer(3)]],
                        constant serenity::shapes::ShapeRecord* records [[buffer(4)]],
                        constant serenity::contracts::Transform* transforms [[buffer(5)]],
                        constant serenity::shapes::BoxData* boxes [[buffer(6)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= count) {
        return;
    }
    const Shapes shapes{records, transforms, boxes};
    const Hit hit = trace(structure, shapes,
                          metal::raytracing::ray(to_float3(rays[i].origin), to_float3(rays[i].direction), 0.0f,
                                                 unbounded));
    out[i] = hit.found ? TraceHit{hit.t, hit.primitive, 1u} : TraceHit{0.0f, 0u, 0u};
}
