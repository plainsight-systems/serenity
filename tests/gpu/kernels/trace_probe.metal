// Runs the tracing loop (metal/acceleration/trace.metal.h) over the scene's
// two-level structure, for tests/gpu/motion_test.cpp: thread i traces ray i
// and writes the hit's t and its shape, or t = -1 for none. What it checks:
// a world ray carried into a shape's object space by its transform keeps
// the world's t, at any scale, and a hit names its shape by its primitive
// index (scene_acceleration.h).

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/transform.h"
#include "core/shapes/box.h"
#include "core/shapes/primitive.h"
#include "metal/acceleration/trace.metal.h"

using namespace serenity::shaders;

struct ProbeRay {
    float4 origin;     // xyz
    float4 direction;  // xyz, unit length
};

kernel void trace_probe(device float2* out [[buffer(0)]],
                        device const ProbeRay* rays [[buffer(1)]],
                        constant uint& count [[buffer(2)]],
                        metal::raytracing::primitive_acceleration_structure structure [[buffer(3)]],
                        device const serenity::shapes::ShapeRecord* records [[buffer(4)]],
                        device const serenity::contracts::Transform* transforms [[buffer(5)]],
                        device const serenity::shapes::BoxData* boxes [[buffer(6)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= count) {
        return;
    }
    const Shapes shapes{records, transforms, boxes};
    const Hit hit = trace(structure, shapes, rays[i].origin.xyz, rays[i].direction.xyz, 0.0f, INFINITY);
    out[i] = hit.found ? float2(hit.t, float(hit.primitive)) : float2(-1.0f, 0.0f);
}
