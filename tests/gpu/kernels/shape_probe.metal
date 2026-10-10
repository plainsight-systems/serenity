// Runs the shapes' surface interaction (metal/shapes/shapes.metal.h) for
// tests/gpu/shape_test.cpp: thread i fills contract 1 at queries[i].xyz on
// shape uint(queries[i].w), a ray arriving from +z.

#include <metal_stdlib>

#include "metal/shapes/shapes.metal.h"

using namespace serenity::shaders;

kernel void interaction_probe(device serenity::contracts::SurfaceInteraction* out [[buffer(0)]],
                              device const float4* queries [[buffer(1)]],
                              constant uint& count [[buffer(2)]],
                              constant serenity::shapes::ShapeRecord* records [[buffer(3)]],
                              constant serenity::contracts::Transform* transforms [[buffer(4)]],
                              constant serenity::shapes::BoxData* boxes [[buffer(5)]],
                              uint i [[thread_position_in_grid]]) {
    if (i < count) {
        out[i] = surface_interaction(Shapes{records, transforms, boxes}, uint(queries[i].w), queries[i].xyz,
                                     float3(0.0f, 0.0f, -1.0f));
    }
}
