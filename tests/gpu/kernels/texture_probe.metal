// Runs the procedural textures' shader halves for
// tests/gpu/texture_test.cpp: thread i evaluates point i. The noise
// (metal/textures/noise.metal.h) at points[i].xyz with the seed in
// points[i].w's bits; the wood (metal/textures/wood.metal.h) at
// points[i].xyz.

#include <metal_stdlib>

#include "metal/textures/noise.metal.h"
#include "metal/textures/wood.metal.h"

using namespace serenity::shaders;

kernel void noise_probe(device float* out [[buffer(0)]],
                        device const float4* points [[buffer(1)]],
                        constant uint& count [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
    if (i < count) {
        out[i] = gradient_noise(points[i].xyz, as_type<uint>(points[i].w));
    }
}

kernel void wood_probe(device float4* out [[buffer(0)]],
                       device const float4* points [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       constant serenity::textures::WoodData& wood_data [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < count) {
        out[i] = float4(wood(wood_data, points[i].xyz), 1.0f);
    }
}
