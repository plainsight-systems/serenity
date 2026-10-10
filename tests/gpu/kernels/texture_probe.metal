// Runs the procedural textures' shader halves for tests/gpu/texture_test.cpp:
// thread i evaluates point i. The noise (metal/textures/noise.metal.h) at
// points[i].point with seed points[i].seed; the wood (metal/textures/
// wood.metal.h) and the swirl (metal/textures/swirl.metal.h) at
// points[i].point, given their data. Bindings: the output at 0, then the
// points, their count, and the texture's data (probes.h).

#include <metal_stdlib>

#include "metal/device/layout.metal.h"
#include "metal/textures/noise.metal.h"
#include "metal/textures/swirl.metal.h"
#include "metal/textures/wood.metal.h"
#include "probes.h"

using namespace serenity::shaders;
using serenity::tests::TexturePoint;

kernel void noise_probe(device float* out [[buffer(0)]],
                        device const TexturePoint* points [[buffer(1)]],
                        constant uint& count [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
    if (i < count) {
        out[i] = gradient_noise(to_float3(points[i].point), points[i].seed);
    }
}

kernel void wood_probe(device serenity::contracts::Float3* out [[buffer(0)]],
                       device const TexturePoint* points [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       constant serenity::textures::WoodData& wood_data [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < count) {
        out[i] = to_packed(wood(wood_data, to_float3(points[i].point)));
    }
}

// The swirl at a point in its shape's own coordinates.
kernel void swirl_probe(device serenity::contracts::Float3* out [[buffer(0)]],
                        device const TexturePoint* points [[buffer(1)]],
                        constant uint& count [[buffer(2)]],
                        constant serenity::textures::SwirlData& swirl_data [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    if (i < count) {
        out[i] = to_packed(swirl(swirl_data, to_float3(points[i].point)));
    }
}
