// Reads the scene's arrays through its block (metal/scene/scene_block.h), as
// a shader reads them, for tests/gpu/scene_block_test.cpp: query i names a
// field of the block, an element of the array it points to, and a 32-bit
// word of that element, and the probe writes that word. Each element is
// found through the field's own type, so its stride is the shader's size of
// that type. Bindings: the output at 0, then the queries, their count and
// the block (probes.h).

#include <metal_stdlib>

#include "metal/scene/scene_block.h"
#include "probes.h"

using serenity::tests::BlockQuery;

namespace {

// Word `word` of element `element` of the array at `array`.
template <typename T>
uint word_of(constant T* array, uint element, uint word) {
    return reinterpret_cast<constant uint*>(array + element)[word];
}

}  // namespace

kernel void scene_block_probe(device uint* out [[buffer(0)]],
                              device const BlockQuery* queries [[buffer(1)]],
                              constant uint& count [[buffer(2)]],
                              constant serenity::gpu::SceneBlock& block [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
    if (i >= count) {
        return;
    }
    const BlockQuery q = queries[i];
    uint word = 0xdeadbeefu;  // a field the block does not have
    switch (q.field) {
        case 0u: word = word_of(block.environment, q.element, q.word); break;
        case 1u: word = word_of(block.textures, q.element, q.word); break;
        case 2u: word = word_of(block.checkers, q.element, q.word); break;
        case 3u: word = word_of(block.woods, q.element, q.word); break;
        case 4u: word = word_of(block.swirls, q.element, q.word); break;
        case 5u: word = word_of(block.materials, q.element, q.word); break;
        case 6u: word = word_of(block.rough, q.element, q.word); break;
        case 7u: word = word_of(block.dielectrics, q.element, q.word); break;
        case 8u: word = word_of(block.conductors, q.element, q.word); break;
        case 9u: word = word_of(block.emissives, q.element, q.word); break;
        case 10u: word = word_of(block.coated, q.element, q.word); break;
        case 11u: word = word_of(block.media, q.element, q.word); break;
        case 12u: word = word_of(block.absorbing, q.element, q.word); break;
        case 13u: word = word_of(block.shapes, q.element, q.word); break;
        case 14u: word = word_of(block.boxes, q.element, q.word); break;
        case 15u: word = word_of(block.light_records, q.element, q.word); break;
        case 16u: word = word_of(block.shape_lights, q.element, q.word); break;
        case 17u: word = word_of(block.sphere_lights, q.element, q.word); break;
        case 18u: word = word_of(block.light_counts, q.element, q.word); break;
        default: break;
    }
    out[i] = word;
}
