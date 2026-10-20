// Reads the scene's arrays through its block (core/scene/scene_block.h), as
// a shader reads them, for tests/gpu/scene_block_test.cpp: query i names a
// field of the block, an element of the array it points to, and a 32-bit
// word of that element, and the probe writes that word. Each element is
// found through the field's own type, so its stride is the shader's size of
// that type. Bindings: the output at 0, then the queries, their count and
// the block (scene_block_probe.h).

#include "core/scene/scene_block.h"
#include "cuda/math/vector.cuh"
#include "gpu/kernels/scene_block_probe.h"

namespace serenity {
namespace shaders {

using serenity::tests::BlockQuery;

namespace {

// Word `word` of element `element` of the array at `array`.
template <typename T>
__device__ uint word_of(const T* array, uint element, uint word) {
  return reinterpret_cast<const uint*>(array + element)[word];
}

}  // namespace

extern "C" __global__ void scene_block_probe(
    uint* out, const BlockQuery* queries, const uint* count_at,
    const serenity::gpu::SceneBlock* block_at) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint& count = *count_at;
  const serenity::gpu::SceneBlock& block = *block_at;
  if (i >= count) {
    return;
  }
  const BlockQuery q = queries[i];
  uint word = 0xdeadbeefu;  // a field the block does not have
  switch (q.field) {
    case 0u:
      word = word_of(block.environment, q.element, q.word);
      break;
    case 1u:
      word = word_of(block.textures, q.element, q.word);
      break;
    case 2u:
      word = word_of(block.checkers, q.element, q.word);
      break;
    case 3u:
      word = word_of(block.woods, q.element, q.word);
      break;
    case 4u:
      word = word_of(block.swirls, q.element, q.word);
      break;
    case 5u:
      word = word_of(block.materials, q.element, q.word);
      break;
    case 6u:
      word = word_of(block.roughs, q.element, q.word);
      break;
    case 7u:
      word = word_of(block.dielectrics, q.element, q.word);
      break;
    case 8u:
      word = word_of(block.conductors, q.element, q.word);
      break;
    case 9u:
      word = word_of(block.emissives, q.element, q.word);
      break;
    case 10u:
      word = word_of(block.coateds, q.element, q.word);
      break;
    case 11u:
      word = word_of(block.media, q.element, q.word);
      break;
    case 12u:
      word = word_of(block.absorbings, q.element, q.word);
      break;
    case 13u:
      word = word_of(block.shapes, q.element, q.word);
      break;
    case 14u:
      word = word_of(block.boxes, q.element, q.word);
      break;
    case 15u:
      word = word_of(block.light_records, q.element, q.word);
      break;
    case 16u:
      word = word_of(block.shape_lights, q.element, q.word);
      break;
    case 17u:
      word = word_of(block.sphere_lights, q.element, q.word);
      break;
    case 18u:
      word = word_of(block.light_counts, q.element, q.word);
      break;
    default:
      break;
  }
  out[i] = word;
}

}  // namespace shaders
}  // namespace serenity
