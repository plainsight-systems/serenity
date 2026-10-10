#include "metal/scene/scene_buffers.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <span>

#include "metal/device/error.h"

namespace serenity::metal {

namespace {

template <typename T>
std::span<const std::byte> bytes(const std::vector<T>& array) {
    return std::as_bytes(std::span(array));
}

template <typename T>
std::span<const std::byte> bytes(const T& one) {
    return std::as_bytes(std::span(&one, 1));
}

// The one list of the scene's arrays: each with the field of the scene's
// block (scene_block.h) that holds its address, named, so no array can be
// given another's address. Every field has an entry: the assertion below
// counts them, at compile time rather than at run time (P.5).
struct Entry {
    std::uint64_t gpu::SceneBlock::*field;
    std::span<const std::byte> bytes;
};

using A = gpu::SceneBlock;
constexpr std::size_t array_count = 19;
static_assert(sizeof(A) == array_count * sizeof(std::uint64_t),
              "every field of gpu::SceneBlock needs its entry in entries_of()");

std::array<Entry, array_count> entries_of(const scene::SceneDescription& scene) {
    return {{
        {&A::environment, bytes(scene.environment)},
        {&A::textures, bytes(scene.textures)},
        {&A::checkers, bytes(scene.checkers)},
        {&A::woods, bytes(scene.woods)},
        {&A::swirls, bytes(scene.swirls)},
        {&A::materials, bytes(scene.materials)},
        {&A::rough, bytes(scene.rough)},
        {&A::dielectrics, bytes(scene.dielectrics)},
        {&A::conductors, bytes(scene.conductors)},
        {&A::emissives, bytes(scene.emissives)},
        {&A::coated, bytes(scene.coated)},
        {&A::media, bytes(scene.media)},
        {&A::absorbing, bytes(scene.absorbing)},
        {&A::shapes, bytes(scene.shapes.records)},
        {&A::boxes, bytes(scene.shapes.boxes)},
        {&A::light_records, bytes(scene.lights)},
        {&A::shape_lights, bytes(scene.shape_lights)},
        {&A::sphere_lights, bytes(scene.sphere_lights)},
        {&A::light_counts, bytes(scene.light_counts)},
    }};
}

std::array<std::span<const std::byte>, array_count> arrays_of(const scene::SceneDescription& scene) {
    std::array<std::span<const std::byte>, array_count> arrays;
    const auto entries = entries_of(scene);
    for (std::size_t i = 0; i < array_count; ++i) {
        arrays[i] = entries[i].bytes;
    }
    return arrays;
}

}  // namespace

SceneBuffers::SceneBuffers(const Device& device, Submission& submission, const scene::SceneDescription& scene)
    : arrays_(device, submission, arrays_of(scene)) {
    const auto entries = entries_of(scene);
    for (std::size_t i = 0; i < array_count; ++i) {
        block_.*entries[i].field = arrays_.address(i);
    }
    // The block itself, written once (scene_block.h); shared storage, which
    // Apple silicon's unified memory lets the GPU read where the CPU wrote.
    block_buffer_ = NS::TransferPtr(device.handle()->newBuffer(sizeof(block_), MTL::ResourceStorageModeShared));
    if (!block_buffer_) {
        throw Error("SceneBuffers: the device made no buffer for the scene's block");
    }
    std::memcpy(block_buffer_->contents(), &block_, sizeof(block_));
    resident_ = submission.keep_resident(block_buffer_.get());
}

}  // namespace serenity::metal
