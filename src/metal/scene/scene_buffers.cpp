#include "metal/scene/scene_buffers.h"

#include <array>
#include <span>

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

// The one list of the scene's arrays: each with the field of Addresses that
// holds its address, named, so no array can be given another's address.
// Every field has an entry: the assertion below counts them.
struct Entry {
    MTL::GPUAddress SceneBuffers::Addresses::*field;
    std::span<const std::byte> bytes;
};

using A = SceneBuffers::Addresses;
constexpr std::size_t array_count = 15;
static_assert(sizeof(A) == array_count * sizeof(MTL::GPUAddress),
              "every field of SceneBuffers::Addresses needs its entry in entries_of()");

std::array<Entry, array_count> entries_of(const scene::SceneDescription& scene) {
    return {{
        {&A::environment, bytes(scene.environment)},
        {&A::textures, bytes(scene.textures)},
        {&A::checkers, bytes(scene.checkers)},
        {&A::woods, bytes(scene.woods)},
        {&A::materials, bytes(scene.materials)},
        {&A::rough, bytes(scene.rough)},
        {&A::dielectrics, bytes(scene.dielectrics)},
        {&A::conductors, bytes(scene.conductors)},
        {&A::emissives, bytes(scene.emissives)},
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
        addresses_.*entries[i].field = arrays_.address(i);
    }
}

}  // namespace serenity::metal
