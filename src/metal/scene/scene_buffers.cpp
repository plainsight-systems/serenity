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

// The scene's arrays, in the order of Addresses' fields.
std::array<std::span<const std::byte>, 13> arrays_of(const scene::SceneDescription& scene) {
    return {
        bytes(scene.environment),   bytes(scene.textures),      bytes(scene.checkers),
        bytes(scene.materials),     bytes(scene.rough),         bytes(scene.dielectrics),
        bytes(scene.conductors),    bytes(scene.emissives),     bytes(scene.shapes.records),
        bytes(scene.shapes.spheres), bytes(scene.shapes.boxes), bytes(scene.sphere_lights),
        bytes(scene.light_counts),
    };
}

}  // namespace

SceneBuffers::SceneBuffers(const Device& device, Submission& submission, const scene::SceneDescription& scene)
    : arrays_(device, submission, arrays_of(scene)) {
    addresses_ = Addresses{
        arrays_.address(0), arrays_.address(1),  arrays_.address(2),  arrays_.address(3),  arrays_.address(4),
        arrays_.address(5), arrays_.address(6),  arrays_.address(7),  arrays_.address(8),  arrays_.address(9),
        arrays_.address(10), arrays_.address(11), arrays_.address(12),
    };
}

}  // namespace serenity::metal
