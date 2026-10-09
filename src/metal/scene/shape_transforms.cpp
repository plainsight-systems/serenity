#include "metal/scene/shape_transforms.h"

#include "metal/device/error.h"

namespace serenity::metal {

namespace {

std::span<const contracts::Transform> some(std::span<const contracts::Transform> at_rest) {
    if (at_rest.empty()) {
        throw Error("ShapeTransforms: the scene has no shapes");
    }
    return at_rest;
}

}  // namespace

ShapeTransforms::ShapeTransforms(const Device& device, Submission& submission,
                                 std::span<const contracts::Transform> at_rest, bool moves)
    : array_(device, submission, std::as_bytes(some(at_rest)), moves ? frames_in_flight : 1u) {}

std::span<contracts::Transform> ShapeTransforms::transforms(std::uint32_t slot) const {
    const std::span<std::byte> bytes = array_.bytes(slot);
    // The bytes were made from transforms (the constructor), in a buffer
    // whose copies start 256-byte aligned: they are transforms.
    return {reinterpret_cast<contracts::Transform*>(bytes.data()), bytes.size() / sizeof(contracts::Transform)};
}

}  // namespace serenity::metal
