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
    // The bytes were made from transforms (the constructor).
    return array_.view<contracts::Transform>(slot);
}

}  // namespace serenity::metal
