#include "metal/scene/shape_transforms.h"

#include "metal/device/error.h"

namespace serenity::metal {

namespace {

std::span<const contracts::Transform> nonempty(std::span<const contracts::Transform> at_rest) {
    if (at_rest.empty()) {
        throw MetalError("ShapeTransforms: the scene has no shapes");
    }
    return at_rest;
}

}  // namespace

ShapeTransforms::ShapeTransforms(const Device& device, Submission& submission,
                                 std::span<const contracts::Transform> at_rest, FrameArray::Copies copies)
    : array_(device, submission, std::as_bytes(nonempty(at_rest)),
             copies) {}

std::span<contracts::Transform> ShapeTransforms::transforms(std::uint32_t slot) {
    // The bytes were made from transforms (the constructor).
    return array_.view<contracts::Transform>(slot);
}

}  // namespace serenity::metal
