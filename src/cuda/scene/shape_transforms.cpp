#include "cuda/scene/shape_transforms.h"

#include "cuda/device/error.h"

namespace serenity::cuda {

namespace {

std::span<const std::byte> shapes_at_rest(
    std::span<const contracts::Transform> at_rest) {
  if (at_rest.empty()) {
    throw CudaError("ShapeTransforms: a scene of no shapes");
  }
  return std::as_bytes(at_rest);
}

}  // namespace

ShapeTransforms::ShapeTransforms(Submission& submission,
                                 std::span<const contracts::Transform> at_rest)
    : array_(submission, shapes_at_rest(at_rest)) {}

}  // namespace serenity::cuda
