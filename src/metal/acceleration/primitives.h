#pragma once

#include <span>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/shapes/primitive.h"
#include "metal/device/device.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: Acceleration.
//
// The structure rays are traced against: one primitive acceleration
// structure of axis-aligned bounding boxes, one box per shape, in primitive
// order, so primitive i is shape i (shapes/primitive.h). It is given the
// boxes, as the Shape family computes them (shapes/shapes.h), and names no
// shape kind: a new kind changes nothing here. Metal traces the boxes in
// hardware; the exact hit inside a box is the shape kind's own test, which
// the tracing loop in trace.metal.h calls for each candidate box.
//
// Why boxes, not triangles: a sphere's silhouette and normal are then exact,
// which glass magnifies; and a firefly is one box, cheap to move, where a
// tessellated sphere is hundreds of triangles. Metal has no sphere primitive
// (its geometry is triangles, boxes and curves).
//
// Built once, at construction, with Metal 4's buildAccelerationStructure, in
// a submission of its own that is waited for, so it is complete before any
// frame traces against it and no barrier is needed per frame. The scene is
// static: nothing moves yet, so nothing is refit (CDSA.29). When fireflies
// move, their boxes are rewritten and the structure refit each frame, with a
// rebuild when refitting has degraded it (CDSA.30); that is a change to this
// file, designed when it comes.
//
// One level: no instance structure above it, because no shape here is
// instanced. Instancing arrives with the first shape that is.
//
// Throws Error if there are no boxes, if the device cannot make the
// structure, its scratch memory or the box buffer, or if the build fails on
// the GPU (submission.h). Not
// performance-sensitive in this form: one build, at start-up.
class PrimitiveAcceleration {
public:
    // Builds the structure over `bounds`, one box per primitive, in order.
    PrimitiveAcceleration(const Device& device, Submission& submission, std::span<const shapes::Bounds> bounds);

    PrimitiveAcceleration(const PrimitiveAcceleration&) = delete;
    PrimitiveAcceleration& operator=(const PrimitiveAcceleration&) = delete;
    PrimitiveAcceleration(PrimitiveAcceleration&&) = delete;
    PrimitiveAcceleration& operator=(PrimitiveAcceleration&&) = delete;
    ~PrimitiveAcceleration() = default;

    // The structure, as a shader binds it.
    MTL::ResourceID resource() const { return structure_->gpuResourceID(); }

private:
    NS::SharedPtr<MTL::Buffer> boxes_;
    NS::SharedPtr<MTL::AccelerationStructure> structure_;
};

}  // namespace serenity::metal
