#pragma once

#include "core/animation/extent.h"
#include "core/contracts/float3.h"

namespace serenity::animation {

// Axis: Animation.
//
// What a motion must keep clear of: the scene's still shapes, as two
// questions a motion asks while it is made, at load. The Animation family
// asks; the scene answers, from its still shapes and each shape kind's exact
// tests (core/scene/scene.h, core/shapes/shapes.h). So every motion kind
// guarantees, when it is made, that what it moves never touches a still
// shape, for all time, and the family depends on no shape kind and no scene
// type: an interface, not a shape (I.25).
//
// Distances are to surfaces: 0 on one, negative inside. Both questions are
// exact for the kinds there are, a sphere and a box. Asked only at load:
// not performance-sensitive per call, though a flight asks some thousands of
// times (flight.h gives the count).
class Obstacles {
public:
    virtual ~Obstacles() = default;

    // The distance from `point` to the nearest still shape's surface.
    virtual double distance(contracts::Float3 point) const = 0;

    // Whether any still shape meets or touches the box `box`.
    virtual bool touches(const Extent& box) const = 0;
};

}  // namespace serenity::animation
