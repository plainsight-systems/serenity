#pragma once

// Contract 11: obstacles. Owned by Animation; answered by Scene content.
//
// What a motion must keep clear of: the scene's still shapes, as two
// questions a motion asks while it is made, at load. Animation asks, and
// says what it needs to know; the scene answers, from its still shapes and
// each shape kind's exact tests (core/scene/scene.h, core/shapes/shapes.h).
// So every motion kind guarantees, when it is made, that what it moves never
// touches a still shape, for all time (core/animation/motion.h), and neither
// family depends on the other: Animation names no shape kind and no scene
// type, and the scene names no motion kind (I.25).
//
// Distances are to surfaces: 0 on one, negative inside. Both questions are
// exact for the kinds there are, a sphere and a box. Asked only at load:
// not performance-sensitive per call, though a flight asks some thousands of
// times (core/animation/flight.h gives the count). CPU only.
//
// Asked from several threads at once: flights are made in parallel
// (flight.h, make_flights), so an answer must be safe to ask concurrently
// and the same from any thread. An implementation keeps no state that a
// question changes: both are const, and read only what was fixed before the
// first was asked.
//
// An interface: used through a reference to it, never copied, so its copy
// and move are deleted, and nothing sliced (C.67, COPY.4).

#include "core/contracts/float3.h"

namespace serenity {
namespace contracts {

// An axis-aligned box in the world, min below max on every axis.
struct Box {
    Float3 min;
    Float3 max;
};

class Obstacles {
public:
    virtual ~Obstacles() = default;
    Obstacles(const Obstacles&) = delete;
    Obstacles& operator=(const Obstacles&) = delete;
    Obstacles(Obstacles&&) = delete;
    Obstacles& operator=(Obstacles&&) = delete;

    // The distance from `point` to the nearest still shape's surface.
    virtual double distance(Float3 point) const = 0;

    // Whether any still shape meets or touches `box`.
    virtual bool touches(const Box& box) const = 0;

protected:
    Obstacles() = default;
};

}  // namespace contracts
}  // namespace serenity
