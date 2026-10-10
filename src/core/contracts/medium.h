#pragma once

// Contract 12: the medium. Owned by Medium; read by the Integrator, and
// named by Shape.
//
// What fills a shape's inside, between its surfaces: a participating
// medium, which light crossing it interacts with on the way. A shape names
// the medium inside it (its interior, shapes/primitive.h), as pbrt-v4's
// shapes carry a MediumInterface; the shape's material says only what its
// boundary does (contract 2), so a glass's Fresnel and the color of the
// glass behind it change apart (change-axes.md: Material, Medium). Outside
// every shape is air, no medium: light crosses it unchanged.
//
// A path learns which medium it is in by crossing boundaries: passing
// through a surface by a transmission lobe (contract 2) into a shape enters
// that shape's interior, out of it leaves into air. One medium at a time:
// media inside media are not modelled. An opaque shape inside a medium (a
// marble's core in its glass) reflects light without leaving the medium.
//
// The question, the one the integrator asks of whatever medium a stretch of
// path crosses:
//
//   float3 transmittance(medium, t)   the share of light, per channel, kept
//                                     over a stretch of length t inside it,
//                                     in (0, 1]. 1 for a stretch in air.
//
// Absorption alone, so far: a medium that scatters light as well (smoke,
// milky glass) asks the integrator to sample where along a stretch light
// scatters, which joins this contract when that medium does (change-axes.md:
// the first scattering medium changes the integrator again).
//
// The kinds and their records are media/medium.h; each kind's data its own
// file there, its shader half in metal/media/.
//
// Layout rules as for every shared contract (contracts/frame_constants.h).

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

namespace serenity {
namespace contracts {

// A shape's interior, or a path's medium, when it is air: no medium.
#if defined(__METAL_VERSION__)
constant constexpr uint32_t no_medium = 0xffffffffu;
#else
constexpr uint32_t no_medium = 0xffffffffu;
#endif

}  // namespace contracts
}  // namespace serenity
