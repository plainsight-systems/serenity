#pragma once

// Contract 2: the BSDF. Owned by Material; read by the Integrator and,
// later, Sample reuse.
//
// How a surface scatters light: given the direction light leaves toward
// (wo) and a direction it arrives from (wi), how much of it scatters, how to
// choose a wi for a given wo, and with what probability. Every material kind
// answers the same four questions, so an estimator never names a kind, and
// adding a kind changes no estimator. The interface is pbrt-v4's (BxDF:
// f, Sample_f, PDF, Flags) and Falcor's (IBSDF: eval, sample, evalPdf).
//
// The data here is shared, one definition for both backends' shaders; the
// functions are each backend's, written to the semantics below, one file per
// kind in metal/materials/ and a dispatch on the kind.
//
// Resolving. A material is resolved once at a surface:
//
//   Bsdf resolve(material record, SurfaceInteraction, the scene's arrays)
//
// reads its textures there and fills a Bsdf: the kind, the shading normal
// and the parameters at that point. Every later question about that surface
// reads the Bsdf, so textures are read once per hit, not once per light
// sample, and ReSTIR, which asks about the same surface many times, never
// reads them again.
//
// The questions. Directions are unit vectors in world space, pointing away
// from the surface: wo toward where the light goes (the camera, or the
// previous vertex of a path), wi toward where it comes from. Each kind works
// in its own local frame about the shading normal.
//
//   float3 evaluate(Bsdf, wo, wi)   f(wo, wi): the scattered radiance per
//                                   unit irradiance, without the cosine,
//                                   which the caller applies. 0 for a delta
//                                   lobe, which no wi chosen apart from it
//                                   can reach.
//   BsdfSample sample(Bsdf, wo, u)  a wi drawn for wo from u, three numbers
//                                   in [0, 1): u.x chooses among a kind's
//                                   lobes (glass: reflect or refract), u.yz
//                                   the direction. pdf 0 means no sample
//                                   (wo below a reflector's horizon); the
//                                   caller ends the path.
//   float pdf(Bsdf, wo, wi)         the density with which sample() draws wi,
//                                   per unit solid angle. 0 for a delta lobe.
//   uint lobes(Bsdf)                every lobe the surface has, as the union
//                                   of their bits (below). A surface with no
//                                   diffuse or glossy lobe, all of whose
//                                   lobes are delta, cannot be lit by aiming
//                                   at a light: the estimator skips that,
//                                   and ReSTIR does not reuse there. The test
//                                   is on the bits, never on the kind, so it
//                                   holds for any kind, mixed ones included.
//
// For every sample, value x |cos(wi, shading normal)| / pdf is the sample's
// weight, its contribution per unit of the radiance arriving along wi. For a
// delta lobe, pdf is the discrete probability of choosing it, and value is
// set so that the weight comes out right: F / |cos| for a mirror, chosen
// with probability F. The weight is what a path's throughput is multiplied
// by.
//
// Emission is not here. A glowing surface scatters nothing (kind none) and
// its light is the emitter contract's (contract 3).
//
// Layout rules as for every shared contract (contracts/frame_constants.h).

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "core/contracts/float3.h"

namespace serenity {
namespace contracts {

// The kinds a Bsdf resolves to: one per way of scattering, not per material
// kind. A material kind resolves to one of these (materials/material.h).
enum class BsdfKind : uint32_t {
    none = 0,        // scatters nothing: an emissive surface
    lambert = 1,     // rough: albedo / pi, every way
    conductor = 2,   // metal: GGX microfacets, Schlick's Fresnel from f0
    dielectric = 3,  // smooth glass: mirror reflection and refraction, by Fresnel
};

// Bits of lobes() and BsdfSample::lobe, as pbrt-v4's BxDFFlags: each lobe
// is one of reflection or transmission, which side wi is on, and one of
// diffuse, glossy or delta, how it spreads. lobes() is the union over a
// surface's lobes; a sample's lobe has one bit of each pair.
//
//   diffuse  spreads over the hemisphere (Lambert);
//   glossy   spreads about a direction (rough metal);
//   delta    one direction only: evaluate() and pdf() are 0 for it.
#if defined(__METAL_VERSION__)
constant constexpr uint32_t lobe_reflection = 1u;    // wi on wo's side
constant constexpr uint32_t lobe_transmission = 2u;  // wi on the other side
constant constexpr uint32_t lobe_diffuse = 4u;
constant constexpr uint32_t lobe_glossy = 8u;
constant constexpr uint32_t lobe_delta = 16u;
#else
constexpr uint32_t lobe_reflection = 1u;
constexpr uint32_t lobe_transmission = 2u;
constexpr uint32_t lobe_diffuse = 4u;
constexpr uint32_t lobe_glossy = 8u;
constexpr uint32_t lobe_delta = 16u;
#endif

// Whether a surface with `lobes` can be lit by aiming at a light: it has a
// lobe that is not delta.
inline bool aims_at_lights(uint32_t lobes) {
    return (lobes & (lobe_diffuse | lobe_glossy)) != 0u;
}

// A material resolved at one surface.
struct Bsdf {
    Float3 normal;   // the normal to shade with, unit, outward: the surface's shading normal
                     // (contracts/surface_interaction.h), as the material perturbs it, if it does
    BsdfKind kind;
    Float3 color;    // lambert: albedo; conductor: f0; dielectric: unused, 1
    float alpha;     // conductor: GGX alpha = roughness^2; otherwise 0
    float ior;       // dielectric: index of refraction inside, against 1 outside; otherwise 0
    uint32_t padding[3];
};

static_assert(sizeof(Bsdf) == 48, "Bsdf must be the same 48 bytes on the host and in shaders");

// What sample() returns.
struct BsdfSample {
    Float3 direction;  // wi, unit
    float pdf;         // per unit solid angle, or a delta lobe's probability; 0 for no sample
    Float3 value;      // f(wo, wi), or for a delta lobe as described above
    uint32_t lobe;     // the lobe chosen: reflection or transmission, and diffuse, glossy or delta
};

static_assert(sizeof(BsdfSample) == 32, "BsdfSample must be the same 32 bytes on the host and in shaders");

}  // namespace contracts
}  // namespace serenity
