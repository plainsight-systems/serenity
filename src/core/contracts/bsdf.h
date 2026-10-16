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
//
//                                   u.x chooses by cutting [0, 1) into one
//                                   interval per lobe, each as long as that
//                                   lobe's probability of being chosen, in
//                                   an order fixed for the kind. So a
//                                   surface of two lobes yields one at u.x
//                                   = 0 and the other just below 1, each
//                                   where its probability is not 0: an
//                                   estimator that takes every lobe of such
//                                   a surface (the preview) samples both
//                                   ends and tells the two apart by their
//                                   lobe bits, never by the kind. Stated
//                                   here so no estimator relies on a
//                                   kind's sampling order that its
//                                   interface does not promise (I.1).
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
// The packed form. A Bsdf as a G-buffer stores one, 16 bytes against its
// 48, for ReSTIR DI's surface record (contract 5, core/contracts/
// surface_record.h), which its merges read at every neighbor and so is most
// of their traffic. Material owns it, here, beside the Bsdf it packs: one
// definition of what a resolved material is, in two forms, so a field
// Material adds is packed where it is added (ES.3). Two functions, each
// backend's (metal/materials/bsdf.metal.h), to these semantics:
//
//   PackedBsdf pack_bsdf(Bsdf)     the kind as it is; the shading normal
//                                  by the equal-area octahedral map in two
//                                  16-bit numbers (contracts/
//                                  light_sample.h states the map); color,
//                                  and each scalar the kind reads, to the
//                                  nearest IEEE half (binary16). A
//                                  positive value that rounds to 0 is
//                                  stored as half's least, 2^-24, so a lobe
//                                  never loses its support: a conductor's
//                                  alpha never becomes a mirror's 0. The
//                                  scalar word holds the one of alpha (a
//                                  conductor's) and ior (a coated or
//                                  dielectric surface's) the kind reads; a
//                                  lambert's is 0.
//                                  Precondition: each such value v is
//                                  finite with |v| <= 65504, half's largest
//                                  (packed_largest, below).
//   Bsdf unpack_bsdf(PackedBsdf)   the Bsdf the packed values stand for:
//                                  the kind; the decoded normal, unit
//                                  length; the halves widened; every field
//                                  the kind does not read 0, padding 0.
//
// What packing loses, stated (ES.46: deliberate narrowings). For every
// value v the precondition admits, its unpacked h is within:
//
//   |h - v| <= 2^-11 |v|   where |v| >= 2^-14, half's least normal: a
//                          half keeps 11 significant bits, so rounding to
//                          nearest loses at most half a step, a relative
//                          2^-11 (5 x 10^-4);
//   |h - v| <= 2^-24       where |v| < 2^-14: the subnormal halves are
//                          2^-24 apart, so rounding loses at most 2^-25,
//                          and a positive value raised to 2^-24 from below
//                          2^-25 moves less than 2^-24. Absolute, not
//                          relative: 10^-9 becomes 2^-24, 60 times itself,
//                          and is still within 6 x 10^-8 of it.
//
// The bounds (packed_relative, packed_absolute, packed_least_normal) are
// named below, one definition for the shaders' packing and the tests that
// hold it. Every albedo, f0 and escape a material resolves to is in
// [0, 1.12] (a texture's shade at most: core/textures/wood.h), and every
// alpha in [10^-3, 1] (metal/materials/resolve.metal.h, least_alpha), so
// those are within the precondition by the scene's own checks; an ior is
// any finite number above 1, and a rough material's constant color, or a
// checker's, any finite number. A renderer that packs (ReSTIR DI's) refuses,
// by MetalError naming the material, a scene with a value a material can
// resolve to outside the precondition (core/scene/packed_range.h; the path
// graph, which packs nothing, takes it): a value packed past 65504 would be
// infinity, a different surface from the one the reference shades, and
// nothing about it would be within a tolerance.
//
// The normal is within 10^-4 radians (contracts/light_sample.h,
// unit_round_trip_error; 6.2 x 10^-5 measured over the sphere). Every
// question asked of a surface stored packed is asked of its unpacked Bsdf,
// by every pass, the target's and the shade's alike, so each pixel's
// surface is one surface: what packing changes is which surface that is,
// by those amounts, against the full-precision one the reference shades. A
// small bias in the image, as a G-buffer's is in production; it is part of
// what the raytraced mode's check against the reference holds within its
// tolerance (core/contracts/reuse_settings.h, bias_correction). The naive
// path tracer never packs a Bsdf, and shades what it did.
//
// Layout rules as for every shared contract (contracts/frame_constants.h).

#include "core/contracts/shared_layout.h"
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
    coated = 4,      // a smooth clear coat's mirror reflection over a Lambert base (materials/coated.h)
};

// Bits of lobes() and BsdfSample::lobe, as pbrt-v4's BxDFFlags: each lobe
// is one of reflection or transmission, which side wi is on, and one of
// diffuse, glossy or delta, how it spreads. lobes() is the union over a
// surface's lobes; a sample's lobe has one bit of each pair.
//
//   diffuse  spreads over the hemisphere (Lambert);
//   glossy   spreads about a direction (rough metal);
//   delta    one direction only: evaluate() and pdf() are 0 for it.
SERENITY_CONSTANT uint32_t lobe_reflection = 1u;    // wi on wo's side
SERENITY_CONSTANT uint32_t lobe_transmission = 2u;  // wi on the other side
SERENITY_CONSTANT uint32_t lobe_diffuse = 4u;
SERENITY_CONSTANT uint32_t lobe_glossy = 8u;
SERENITY_CONSTANT uint32_t lobe_delta = 16u;

// Whether a surface with `lobes` can be lit by aiming at a light: it has a
// lobe that is not delta.
constexpr bool aims_at_lights(uint32_t lobes) {
    return (lobes & (lobe_diffuse | lobe_glossy)) != 0u;
}

// A material resolved at one surface.
struct Bsdf {
    Float3 normal;   // the normal to shade with, unit, outward: the surface's shading normal
                     // (contracts/surface_interaction.h), as the material perturbs it, if it does
    BsdfKind kind;
    Float3 color;    // lambert, coated: albedo; conductor: f0; dielectric: unused, 1
    float alpha;     // conductor: GGX alpha = roughness^2; otherwise 0
    float ior;       // dielectric: index of refraction inside; coated: the coat's; against 1 outside;
                     // otherwise 0
    float escape;    // coated: 1 - F_in, what the coat lets out of the base's light (materials/coated.h);
                     // otherwise 0
    uint32_t padding[2];
};

static_assert(sizeof(Bsdf) == 48, "Bsdf must be the same 48 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<Bsdf>, "Bsdf is written to the GPU as bytes");
#endif

// PackedBsdf's fields (below), named once (ES.45): a half in a word's low
// 16 bits or, shifted, its high 16; the kind's 8 bits, shifted.
SERENITY_CONSTANT uint32_t packed_half_mask = 0xffffu;
SERENITY_CONSTANT uint32_t packed_high_shift = 16u;
SERENITY_CONSTANT uint32_t packed_kind_shift = 16u;
SERENITY_CONSTANT uint32_t packed_kind_mask = 0xffu;

// The packed form's range and tolerance (above), named once (ES.45).
SERENITY_CONSTANT float packed_largest = 65504.0f;            // half's largest finite value
SERENITY_CONSTANT float packed_least_normal = 6.103515625e-5f;  // 2^-14, half's least normal value
SERENITY_CONSTANT float packed_relative = 4.8828125e-4f;      // 2^-11, at and above packed_least_normal
SERENITY_CONSTANT float packed_absolute = 5.9604644775390625e-8f;  // 2^-24, below packed_least_normal

// A Bsdf packed (above): four words, read whole.
struct PackedBsdf {
    uint32_t normal;           // equal-area octahedral, x in bits 0-15, y in bits 16-31
    uint32_t color_rg;         // halves: red in bits 0-15, green in bits 16-31
    uint32_t color_b_scalar;   // halves: blue in bits 0-15; alpha or ior, as the kind reads, in bits 16-31
    uint32_t escape_kind;      // half: escape (coated; else 0) in bits 0-15; BsdfKind in bits 16-23; 0 above
};

static_assert(sizeof(PackedBsdf) == 16, "PackedBsdf must be the same 16 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<PackedBsdf>, "PackedBsdf is written to the GPU as bytes");
#endif

// The kind alone, read without unpacking the rest: whether a stored surface
// is one at all (kind none: contracts/surface_record.h). Here, with the
// layout, so the Integrator and Sample reuse, which meet at contracts
// alone, read it one way.
constexpr BsdfKind packed_kind(PackedBsdf packed) {
    return static_cast<BsdfKind>((packed.escape_kind >> packed_kind_shift) & packed_kind_mask);
}

// What sample() returns.
struct BsdfSample {
    Float3 direction;  // wi, unit
    float pdf;         // per unit solid angle, or a delta lobe's probability; 0 for no sample
    Float3 value;      // f(wo, wi), or for a delta lobe as described above
    uint32_t lobe;     // the lobe chosen: reflection or transmission, and diffuse, glossy or delta
};

static_assert(sizeof(BsdfSample) == 32, "BsdfSample must be the same 32 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<BsdfSample>, "BsdfSample is written to the GPU as bytes");
#endif

}  // namespace contracts
}  // namespace serenity
