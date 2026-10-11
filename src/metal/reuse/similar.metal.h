#pragma once

// Axis: Sample reuse (which surfaces' reservoirs are worth reusing).
//
// The tests the merges ask of a surface before streaming its reservoir:
// last frame's surface where this one reprojects (temporal.metal.h), and
// each neighbor (spatial.metal.h). One definition, so the two cannot drift
// apart (ES.3); RTXDI keeps its geometric test, by the same two measures,
// in one function for both its passes, and asks its material test in both
// (below).
//
// Geometry. A surface `there` is similar to `here` when
//
//   |depth_there - depth_here| <= depth_threshold x depth_here, and
//   dot(normal_here, normal_there) >= normal_threshold,
//
// each depth along its frame's camera's forward axis, as the caller's
// Frame gives it (unbiased.metal.h), the normals the records' shading
// normals, decoded (contracts/bsdf.h, PackedBsdf::normal; metal/math/
// warp.metal.h), and there's kind not none (contracts/surface_record.h).
// The thresholds are the graph's (contracts/reuse_settings.h).
//
// Material, asked by both merges (RTXDI's RAB_AreMaterialsSimilar, which
// its temporal resampling asks always and its spatial resampling by its
// enableMaterialSimilarityTest, on by default, and so always here): the
// two packed BSDFs are of the same
// kind, the luminances of their colors are within material_color_tolerance
// of each other, and their scalars (alpha, or ior) within
// material_scalar_tolerance. Read from contract 2's packed words as that
// contract lays them out, so this family includes no Material shader.
// RTXDI leaves the comparison to its application; these tolerances are
// Serenity's, named here (CDSA.21), each a value the implementation's
// measurements may move. A neighbor of another material lit by the same
// light takes a different share of it, so its reservoir is a worse
// candidate here, and in basic mode its share of the normalization a
// worse guess.
//
// Depends on contracts 2 and 5, the settings and the shared unit-vector map
// alone.
//
// In raytraced mode these tests decide noise, never correctness: a
// reservoir from a surface they let through is weighed by whether that
// surface could have produced each sample (unbiased.metal.h), so a wrong
// surface let through costs variance, and one kept out costs only the reuse
// it would have given. In basic mode they also bound its bias: what a
// surface they let through cannot see is what the normalization weighs
// wrongly. A surface
// seen through glass sits at a depth the camera's straight ray does not
// see it at; the depth test is then wrong more often, and that, too, is
// noise or, in basic mode, the same edge-bounded bias.
//
// Cost: a normal's decoding and a few dozen flops, on records the merge
// reads anyway.

#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "core/contracts/surface_record.h"
#include "core/contracts/reuse_settings.h"
#include "metal/math/warp.metal.h"

namespace serenity {
namespace shaders {

// The material test's tolerances (above): absolute differences, of a
// luminance in [0, 1] and of an alpha or an ior.
constant constexpr float material_color_tolerance = 0.5f;
constant constexpr float material_scalar_tolerance = 0.5f;

// A record's kind, read from contract 2's packed word as that contract
// lays it out; none for a pixel with no surface (contracts/
// surface_record.h).
inline serenity::contracts::BsdfKind record_kind(serenity::contracts::SurfaceRecord record) {
    return serenity::contracts::BsdfKind((record.bsdf.escape_kind >> serenity::contracts::packed_kind_shift) &
                                         serenity::contracts::packed_kind_mask);
}

// The geometric test; `normal_here` decoded once by the caller, who tests
// its pixel against several surfaces. No two adjacent parameters of one
// type (I.24), as before.
inline bool similar(float depth_here, float3 normal_here, serenity::contracts::SurfaceRecord there, float depth_there,
                    serenity::contracts::ReuseSettings settings) {
    return record_kind(there) != serenity::contracts::BsdfKind::none &&
           metal::abs(depth_there - depth_here) <= settings.depth_threshold * depth_here &&
           metal::dot(normal_here, decode_unit(there.bsdf.normal)) >= settings.normal_threshold;
}

// A packed Bsdf's color luminance (Rec. 709) and its scalar, from its
// halves, as contract 2 lays them out.
struct MaterialLook {
    float luminance;
    float scalar;
};
inline MaterialLook material_look(serenity::contracts::PackedBsdf packed) {
    const half2 rg = as_type<half2>(packed.color_rg);
    const half2 b_scalar = as_type<half2>(packed.color_b_scalar);
    return MaterialLook{0.2126f * float(rg.x) + 0.7152f * float(rg.y) + 0.0722f * float(b_scalar.x),
                        float(b_scalar.y)};
}

// The material test. Symmetric, so its two same-typed parameters cannot be
// passed in the wrong order.
inline bool similar_material(serenity::contracts::PackedBsdf here, serenity::contracts::PackedBsdf there) {
    const uint kind_bits = serenity::contracts::packed_kind_mask
                                                    << serenity::contracts::packed_kind_shift;
    if ((here.escape_kind & kind_bits) != (there.escape_kind & kind_bits)) {
        return false;
    }
    const MaterialLook a = material_look(here);
    const MaterialLook b = material_look(there);
    return metal::abs(a.luminance - b.luminance) <= material_color_tolerance &&
           metal::abs(a.scalar - b.scalar) <= material_scalar_tolerance;
}

}  // namespace shaders
}  // namespace serenity
