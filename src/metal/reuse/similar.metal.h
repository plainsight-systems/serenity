#pragma once

// Axis: Sample reuse (which surfaces' reservoirs are worth reusing).
//
// The tests the merges ask of a surface before streaming its reservoir:
// last frame's surface where this one reprojects (temporal.metal.h), and
// each neighbor (spatial.metal.h). One definition, so the two cannot drift
// apart (ES.3); RTXDI keeps its geometric test, by the same two measures,
// in one function for both its passes, and its material test for the
// spatial pass alone, as here.
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
// Material, the spatial merge's alone (RTXDI's material similarity test,
// on by default, and so always here): the two packed BSDFs are of the same
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

// The geometric test; `normal_here` decoded once by the caller, who tests
// its pixel against several surfaces. No two adjacent parameters of one
// type (I.24), as before.
inline bool similar(float depth_here, float3 normal_here, serenity::contracts::SurfaceRecord there, float depth_there,
                    serenity::contracts::ReuseSettings settings);

// The material test. Symmetric, so its two same-typed parameters cannot be
// passed in the wrong order.
inline bool similar_material(serenity::contracts::PackedBsdf here, serenity::contracts::PackedBsdf there);

}  // namespace shaders
}  // namespace serenity
