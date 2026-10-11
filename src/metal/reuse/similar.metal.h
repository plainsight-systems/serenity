#pragma once

// Axis: Sample reuse (which surfaces' reservoirs are worth reusing).
//
// The test both merges ask of a surface before streaming its reservoir:
// last frame's surface where this one reprojects (temporal.metal.h), and
// each neighbor (spatial.metal.h). One definition, so the two cannot drift
// apart (ES.3); RTXDI keeps the same test, by the same two measures, in one
// function for both its passes.
//
// A surface `there` is similar to `here` when
//
//   |depth_there - depth_here| <= depth_threshold x depth_here, and
//   dot(normal_here, normal_there) >= normal_threshold,
//
// each depth along its frame's camera's forward axis, as the caller's
// Frame gives it (unbiased.metal.h), the normals the records' shading
// normals (bsdf.normal), and there's bsdf not of kind none
// (contracts/surface_record.h). The thresholds are the graph's
// (core/passes/restir_di.h).
//
// Depends on contract 5 and the settings alone.
//
// The test decides noise, never correctness: a reservoir from a surface it
// lets through is weighed by whether that surface could have produced each
// sample (unbiased.metal.h), so a wrong surface let through costs variance,
// and one kept out costs only the reuse it would have given. A surface seen
// through glass sits at a depth the camera's straight ray does not see it
// at; the test is then wrong more often, and that, too, is noise.
//
// Cost: a dozen flops, on records the merge reads anyway.

#include <metal_stdlib>

#include "core/contracts/surface_record.h"
#include "core/passes/restir_di.h"

namespace serenity {
namespace shaders {

inline bool similar(float depth_here, float3 normal_here, serenity::contracts::SurfaceRecord there,
                    float depth_there, serenity::passes::RestirDi settings);

}  // namespace shaders
}  // namespace serenity
