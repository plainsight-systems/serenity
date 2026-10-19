#pragma once

// Axis: Acceleration (shader half of acceleration/scene_acceleration.h).
//
// The nearest shape a ray reaches, and whether any shape blocks a shadow
// ray, among the shapes the ray's mask finds (contract 19,
// core/contracts/ray_mask.h). Metal's hardware walks a structure of
// bounding boxes and hands back each box the ray enters as a candidate;
// the shape's own exact test (shapes/shapes.metal.h) decides whether, and
// where, the ray hits what is inside, and a hit nearer than the nearest so
// far is committed. Kind-blind: it names no shape kind.
//
// One level (scene_acceleration.h): a candidate is a box of the structure
// the mask names, and the shape it holds is the box's primitive index for
// everything (primitive i is shape i, shapes/primitive.h) and the
// occluders' table entry for the occluders (structure_block.h). The exact
// test carries the world's ray into the shape's object space itself
// (shapes/shapes.metal.h), and the t it finds is the world's
// (contracts/transform.h), so it is committed and compared as it is. What a
// hit reports is the shape and t; where on it, in the world, is
// surface_interaction()'s.
//
// The mask is an argument, not a bound value: every site passes a constant
// (I.4: a RayMask, no integer), so the compiler picks the structure and
// drops the table where everything is traced, and the threads of a SIMD
// group at one site all trace the same structure (GPU.4). For a query of
// the caller's own (manifold next event estimation's seed visits every
// shape on a segment), masked() gives the structure a mask traverses and
// how its candidates name shapes, so no caller reads the block or the
// table itself. A two-level structure (Step 3) makes masked() hand back the
// instance structure and the mask bits instead, and candidates name their
// shape by their instance; no caller changes.
//
// A ray is Metal's own bundle, metal::raytracing::ray: origin, unit
// direction, and the open interval (min_distance, max_distance) its hits lie
// in (I.23), which the query takes as it is.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/ray_mask.h"
#include "metal/acceleration/structure_block.h"
#include "metal/shapes/crossing.metal.h"
#include "metal/shapes/shapes.metal.h"

namespace serenity {
namespace shaders {

// The far end of a ray that runs until it meets something: the largest
// finite float. Not INFINITY: every shader compiles with Metal's default
// fast math, under which an infinite operand makes a comparison's result
// undefined (no-infs-fp-math), and the shapes' exact tests compare their t
// against this bound.
constant constexpr float unbounded = metal::numeric_limits<float>::max();

// The structures a frame traces, as a pass binds them (structure_block.h).
using SceneStructure = constant serenity::gpu::StructureBlock*;

// The intersection query a masked traversal runs: one level, so no
// instancing tag.
using MaskedQuery = metal::raytracing::intersection_query<>;

// What a query carrying one mask traverses: the structure, and how a
// candidate names its shape, by its primitive index; `table` is null for
// everything, whose primitive i is shape i. `empty`: the mask finds no
// shape in this scene (occluders, in a scene of lights alone), and no
// query is to be made. A caller's own query starts by reset() and names
// each candidate by candidate_shape(), and so reads neither the structure
// nor the table itself.
struct MaskedStructure {
  metal::raytracing::primitive_acceleration_structure structure;
  constant uint* table;
  bool empty;

  uint shape(uint primitive) const {
    return table == nullptr ? primitive : table[primitive];
  }

  void reset(thread MaskedQuery& query, metal::raytracing::ray r,
             metal::raytracing::intersection_params params) const {
    query.reset(r, structure, params);
  }
  uint candidate_shape(thread MaskedQuery& query) const {
    return shape(query.get_candidate_primitive_id());
  }
  uint committed_shape(thread MaskedQuery& query) const {
    return shape(query.get_committed_primitive_id());
  }
};

inline MaskedStructure masked(SceneStructure structures,
                              serenity::contracts::RayMask mask) {
  if (mask == serenity::contracts::RayMask::everything) {
    return MaskedStructure{structures->everything, nullptr, false};
  }
  return MaskedStructure{structures->occluders, structures->occluder_shapes,
                         structures->occluder_count == 0u};
}

struct Hit {
  bool found;
  float t;  // the ray's parameter: its distance, the direction being unit
  uint primitive;  // the shape, as primitive i is shape i (shapes/primitive.h)
};

// The nearest shape `mask` finds along `r`, its direction unit length.
inline Hit trace(SceneStructure structures, Shapes shapes,
                 metal::raytracing::ray r, serenity::contracts::RayMask mask) {
  using namespace metal::raytracing;
  const MaskedStructure traversed = masked(structures, mask);
  if (traversed.empty) {
    return Hit{false, 0.0f, 0u};
  }
  intersection_params params;
  params.assume_geometry_type(geometry_type::bounding_box);
  params.accept_any_intersection(false);

  MaskedQuery query;
  traversed.reset(query, r, params);
  ray nearer = r;  // hits nearer than the nearest so far
  while (query.next()) {
    if (query.get_candidate_intersection_type() !=
        intersection_type::bounding_box) {
      continue;
    }
    const Crossing crossing =
        intersect_shape(shapes, traversed.candidate_shape(query), nearer);
    if (crossing.found) {
      query.commit_bounding_box_intersection(crossing.t);
      nearer.max_distance = crossing.t;
    }
  }

  const bool found = query.get_committed_intersection_type() ==
                     intersection_type::bounding_box;
  return Hit{found, found ? query.get_committed_distance() : 0.0f,
             found ? traversed.committed_shape(query) : 0u};
}

// Whether any shape `mask` finds but `ignore` lies along `r`: a shadow
// ray's question. It stops at the first such shape, whichever it is,
// rather than searching for the nearest. A shadow ray toward a light
// carries the occluders, which hold no light, so its light's shape is not
// among the candidates at all; `ignore` stays for a ray that carries
// everything, and costs a comparison.
inline bool occluded(SceneStructure structures, Shapes shapes,
                     metal::raytracing::ray r,
                     serenity::contracts::RayMask mask, uint ignore) {
  using namespace metal::raytracing;
  const MaskedStructure traversed = masked(structures, mask);
  if (traversed.empty) {
    return false;
  }
  intersection_params params;
  params.assume_geometry_type(geometry_type::bounding_box);
  params.accept_any_intersection(true);

  MaskedQuery query;
  traversed.reset(query, r, params);
  while (query.next()) {
    if (query.get_candidate_intersection_type() !=
        intersection_type::bounding_box) {
      continue;
    }
    const uint shape = traversed.candidate_shape(query);
    if (shape != ignore && intersect_shape(shapes, shape, r).found) {
      query.abort();
      return true;
    }
  }
  return false;
}

}  // namespace shaders
}  // namespace serenity
