#pragma once

// Axis: Integrator (manifold next event estimation).
//
// Caustics through glass, found the way next event estimation finds direct
// light: manifold next event estimation (MNEE; Hanika, Droske and Fascione
// 2015) as Blender Cycles ships it (its "shadow caustics", intern/cycles/
// kernel/integrator/mnee.h, 2022), its choices taken as they are but where
// stated below. At a rough surface, the receiver, a light point is drawn as
// next event estimation draws it; where the straight segment to it crosses
// smooth refractive casters, the path through them that obeys Snell's law
// at every crossing is found by Newton's method on the specular constraints,
// and its light is counted, its generalized geometry term in place of the
// straight segment's. The path tracer's caustic paths, light found by a
// BSDF-sampled ray after the same refractions, are then not counted again
// (covered(), below), so each caustic is counted once, by MNEE.
//
// What it covers (Cycles' domain): light reaching a receiver through 1 to
// max_interfaces refractions at casters, no reflection among them, seeded by
// the straight segment, so inside the casters' shadow of the light point.
// Here: caustics on the table under clear and tinted glass (2 crossings a
// marble), the tint's absorption along the chain, the cat's-eye cores lit
// through their shell (1 crossing). Not: light reflected off metal onto a
// receiver (no refraction), and anything seen directly through glass or in
// a mirror (no receiver: that is the pixel's sampling, not its lighting).
//
//   receiver  a surface with a diffuse lobe (contract 2: lambert, and the
//             coated base), so the chain starts at a diffuse bounce. Cycles
//             flags receivers per object; derived here from the material, as
//             the scene format has no flag. Not the conductors: their
//             near-mirror lobe (alpha 10^-3) lit by a light sample alone,
//             with no BSDF strategy to weigh against, is the 1 / alpha^2
//             variance path.cuh's MIS exists to avoid.
//   caster    a dielectric sphere (contract 2's smooth glass): smooth normals
//             by construction, as Cycles requires of a caster. A dielectric
//             box is refused (no smooth normal across its edges): a seed
//             through one gives no light, and a chain through one is counted
//             by the path tracer, as before.
//
// The algorithm, for a receiver at `here`, a light point y drawn from the
// point rays leave toward wo (path.cuh, "One origin"), x0:
//
//   1. Seed (seed_segment()): one ray query along x0 -> y, every shape on it
//      visited, none committed: each caster sphere's crossings within the
//      segment, sorted, at most max_interfaces (more refuses the seed);
//      any other shape on it marks the segment blocked. No crossing and not
//      blocked: the segment is clear, the caller's ordinary next event
//      estimation. Other shapes do not stop the seed (Cycles: the solved
//      path may pass where the straight one is blocked); its visibility is
//      tested once solved (step 4). The seed, as the solved chain's
//      segments (step 4), carries the occluders (contract 19): "every
//      shape" is every surface, and no firefly blocks it, as none blocks
//      the shadow ray it stands in for.
//   2. Solve (newton()): Cycles' walk, the constraint at crossing i the
//      generalized half vector H = -(w_i + eta w_o) projected on the
//      surface's tangent plane (eta the index past the crossing over the
//      index before it, seen from the receiver), its derivatives in the
//      sphere's own differential geometry (dn = dp / r), the block
//      tridiagonal system solved for the step, the step started at 1/10
//      and doubled toward 1 after each success, halved where a step leaves
//      the sphere or turns a crossing into a reflection, until every
//      constraint is under 10^-3, or 64 iterations, or a step under 10^-4.
//      Cycles projects a step back onto its object by tracing the scene;
//      a sphere is projected exactly, by its own roots, no ray traced.
//   3. Contribution (manifold_light()): what step 5 of the path tracer adds
//      for a light sample, f |cos| L_e T / (P p_l), with the segment's
//      geometry replaced by the chain's:
//
//        X = f(wo, w0) |cos w0| L_e T prod_i [(1 - F_i) / eta_t,i^2]
//            G(x0 <-> y) d^2 / cos_y
//
//      w0 toward the first crossing; F_i Fresnel at crossing i; eta_t,i^2
//      the radiance's scaling across it, as the dielectric's BSDF scales it
//      (materials/dielectric.cuh), so a core inside glass sees 2.25 times the
//      light outside, as the path tracer does; T the media's transmittance
//      over every segment (contract 12, the chain's medium changing at each
//      crossing as step 6 changes it); G = |cos at x1| / |x1 - x0|^2 x
//      |det dx1/dy|, the generalized geometry term, Cycles' transfer
//      matrix's determinant; d^2 / cos_y converts y's density from the
//      solid angle at x0, where the light sample drew it, to y's area,
//      where G measures it. X is per solid angle at x0, so the caller
//      divides by the density its sample had there, P p_l in the path
//      tracer, or multiplies by ReSTIR PT's W_RIS.
//   4. Visibility: each segment of the solved chain traced, its nearest
//      shape the next crossing's sphere within 10^-3 of its radius of where
//      it should be (Cycles' test, its absolute 1 mm made relative); the
//      last, from the last crossing to y, a shadow ray.
//
// Departures from Cycles, each for this scene's scale or shapes:
//
//   - Cycles clamps G at 2 (sr per m^2, in a scene of meter-sized objects:
//     "it has a tendency to be unstable"). The marbles' G is 10 to 10^4 (a
//     light centimeters off), and the clamp would remove nearly all of every
//     caustic. Not clamped.
//   - Cycles' distances (1 mm least segment, 0.1 mm least progress, 1 mm
//     visibility tolerance) are absolute; here each is that fraction of the
//     caster's radius (a marble is 1.4 to 2.5 cm, Cycles' objects meters).
//   - At most 4 crossings (Cycles allows 6 casters' worth): two marbles in
//     a row; more refuses the seed.
//   - Projection by the sphere's roots, not a traced ray (above).
//   - Cycles gives up when any one crossing moves less than its least
//     progress in a step; here, when none does. A chain's crossings
//     converge at different rates, and the per-crossing test gives up on
//     some a step before convergence (on a grid of 65 floor points under a
//     glass sphere that have a refraction path, 15 converged with it, 18
//     with this; measured while this was built).
//   - The tangent frame: Cycles builds its shading frame from a triangle's
//     edge, fixed across the triangle, and its derivatives assume so; a
//     sphere has no triangle, so each crossing's frame is built from the
//     seed's tangent, fixed for the solve (vertex_at()).
//   - What is not counted again (covered()): Cycles drops a light struck by
//     a ray leaving any caster after a receiver anywhere before; this drops
//     exactly MNEE's domain, a light struck after a receiver's diffuse
//     bounce and 1 to max_interfaces refractions at caster spheres. A light
//     reflected off glass onto a receiver, which MNEE does not find, is kept.
//
// What is biased, stated (as in Cycles; measured in tests/gpu/mnee_test.cpp
// and docs/research/2026-10-12-mnee.md): one solution per light point,
// where a lens gives more than one near its caustic's fold; a solve that
// fails to converge gives none, and from the straight seed it fails over
// most of a ball's caustic's rim, where the path bends far from the seed
// (6.5% of the light in a glass sphere's shadow on a floor, 13% at its
// middle, a quarter of a percent of the whole image); light the casters
// bend outside the straight segment's shadow (an edge ray deviated
// sideways) is neither found by MNEE nor counted by the path tracer; a
// light point drawn on the cap visible from x0 that the chain's last
// segment sees from behind gives none.
//
// Where each step runs (decided 2026-10-12, by measurement; contract 16,
// core/contracts/mnee_request.h): step 1 in the estimator's loop, where the
// light sample is drawn; steps 2 to 4 in the MNEE pass, from a request the
// loop writes (request(), enqueue(); the pass's solve_request(): metal/
// passes/mnee/mnee.h), as Cycles runs its walk in a kernel of its own.
// Solved in the loop, a tenth of the paths solving held up nearly every
// SIMD group: the path pass took 124 ms at 3456 x 2234 against 30 without
// MNEE; with the list, 59 (docs/research/2026-10-12-mnee.md).
//
// Cost: step 1 is one ray query in place of the shadow ray, visiting the
// segment to its end rather than stopping at the first blocker; a request
// is 128 bytes written and read. Per request, per iteration: max_interfaces
// blocks of a few dozen multiply-adds and three normalizations each, a 2 x
// 2 inverse and one sphere's roots a crossing; at most 64 iterations, about
// six from a straight seed that converges; then n + 1 rays for visibility.
// Every loop of the solve over crossings is unrolled to max_interfaces, so
// its crossings' arrays are registers (GPU.3; a 4% gain measured in the
// loop). Not seed_segment()'s insertion, in the estimators' loops: it
// indexes its Seed at run time, and whether the compiler keeps that Seed
// in registers is not measured (docs/research/
// 2026-10-16-codebase-guidelines-pass.md, area C's P3). Pure
// functions of their inputs (F.8) but enqueue(); results returned as
// structs (F.21); every constant named (ES.45).

#include "core/contracts/bsdf.h"
#include "core/contracts/emitter.h"
#include "core/contracts/mnee_request.h"
#include "core/contracts/surface_interaction.h"
#include "core/materials/material.h"
#include "core/shapes/primitive.h"
#include "cuda/acceleration/trace.cuh"
#include "cuda/device/append.cuh"
#include "cuda/integrator/surface.cuh"
#include "cuda/materials/bsdf.cuh"
#include "cuda/materials/dielectric.cuh"
#include "cuda/materials/resolve.cuh"
#include "cuda/math/ray.cuh"
#include "cuda/math/transform.cuh"
#include "cuda/math/vector.cuh"
#include "cuda/math/warp.cuh"
#include "cuda/media/media.cuh"
#include "cuda/shapes/shapes.cuh"

namespace serenity {
namespace shaders {
namespace mnee {

// Every loop of the solve over a chain's crossings runs max_interfaces
// times at most, unrolled, so each array a crossing indexes is held in
// registers, not thread memory (GPU.3); seed_segment()'s insertion is the
// one loop over them that is not (above).

// Cycles' solver's constants (mnee.h), named once (ES.45); the distances
// relative to the caster's radius (above).
// 4, contract 16's
inline constexpr uint max_interfaces = serenity::contracts::mnee_most_crossings;
inline constexpr uint max_iterations = 64u;
inline constexpr float solver_threshold = 1.0e-3f;
inline constexpr float least_step = 1.0e-4f;
inline constexpr float first_step = 0.1f;
inline constexpr float least_determinant = 1.0e-4f;
inline constexpr float projection_reach = 2.0f;
inline constexpr float least_segment = 1.0e-3f;   // times the caster's radius
inline constexpr float least_progress = 1.0e-4f;  // times the caster's radius
inline constexpr float hit_tolerance = 1.0e-3f;   // times the caster's radius
// a root nearer than this times the radius is the start
inline constexpr float same_sphere = 1.0e-4f;

// Whether a surface of `bsdf` receives caustics (above): it has a diffuse
// lobe.
__device__ inline bool receives(serenity::contracts::Bsdf bsdf) {
  return (bsdf_lobes(bsdf) & serenity::contracts::lobe_diffuse) != 0u;
}

// A caster as the solve reads it at one crossing: where its sphere is, its
// index, the medium inside it, its shape, and the crossing's reference
// tangent (vertex_at()), set from the seed and fixed for the solve.
struct Caster {
  float3 center;
  float radius;
  float ior;
  uint interior;
  uint shape;
  float3 reference;
};

// Shape `shape` as a caster, if it is one: a dielectric sphere.
struct CasterAt {
  bool is_caster;   // a dielectric sphere
  bool dielectric;  // a dielectric of any shape
  Caster caster;    // read only where is_caster
};

__device__ inline CasterAt caster_at(Shapes shapes, Materials materials,
                                     uint shape) {
  const serenity::shapes::ShapeRecord record = shapes.records[shape];
  const serenity::materials::MaterialRecord material =
      materials.records[record.material];
  if (material.kind != serenity::materials::MaterialKind::dielectric) {
    return CasterAt{false, false, Caster{}};
  }
  if (record.kind != serenity::shapes::ShapeKind::sphere) {
    return CasterAt{false, true, Caster{}};
  }
  const serenity::contracts::Transform placed = shapes.transforms[shape];
  return CasterAt{true, true,
                  Caster{transform_translation(placed), transform_scale(placed),
                         materials.dielectrics[material.index].ior,
                         record.interior, shape, make_float3(0.0f)}};
}

// The chain a path is on since its last diffuse bounce, for covered()
// (above): 0 off MNEE's domain; 1 just after a receiver's diffuse bounce;
// 1 + n after n refractions at caster spheres since. Carried by the
// estimators' loops from surface to surface, one word.
inline constexpr uint off_chain = 0u;

// The chain after the bounce that drew `lobe` at shape `primitive`. The
// shape's record is read only on a chain, at a refraction, the one case its
// kind decides: a replayed shift's first surface is the destination's primary
// hit, whose primitive is no_primitive (unpack_primary_hit(), shift.cuh),
// always off the chain, and must not be used as an index.
__device__ inline uint chain_after(uint chain, uint lobe,
                                   const serenity::shapes::ShapeRecord* records,
                                   uint primitive) {
  if ((lobe & serenity::contracts::lobe_diffuse) != 0u) {
    return 1u;
  }
  const uint refraction =
      serenity::contracts::lobe_transmission | serenity::contracts::lobe_delta;
  if (chain == off_chain || (lobe & refraction) != refraction ||
      chain > max_interfaces) {
    return off_chain;
  }
  return records[primitive].kind == serenity::shapes::ShapeKind::sphere
             ? chain + 1u
             : off_chain;
}

// Whether a light a BSDF-sampled ray strikes at the end of `chain` is MNEE's
// to count, and so not the path's: after a receiver's diffuse bounce and 1
// to max_interfaces refractions at caster spheres.
__device__ inline bool covered(uint chain) { return chain >= 2u; }

// Both roots of |o + t d - c|^2 = r^2 for a caster's sphere and a ray from
// `origin` along unit `direction`, nearer first, in the stable form of
// shapes/sphere.cuh; found false where the line misses the sphere.
struct Roots {
  bool found;
  float near;
  float far;
};

__device__ inline Roots sphere_roots(float3 origin, float3 direction,
                                     Caster caster) {
  const float3 to_origin = origin - caster.center;
  const float b = dot(to_origin, direction);
  const float3 nearest = to_origin - b * direction;
  const float discriminant =
      caster.radius * caster.radius - dot(nearest, nearest);
  if (!(discriminant > 0.0f)) {
    return Roots{false, 0.0f, 0.0f};
  }
  const float q = -(b + copysign(sqrt(discriminant), b));
  const float c = dot(to_origin, to_origin) - caster.radius * caster.radius;
  return Roots{true, min(c / q, q), max(c / q, q)};
}

// Step 1's result (above).
struct Seed {
  uint count;    // crossings of caster spheres along the segment, at most
                 // max_interfaces
  bool blocked;  // some other shape on it
  bool usable;   // false: too many crossings, or a dielectric that is no sphere
  float t[max_interfaces];
  uint shape[max_interfaces];
};

// Step 1, on OptiX (docs/research/2026-10-19-cuda-backend.md, "Phase 4's
// design"): Metal walks every box the segment enters with its own query
// loop; here OptiX walks them and runs the seed's intersection program for
// each (integrator/mnee_seed.cuh), which runs the loop's body,
// seed_candidate(), on the seed this function keeps as a local and hands it
// by a pointer in payload words 1 and 2 (the shape to ignore in word 0, as
// a shadow ray's). Where the loop stopped, the program reports a hit, which
// ends the traversal (TERMINATE_ON_FIRST_HIT); every other box it leaves
// unreported, so OptiX goes on to the next. The seed is the same whatever
// order the boxes come in, as Metal's is.
//
// The segment, and what the program reads of the scene to test a box,
// beside the seed it fills.
struct SeedQuery {
  Shapes shapes;
  Materials materials;
  float3 origin;
  float3 direction;
  float distance;
  Seed seed;
};

// The loop's body for the box holding `shape`, `ignore` already skipped:
// Metal's seed_segment()'s, a shape seen before skipped (OptiX may test a
// box more than once; the seed is the same either way). Whether the walk
// stops here.
__device__ inline bool seed_candidate(SeedQuery& query, uint shape) {
  Seed& seed = query.seed;
  for (uint i = 0u; i < seed.count; ++i) {
    if (seed.shape[i] == shape) {
      return false;  // its crossings are in already
    }
  }
  const Ray segment =
      make_ray(query.origin, query.direction, 0.0f, query.distance);
  const CasterAt at = caster_at(query.shapes, query.materials, shape);
  if (!at.is_caster) {
    if (at.dielectric && intersect_shape(query.shapes, shape, segment).found) {
      // glass with no smooth normal: no seed through it (above)
      seed.usable = false;
      seed.blocked = true;
      return true;
    }
    if (!seed.blocked && intersect_shape(query.shapes, shape, segment).found) {
      seed.blocked = true;
    }
    return false;
  }
  const Roots crossings =
      sphere_roots(query.origin, query.direction, at.caster);
  if (!crossings.found) {
    return false;
  }
  for (uint r = 0u; r < 2u; ++r) {
    const float t = r == 0u ? crossings.near : crossings.far;
    if (!(t > 0.0f && t < query.distance)) {
      continue;
    }
    if (seed.count == max_interfaces) {
      seed.usable = false;  // more crossings than the solve takes (above)
      break;
    }
    uint i = seed.count++;
    for (; i > 0u && seed.t[i - 1u] > t; --i) {
      seed.t[i] = seed.t[i - 1u];
      seed.shape[i] = seed.shape[i - 1u];
    }
    seed.t[i] = t;
    seed.shape[i] = shape;
  }
  if (!seed.usable) {
    seed.blocked = true;
    return true;
  }
  return false;
}

// Step 1: the crossings of caster spheres along `direction` (unit) from
// `origin` within `distance`, `ignore` (the light's shape) ignored, among
// the surfaces alone (contract 19, the occluders), as next event
// estimation's shadow ray, in whose place it is.
template <typename SceneT>
__device__ inline Seed seed_segment(SceneT scene, float3 origin,
                                    float3 direction, float distance,
                                    uint ignore) {
  SeedQuery query{scene.shapes(), scene.materials(), origin,
                  direction,      distance,          Seed{}};
  query.seed.usable = true;
  const MaskedStructure traversed =
      masked(scene.structure, serenity::contracts::RayMask::occluders);
  if (traversed.empty) {
    return query.seed;  // no surface: clear
  }
  const auto address = reinterpret_cast<unsigned long long>(&query);
  uint ignored = ignore;
  uint high = uint(address >> 32u);
  uint low = uint(address & 0xffffffffull);
  optixTraverse(
      traversed.handle, origin, direction, 0.0f, distance, 0.0f,
      OptixVisibilityMask(0xffu),
      OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
      serenity::cuda::seed_record, 1u, 0u, ignored, high, low);
  return query.seed;
}

// A crossing's differential geometry (Cycles' ManifoldVertex, a sphere's):
// position, outward normal, an orthonormal tangent frame, the normal's
// derivatives along it, dn = dp / r, and the reference the frame is built
// from.
//
// The frame is Cycles' shading frame, s the reference tangent made normal
// to n, t = n x s, its derivatives Cycles' (ds from dn with the reference
// held fixed, as Cycles holds a triangle's dp_du fixed). The reference is
// the seed's tangent, fixed for the solve (set_references()): a frame
// rebuilt from n alone at each point (an orthonormal basis of n) has
// derivatives that formula does not give, and a Jacobian off by some 10%
// confines Newton's method to seeds near their solution (measured against
// finite differences while this was built).
struct Vertex {
  float3 p;
  float3 n;
  float3 dp_du;
  float3 dp_dv;
  float3 dn_du;
  float3 dn_dv;
  float3 reference;
};

// Each crossing's reference tangent, from the chain `p` as seeded.
__device__ inline void set_references(uint count, const float3* p,
                                      Caster* casters) {
#pragma unroll
  for (uint i = 0u; i < max_interfaces; ++i) {
    if (i >= count) {
      break;
    }
    casters[i].reference = tangents(normalize(p[i] - casters[i].center)).t;
  }
}

__device__ inline Vertex vertex_at(float3 p, Caster caster) {
  const float3 n = normalize(p - caster.center);
  const float3 s = normalize(caster.reference - dot(caster.reference, n) * n);
  const float3 t = cross(n, s);
  return Vertex{
      p, n, s, t, s / caster.radius, t / caster.radius, caster.reference};
}

// One crossing's blocks of the constraint's derivative, with respect to the
// crossing before (a), itself (b) and the one after (c), each a 2 x 2
// matrix in row order, and its constraint.
struct Blocks {
  float4 a;
  float4 b;
  float4 c;
  float2 constraint;
};

__device__ inline float2 mul(float4 m, float2 v) {
  return make_float2(m.x * v.x + m.y * v.y, m.z * v.x + m.w * v.y);
}
__device__ inline float4 mul(float4 a, float4 b) {
  return make_float4(a.x * b.x + a.y * b.z, a.x * b.y + a.y * b.w,
                     a.z * b.x + a.w * b.z, a.z * b.y + a.w * b.w);
}
__device__ inline float determinant(float4 m) { return m.x * m.w - m.y * m.z; }

// m's inverse, and its determinant; 0 where the determinant is under
// least_determinant, the inverse then unread.
struct Inverse {
  float determinant;
  float4 inverse;
};
__device__ inline Inverse invert(float4 m) {
  const float det = determinant(m);
  if (abs(det) < least_determinant) {
    return Inverse{0.0f, make_float4(0.0f)};
  }
  return Inverse{det, make_float4(m.w, -m.y, -m.z, m.x) / det};
}

// The tangent part of `x` against unit `h`, negated: Cycles' dH update.
__device__ inline float3 against(float3 x, float3 h) {
  return -(x - h * dot(x, h));
}

// Cycles' mnee_compute_constraint_derivatives() for the chain `p` of
// `count` crossings from `x0` to `y`; false where a segment is shorter than
// least_segment of its caster's radius.
__device__ inline bool constraint_blocks(uint count, const float3* p,
                                         const Caster* casters, float3 x0,
                                         float3 y, Blocks* blocks) {
#pragma unroll
  for (uint i = 0u; i < max_interfaces; ++i) {
    if (i >= count) {
      break;
    }
    const Vertex v = vertex_at(p[i], casters[i]);
    float3 wi = (i == 0u ? x0 : p[i - 1u]) - v.p;
    float3 wo = (i + 1u == count ? y : p[i + 1u]) - v.p;
    const float li = length(wi);
    const float lo = length(wo);
    const float shortest = least_segment * casters[i].radius;
    if (li < shortest || lo < shortest) {
      return false;
    }
    float ili = 1.0f / li;
    float ilo = 1.0f / lo;
    wi *= ili;
    wo *= ilo;
    const float eta =
        dot(wi, v.n) < 0.0f ? 1.0f / casters[i].ior : casters[i].ior;
    float3 h = -(wi + eta * wo);
    const float ilh = 1.0f / length(h);
    h *= ilh;
    ilo *= eta * ilh;
    ili *= ilh;

    // The local shading frame (s, t) and its derivatives.
    const float dpu_n = dot(v.reference, v.n);
    float3 s = v.reference - dpu_n * v.n;
    const float inv_len_s = 1.0f / length(s);
    s *= inv_len_s;
    const float3 t = cross(v.n, s);

    if (i > 0u) {
      const Vertex before = vertex_at(p[i - 1u], casters[i - 1u]);
      const float3 du =
          against((before.dp_du - wi * dot(wi, before.dp_du)) * ili, h);
      const float3 dv =
          against((before.dp_dv - wi * dot(wi, before.dp_dv)) * ili, h);
      blocks[i].a = make_float4(dot(du, s), dot(dv, s), dot(du, t), dot(dv, t));
    } else {
      blocks[i].a = make_float4(0.0f);
    }

    const float3 du =
        against(-v.dp_du * (ili + ilo) + wi * (dot(wi, v.dp_du) * ili) +
                    wo * (dot(wo, v.dp_du) * ilo),
                h);
    const float3 dv =
        against(-v.dp_dv * (ili + ilo) + wi * (dot(wi, v.dp_dv) * ili) +
                    wo * (dot(wo, v.dp_dv) * ilo),
                h);
    float3 ds_du =
        -inv_len_s * (dot(v.reference, v.dn_du) * v.n + dpu_n * v.dn_du);
    float3 ds_dv =
        -inv_len_s * (dot(v.reference, v.dn_dv) * v.n + dpu_n * v.dn_dv);
    ds_du -= s * dot(s, ds_du);
    ds_dv -= s * dot(s, ds_dv);
    const float3 dt_du = cross(v.dn_du, s) + cross(v.n, ds_du);
    const float3 dt_dv = cross(v.dn_dv, s) + cross(v.n, ds_dv);
    blocks[i].b =
        make_float4(dot(du, s) + dot(h, ds_du), dot(dv, s) + dot(h, ds_dv),
                    dot(du, t) + dot(h, dt_du), dot(dv, t) + dot(h, dt_dv));

    if (i + 1u < count) {
      const Vertex after = vertex_at(p[i + 1u], casters[i + 1u]);
      const float3 du_next =
          against((after.dp_du - wo * dot(wo, after.dp_du)) * ilo, h);
      const float3 dv_next =
          against((after.dp_dv - wo * dot(wo, after.dp_dv)) * ilo, h);
      blocks[i].c = make_float4(dot(du_next, s), dot(dv_next, s),
                                dot(du_next, t), dot(dv_next, t));
    } else {
      blocks[i].c = make_float4(0.0f);
    }
    blocks[i].constraint = make_float2(dot(s, h), dot(t, h));
  }
  return true;
}

// Cycles' mnee_solve_matrix_h_to_x(): the step dx at every crossing, by the
// block tridiagonal system's LU factorization; false where a block is
// singular.
__device__ inline bool solve_steps(uint count, const Blocks* blocks,
                                   float2* dx) {
  float4 inverses[max_interfaces];
  float2 c[max_interfaces];
  Inverse l = invert(blocks[0].b);
  if (l.determinant == 0.0f) {
    return false;
  }
  inverses[0] = l.inverse;
  c[0] = blocks[0].constraint;
#pragma unroll
  for (uint k = 1u; k < max_interfaces; ++k) {
    if (k >= count) {
      break;
    }
    const float4 a = mul(blocks[k].a, inverses[k - 1u]);
    l = invert(blocks[k].b - mul(a, blocks[k - 1u].c));
    if (l.determinant == 0.0f) {
      return false;
    }
    inverses[k] = l.inverse;
    c[k] = blocks[k].constraint - mul(a, c[k - 1u]);
  }
  dx[count - 1u] = mul(inverses[count - 1u], c[count - 1u]);
#pragma unroll
  for (uint down = 1u; down < max_interfaces; ++down) {
    const uint k = max_interfaces - 1u - down;
    if (k >= count - 1u) {
      continue;
    }
    dx[k] = mul(inverses[k], c[k] - mul(blocks[k].c, dx[k + 1u]));
  }
  return true;
}

// The point where the ray from `from` toward `toward` meets `caster`'s
// sphere within projection_reach times their distance, the start itself
// not counted where it lies on the sphere; found false where it does not.
struct Projected {
  bool found;
  float3 p;
};

__device__ inline Projected project(float3 from, float3 toward, Caster caster) {
  float3 d = toward - from;
  const float reach = length(d) * projection_reach;
  d = normalize(d);
  const Roots roots = sphere_roots(from, d, caster);
  if (!roots.found) {
    return Projected{false, make_float3(0.0f)};
  }
  const float least = same_sphere * caster.radius;
  const float t = roots.near > least ? roots.near : roots.far;
  if (!(t > least && t < reach)) {
    return Projected{false, make_float3(0.0f)};
  }
  // Put back on the sphere, as surface_interaction() puts a hit back.
  return Projected{
      true,
      caster.center + caster.radius * normalize(from + t * d - caster.center)};
}

// Whether every crossing of the chain `p` from `x0` to `y` refracts: its
// two neighbors on opposite sides of its surface.
__device__ inline bool transmits(uint count, const float3* p,
                                 const Caster* casters, float3 x0, float3 y) {
#pragma unroll
  for (uint i = 0u; i < max_interfaces; ++i) {
    if (i >= count) {
      break;
    }
    const float3 n = p[i] - casters[i].center;
    const float3 wi = (i == 0u ? x0 : p[i - 1u]) - p[i];
    const float3 wo = (i + 1u == count ? y : p[i + 1u]) - p[i];
    if (dot(n, wi) * dot(n, wo) >= 0.0f) {
      return false;
    }
  }
  return true;
}

// Step 2, Cycles' mnee_newton_solver(): `p` moved onto the chain that obeys
// Snell's law at each crossing, `blocks` its derivatives there; false where
// none is found. `facing` the receiver's normal on the side x0 lies.
__device__ inline bool newton(uint count, float3* p, const Caster* casters,
                              float3 x0, float3 facing, float3 y,
                              Blocks* blocks) {
  float2 dx[max_interfaces];
  float3 tentative[max_interfaces];
  float beta = first_step;
  bool resolve = true;
  for (uint iteration = 0u; iteration < max_iterations; ++iteration) {
    if (resolve) {
      if (!constraint_blocks(count, p, casters, x0, y, blocks)) {
        return false;
      }
      float norm = 0.0f;
#pragma unroll
      for (uint i = 0u; i < max_interfaces; ++i) {
        if (i >= count) {
          break;
        }
        norm = max(norm, length(blocks[i].constraint));
      }
      if (norm < solver_threshold) {
        return true;
      }
      if (!solve_steps(count, blocks, dx)) {
        return false;
      }
    }
    bool reduce = false;
    bool moving = false;
#pragma unroll
    for (uint i = 0u; i < max_interfaces && !reduce; ++i) {
      if (i >= count) {
        break;
      }
      const Vertex v = vertex_at(p[i], casters[i]);
      const float3 step = beta * (dx[i].x * v.dp_du + dx[i].y * v.dp_dv);
      float3 moved = v.p - step;
      if (i == 0u && dot(facing, moved - x0) <= 0.0f) {
        moved =
            v.p +
            step;  // the first solve below the receiver: the other way (Cycles)
      }
      const Projected on = project(i == 0u ? x0 : p[i - 1u], moved, casters[i]);
      if (!on.found) {
        reduce = true;
        break;
      }
      moving =
          moving || distance(on.p, v.p) >= least_progress * casters[i].radius;
      tentative[i] = on.p;
    }
    if (!reduce && !moving) {
      return false;  // no crossing moves: stuck (Cycles, but for the whole
                     // chain; above)
    }
    if (!reduce && !transmits(count, tentative, casters, x0, y)) {
      reduce = true;
    }
    if (reduce) {
      resolve = false;
      beta *= 0.5f;
      if (beta < least_step) {
        return false;
      }
      continue;
    }
#pragma unroll
    for (uint i = 0u; i < max_interfaces; ++i) {
      if (i >= count) {
        break;
      }
      p[i] = tentative[i];
    }
    resolve = true;
    beta = min(1.0f, 2.0f * beta);
  }
  return false;
}

// Cycles' mnee_compute_transfer_matrix(), for a light point y on a surface
// of unit normal `light_normal`: |det dx1 / dy|, area on the first
// crossing's tangent plane per area on the light; 0 where a block is
// singular.
__device__ inline float transfer(uint count, const float3* p,
                                 const Caster* casters, const Blocks* blocks,
                                 float3 x0, float3 y, float3 light_normal) {
  float4 u[max_interfaces];
  Inverse l = invert(blocks[0].b);
  if (l.determinant == 0.0f) {
    return 0.0f;
  }
#pragma unroll
  for (uint k = 1u; k < max_interfaces; ++k) {
    if (k >= count) {
      break;
    }
    u[k - 1u] = mul(l.inverse, blocks[k - 1u].c);
    l = invert(blocks[k].b - mul(blocks[k].a, u[k - 1u]));
    if (l.determinant == 0.0f) {
      return 0.0f;
    }
  }
  const uint last = count - 1u;
  const Vertex m = vertex_at(p[last], casters[last]);
  const float3 s = normalize(m.dp_du - dot(m.dp_du, m.n) * m.n);
  const float3 t = cross(m.n, s);
  const Tangents light = tangents(light_normal);
  const float3 wi = normalize((count == 1u ? x0 : p[last - 1u]) - m.p);
  const float eta =
      dot(wi, m.n) < 0.0f ? 1.0f / casters[last].ior : casters[last].ior;
  float3 wo = y - m.p;
  float ilo = 1.0f / length(wo);
  wo *= ilo;
  float3 h = -(wi + eta * wo);
  const float ilh = 1.0f / length(h);
  h *= ilh;
  ilo *= eta * ilh;
  const float3 du = against((light.t - wo * dot(wo, light.t)) * ilo, h);
  const float3 dv = against((light.b - wo * dot(wo, light.b)) * ilo, h);
  float4 tp = -mul(l.inverse,
                   make_float4(dot(du, s), dot(dv, s), dot(du, t), dot(dv, t)));
#pragma unroll
  for (uint down = 1u; down < max_interfaces; ++down) {
    const uint k = max_interfaces - 1u - down;
    if (k >= last) {
      continue;
    }
    tp = -mul(u[k], tp);
  }
  return abs(determinant(tp));
}

// The generalized geometry term G(x0 <-> y) of the solved chain `p`, its
// derivatives `blocks`: solid angle at x0 per area at the light, |cos at
// x1| / |x1 - x0|^2 |det dx1 / dy| (step 3).
__device__ inline float geometry_term(uint count, const float3* p,
                                      const Caster* casters,
                                      const Blocks* blocks, float3 x0, float3 y,
                                      float3 light_normal) {
  const float3 to_first = p[0] - x0;
  const float distance2 = dot(to_first, to_first);
  const float cos_first =
      abs(dot(to_first, normalize(p[0] - casters[0].center)));
  return cos_first * rsqrt(distance2) / distance2 *
         transfer(count, p, casters, blocks, x0, y, light_normal);
}

// Whether the nearest surface along the segment from `origin` (a point just
// off the surface it leaves, on the segment's side) to `to` is shape
// `shape`, a caster, within `tolerance` of `to`. Among the surfaces alone
// (contract 19, the occluders): the chain's visibility, as a shadow ray's.
__device__ inline bool reaches(SceneStructure structure, Shapes shapes,
                               float3 origin, float3 to, uint shape,
                               float tolerance) {
  const float length = distance(origin, to);
  const float3 d = (to - origin) / length;
  const Hit hit =
      trace(structure, shapes, make_ray(origin, d, 0.0f, length + tolerance),
            serenity::contracts::RayMask::occluders);
  return hit.found && hit.primitive == shape &&
         abs(hit.t - length) <= tolerance;
}

// The point a segment from `p`, on a caster whose outward normal there is
// unit `n`, toward `to` leaves from (integrator/surface.cuh).
__device__ inline float3 leaving(float3 p, float3 n, float3 to) {
  return offset_ray(p, dot(to - p, n) >= 0.0f ? n : -n);
}

// Step 3's light carried along a solved chain of `count` crossings `p` at
// `casters`, from the receiver's point `origin` in `medium` to the light's
// `y`: each crossing's Fresnel transmittance and radiance scaling, and each
// segment's medium. Not found where a crossing reflects whole.
struct Carried {
  bool found;
  float3 kept;
};

template <typename SceneT>
__device__ __forceinline__ Carried
chain_transmittance(SceneT scene, uint medium, float3 origin, float3 y,
                    uint count, const float3* p, const Caster* casters) {
  float3 kept = make_float3(1.0f);
  float3 from = origin;
#pragma unroll
  for (uint i = 0u; i < max_interfaces; ++i) {
    if (i >= count) {
      break;
    }
    const float3 n = normalize(p[i] - casters[i].center);
    const Boundary boundary =
        dielectric_boundary(casters[i].ior, normalize(p[i] - from), n);
    if (boundary.total_internal) {
      return Carried{false, make_float3(0.0f)};
    }
    const float eta_t = transmitted_eta(casters[i].ior, boundary.entering);
    kept *= transmittance(scene.media(), medium, distance(from, p[i])) *
            (1.0f - boundary.reflectance) / (eta_t * eta_t);
    medium = boundary.entering ? casters[i].interior
                               : serenity::contracts::no_medium;
    from = p[i];
  }
  kept *= transmittance(scene.media(), medium, distance(p[count - 1u], y));
  return Carried{true, kept};
}

// Step 4 along the same chain: each segment traced, nearest first, the
// last to the light's `y`, its shape `light` ignored.
template <typename SceneT>
__device__ __forceinline__ bool chain_visible(SceneT scene, Shapes shapes,
                                              float3 origin, float3 y,
                                              uint light, uint count,
                                              const float3* p,
                                              const Caster* casters) {
  const uint last = count - 1u;
  if (!reaches(scene.structure, shapes, origin, p[0], casters[0].shape,
               hit_tolerance * casters[0].radius)) {
    return false;
  }
#pragma unroll
  for (uint i = 0u; i < max_interfaces; ++i) {
    if (i >= last) {
      break;
    }
    const float3 n = normalize(p[i] - casters[i].center);
    if (!reaches(scene.structure, shapes, leaving(p[i], n, p[i + 1u]),
                 p[i + 1u], casters[i + 1u].shape,
                 hit_tolerance * casters[i + 1u].radius)) {
      return false;
    }
  }
  const float3 leave_from =
      leaving(p[last], normalize(p[last] - casters[last].center), y);
  return !occluded(scene.structure, shapes,
                   make_ray(leave_from, normalize(y - leave_from), 0.0f,
                            distance(leave_from, y)),
                   serenity::contracts::RayMask::occluders, light);
}

// Steps 2 to 4 at `here` for the light sample `sample` drawn from `origin`
// (x0) toward a sphere light centered at `light_center`, whose segment
// `seed` crosses casters: X (above), per solid angle at x0; 0 where no
// chain is found, it is blocked, or the light faces away. Precondition:
// seed.usable, seed.count > 0, sample.pdf > 0.
template <typename SceneT>
__device__ inline float3 manifold_light(SceneT scene, Shading here,
                                        float3 origin, float3 light_center,
                                        serenity::contracts::LightSample sample,
                                        Seed seed) {
  const Shapes shapes = scene.shapes();
  const Materials materials = scene.materials();
  const float3 wi = to_float3(sample.direction);
  const float3 y = origin + sample.distance * wi;
  const float3 light_normal = normalize(y - light_center);
  const float cos_y = -dot(light_normal, wi);
  if (!(cos_y > 0.0f)) {
    return make_float3(0.0f);
  }
  const uint count = seed.count;
  Caster casters[max_interfaces];
  float3 p[max_interfaces];
#pragma unroll
  for (uint i = 0u; i < max_interfaces; ++i) {
    if (i >= count) {
      break;
    }
    casters[i] = caster_at(shapes, materials, seed.shape[i]).caster;
    p[i] = casters[i].center +
           casters[i].radius *
               normalize(origin + seed.t[i] * wi - casters[i].center);
  }
  set_references(count, p, casters);
  const float3 ng = to_float3(here.surface.geometric_normal);
  const float3 facing = dot(ng, here.wo) >= 0.0f ? ng : -ng;
  Blocks blocks[max_interfaces];
  if (!newton(count, p, casters, origin, facing, y, blocks)) {
    return make_float3(0.0f);
  }
  const uint last = count - 1u;
  if (dot(light_normal, p[last] - y) <= 0.0f) {
    return make_float3(
        0.0f);  // the light's point seen from behind by the chain
  }
  const float3 w0 = normalize(p[0] - origin);
  if (!leave_alike(here.surface, w0, here.wo)) {
    return make_float3(0.0f);
  }
  const float3 f = bsdf_evaluate(here.bsdf, here.wo, w0) *
                   abs(dot(w0, to_float3(here.bsdf.normal)));
  if (!any(f > 0.0f)) {
    return make_float3(0.0f);
  }

  const Carried carried =
      chain_transmittance(scene, here.medium, origin, y, count, p, casters);
  if (!carried.found) {
    return make_float3(0.0f);
  }

  const float geometry =
      geometry_term(count, p, casters, blocks, origin, y, light_normal);
  const float3 unshadowed = f * to_float3(sample.radiance) * carried.kept *
                            geometry *
                            (sample.distance * sample.distance / cos_y);
  if (!any(unshadowed > 0.0f) || !all(isfinite(unshadowed))) {
    return make_float3(0.0f);
  }
  return chain_visible(scene, shapes, origin, y, sample.primitive, count, p,
                       casters)
             ? unshadowed
             : make_float3(0.0f);
}

// The work list (contract 16, core/contracts/mnee_request.h): the
// estimators' loops write a request where the seed crosses casters; the
// MNEE pass solves them together.

// The list as a loop writes it: the requests and the header's count and
// capacity.
struct Queue {
  serenity::contracts::MneeRequest* requests;
  uint* header;  // the count and the capacity (contract 16)
};

// The request for the light sample `sample`, drawn from `origin` at `here`,
// whose segment `seed` crosses casters, for pixel `pixel`, its light to be
// multiplied by `weight`. Precondition: seed.usable, seed.count > 0.
__device__ inline serenity::contracts::MneeRequest request(
    Shading here, float3 origin, serenity::contracts::LightSample sample,
    Seed seed, float3 weight, uint pixel) {
  namespace c = serenity::contracts;
  c::MneeRequest r{};
  r.origin = to_packed(origin);
  r.pixel = pixel;
  r.weight = to_packed(weight);
  r.light = sample.primitive;
  r.direction = sample.direction;
  r.distance = sample.distance;
  r.radiance = sample.radiance;
  r.medium = here.medium;
  r.wo = to_packed(here.wo);
  r.crossings = seed.count;
#pragma unroll
  for (uint i = 0u; i < max_interfaces; ++i) {
    r.t[i] = seed.t[i];
    r.shapes[i / c::mnee_shapes_a_word] |=
        seed.shape[i] << (c::mnee_shape_bits * (i % c::mnee_shapes_a_word));
  }
  r.bsdf = pack_bsdf(here.bsdf);
  return r;
}

// `r` appended to the list, one atomic for the warp's lanes that append
// together (GDSA.10; the lanes here are those active at the call;
// device/append.cuh).
// Where the list is full it is not written, and its light is lost: not
// silently, for the count it added to the header passes the capacity by
// it, and the MNEE pass's grid kernel counts what passes (contract 16).
// Nothing travels back through the path's light: a NaN, as the loss was
// once reported, is not one under fast math, which may assume none.
__device__ inline void enqueue(Queue queue,
                               serenity::contracts::MneeRequest r) {
  const uint slot =
      warp_reserve(queue.header + serenity::contracts::mnee_count_word, 1u);
  // Written once by the host, before any frame (cuda/frame/mnee_list.h):
  // a plain load.
  const uint capacity = queue.header[serenity::contracts::mnee_capacity_word];
  if (slot < capacity) {
    queue.requests[slot] = r;
  }
}

// The light of request `r` (steps 2 to 4), X times its weight: the
// receiver and the light sample rebuilt from it, the receiver's
// SurfaceInteraction holding only what manifold_light() reads of it (its
// geometric normal, the BSDF's: contract 16).
template <typename SceneT>
__device__ inline float3 solve_request(SceneT scene,
                                       serenity::contracts::MneeRequest r) {
  namespace c = serenity::contracts;
  Seed seed{};
  seed.count = r.crossings;
  seed.usable = true;
#pragma unroll
  for (uint i = 0u; i < max_interfaces; ++i) {
    seed.t[i] = r.t[i];
    seed.shape[i] = (r.shapes[i / c::mnee_shapes_a_word] >>
                     (c::mnee_shape_bits * (i % c::mnee_shapes_a_word))) &
                    c::mnee_largest_shape;
  }
  Shading here{};
  here.bsdf = unpack_bsdf(r.bsdf);
  here.surface.geometric_normal = here.bsdf.normal;
  here.surface.shading_normal = here.bsdf.normal;
  here.wo = to_float3(r.wo);
  here.medium = r.medium;
  serenity::contracts::LightSample sample{};
  sample.direction = r.direction;
  sample.distance = r.distance;
  sample.radiance = r.radiance;
  sample.primitive = r.light;
  sample.pdf = 1.0f;  // unread: the weight holds the density
  const float3 center =
      transform_translation(scene.shapes().transforms[r.light]);
  return to_float3(r.weight) *
         manifold_light(scene, here, to_float3(r.origin), center, sample, seed);
}

}  // namespace mnee
}  // namespace shaders
}  // namespace serenity
