#pragma once

// Axis: Integrator (ReSTIR PT's hybrid shift).
//
// A stored path (contract 15, core/contracts/path_reservoir.h) carried to
// another pixel: the hybrid shift of ReSTIR PT (Lin et al. 2022, §6.2) as
// ReSTIR PT Enhanced runs it (Lin, Kettunen and Wyman 2026, §2.3, §4,
// §6.2.3; supplemental §1, eq. 4). From the destination pixel's own primary
// hit y1, the source path's numbers are drawn again, vertex by vertex, up
// to y_{k-1} (random replay: the identity in the primary sample space, its
// Jacobian 1); then y_{k-1} is joined to the source's own x_k
// (reconnection), whose suffix after x_k's BSDF the reservoir holds. A path
// with no reconnection vertex (k = 0) is replayed to its end.
//
// The numbers (decided 2026-10-11 with contract 15). Every number a path
// tree draws is at a dimension fixed by what it is for and the vertex it is
// at, in the tree's stream (pixel, frame) (metal/sampler/sampler.metal.h,
// purpose_numbers()), so replay redraws a vertex's numbers without drawing
// the ones before it, and the numbers NEE spends at a vertex do not move
// the BSDF's (GDSA.3). Surface s (0 the primary hit, vertex s + 1):
//
//   bsdf_first + 4 s, + 1, + 2   step 6's three numbers
//   bsdf_first + 4 s + 3          Russian roulette's, initial sampling only
//   nee_first + 128 s + 4 i ...   NEE's candidate i: its light, its point's
//                                 two, the pick's
//   pick_first + c                the tree's pick of its candidate path c
//
// Above every dimension a path tracer's stream reaches (5, 6
// and 7 x 2^24; sampler.metal.h's table holds every purpose), and pcg3d is a
// bijection of (key, dimension), so these are distinct words from all of
// theirs. Russian roulette is never replayed (Enhanced §6.2.4): the source
// path's survival is in its source density, not in its numbers.
//
// The shift, step by step, as shift_path() runs it:
//
//   1. The destination's primary hit y1, read from the G-buffer the
//      initial pass wrote (contract 5's PrimaryHit, unpacked by
//      unpack_primary_hit() below); none, or a light, fails (the source
//      path had a surface to scatter from). Its primary footprint's
//      threshold is the record's.
//   2. Replay: at y_j, j from 1 to k - 2 (to d - 1 for k = 0), the source's
//      step 6 numbers drawn at y_j's BSDF; the ray traced to y_{j+1}. Any
//      ray that escapes or strikes a light before the end fails.
//   3. The invertibility check: the shifted path must have its own
//      reconnection vertex at k too, or the shift is not a bijection
//      (Enhanced §2.3). Each pair (y_{j-1}, y_j) replay passes, j >= 2, must
//      fail the criteria (metal/integrator/reconnection.metal.h), and
//      (y_{k-1}, x_k) must pass them; with the variant the source's pair had
//      (the full test where x_k's direction was drawn by its BSDF, without
//      the inverse test where NEE drew it or x_k is a light).
//   4. Forced NEE reconnection (Enhanced §6.2.3): a path ending on a light
//      NEE drew, with no reconnection vertex before, reconnects to that
//      light's point from y_{d-1} with no criteria, so replay never draws
//      NEE's numbers again. Its pair (y_{d-2}, y_{d-1}) must still fail the
//      NEE variant.
//   5. Reconnection: y_{k-1} joined to x_k, decoded where its shape is now
//      (metal/shapes/shape_point.metal.h); visibility between them traced;
//      the medium's transmittance over the segment, y_{k-1}'s medium.
//
//   Which shapes each ray finds (contract 19): the primary hit and the
//   replay's rays carry everything, since a light they strike ends the
//   shift as it ended the source path; the reconnection's visibility, to
//   a vertex or to a light, carries the occluders: a visibility question.
//   To a light it is the source path's own shadow ray's (nee.metal.h), so
//   the shift tests the visibility its path was sampled with. To a
//   surface x_k, whose source segment a ray carrying everything found
//   clear of fireflies, a firefly on the shifted segment does not fail
//   the shift: the fireflies' shadows left out once more, the same small
//   change as the shadow rays' (Milestone 4's Step 2 measures it).
//   6. The shifted integrand, and the Jacobian of supplemental eq. 4,
//
//        |du_y / du_x| = p_{k-1}^y(w', l_{k-1}) G(y_{k-1} -> x_k) p_k^y(w_k,
//        l_k)
//                        / [p_{k-1}^x(w, l_{k-1}) G(x_{k-1} -> x_k) p_k^x(w_k,
//                        l_k)]
//
//      the denominator the source's, cached (contract 15, `jacobian`); p_k
//      1 where x_k's direction was drawn by NEE (its density is the
//      light's, which x_{k-1} does not change) or x_k is the light; the
//      lobe at x_{k-1} and x_k kept (supplemental §1, lobe-indexed paths:
//      every kind here has one lobe that is not delta, so a lobe's density
//      is bsdf_pdf()). MIS weights that x_{k-1} or x_k's new direction
//      changes are recomputed (the M-sample balance heuristic,
//      metal/integrator/nee.metal.h); the ones after x_k are in the suffix.
//
// A failed shift is a value of the interface, Shifted::shifted false with F
// and the Jacobian 0 (I.1): its caller weighs it as a path that does not
// exist at the destination, which pairwise MIS needs to know apart from a
// neighbor it never shifted to.
//
// What is approximate, stated. A medium: the reconnection segment is dimmed
// by y_{k-1}'s medium, the cached suffix by x's; they differ only where a
// rough surface sits inside an absorbing medium, which no scene has. The
// reconnection vertex is the stored point, quantized (contract 15).
//
// Cost: the destination's 48-byte record, unpacked; a ray, a surface's
// resolve and a BSDF sample per replayed vertex; then a shadow ray and two BSDF
// evaluations for the reconnection. Pairs that replay are compacted before they
// run (metal/passes/restir_pt/restir_pt.metal.h), so a SIMD group that only
// reconnects does not wait on one that replays (GPU.4).

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "core/contracts/path_reservoir.h"
#include "core/contracts/path_reuse_settings.h"
#include "core/contracts/primary_hit.h"
#include "core/contracts/surface_interaction.h"
#include "metal/acceleration/trace.metal.h"
#include "metal/camera/thin_lens.metal.h"
#include "metal/integrator/mnee.metal.h"
#include "metal/integrator/nee.metal.h"
#include "metal/integrator/path.metal.h"
#include "metal/integrator/reconnection.metal.h"
#include "metal/integrator/surface.metal.h"
#include "metal/lights/emitter.metal.h"
#include "metal/lights/gradient_sky.metal.h"
#include "metal/materials/bsdf.metal.h"
#include "metal/materials/resolve.metal.h"
#include "metal/media/media.metal.h"
#include "metal/sampler/sampler.metal.h"
#include "metal/shapes/shape_point.metal.h"
#include "metal/shapes/shapes.metal.h"

namespace serenity {
namespace shaders {
namespace shift {

// The numbers' dimensions (above, "The numbers"; sampler.metal.h's table),
// and each span's bound within it: a surface's four BSDF numbers and its
// candidates' four each, at most max_path_surfaces surfaces; a pick for at
// most two candidates a surface (its NEE's, and the light or sky its BSDF
// ray finds) and the escape.
constant constexpr uint bsdf_first = purpose::path_bsdf;
constant constexpr uint nee_first = purpose::path_nee;
constant constexpr uint pick_first = purpose::path_pick;
constant constexpr uint bsdf_stride = 4u;
constant constexpr uint nee_stride = 128u;  // 4 x max_nee_candidates
static_assert(nee_stride >= 4u * serenity::contracts::max_nee_candidates,
              "a surface's candidates fit its stride");
static_assert(
    bsdf_stride * serenity::contracts::max_path_surfaces <= purpose_span &&
        nee_stride * serenity::contracts::max_path_surfaces <= purpose_span &&
        2u * (serenity::contracts::max_path_surfaces + 1u) <= purpose_span,
    "the tree's numbers stay within their purposes' spans");

// The stream of surface s's step 6 numbers, and its roulette's after them.
inline PathNumbers bsdf_numbers(PathNumbers stream, uint surface) {
  return purpose_numbers(stream, bsdf_first + bsdf_stride * surface);
}

// The stream of surface s's NEE candidates.
inline PathNumbers nee_numbers(PathNumbers stream, uint surface) {
  return purpose_numbers(stream, nee_first + nee_stride * surface);
}

// A source path's stream.
inline PathNumbers source_stream(serenity::contracts::PathReservoir source) {
  return PathNumbers{uint2(source.seed_pixel, source.seed_frame), 0u};
}

// What a shift returns (above).
struct Shifted {
  bool shifted;
  float3 integrand;  // F at the destination, in the primary sample space
  float jacobian;    // |du_y / du_x|
};

inline Shifted failed() { return Shifted{false, float3(0.0f), 0.0f}; }

// The destination's primary hit y1, as a shift starts from it: whether
// there is one that scatters, it, and the primary footprint's threshold for
// the criteria there (reconnection.metal.h).
struct Primary {
  bool found;
  Shading here;
  float threshold;
};

// The primary hit a camera ray finds, as the tree's walk finds it
// (path.metal.h, steps 1 to 4 at surface 0), traced and resolved: what a
// shift read before the G-buffer (contract 5's PrimaryHit), kept for the
// test that holds the G-buffer's shift to it (tests/gpu/restir_pt_test.cpp).
inline Primary primary_hit(path::Scene scene, CameraRay camera,
                           float footprint) {
  const metal::raytracing::ray along(camera.origin, camera.direction, 0.0f,
                                     unbounded);
  const Hit hit = trace(scene.structure, scene.shapes(), along,
                        serenity::contracts::RayMask::everything);
  if (!hit.found || light_at(scene.lights(), hit.primitive).is_light) {
    return Primary{false, Shading{}, 0.0f};
  }
  const float3 point = along.origin + hit.t * along.direction;
  const serenity::contracts::SurfaceInteraction surface = surface_interaction(
      scene.shapes(), hit.primitive, point, along.direction);
  const Shading here{surface,
                     resolve_bsdf(scene.materials(), scene.textures(), surface),
                     -along.direction, serenity::contracts::no_medium};
  return Primary{
      here.bsdf.kind != serenity::contracts::BsdfKind::none, here,
      reconnection::footprint_threshold(footprint, camera.origin, point,
                                        to_float3(surface.geometric_normal))};
}

// contract 5's PrimaryHit, the G-buffer's record of a pixel's primary hit
// `here` seen from the lens point `lens`, its footprint's `threshold`
// (core/contracts/primary_hit.h), packed by the initial pass; and back, for
// a shift. A pixel with no surface
// a path starts from is no_primary_hit(): distance 0, kind none.
inline serenity::contracts::PrimaryHit no_primary_hit() {
  serenity::contracts::PrimaryHit record{};
  record.sides = serenity::contracts::primary_no_medium
                 << serenity::contracts::primary_interior_shift;
  return record;
}

inline serenity::contracts::PrimaryHit pack_primary_hit(Shading here,
                                                        float3 lens,
                                                        float threshold) {
  namespace c = serenity::contracts;
  if (here.bsdf.kind == c::BsdfKind::none) {
    return no_primary_hit();
  }
  const float3 at = to_float3(here.surface.position);
  const float3 normal = to_float3(here.surface.geometric_normal);
  const uint interior = here.surface.interior == c::no_medium
                            ? c::primary_no_medium
                            : here.surface.interior;
  return c::PrimaryHit{here.surface.position,
                       encode_unit(normal),
                       pack_bsdf(here.bsdf),
                       encode_unit(here.wo),
                       metal::distance(lens, at),
                       threshold,
                       (here.surface.flags & c::primary_flags_mask) |
                           (interior << c::primary_interior_shift)};
}

// The surface a shift rebuilds of a record: its point, its geometric normal
// decoded and taken as the shading one too (contract 1: equal for every
// shape so far), its flags and interior; its material, primitive and own
// coordinates 0, which no step of a shift reads at y1 (shift_path() below).
inline Primary unpack_primary_hit(serenity::contracts::PrimaryHit record) {
  namespace c = serenity::contracts;
  if (!(record.distance > 0.0f)) {
    return Primary{false, Shading{}, 0.0f};
  }
  const c::Float3 normal = to_packed(decode_unit(record.normal));
  const uint interior = record.sides >> c::primary_interior_shift;
  c::SurfaceInteraction surface{};
  surface.position = record.position;
  surface.geometric_normal = normal;
  surface.primitive = c::no_primitive;
  surface.shading_normal = normal;
  surface.flags = record.sides & c::primary_flags_mask;
  surface.interior = interior == c::primary_no_medium ? c::no_medium : interior;
  return Primary{true,
                 Shading{surface, unpack_bsdf(record.bsdf),
                         decode_unit(record.wo), c::no_medium},
                 record.threshold};
}

// The vertex before the current one, as a pair test reads it.
struct Before {
  float3 position;
  float3 normal;    // geometric
  float roughness;  // of the lobe sampled there
  bool delta;       // whether that lobe is delta
  float density;    // the density it drew its direction with
};

// The pair (before, here) for the full test, `here`'s direction drawn by
// `lobe` with density `density` (reconnection.metal.h).
inline reconnection::Pair full_pair(Before before, Shading here, uint lobe,
                                    float density) {
  const float3 at = to_float3(here.surface.position);
  const float3 normal = to_float3(here.surface.geometric_normal);
  return reconnection::Pair{
      before.roughness,
      reconnection::footprint(before.delta, before.density, before.position, at,
                              normal),
      reconnection::inverse_skipped(lobe)
          ? reconnection::largest
          : reconnection::footprint(
                (lobe & serenity::contracts::lobe_delta) != 0u, density, at,
                before.position, before.normal)};
}

// The pair (before, here) with the inverse test skipped: here's direction
// drawn by NEE, or here a light.
inline reconnection::Pair forward_pair(Before before, float3 at,
                                       float3 normal) {
  return reconnection::Pair{
      before.roughness,
      reconnection::footprint(before.delta, before.density, before.position, at,
                              normal),
      reconnection::largest};
}

// Whether nothing blocks the segment between two surfaces, each left on the
// side facing the other (integrator/surface.metal.h): a reconnection's
// visibility.
inline bool segment_clear(path::Scene scene,
                          serenity::contracts::SurfaceInteraction from,
                          serenity::contracts::SurfaceInteraction to) {
  const float3 a = to_float3(from.position);
  const float3 b = to_float3(to.position);
  const float3 direction = metal::normalize(b - a);
  const float3 start = leaving_point(from, direction);
  const float3 end = leaving_point(to, -direction);
  const float3 span = end - start;
  const float length = metal::length(span);
  return !(length > 0.0f) ||
         !occluded(scene.structure, scene.shapes(),
                   metal::raytracing::ray(start, span / length, 0.0f, length),
                   serenity::contracts::RayMask::occluders,
                   serenity::contracts::no_primitive);
}

// A light at a stored point on shape `primitive`, as seen from `origin`:
// its answer (contract 4), and the density P p_l with which NEE from there
// draws its direction.
struct StoredLight {
  bool is_light;
  serenity::contracts::LightReceived light;
  float density;
};

inline StoredLight stored_light(path::Scene scene, uint primitive, uint point,
                                float3 origin) {
  const uint index = light_index_at(scene.lights(), primitive);
  if (index == serenity::lights::no_light) {
    return StoredLight{false, serenity::contracts::LightReceived{}, 0.0f};
  }
  const PlacedLight placed = place_light(scene.lights(), index);
  const serenity::contracts::LightReceived light =
      evaluate_light_point(placed,
                           serenity::contracts::LightPoint{
                               serenity::contracts::light_valid | index, point},
                           origin);
  return StoredLight{true, light,
                     power_probability(scene.power(), index) *
                         light_point_pdf(placed, origin, light)};
}

// A shift's result where F and the Jacobian are finite and the Jacobian
// above 0; failed otherwise.
inline Shifted shifted_to(float3 integrand, float jacobian) {
  const bool finite =
      metal::all(metal::isfinite(integrand)) && metal::isfinite(jacobian);
  return finite && jacobian > 0.0f ? Shifted{true, integrand, jacobian}
                                   : failed();
}

// What reconnect()'s steps 3 to 6 share, once its prologue has found it
// (below): the source's vertex x_k, the direction from y_{k-1} toward it
// and the squared distance, the density y_{k-1}'s BSDF draws that direction
// with, the roughness of the lobe the source drew there, whether NEE drew
// it, forced, and the candidate count at y_{k-1}.
struct Reconnecting {
  serenity::contracts::SurfaceInteraction at_k;
  float3 x_k;
  float3 w;
  float distance2;
  float density;
  float roughness;
  bool forced;
  uint m_before;
};

// Steps 4 to 6 where x_k = x_d, a light: struck by y_{k-1}'s BSDF, or NEE's,
// forced (above).
[[gnu::always_inline]] inline Shifted reconnect_to_light(
    path::Scene scene, Shading here, float3 beta, float threshold,
    serenity::contracts::PathReservoir source,
    serenity::contracts::PathReuseSettings settings, Reconnecting r) {
  const float3 w = r.w;
  const float density = r.density;
  const float roughness = r.roughness;
  const bool forced = r.forced;
  const uint m_before = r.m_before;
  const float3 origin = leaving_point(here.surface, w);
  const StoredLight stored =
      stored_light(scene, source.primitive, source.point, origin);
  if (!stored.is_light || stored.light.cos_light <= 0.0f) {
    return failed();
  }
  const float geometry =
      stored.light.cos_light / (stored.light.distance * stored.light.distance);
  const float forward = density > 0.0f ? 1.0f / (density * geometry) : 0.0f;
  if (!forced &&
      !reconnection::connectable(
          reconnection::Pair{roughness, forward, reconnection::largest},
          threshold, settings.roughness)) {
    return failed();
  }
  if (!nee::visible(scene.structure, scene.shapes(), origin, stored.light)) {
    return failed();
  }
  const float3 kept =
      transmittance(scene.media(), here.medium, stored.light.distance);
  if (forced) {
    const float3 unshadowed = nee::unshadowed(here, stored.light);
    const float bsdf_density =
        bsdf_pdf(here.bsdf, here.wo, to_float3(stored.light.direction));
    if (!(stored.density > 0.0f)) {
      return failed();
    }
    return shifted_to(
        beta * nee::light_weight(m_before, stored.density, bsdf_density) *
            unshadowed * kept / stored.density,
        stored.density * geometry / source.jacobian);
  }
  if (!(density > 0.0f)) {
    return failed();
  }
  const float3 f = bsdf_evaluate(here.bsdf, here.wo, w) *
                   metal::abs(metal::dot(w, to_float3(here.bsdf.normal)));
  return shifted_to(beta * f / density * kept *
                        to_float3(stored.light.radiance) *
                        nee::bsdf_weight(m_before, stored.density, density),
                    density * geometry / source.jacobian);
}

// Steps 4 to 6 where x_k is a surface, whose direction toward x_{k+1} the
// source keeps (above).
[[gnu::always_inline]] inline Shifted reconnect_to_surface(
    path::Scene scene, Shading here, float3 beta, float threshold,
    serenity::contracts::PathReservoir source,
    serenity::contracts::PathReuseSettings settings, Reconnecting r) {
  namespace c = serenity::contracts;
  const uint d = c::path_length(source.flags);
  const uint k = c::path_rc(source.flags);
  const c::PathEnd end = c::path_end(source.flags);
  const uint lobe_at = c::path_lobe_at(source.flags);
  const float3 y_normal = to_float3(here.surface.geometric_normal);
  const c::SurfaceInteraction at_k = r.at_k;
  const float3 x_k = r.x_k;
  const float3 y = to_float3(here.surface.position);
  const float3 w = r.w;
  const float distance2 = r.distance2;
  const float density = r.density;
  const float roughness = r.roughness;
  if (light_at(scene.lights(), source.primitive).is_light ||
      !(density > 0.0f)) {
    return failed();
  }
  const Shading xk{at_k,
                   resolve_bsdf(scene.materials(), scene.textures(), at_k), -w,
                   here.medium};
  const float3 wk = decode_unit(source.incoming);
  const float3 f_k = bsdf_evaluate(xk.bsdf, xk.wo, wk) *
                     metal::abs(metal::dot(wk, to_float3(xk.bsdf.normal)));
  const float density_k = bsdf_pdf(xk.bsdf, xk.wo, wk);
  const bool nee_at_k = lobe_at == 0u;
  const float geometry =
      metal::abs(metal::dot(to_float3(at_k.geometric_normal), w)) / distance2;
  const reconnection::Pair pair{
      roughness,
      1.0f / (density * metal::max(geometry, reconnection::least_geometry)),
      reconnection::inverse_skipped(lobe_at)
          ? reconnection::largest
          : reconnection::footprint((lobe_at & c::lobe_delta) != 0u, density_k,
                                    x_k, y, y_normal)};
  if (!reconnection::connectable(pair, threshold, settings.roughness) ||
      !segment_clear(scene, here.surface, at_k)) {
    return failed();
  }
  const float3 f = bsdf_evaluate(here.bsdf, here.wo, w) *
                   metal::abs(metal::dot(w, to_float3(here.bsdf.normal)));
  const float3 prefix =
      beta * f / density *
      transmittance(scene.media(), here.medium, metal::sqrt(distance2));
  const uint m_k = nee::candidate_count(settings.nee_candidates, k);
  if (nee_at_k) {
    // NEE at x_k drew the last segment: its density is the light's,
    // which y_{k-1} does not change; its MIS weight is reweighed.
    return shifted_to(
        prefix * nee::light_weight(m_k, source.light_density, density_k) * f_k *
            to_float3(source.radiance),
        density * geometry / source.jacobian);
  }
  if (!(density_k > 0.0f)) {
    return failed();
  }
  // A light the BSDF ray from x_k struck: its MIS weight is reweighed by
  // x_k's new density; every weight after x_k is in the suffix.
  const bool reweighed =
      end == c::PathEnd::struck && d == k + 1u && source.light_density > 0.0f;
  const float weight =
      reweighed ? nee::bsdf_weight(m_k, source.light_density, density_k) : 1.0f;
  return shifted_to(
      prefix * f_k / density_k * to_float3(source.radiance) * weight,
      density * geometry * density_k / source.jacobian);
}

// Steps 3 to 6 (above): y_{k-1} at `here`, its throughput `beta` and the
// vertex before it, joined to the source's x_k.
inline Shifted reconnect(path::Scene scene, Shading here, float3 beta,
                         Before before, float threshold,
                         serenity::contracts::PathReservoir source,
                         serenity::contracts::PathReuseSettings settings) {
  namespace c = serenity::contracts;
  const uint d = c::path_length(source.flags);
  const uint k = c::path_rc(source.flags);
  const c::PathEnd end = c::path_end(source.flags);
  const uint lobe_before = c::path_lobe_before(source.flags);
  const bool to_light = k == d;
  const bool forced = to_light && end == c::PathEnd::nee;
  if (!(source.jacobian > 0.0f) || !c::aims_at_lights(bsdf_lobes(here.bsdf))) {
    return failed();
  }
  const float3 y = to_float3(here.surface.position);
  const float3 y_normal = to_float3(here.surface.geometric_normal);
  const float3 decoded = decode_shape_point(
      scene.shapes(), source.primitive,
      c::ShapePoint{source.point, c::path_face(source.flags)});
  const c::SurfaceInteraction at_k = surface_interaction(
      scene.shapes(), source.primitive, decoded, decoded - y);
  const float3 x_k = to_float3(at_k.position);
  const float3 toward = x_k - y;
  const float distance2 = metal::dot(toward, toward);
  if (!(distance2 > 0.0f)) {
    return failed();
  }
  const float3 w = toward * metal::rsqrt(distance2);
  if (!leave_alike(here.surface, w, here.wo)) {
    return failed();  // every lobe that is not delta reflects (contract 2)
  }
  const float density = bsdf_pdf(here.bsdf, here.wo, w);
  const float roughness = reconnection::lobe_roughness(lobe_before, here.bsdf);

  // Step 3 at (y_{k-2}, y_{k-1}): y_{k-1}'s direction is the one to x_k,
  // drawn by the lobe the source drew there, or by NEE where forced.
  if (k >= 3u) {
    const reconnection::Pair pair =
        forced ? forward_pair(before, y, y_normal)
               : full_pair(before, here, lobe_before, density);
    if (reconnection::connectable(pair, threshold, settings.roughness)) {
      return failed();
    }
  }
  const uint m_before = nee::candidate_count(settings.nee_candidates, k - 1u);

  const Reconnecting r{at_k,    x_k,       w,      distance2,
                       density, roughness, forced, m_before};
  return to_light ? reconnect_to_light(scene, here, beta, threshold, source,
                                       settings, r)
                  : reconnect_to_surface(scene, here, beta, threshold, source,
                                         settings, r);
}

// Replay's last step for a path with no reconnection vertex: at y_{d-1},
// `here`, the source's numbers drawn once more, and whatever the ray finds
// at y_d counted as the path tracer counts it (the identity in the primary
// sample space: Jacobian 1). `chain` the replayed path's since its last
// diffuse bounce (mnee::chain_after()): with MNEE, a light struck at the
// end of MNEE's chain is not the path's (mnee::covered()), so the shift
// fails, as the tree at the destination would not have offered it.
inline Shifted replay_end(path::Scene scene, Shading here, float3 beta,
                          Before before, float threshold,
                          serenity::contracts::PathReservoir source,
                          serenity::contracts::PathReuseSettings settings,
                          PathNumbers stream, uint chain) {
  namespace c = serenity::contracts;
  const uint d = c::path_length(source.flags);
  const c::PathEnd end = c::path_end(source.flags);
  PathNumbers numbers = bsdf_numbers(stream, d - 2u);
  const c::BsdfSample next =
      bsdf_sample(here.bsdf, here.wo, next_numbers3(numbers));
  if (next.pdf <= 0.0f || end == c::PathEnd::nee) {
    return failed();
  }
  const float3 wi = to_float3(next.direction);
  beta *= to_float3(next.value) *
          metal::abs(metal::dot(wi, to_float3(here.bsdf.normal))) / next.pdf;
  if (d - 1u >= 2u &&
      reconnection::connectable(full_pair(before, here, next.lobe, next.pdf),
                                threshold, settings.roughness)) {
    return failed();
  }
  uint medium = here.medium;
  if ((next.lobe & c::lobe_transmission) != 0u) {
    medium = (here.surface.flags & c::arrived_from_outside) != 0u
                 ? here.surface.interior
                 : c::no_medium;
  }
  const float emission_pdf = path::handed_on(next, here);
  const metal::raytracing::ray along = leave(here.surface, wi);
  const Hit hit =
      trace(scene.structure, scene.shapes(), along, c::RayMask::everything);
  if (end == c::PathEnd::sky) {
    return hit.found ? failed()
                     : shifted_to(beta * gradient_sky(scene.sky(), wi), 1.0f);
  }
  if (!hit.found) {
    return failed();
  }
  const ShapeLight glowing = light_at(scene.lights(), hit.primitive);
  if (!glowing.is_light ||
      mnee::covered(mnee::chain_after(chain, next.lobe, scene.shapes().records,
                                      here.surface.primitive))) {
    return failed();
  }
  const float3 point = along.origin + hit.t * along.direction;
  beta *=
      transmittance(scene.media(), medium,
                    metal::distance(to_float3(here.surface.position), point));
  // The struck light must not pass the criteria from y_{d-1}: the source's
  // did not.
  const c::SurfaceInteraction lit =
      surface_interaction(scene.shapes(), hit.primitive, point, wi);
  const Before last{to_float3(here.surface.position),
                    to_float3(here.surface.geometric_normal),
                    reconnection::lobe_roughness(next.lobe, here.bsdf),
                    (next.lobe & c::lobe_delta) != 0u, next.pdf};
  if (reconnection::connectable(forward_pair(last, to_float3(lit.position),
                                             to_float3(lit.geometric_normal)),
                                threshold, settings.roughness)) {
    return failed();
  }
  const float weight =
      emission_pdf == path::counted_in_full
          ? 1.0f
          : nee::bsdf_weight(
                nee::candidate_count(settings.nee_candidates, d - 1u),
                power_probability(
                    scene.power(),
                    light_index_at(scene.lights(), hit.primitive)) *
                    light_pdf(scene.lights(), glowing.light, along.origin, wi),
                emission_pdf);
  return shifted_to(
      beta * light_emitted(scene.lights(), glowing.light, point, -wi) * weight,
      1.0f);
}

// The hybrid shift of `source` to the pixel whose primary hit is
// `destination` (above).
inline Shifted shift_path(path::Scene scene, Primary destination,
                          serenity::contracts::PathReservoir source,
                          serenity::contracts::PathReuseSettings settings) {
  namespace c = serenity::contracts;
  const uint d = c::path_length(source.flags);
  const uint k = c::path_rc(source.flags);
  if (c::path_end(source.flags) == c::PathEnd::none || d < 2u ||
      (k != 0u && (k < 2u || k > d)) || !destination.found) {
    return failed();
  }
  Shading here = destination.here;
  const float threshold = destination.threshold;
  const PathNumbers stream = source_stream(source);
  float3 beta = float3(1.0f);
  Before before{float3(0.0f), float3(0.0f), 0.0f, true,
                0.0f};  // the camera: never in a pair
  const uint last = k == 0u ? d - 1u : k - 1u;
  uint chain = mnee::off_chain;  // replay_end()'s
  for (uint index = 1u; index < last; ++index) {
    // Step 2 at y_vertex, surface vertex - 1.
    PathNumbers numbers = bsdf_numbers(stream, index - 1u);
    const c::BsdfSample next =
        bsdf_sample(here.bsdf, here.wo, next_numbers3(numbers));
    if (next.pdf <= 0.0f) {
      return failed();
    }
    const float3 wi = to_float3(next.direction);
    beta *= to_float3(next.value) *
            metal::abs(metal::dot(wi, to_float3(here.bsdf.normal))) / next.pdf;
    chain = mnee::chain_after(chain, next.lobe, scene.shapes().records,
                              here.surface.primitive);
    // Step 3: the pair replay passes must not be one the criteria take.
    if (index >= 2u &&
        reconnection::connectable(full_pair(before, here, next.lobe, next.pdf),
                                  threshold, settings.roughness)) {
      return failed();
    }
    uint medium = here.medium;
    if ((next.lobe & c::lobe_transmission) != 0u) {
      medium = (here.surface.flags & c::arrived_from_outside) != 0u
                   ? here.surface.interior
                   : c::no_medium;
    }
    before = Before{to_float3(here.surface.position),
                    to_float3(here.surface.geometric_normal),
                    reconnection::lobe_roughness(next.lobe, here.bsdf),
                    (next.lobe & c::lobe_delta) != 0u, next.pdf};
    const metal::raytracing::ray along = leave(here.surface, wi);
    const Hit hit =
        trace(scene.structure, scene.shapes(), along, c::RayMask::everything);
    if (!hit.found || light_at(scene.lights(), hit.primitive).is_light) {
      return failed();
    }
    const float3 point = along.origin + hit.t * along.direction;
    beta *= transmittance(scene.media(), medium,
                          metal::distance(before.position, point));
    const c::SurfaceInteraction surface =
        surface_interaction(scene.shapes(), hit.primitive, point, wi);
    here = Shading{surface,
                   resolve_bsdf(scene.materials(), scene.textures(), surface),
                   -wi, medium};
  }
  return k == 0u ? replay_end(scene, here, beta, before, threshold, source,
                              settings, stream, chain)
                 : reconnect(scene, here, beta, before, threshold, source,
                             settings);
}

}  // namespace shift
}  // namespace shaders
}  // namespace serenity
