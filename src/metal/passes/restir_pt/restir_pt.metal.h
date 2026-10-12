#pragma once

// Axis: Pass (ReSTIR PT: the composition its passes run).
//
// ReSTIR PT (Lin et al. 2022) as ReSTIR PT Enhanced runs it at its defaults
// (Lin, Kettunen and Wyman 2026; docs/research/2026-10-11-restir-pt-
// layout.md, the design this implements, Phase 1): every pixel's light by
// resampling whole paths, reused across pixels by the hybrid shift, so the
// reuse reaches through glass and off mirrors, which ReSTIR DI's does not.
//
// Where the families meet, as ReSTIR DI's composition has it
// (metal/passes/restir_di/restir_di.metal.h): this header composes and
// decides nothing they decide.
//
//   Integrator    the path loop (path.metal.h, walk_path, whose gather this
//                 header supplies), NEE's candidates (nee.metal.h), the
//                 reconnection criteria (reconnection.metal.h) and the
//                 hybrid shift (shift.metal.h). Sees a path reservoir only
//                 as contract 15's data.
//   Sample reuse  the softmax pick (reservoir.metal.h, stream_pick), the
//                 paired neighbors (pairing.metal.h) and the pairwise MIS
//                 merge (path_merge.metal.h), on paths already shifted.
//   Pass          this header and the kernels of restir_pt_initial,
//                 restir_pt_spatial and restir_pt_shade (metal/passes/
//                 restir_pt_*/): the order, the numbers, the bindings.
//
// The algorithm, by pass (core/frame/schedule.h):
//
// restir_pt_initial, one thread a pixel:
//
//   1. The camera's ray, its numbers the path tracer's (pixel, frame), and
//      the path tree along it: the path loop (walk_path) with TreeGather
//      below. At every vertex x1 ... x_n, n at most max_surfaces, NEE by
//      resampled importance sampling (nee.metal.h), a candidate path ending
//      on its light; every ray a BSDF draws that strikes a light, or the
//      sky, a candidate path ending there; one lobe sampled per vertex
//      (contract 2: each kind's sample() draws one), so every path is
//      lobe-indexed (Enhanced supplemental §1). Direct light and indirect
//      in the same tree (Enhanced §6.1): the paths of length 2 compete with
//      the longer.
//   2. Each candidate streamed into the pixel's path reservoir with weight
//      luminance(F) times its unbiased contribution weight (1, or NEE's
//      W_RIS p1) over its source density, Russian roulette's survival
//      folded into that density (Enhanced supplemental §6: roulette never
//      replayed). W = w_sum / p̂(F) of the kept path; M = 1.
//   3. Along the way, each path's reconnection vertex: the first pair of
//      the tree's main path that passes the criteria, the variant by how the
//      path goes on from x_k (shift.metal.h, step 3), else for a path
//      ending on NEE's light the light itself (forced). The reservoir keeps
//      x_k, its direction on, the suffix after it, and the cached Jacobian
//      denominator (contract 15).
//   4. The light the camera sees directly (path length 1), not resampled,
//      into the frame's light buffer; the pixel's primary hit's distance
//      and normal, for the merge's similarity test.
//
// restir_pt_spatial, four dispatches (Enhanced §3, §6.2.2, §7):
//
//   5. Pairs: for each of the 3 reuse textures, the pixel's paired
//      neighbor (pairing.metal.h); a pair whose primary hits are not
//      similar (the 2022 code's test: normals within 60 degrees, distances
//      within 10%, made symmetric so both ends of a pair agree) is not
//      merged. Each pixel's path shifted to each similar partner: here,
//      where the path reconnects at x2 and needs no replay; or, where it
//      replays, the pair appended to a list, compacted by a SIMD-group scan
//      and one atomic a group (GDSA.10: the count left on the device,
//      GDSA.13), so the lanes that only reconnect do not wait on the ones
//      that replay (GPU.4).
//   6. The list's length turned into an indirect dispatch's grid, and the
//      count set back to 0 for the next frame.
//   7. The replaying pairs, one thread each, from the list.
//   8. The merge, one thread a pixel: its own path and its partners',
//      each shifted both ways by the dispatches above (one shift per pair
//      serves both ends: Enhanced §3), weighed by pairwise MIS and summed
//      vector-valued for shading (path_merge.metal.h, §6.3), added to the
//      light buffer.
//
// restir_pt_shade, one thread a pixel:
//
//   9. The light buffer, plus the initial reservoir's F W where no spatial
//      pass ran, folded into the accumulated image as the path pass folds
//      its own (metal/film/accumulate.metal.h).
//
// What it reads and writes, per pixel: the initial pass writes a 64-byte
// reservoir, 8 bytes of primary hit and 16 of light; the shift prepass
// reads the reservoir and 3 partners' primary hits and writes 3 x 16 bytes
// of shifted paths, and 4 bytes a replaying pair; the merge reads its own
// and 3 partners' reservoirs and shifted paths (64 + 3 x (64 + 2 x 16)
// bytes) and adds to its 16 bytes of light; the shade reads them. Some
// 0.7 KB a pixel a frame, 2.4 GB at 3456 x 2234, a floor of 6 ms at the M3
// Max's 400 GB/s; the frame time is measured in docs/research/
// 2026-10-11-restir-pt-phase1.md. Every shift traces its destination's
// primary ray again rather than reading a stored primary hit: the shift
// starts from exactly the surface the destination's own tree started from,
// to the bit, at a ray a shift (queued: a G-buffer in its place).

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/camera.h"
#include "core/contracts/path_reservoir.h"
#include "core/contracts/path_reuse_settings.h"
#include "metal/camera/thin_lens.metal.h"
#include "metal/integrator/nee.metal.h"
#include "metal/integrator/path.metal.h"
#include "metal/integrator/reconnection.metal.h"
#include "metal/integrator/shift.metal.h"
#include "metal/integrator/target.metal.h"
#include "metal/math/warp.metal.h"
#include "metal/reuse/pairing.metal.h"
#include "metal/reuse/path_merge.metal.h"
#include "metal/reuse/reservoir.metal.h"
#include "metal/sampler/sampler.metal.h"
#include "metal/shapes/shape_point.metal.h"

namespace serenity {
namespace shaders {
namespace restir_pt {

using Scene = path::Scene;

// The camera's ray through `pixel` in frame `frame`, its numbers the path
// pass's (metal/passes/path/path.metal): the one ray a pixel's tree starts
// from and every shift to that pixel starts from again.
inline CameraRay primary_ray(constant serenity::contracts::CameraData& camera, uint2 pixel, uint frame, uint2 size,
                             thread PathNumbers& numbers) {
    numbers = path_numbers(pixel, frame);
    const float2 within = next_numbers2(numbers);
    const float2 lens = has_lens(camera) ? next_numbers2(numbers) : float2(0.0f);
    return camera_ray(camera, float2(pixel) + within, lens, size);
}

inline CameraRay primary_ray(constant serenity::contracts::CameraData& camera, uint2 pixel, uint frame, uint2 size) {
    PathNumbers numbers;
    return primary_ray(camera, pixel, frame, size, numbers);
}

// What the reservoir keeps of a candidate path's shape and reconnection
// vertex (contract 15), as it is offered.
struct Stored {
    uint length;
    uint rc;
    serenity::contracts::PathEnd end;
    uint lobe_before;
    uint lobe_at;
    uint primitive;
    serenity::contracts::ShapePoint point;
    uint incoming;
    float3 radiance;
    float jacobian;
    float light_density;
};

// Steps 1 to 4's state: the path loop's gather (path.metal.h, "What the
// loop does with the light it finds"). Its hooks are the step's; its state
// the caller's local, so the loop and it are one kernel's registers.
struct TreeGather {
    serenity::contracts::PathReuseSettings settings;
    PathNumbers stream;  // the tree's: (pixel, frame)
    Shapes shapes;       // where a reconnection vertex's shape is
    // What the pass writes.
    float3 seen;  // the light the camera sees directly, step 4
    float primary_distance;
    float3 primary_normal;
    float weight_sum;
    serenity::contracts::PathReservoir chosen;
    uint offered;  // candidates so far, numbering the pick's numbers
    // The main path: Russian roulette's survival so far, the primary
    // footprint's threshold, the vertex before the current one and the
    // lobe it sampled, and the current vertex's ray footprint from it.
    float survival;
    float threshold;
    shift::Before before;
    uint before_lobe;
    float forward;
    // The main path's reconnection vertex, k, 0 until found, and what the
    // reservoir keeps of it; the suffix after x_k's BSDF.
    uint rc;
    uint rc_primitive;
    serenity::contracts::ShapePoint rc_point;
    uint rc_incoming;
    uint rc_lobe_before;
    uint rc_lobe_at;
    float rc_jacobian;
    float3 suffix;

    // Step 2: a candidate, kept or not by the pick (reservoir.metal.h).
    [[gnu::always_inline]] void offer(float3 integrand, float contribution_weight, Stored stored) {
        const float weight = luminance(integrand) * contribution_weight / survival;
        PathNumbers pick = purpose_numbers(stream, shift::pick_first + offered++);
        if (!stream_pick(weight_sum, weight, next_number(pick))) {
            return;
        }
        chosen.integrand = to_packed(integrand);
        chosen.flags = serenity::contracts::pack_path_flags(serenity::contracts::PathShape{
            1u, stored.length, stored.rc, stored.end, stored.lobe_before, stored.lobe_at, stored.point.face});
        chosen.primitive = stored.primitive;
        chosen.point = stored.point.point;
        chosen.incoming = stored.incoming;
        chosen.radiance = to_packed(stored.radiance);
        chosen.jacobian = stored.jacobian;
        chosen.light_density = stored.light_density;
    }

    // The main path's reconnection vertex as a candidate ending at vertex
    // `length` keeps it, its suffix ending in `tail` (the light and every
    // weight after x_k), and the light's NEE density where x_k's own MIS
    // weight is reweighed.
    [[gnu::always_inline]] Stored found(uint length, serenity::contracts::PathEnd end, float3 tail,
                                        float light_density) const {
        return Stored{length,      rc,          end,         rc_lobe_before, rc_lobe_at, rc_primitive,
                      rc_point,    rc_incoming, suffix * tail, rc_jacobian,  light_density};
    }

    [[gnu::always_inline]] static Stored replayed(uint length, serenity::contracts::PathEnd end) {
        return Stored{length, 0u, end, 0u, 0u, 0u, serenity::contracts::ShapePoint{0u, 0u}, 0u, float3(0.0f),
                      0.0f, 0.0f};
    }

    // Step 2 of the loop: the sky. Seen directly, or a candidate (no
    // reconnection at the sky: a path with none before it is replayed).
    [[gnu::always_inline]] void escaped(Scene scene, thread float3&, float3 beta, metal::raytracing::ray along,
                                        uint surface) {
        const float3 sky = gradient_sky(scene.sky(), along.direction);
        if (surface == 0u) {
            seen += beta * sky;
            return;
        }
        const uint length = surface + 1u;
        offer(beta * sky, 1.0f,
              rc != 0u ? found(length, serenity::contracts::PathEnd::sky, sky, 0.0f)
                       : replayed(length, serenity::contracts::PathEnd::sky));
    }

    [[gnu::always_inline]] void crossed(float3 kept) {
        if (rc != 0u) {
            suffix *= kept;
        }
    }

    // Step 3 of the loop: a light struck, seen directly, or a candidate
    // weighed against NEE at the vertex it left (nee.metal.h).
    [[gnu::always_inline]] void emitted(Scene scene, thread float3&, float3 beta, serenity::lights::LightRecord light,
                                        float3 point, metal::raytracing::ray along, float emission_pdf, uint surface,
                                        serenity::contracts::SurfaceInteraction at) {
        const float3 emitted = light_emitted(scene.lights(), light, point, -along.direction);
        if (surface == 0u) {
            seen += beta * emitted;
            return;
        }
        const uint length = surface + 1u;
        const bool weighed = emission_pdf != path::counted_in_full;
        const float light_density =
            weighed ? light_probability(scene.selection(), light) *
                          light_pdf(scene.lights(), light, along.origin, along.direction)
                    : 0.0f;
        const float weight =
            weighed ? nee::bsdf_weight(nee::candidate_count(settings.nee_candidates, length - 1u), light_density,
                                       emission_pdf)
                    : 1.0f;
        const float3 integrand = beta * emitted * weight;
        if (rc != 0u) {
            // x_{d-1} is x_k: its weight is the shift's to reweigh.
            const bool at_k = rc == length - 1u;
            offer(integrand, 1.0f,
                  found(length, serenity::contracts::PathEnd::struck, at_k ? emitted : emitted * weight,
                        at_k ? light_density : 0.0f));
            return;
        }
        const float3 at_light = to_float3(at.position);
        const reconnection::Pair pair = shift::forward_pair(before, at_light, to_float3(at.geometric_normal));
        if (reconnection::connectable(pair, threshold, settings.roughness)) {
            offer(integrand, 1.0f,
                  Stored{length, length, serenity::contracts::PathEnd::struck, before_lobe, 0u, at.primitive,
                         encode_shape_point(shapes, at.primitive, at_light), 0u, emitted,
                         1.0f / pair.forward, 0.0f});
            return;
        }
        offer(integrand, 1.0f, replayed(length, serenity::contracts::PathEnd::struck));
    }

    [[gnu::always_inline]] bool continues_at(uint surface) const { return surface < settings.max_surfaces; }

    // After step 4: the primary footprint at x1; the ray footprint of every
    // later vertex from the one before.
    [[gnu::always_inline]] void reached(Shading here, float3 from, uint surface) {
        const float3 at = to_float3(here.surface.position);
        const float3 normal = to_float3(here.surface.geometric_normal);
        if (surface == 0u) {
            threshold = reconnection::footprint_threshold(settings.footprint, from, at, normal);
            // A light seen directly is no surface a path starts from.
            const bool scatters = here.bsdf.kind != serenity::contracts::BsdfKind::none;
            primary_distance = scatters ? metal::distance(from, at) : 0.0f;
            primary_normal = normal;
            forward = 0.0f;
            return;
        }
        forward = reconnection::footprint(before.delta, before.density, before.position, at, normal);
    }

    // Step 5 of the loop: NEE by resampled importance sampling, a
    // candidate path ending on the light it keeps.
    [[gnu::always_inline]] void next_event(Scene scene, thread float3&, float3 beta, Shading here, uint surface,
                                           thread PathNumbers&) {
        const uint index = surface + 1u;
        const uint m = nee::candidate_count(settings.nee_candidates, index);
        const float3 origin = path::next_event_origin(here.surface, here.wo);
        PathNumbers numbers = shift::nee_numbers(stream, surface);
        float sum = 0.0f;
        nee::Candidate kept = nee::no_candidate();
        for (uint i = 0u; i < m; ++i) {
            const float choose = next_number(numbers);
            const float2 u = next_numbers2(numbers);
            const float pick = next_number(numbers);
            const nee::Candidate candidate =
                nee::draw_candidate(scene.lights(), scene.selection(), here, origin, choose, u);
            if (stream_pick(sum, candidate.target > 0.0f ? candidate.target / candidate.density : 0.0f, pick)) {
                kept = candidate;
            }
        }
        const float3 wi = to_float3(kept.sample.direction);
        if (!(kept.target > 0.0f) ||
            !nee::visible(scene.structure, scene.shapes(), origin, wi, kept.sample.distance, kept.sample.primitive)) {
            return;
        }
        const float3 kept_light = nee::unshadowed(here, wi, to_float3(kept.sample.radiance)) *
                                  transmittance(scene.media(), here.medium, kept.sample.distance) / kept.density;
        const float weight = nee::light_weight(m, kept.density, bsdf_pdf(here.bsdf, here.wo, wi));
        const float contribution_weight = sum / (float(m) * kept.target) * kept.density;  // W_RIS p1
        const uint length = index + 1u;
        if (rc != 0u) {
            offer(beta * weight * kept_light, contribution_weight,
                  found(length, serenity::contracts::PathEnd::nee, weight * kept_light, 0.0f));
            return;
        }
        const float3 at = to_float3(here.surface.position);
        if (index >= 2u &&
            reconnection::connectable(reconnection::Pair{before.roughness, forward, reconnection::largest}, threshold,
                                      settings.roughness)) {
            // x_k = this vertex, its direction NEE's (lobe 0): the suffix
            // the light over its density, the weight the shift's.
            offer(beta * weight * kept_light, contribution_weight,
                  Stored{length, index, serenity::contracts::PathEnd::nee, before_lobe, 0u, here.surface.primitive,
                         encode_shape_point(shapes, here.surface.primitive, at), encode_unit(wi),
                         to_float3(kept.sample.radiance) *
                             transmittance(scene.media(), here.medium, kept.sample.distance) / kept.density,
                         1.0f / forward, kept.density});
            return;
        }
        // Forced: x_k the light's point, as NEE drew it, found once.
        const nee::KeptPoint point = nee::kept_point(scene.lights(), kept, origin);
        const float geometry = point.light.cos_light / (point.light.distance * point.light.distance);
        offer(beta * weight * kept_light, contribution_weight,
              Stored{length, length, serenity::contracts::PathEnd::nee, 0u, 0u, kept.sample.primitive,
                     serenity::contracts::ShapePoint{point.point.point, 0u}, 0u, float3(0.0f),
                     kept.density * geometry, 0.0f});
    }

    [[gnu::always_inline]] float3 bsdf_numbers(thread PathNumbers&, uint surface) const {
        PathNumbers numbers = shift::bsdf_numbers(stream, surface);
        return next_numbers3(numbers);
    }

    // After step 6: the pair (x_{j-1}, x_j) tested with x_j's lobe, until
    // the main path's reconnection vertex is found; the suffix after it.
    [[gnu::always_inline]] void scattered(Shading here, serenity::contracts::BsdfSample next, uint surface, float3,
                                          float3 factor) {
        const uint index = surface + 1u;
        const float3 at = to_float3(here.surface.position);
        const bool delta = (next.lobe & serenity::contracts::lobe_delta) != 0u;
        if (rc != 0u) {
            suffix *= factor;
        } else if (index >= 2u) {
            const reconnection::Pair pair{
                before.roughness, forward,
                reconnection::inverse_skipped(next.lobe)
                    ? reconnection::largest
                    : reconnection::footprint(delta, next.pdf, at, before.position, before.normal)};
            if (reconnection::connectable(pair, threshold, settings.roughness)) {
                rc = index;
                rc_primitive = here.surface.primitive;
                rc_point = encode_shape_point(shapes, here.surface.primitive, at);
                rc_incoming = encode_unit(to_float3(next.direction));
                rc_lobe_before = before_lobe;
                rc_lobe_at = next.lobe;
                rc_jacobian = next.pdf / forward;
                suffix = float3(1.0f);
            }
        }
        before = shift::Before{at, to_float3(here.surface.geometric_normal),
                               reconnection::lobe_roughness(next.lobe, here.bsdf), delta, next.pdf};
        before_lobe = next.lobe;
    }

    // Step 7: roulette as the path tracer runs it, its survival kept as the
    // source density's factor, the throughput not divided (Enhanced
    // supplemental §6); its number at the surface's fixed dimension.
    [[gnu::always_inline]] bool survives(thread float3& beta, float eta_scale, uint surface, thread PathNumbers&) {
        if (surface >= path::roulette_from) {
            const float energy = metal::max(metal::max(beta.r, beta.g), beta.b) * eta_scale;
            const float q = metal::min(energy, path::max_survival);
            PathNumbers numbers = purpose_numbers(stream, shift::bsdf_first + shift::bsdf_stride * surface + 3u);
            if (next_number(numbers) >= q) {
                return false;
            }
            survival *= q;
        }
        return true;
    }
};

// A tree's gather at its start: nothing found, nothing kept.
inline TreeGather start_tree(serenity::contracts::PathReuseSettings settings, PathNumbers stream, Shapes shapes) {
    TreeGather tree{};
    tree.settings = settings;
    tree.stream = stream;
    tree.shapes = shapes;
    tree.seen = float3(0.0f);
    tree.chosen.seed_pixel = stream.key.x;
    tree.chosen.seed_frame = stream.key.y;
    tree.chosen.flags = serenity::contracts::pack_path_flags(
        serenity::contracts::PathShape{1u, 0u, 0u, serenity::contracts::PathEnd::none, 0u, 0u, 0u});
    tree.survival = 1.0f;
    tree.suffix = float3(1.0f);
    return tree;
}

// Step 2's end: the kept path's W = w_sum / p̂(F); an empty reservoir, its
// count 1 (one tree was drawn), where nothing was kept.
inline serenity::contracts::PathReservoir finish_tree(TreeGather tree) {
    serenity::contracts::PathReservoir reservoir = tree.chosen;
    const float target = luminance(to_float3(reservoir.integrand));
    if (!(target > 0.0f && tree.weight_sum > 0.0f) ||
        serenity::contracts::path_end(reservoir.flags) == serenity::contracts::PathEnd::none) {
        reservoir = serenity::contracts::PathReservoir{};
        reservoir.seed_pixel = tree.stream.key.x;
        reservoir.seed_frame = tree.stream.key.y;
        reservoir.flags = serenity::contracts::pack_path_flags(
            serenity::contracts::PathShape{1u, 0u, 0u, serenity::contracts::PathEnd::none, 0u, 0u, 0u});
        return reservoir;
    }
    reservoir.weight = tree.weight_sum / target;
    return reservoir;
}

// The reservoir's estimate on its own, F W: what the shade adds where no
// spatial pass merged.
inline float3 estimate(serenity::contracts::PathReservoir reservoir) {
    return to_float3(reservoir.integrand) * reservoir.weight;
}

// A pixel's primary hit as the merge's similarity test reads it: its
// distance from the camera (0: no surface a path starts from) and its
// geometric normal, octahedral.
inline uint2 pack_primary(float distance, float3 normal) {
    return uint2(as_type<uint>(distance), encode_unit(normal));
}

// The 2022 code's isValidGeometry (above, step 5), made symmetric: normals
// within 60 degrees, and distances within 10% of the larger, so a pair's
// two ends agree whether to merge.
constant constexpr float similar_cosine = 0.5f;
constant constexpr float similar_distance = 0.1f;

inline bool similar(uint2 a, uint2 b) {
    const float da = as_type<float>(a.x);
    const float db = as_type<float>(b.x);
    return da > 0.0f && db > 0.0f && metal::abs(da - db) <= similar_distance * metal::max(da, db) &&
           metal::dot(decode_unit(a.y), decode_unit(b.y)) >= similar_cosine;
}

// Whether shifting `reservoir`'s path replays any vertex (step 5): no
// reconnection vertex, or one past x2.
inline bool replays(serenity::contracts::PathReservoir reservoir) {
    const uint k = serenity::contracts::path_rc(reservoir.flags);
    return k != 2u;
}

// A shifted path as the shift prepass stores it (step 5): F and the
// Jacobian; a Jacobian of -1 for a partner not merged (outside the image, or
// not similar), told apart from a shift that failed (0).
constant constexpr float not_paired = -1.0f;

inline float4 stored_shift(shift::Shifted shifted) {
    return float4(shifted.integrand, shifted.jacobian);
}

}  // namespace restir_pt
}  // namespace shaders
}  // namespace serenity
