#pragma once

// Axis: Pass (ReSTIR PT: the composition its passes run).
//
// ReSTIR PT (Lin et al. 2022) as ReSTIR PT Enhanced runs it at its defaults
// (Lin, Kettunen and Wyman 2026; docs/research/2026-10-11-restir-pt-
// layout.md, the design this implements, Phases 1 and 2): every pixel's light by
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
//                 paired neighbors (pairing.metal.h), the pairwise and
//                 Talbot MIS weights and the cap (path_merge.metal.h), on
//                 paths already shifted, and the duplication map
//                 (duplication.metal.h).
//   Pass          this header and the kernels of restir_pt_initial,
//                 restir_pt_temporal, restir_pt_spatial, restir_pt_shade
//                 and restir_pt_duplication (metal/passes/restir_pt_*/):
//                 the order, the numbers, the bindings.
//
// The algorithm, by pass (core/frame/schedule.h):
//
// restir_pt_initial, a light tiles' dispatch, then one thread a pixel:
//
//   1. The frame's light tiles (contracts/light_tile.h): 128 tiles of 1024
//      lights drawn by the light selection, each placed and packed; then
//      the camera's ray, its numbers the path tracer's (pixel, frame), and
//      the path tree along it: the path loop (walk_path) with TreeGather
//      below. At every vertex x1 ... x_n, n at most max_surfaces, NEE by
//      resampled importance sampling (nee.metal.h), its candidates drawn
//      from the tile the pixel's 8 x 8 block draws, a candidate path ending
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
//      into the frame's light buffer; the pixel's primary hit packed into
//      the G-buffer (contract 5's PrimaryHit), which every shift to the
//      pixel starts from; and, with temporal reuse, where that hit was last
//      frame (its point on its shape, placed by last frame's transform).
//
// restir_pt_temporal, four dispatches (Lin et al. 2022, TemporalReuse and
// TemporalPathRetrace; Enhanced §5, §7):
//
//   5. The temporal input: the primary hit's last-frame point through last
//      frame's camera, rounded to a pixel, it or the first of the 8 around
//      it whose last-frame primary hit is similar (previous_pixel(), as
//      ReSTIR DI's temporal merge searches); none at a history's first
//      frame, off the image, or where nothing near is the same surface.
//      Two shifts: last frame's final path there, shifted here in this
//      frame's scene; this pixel's path, shifted there in last frame's
//      (last frame's transforms and glows, for the Talbot weight's domain).
//      Each done here where it reconnects at x2; appended to the list
//      where it replays, as step 8 appends.
//   6. The list's length as an indirect grid (step 9's kernel), and the
//      replaying shifts, one thread each, each lane's frame data.
//   7. The Talbot merge (path_merge.metal.h): the own reservoir and last
//      frame's, its confidence capped at c_cap times the own one's, c_cap
//      lowered by last frame's duplication map at the input's pixel
//      (Enhanced §5); the path kept by the softmax pick; into the
//      reservoirs in place where a spatial reuse follows, else into the
//      frame's final reservoirs, its vector-valued estimate added to the
//      light buffer.
//
// restir_pt_spatial, four dispatches (Enhanced §3, §6.2.2, §7):
//
//   8. Pairs: for each of the 3 reuse textures, the pixel's paired
//      neighbor (pairing.metal.h); a pair whose primary hits are not
//      similar (the 2022 code's test: normals within 60 degrees, distances
//      within 10%, made symmetric so both ends of a pair agree) is not
//      merged. Each pixel's path shifted to each similar partner: here,
//      where the path reconnects at x2 and needs no replay; or, where it
//      replays, the pair appended to a list, compacted by a SIMD-group scan
//      and one atomic a group (GDSA.10: the count left on the device,
//      GDSA.13), so the lanes that only reconnect do not wait on the ones
//      that replay (GPU.4).
//   9. The list's length turned into an indirect dispatch's grid, and the
//      count set back to 0 for the next list.
//  10. The replaying pairs, one thread each, from the list.
//  11. The merge, one thread a pixel: its own path and its partners',
//      each shifted both ways by the dispatches above (one shift per pair
//      serves both ends: Enhanced §3), weighed by pairwise MIS and summed
//      vector-valued for shading (path_merge.metal.h, §6.3), added to the
//      light buffer; with temporal reuse in the graph, the path kept by the
//      same weights written as the frame's final reservoir.
//
// restir_pt_shade, one thread a pixel:
//
//  12. The light buffer, plus the initial reservoir's F W where no reuse
//      ran, folded into the accumulated image as the path pass folds its
//      own (metal/film/accumulate.metal.h).
//
// restir_pt_duplication, one thread a pixel in groups of 24 x 24:
//
//  13. The duplication map (metal/reuse/duplication.metal.h): each pixel's
//      share of the final reservoirs within 17 x 17 that hold copies of
//      its path, for the next frame's step 7.
//
// Manifold next event estimation (decided 2026-10-11, by the user's
// direction; settings.mnee, the graph's: metal/integrator/mnee.metal.h
// states the algorithm, docs/research/2026-10-12-mnee.md the measurements).
// MNEE's paths are taken out of ReSTIR PT's path space and estimated per
// pixel each frame, not resampled or reused: at each receiver of the tree's
// main path (step 1), NEE's kept candidate's segment is visited whole in
// place of its shadow ray; where it crosses casters, a request goes on
// MNEE's work list (contract 16), its weight the path's throughput over its
// survival times the candidate's W_RIS, and no candidate path is offered;
// the MNEE pass, recorded by the initial pass after its kernel, solves the
// requests and adds their light to the light buffer, which the shade folds
// in (metal/passes/mnee/mnee.h); where it is clear, the candidate is
// offered as before. The tree with MNEE is TreeGatherOf<true>, a kernel of
// its own (restir_pt_initial_mnee), so the tree without it compiles and
// costs what it did. A light a BSDF ray strikes at the end of MNEE's
// chain (mnee::covered()) is offered by no tree (step 3 of the loop) and
// found by no shift (shift.metal.h, replay_end()), so the path space every
// pixel resamples and every shift maps into is the same one, MNEE's paths
// out of it. Reconnection keeps a path's chain as the source's (x_k is
// rough, so the chain after it is the suffix's), so only replay is asked.
// Considered and not taken: MNEE's chains as candidate paths of the tree,
// resampled and shifted (no production system shifts a manifold walk; its
// Jacobian under a shift is the solve's own, not built here).
//
// What is approximate, stated. Last frame's domain, where step 5 shifts
// this pixel's path for its Talbot weight, is last frame's camera, primary
// hits, transforms and glows, traced against this frame's structure: a ray
// a firefly crossed last frame and not this one (or the reverse) is
// answered by this frame's. The structure kept a frame longer would make it
// exact, as ReSTIR DI's raytraced mode needs (docs/process/QUEUE.md). And a reconnection keeps its suffix after x_k as its
// source frame lit it (contract 15's cached radiance), as the 2022 code
// does: a path kept across frames carries a moved firefly's last light
// after x_k until it is resampled away, which the cap bounds; a path whose
// x_k is the light is lit as it is now.
//
// What it reads and writes, per pixel, the whole graph: the initial pass
// writes a 64-byte reservoir, 48 bytes of primary hit, 12 of previous
// position and 16 of light; the temporal reuse reads 2 primary hits, 2
// reservoirs and the previous position, and its shifts' 2 x 16 bytes, and
// writes 4 bytes of previous pixel and a reservoir; the spatial shift
// prepass reads the reservoir and 3 partners' primary hits (3 x 48) and
// writes 3 x 16 bytes of shifted paths, and 4 bytes a replaying pair; the
// merge reads its own and 3 partners' reservoirs and shifted paths
// (64 + 3 x (64 + 2 x 16) bytes), adds to its 16 bytes of light and writes
// a final reservoir; the shade reads the light; the duplication map 2.8
// reservoirs' first halves and writes 4. Some 1.3 KB a pixel a frame, 10 GB
// at 3456 x 2234, a floor of 25 ms at the M3 Max's 400 GB/s if none of it
// stayed in cache; the frame time is measured in docs/research/
// 2026-10-11-restir-pt-phase2.md, where shifts' shadow rays are most of it.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/camera.h"
#include "core/contracts/path_reservoir.h"
#include "core/contracts/light_tile.h"
#include "core/contracts/path_reuse_settings.h"
#include "core/contracts/primary_hit.h"
#include "core/contracts/transform.h"
#include "metal/camera/thin_lens.metal.h"
#include "metal/device/append.metal.h"
#include "metal/device/indirect_grid.h"
#include "metal/integrator/mnee.metal.h"
#include "metal/integrator/nee.metal.h"
#include "metal/integrator/path.metal.h"
#include "metal/integrator/reconnection.metal.h"
#include "metal/integrator/shift.metal.h"
#include "metal/integrator/target.metal.h"
#include "metal/math/transform.metal.h"
#include "metal/math/warp.metal.h"
#include "metal/reuse/duplication.metal.h"
#include "metal/reuse/pairing.metal.h"
#include "metal/reuse/path_merge.metal.h"
#include "metal/reuse/reservoir.metal.h"
#include "metal/reuse/temporal.metal.h"
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
    return pixel_ray(camera, pixel, size, numbers);
}

inline CameraRay primary_ray(constant serenity::contracts::CameraData& camera, uint2 pixel, uint frame, uint2 size) {
    PathNumbers numbers;
    return primary_ray(camera, pixel, frame, size, numbers);
}

// The numbers the passes draw beyond the tree's and the shift's
// (shift.metal.h, "The numbers"), each a purpose of its own stream, above
// every dimension a path reaches, as theirs are (metal/sampler/
// sampler.metal.h): a light tile's entry, at (entry, tile) and the frame;
// the tile an 8 x 8 block draws, at (block, frame).
constant constexpr uint tile_fill_first = purpose::tile_fill;
constant constexpr uint tile_pick_first = purpose::tile_pick;
// The temporal merge's two picks and the spatial merge's, at (pixel,
// frame); the frame's start of the temporal search's ring, at ((0, 0),
// frame), the frame's alone.
constant constexpr uint temporal_pick_first = purpose::temporal_pick;
constant constexpr uint spatial_pick_first = purpose::spatial_pick;
constant constexpr uint ring_first = purpose::ring;

// Entry `n` of frame `frame`'s light tiles, entry n % 1024 of tile
// n / 1024: a light the selection draws, placed and packed as this frame
// places and lights it (contracts/light_tile.h). Precondition: the scene
// has a light (the caller tests it).
inline serenity::contracts::LightTileEntry tile_entry(Scene scene, uint n, uint frame) {
    constexpr uint size = serenity::contracts::light_tile_size;
    PathNumbers numbers = purpose_numbers(path_numbers(uint2(n % size, n / size), frame), tile_fill_first);
    return compact_light(scene.lights(), select_light_index(scene.selection(), next_number(numbers)).index);
}

// The light tile the block holding `pixel` draws in frame `frame`, among
// `tiles` (contracts/light_tile.h).
inline const device serenity::contracts::LightTileEntry* block_tile(
    const device serenity::contracts::LightTileEntry* tiles, uint2 pixel, uint frame) {
    PathNumbers numbers =
        purpose_numbers(path_numbers(pixel / serenity::contracts::light_tile_block, frame), tile_pick_first);
    const uint tile = metal::min(uint(next_number(numbers) * float(serenity::contracts::light_tiles)),
                                 serenity::contracts::light_tiles - 1u);
    return tiles + tile * serenity::contracts::light_tile_size;
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
// `manifold`, the graph's mnee (above, "Manifold next event estimation"),
// a template parameter, so the tree without MNEE compiles what it did
// (ES.3; GPU.3): the initial pass has a kernel for each.
template <bool manifold>
struct TreeGatherOf {
    serenity::contracts::PathReuseSettings settings;
    PathNumbers stream;  // the tree's: (pixel, frame)
    Shapes shapes;       // where a reconnection vertex's shape is
    const device serenity::contracts::LightTileEntry* tile;  // the block's light tile (contracts/light_tile.h)
    // What the pass writes.
    float3 seen;  // the light the camera sees directly, step 4
    serenity::contracts::PrimaryHit primary;  // y1, for the G-buffer, step 4
    // y1's shape, for where its point was last frame (step 4,
    // previous_position()): one word carried across the walk's ray queries,
    // not the point and last frame's transforms (GPU.3).
    uint primary_shape;
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
    // The main path's chain since its last diffuse bounce, with MNEE
    // (mnee::chain_after(); above, "Manifold next event estimation"); MNEE's
    // work list and the pixel its requests are for.
    uint chain;
    mnee::Queue queue;
    uint pixel;

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

    // Step 1 of the loop: the medium's share over the stretch crossed,
    // after x_k the suffix's (asked again here, so the path pass's step 1
    // stays the line it was).
    [[gnu::always_inline]] void crossed(Scene scene, uint medium, float stretch) {
        if (rc != 0u) {
            suffix *= transmittance(scene.media(), medium, stretch);
        }
    }

    // Step 3 of the loop: a light struck, seen directly, or a candidate
    // weighed against NEE at the vertex it left (nee.metal.h).
    [[gnu::always_inline]] void emitted(Scene scene, thread float3&, float3 beta, serenity::lights::LightRecord light,
                                        float3 point, metal::raytracing::ray along, float emission_pdf, uint surface,
                                        serenity::contracts::SurfaceInteraction at) {
        const float3 glow = light_emitted(scene.lights(), light, point, -along.direction);
        if (surface == 0u) {
            seen += beta * glow;
            return;
        }
        if (manifold && mnee::covered(chain)) {
            return;  // MNEE's, counted at the receiver (next_event())
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
        const float3 integrand = beta * glow * weight;
        if (rc != 0u) {
            // x_{d-1} is x_k: its weight is the shift's to reweigh.
            const bool at_k = rc == length - 1u;
            offer(integrand, 1.0f,
                  found(length, serenity::contracts::PathEnd::struck, at_k ? glow : glow * weight,
                        at_k ? light_density : 0.0f));
            return;
        }
        const float3 at_light = to_float3(at.position);
        const reconnection::Pair pair = shift::forward_pair(before, at_light, to_float3(at.geometric_normal));
        if (reconnection::connectable(pair, threshold, settings.roughness)) {
            offer(integrand, 1.0f,
                  Stored{length, length, serenity::contracts::PathEnd::struck, before_lobe, 0u, at.primitive,
                         encode_shape_point(shapes, at.primitive, at_light), 0u, glow,
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
            // A light seen directly is no surface a path starts from:
            // pack_primary_hit() records none for it.
            primary = shift::pack_primary_hit(here, from, threshold);
            primary_shape = here.surface.primitive;
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
        const float probability = selection_probability(scene.selection());
        for (uint i = 0u; i < m; ++i) {
            const float choose = next_number(numbers);
            const float2 u = next_numbers2(numbers);
            const float pick = next_number(numbers);
            const nee::Candidate candidate = nee::draw_tile_candidate(
                tile[metal::min(uint(choose * float(serenity::contracts::light_tile_size)),
                                serenity::contracts::light_tile_size - 1u)],
                probability, here, origin, u);
            if (stream_pick(sum, candidate.target > 0.0f ? candidate.target / candidate.density : 0.0f, pick)) {
                kept = candidate;
            }
        }
        const float3 wi = to_float3(kept.sample.direction);
        if (!(kept.target > 0.0f)) {
            return;
        }
        if (manifold && mnee::receives(here.bsdf)) {
            // With MNEE: the kept candidate's segment visited whole, in
            // place of its shadow ray; crossing casters, MNEE's light
            // along the solved chain, weighed by W_RIS, into the pixel's
            // light, not resampled (above, "Manifold next event
            // estimation").
            const mnee::Seed seed =
                mnee::seed_segment(scene, origin, wi, kept.sample.distance, kept.sample.primitive);
            if (seed.count != 0u) {
                if (seed.usable) {
                    // Where the list is full, counted lost by the MNEE pass
                    // (mnee::enqueue()).
                    mnee::enqueue(queue, mnee::request(here, origin, kept.sample, seed,
                                                       beta / survival * (sum / (float(m) * kept.target)), pixel));
                }
                return;
            }
            if (seed.blocked) {
                return;
            }
        } else if (!nee::visible(scene.structure, scene.shapes(), origin, wi, kept.sample.distance,
                                 kept.sample.primitive)) {
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
        if constexpr (manifold) {
            chain = mnee::chain_after(chain, next.lobe, shapes.records, here.surface.primitive);
        }
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

    static constant constexpr bool divides_by_survival = false;
    static constant constexpr bool observes_vertices = true;

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

// The tree without MNEE, as the probes ask it.
using TreeGather = TreeGatherOf<false>;

// A tree's gather at its start: nothing found, nothing kept.
template <bool manifold>
inline TreeGatherOf<manifold> start_tree(serenity::contracts::PathReuseSettings settings, PathNumbers stream,
                                         Shapes shapes, const device serenity::contracts::LightTileEntry* tile,
                                         mnee::Queue queue, uint pixel) {
    TreeGatherOf<manifold> tree{};
    tree.settings = settings;
    tree.queue = queue;
    tree.pixel = pixel;
    tree.stream = stream;
    tree.shapes = shapes;
    tree.tile = tile;
    tree.primary_shape = 0u;
    tree.seen = float3(0.0f);
    tree.primary = shift::no_primary_hit();
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
template <bool manifold>
inline serenity::contracts::PathReservoir finish_tree(TreeGatherOf<manifold> tree) {
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

// Where the tree's primary hit was last frame (step 4): its point taken
// into its shape's own coordinates by this frame's transform and placed by
// last frame's (contract 10; as ReSTIR DI's restir::previous_position()
// places the object-space point its surface carries). Read only where the
// tree has a primary hit.
template <bool manifold>
inline float3 previous_position(TreeGatherOf<manifold> tree, constant serenity::contracts::Transform* transforms,
                                constant serenity::contracts::Transform* last_transforms) {
    const float3 object = transform_to_object(transforms[tree.primary_shape], to_float3(tree.primary.position));
    return transform_to_world(last_transforms[tree.primary_shape], object);
}

// The reservoir's estimate on its own, F W: what the shade adds where no
// spatial pass merged.
inline float3 estimate(serenity::contracts::PathReservoir reservoir) {
    return to_float3(reservoir.integrand) * reservoir.weight;
}

// The 2022 code's isValidGeometry (above, step 8), made symmetric: normals
// within 60 degrees, and distances within 10% of the larger, so a pair's
// two ends agree whether to merge. Of two primary hits (contract 5), or of
// one's distance and geometric normal as another frame saw it.
constant constexpr float similar_cosine = 0.5f;
constant constexpr float similar_distance = 0.1f;

inline bool similar(float da, float3 na, serenity::contracts::PrimaryHit b) {
    const float db = b.distance;
    return da > 0.0f && db > 0.0f && metal::abs(da - db) <= similar_distance * metal::max(da, db) &&
           metal::dot(na, decode_unit(b.normal)) >= similar_cosine;
}

inline bool similar(serenity::contracts::PrimaryHit a, serenity::contracts::PrimaryHit b) {
    return a.distance > 0.0f && similar(a.distance, decode_unit(a.normal), b);
}

// Whether shifting `reservoir`'s path replays any vertex (steps 5 and 8): no
// reconnection vertex, or one past x2.
inline bool replays(serenity::contracts::PathReservoir reservoir) {
    const uint k = serenity::contracts::path_rc(reservoir.flags);
    return k != 2u;
}

// A shifted path as the shift prepasses store it (steps 5 and 8): F and the
// Jacobian; a Jacobian of -1 for a partner not merged (outside the image, or
// not similar), told apart from a shift that failed (0).
constant constexpr float not_paired = -1.0f;

inline float4 stored_shift(shift::Shifted shifted) {
    return float4(shifted.integrand, shifted.jacobian);
}

// A replaying shift as the replay list holds it (steps 5, 6, 8 and 10): the
// pixel whose shifted paths it writes and the slot among them, one word,
// replay_slots to a pixel; its grid one of metal/device/indirect_grid.h's,
// in threadgroups of contract 15's replay_group. The spatial reuse's slots
// are its partners, the temporal reuse's its two shifts (last frame's path
// here, this pixel's there), each fewer than replay_slots.
constant constexpr uint replay_slots = 4u;
static_assert(serenity::contracts::max_path_neighbors <= replay_slots, "a pixel's partners fit its replay slots");

inline uint replay_item(uint pixel, uint slot) {
    return pixel * replay_slots + slot;
}
inline uint replay_pixel(uint item) {
    return item / replay_slots;
}
inline uint replay_slot(uint item) {
    return item % replay_slots;
}

// Whether a reservoir holds a path worth shifting: one, with a weight.
inline bool holds_path(serenity::contracts::PathReservoir reservoir) {
    return serenity::contracts::path_end(reservoir.flags) != serenity::contracts::PathEnd::none &&
           reservoir.weight > 0.0f;
}

// A merge's result: the reservoir it keeps, and its estimate for shading,
// the vector-valued sum (path_merge.metal.h).
struct Merged {
    serenity::contracts::PathReservoir reservoir;
    float3 estimate;
};

// Step 5's search (the temporal input's pixel), as ReSTIR DI's temporal
// merge finds its own (metal/reuse/temporal.metal.h, steps 2 and 3): the
// pixel's primary hit where it was last frame, `previous_position`,
// through last frame's camera, rounds to a pixel p of last frame's image of
// `size`; p, then the 8 around it in ring order from the frame's
// `ring_start`, until one whose last-frame primary hit is similar to this
// pixel's (above, similar(): its distance from last frame's camera, and
// `normal`, its geometric normal this frame). That pixel's index, or
// no_pixel: the surface was out of last frame's image, or nothing near
// was it.
constant constexpr uint no_pixel = 0xffffffffu;

inline uint previous_pixel(constant serenity::contracts::CameraData& camera, float3 previous_position, float3 normal,
                           const device serenity::contracts::PrimaryHit* last, uint2 size, uint ring_start) {
    if (!(camera_depth(camera, previous_position) > 0.0f)) {
        return no_pixel;
    }
    const float2 place = camera_pixel(camera, previous_position, size);
    if (!(metal::all(place >= 0.0f) && metal::all(place < float2(size)))) {
        return no_pixel;
    }
    const int2 p = int2(metal::min(uint2(place), size - 1u));
    const float distance = metal::distance(to_float3(camera.origin), previous_position);
    for (uint t = 0u; t < temporal_tries; ++t) {
        const int2 x = t == 0u ? p : ring_neighbor(p, t, ring_start);
        if (metal::any(x < int2(0)) || metal::any(x >= int2(size))) {
            continue;
        }
        const uint j = uint(x.y) * size.x + uint(x.x);
        if (similar(distance, normal, last[j])) {
            return j;
        }
    }
    return no_pixel;
}

// The frame's start of step 5's ring, the same for every pixel: drawn, so a
// pixel whose first try fails does not take the same side's neighbor frame
// after frame (temporal.metal.h, step 3, which found that drift).
inline uint ring_start(uint frame) {
    PathNumbers numbers = purpose_numbers(path_numbers(uint2(0u), frame), ring_first);
    return draw_ring_start(next_number(numbers));
}

// Step 7: the Talbot merge of the pixel's own reservoir `own` and last
// frame's `last` (path_merge.metal.h): `here` last's path shifted to this
// pixel in this frame, `there` own's shifted to last's pixel in last frame,
// each F and the Jacobian (stored_shift(); a Jacobian of 0 for a shift that
// failed or a path not held); `cap` c_cap at last's pixel; `u` the merge's
// two numbers.
inline Merged temporal_merge(serenity::contracts::PathReservoir own, serenity::contracts::PathReservoir last,
                             float4 here, float4 there, float cap, float2 u) {
    namespace c = serenity::contracts;
    const float own_count = float(c::path_count(own.flags));
    const float last_count = path_merge::temporal_confidence(float(c::path_count(last.flags)), own_count, cap);
    const float3 own_f = to_float3(own.integrand);
    const float own_target = luminance(own_f);
    const float last_target = luminance(to_float3(last.integrand));
    const float last_here = here.w > 0.0f ? luminance(here.xyz) * here.w : 0.0f;
    const float own_there = there.w > 0.0f ? luminance(there.xyz) * there.w : 0.0f;
    const float m_own = path_merge::talbot_weight(own_count * own_target, last_count * own_there);
    const float m_last = path_merge::talbot_weight(last_count * last_target, own_count * last_here);
    const float w_own = m_own * own_target * own.weight;
    const float w_last = m_last * last_here * last.weight;
    const float3 estimate =
        m_own * own.weight * own_f + (here.w > 0.0f ? m_last * last.weight * here.w * here.xyz : float3(0.0f));
    float sum = 0.0f;
    c::PathReservoir kept = own;
    (void)stream_pick(sum, w_own, u.x);
    if (stream_pick(sum, w_last, u.y)) {
        kept = path_merge::shifted_reservoir(last, here.xyz, here.w);
    }
    const float target = luminance(to_float3(kept.integrand));
    kept.weight = target > 0.0f && sum > 0.0f ? sum / target : 0.0f;
    kept.flags = path_merge::with_count(kept.flags, own_count + last_count);
    return Merged{kept, estimate};
}

}  // namespace restir_pt
}  // namespace shaders
}  // namespace serenity
