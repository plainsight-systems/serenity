#pragma once

// Axis: Integrator (the naive path tracer).
//
// The radiance arriving along one camera ray, by the path tracer with next
// event estimation and multiple importance sampling (MIS): pbrt-v4's
// PathIntegrator (section 13.4), its SampleLd() and its emission at a hit.
// At every surface with a lobe that is not delta, a firefly's light is found
// two ways, by a light sample (next event estimation) and by the ray the
// BSDF sends on, striking the firefly, and each is weighted against the
// other by the power heuristic (Veach 1997, chapter 9), as
// logical-overview.md's principle 7 has said since 2026-10-11. Milestone 1's
// baseline, run once per pixel per frame by the path pass
// (metal/passes/path/path.h); its mean over frames converges to the scene's
// full light transport, so it is what every later estimator is measured
// against.
//
// Why two strategies, weighted (decided 2026-10-11, by the user's direction:
// docs/research/2026-10-11-restir-ris-results.md, "Verdict"; measured in
// docs/research/2026-10-11-path-mis.md). A light sample is drawn uniformly
// over the light's cone, so at a near-mirror conductor (GGX alpha down to
// 10^-3) one that lands in the lobe returns f cos L_e / (P p) with f up to
// about 1 / (4 pi alpha^2): unbiased, its variance growing as 1 / alpha^2,
// the rare samples of 10^5 that set the error over thousands of frames. The
// BSDF's density matches such a lobe, and the light's matches a rough
// surface; the power heuristic hands each direction to the strategy whose
// density there is the higher, as production path tracers do (pbrt-v4's
// PathIntegrator, Falcor's). Considered and not taken: the balance
// heuristic (the power heuristic, exponent 2, has the lower variance where
// one density dominates the other, as at a near mirror, Veach 9.2.4, and is
// pbrt-v4's); clamping a sample's radiance (biased, and the reference must
// not be).
//
// The algorithm, whose steps the code below carries by number:
//
//   L = 0, the radiance found; beta = 1, the path's throughput; the ray from
//   the camera; emission_pdf = counted_in_full, how the next surface counts
//   a light it is (step 3), for the camera's own ray; from, the point the
//   ray truly left, the camera's; and the medium the path is in (contract
//   12), no_medium, air, at the camera.
//   For each surface, until Russian roulette ends the path (step 7):
//
//   Step 1  Trace: the nearest surface along the ray; beta *=
//           transmittance(medium, t) (contract 12), what the medium the ray
//           crossed kept of it, t the stretch from the surface the ray truly
//           left, not from the point just off it the ray starts at
//           (integrator/surface.metal.h), so the medium's light does not
//           hang on the offset's size. 1 in air.
//   Step 2  Escape: if there is none, L += beta x sky(direction), and stop.
//           The sky is not aimed at, so this is the one way it is counted.
//   Step 3  Emission: if the surface is a light (contract 3, light_at),
//           L += beta x L_e x w_b, L_e the emitter's, w_b by emission_pdf:
//             - counted_in_full, for the camera's own ray and a ray a delta
//               lobe sent, which no shadow ray could have reached the light
//               along (glass blocks shadow rays): w_b = 1. Light that glass
//               focuses onto a rough surface is counted so, the one way it
//               can be, until manifold next event estimation takes it over.
//               And for a ray a lobe that is not delta sent to wo's other
//               side, which step 5 there does not draw (below, "One
//               origin"): every such lobe reflects, so only a direction
//               that rounding puts just past the horizon.
//             - a density p_b > 0, for a ray that a lobe that is not delta
//               drew with density p_b, from a surface whose step 5 aimed at
//               lights: w_b = p_b^2 / (p_b^2 + (P p_l)^2), the power
//               heuristic, P = 1 / N the light's chance of being chosen
//               (light_selection/uniform_light.metal.h, light_probability)
//               and p_l the density with which step 5 draws this direction
//               toward this light, asked from the point the ray started at,
//               which is where step 5 at that surface draws from (contract
//               3's light_pdf(): the cone's 1 / its solid angle, and 0
//               outside it, where w_b is 1; below, "One origin").
//             - not_counted, ReSTIR's continuation only (below): nothing.
//           A light's surface scatters what its BSDF scatters: a glowing
//           sphere's nothing, so its path ends at step 6.
//   Step 4  Resolve the surface's BSDF (contract 2).
//   Step 5  Next event estimation, where the BSDF has a lobe that is not
//           delta (aims_at_lights): choose one light uniformly, P = 1 / N
//           (light_selection/uniform_light.metal.h); draw a direction toward
//           it, pdf p_l (contract 3), from the point a ray leaving toward
//           wo's side leaves from (next_event_origin(); below, "One
//           origin"); if it gives no light (L_e = 0, a
//           firefly not yet awake, core/animation/glow.h), stop there, as
//           pbrt-v4's PathIntegrator does (its test of ls->L): the term is 0
//           whatever the shadow ray finds, so the ray is not traced; if the
//           direction leaves on wo's other side, stop there too: it is not
//           step 5's to draw (below); else trace one shadow ray, from that
//           same point to the light's surface, the light itself ignored; if
//           nothing blocks it,
//             L += beta x f(wo, wi) |cos(wi)| x L_e x T / (P x p_l) x w_l,
//           w_l = (P p_l)^2 / ((P p_l)^2 + p_b^2), p_b = pdf(wo, wi), the
//           density with which step 6 here would draw wi (contract 2): the
//           same two densities step 3 weighs a BSDF-drawn ray by, asked from
//           the same point, so for any direction both strategies can draw,
//           w_l + w_b = 1, and the light is counted once in expectation
//           (below, "One origin"). T is the transmittance of the medium the
//           path is in over the shadow ray, from the point it leaves from to
//           the light's: 1 in air; a light inside the same glass as the
//           surface, dimmed as the glass dims. Not from the surface's point,
//           as step 1 measures a stretch: short of it by at most the step
//           the point leaves from (surface.metal.h), which keeps some
//           sigma x step more of the light, 0.2% for 15 um of the
//           marbles' ruby (140 per meter), and only for a light in the
//           surface's own medium, which no scene has. As pbrt-v4 and Falcor
//           measure a shadow ray's medium. Measured from the surface's
//           point it was slower (docs/research/2026-10-11-ray-offset.md).
//           The term is computed whole before the shadow ray is
//           traced, so across that query the step holds one float3 beside
//           the walk's own state, not f, wi, L_e, T and the densities
//           (GPU.3; ReSTIR's shade does the same,
//           metal/passes/restir_di/restir_di.metal.h). The cost per surface
//           is the same for any number of lights (principle 9).
//   Step 6  Sample the BSDF: wi, f and pdf (contract 2). If pdf is 0, stop.
//           beta *= f |cos(wi)| / pdf. emission_pdf = counted_in_full if
//           the lobe was delta, or if wi leaves on wo's other side, where
//           step 5 draws nothing (below, "One origin"), else pdf: what step
//           3 at the next surface weighs a light by (handed_on()). If the
//           lobe was a
//           transmission, the path crossed the surface into or out of its
//           shape: arriving from outside, it enters the shape's interior
//           (SurfaceInteraction::interior, contract 1); from inside, it
//           leaves into air. One medium at a time (contract 12): an opaque
//           core inside a tinted marble reflects without leaving the
//           marble's medium, so the stretches between glass and core are
//           dimmed as the glass's. A path in air asks no medium anything;
//           lanes diverge only where a SIMD group straddles glass and air,
//           a marble's few pixels (GPU.4). What a scene with no medium
//           pays for it: docs/research/2026-10-09-medium-cost.md.
//           A transmission also multiplies eta_scale, 1 at the camera, by
//           eta_t^2 (bsdf_eta), undoing the 1 / eta_t^2 its value scaled
//           radiance by (dielectric.metal.h): beta x eta_scale is the
//           throughput as energy sees it, which no boundary changes.
//   Step 7  Russian roulette, from the 4th surface: survive with
//           q = min(the largest channel of beta x eta_scale, 0.95), else
//           stop; beta /= q, so the mean is unchanged. This, not a depth
//           limit, ends paths: every path length keeps a chance, so the
//           mean converges to the full light transport (Veach 1997, 2.4).
//           By beta x eta_scale, not beta, as pbrt-v4's PathIntegrator
//           (etaScale): inside glass of ior 1.5, beta alone is 1 / 2.25 of
//           the energy the path carries, and roulette would end the paths
//           inside a marble twice as often as their light warrants. The
//           mean is the same either way; the noise inside glass is not.
//           Roulette enters neither strategy's density, nor any MIS
//           weight. Not because it is independent of the direction drawn:
//           it is not, q being found from beta after step 6 multiplied it
//           by f |cos| / pdf, which varies with wi at a conductor or a
//           coated surface. Because it is conditionally unbiased: given the
//           direction drawn and all before it, the path goes on with
//           chance q and a survivor's beta is divided by q, so what the
//           path goes on to gather has the expectation it had without
//           roulette, whatever q is and whatever it depends on (Veach 1997,
//           2.4). The weights at the next surface weigh the densities with
//           which step 6 drew the direction, which roulette does not
//           change; step 5 here is before it.
//   Step 8  Continue: the ray leaves the surface along wi, and from is the
//           surface's point.
//
// What the loop carries between surfaces for MIS: emission_pdf, one float
// in place of the bool ("counts emission") it replaces. Nothing else
// crosses a trace for it: step 3 finds p_l at the hit from the ray itself,
// its origin and direction, and the light struck.
//
// One origin (decided 2026-10-11, by the review of 7751a65). Every ray
// leaves a surface from leaving_point(), just off its point on the side
// the ray leaves toward, by the point's own float rounding
// (integrator/surface.metal.h). Both strategies draw, and weigh, a
// direction from the point its ray leaves from:
//
//   - step 5 draws its light sample from next_event_origin(), the point a
//     ray toward wo's side leaves from, its density P p_l the cone's from
//     there, and its shadow ray starts there; a direction it draws that
//     leaves on wo's other side it does not take (density 0 there). Every
//     lobe that is not delta is a reflection (contract 2: lambert,
//     conductor, the coated base), so f is 0 there anyway, and step 5 loses
//     nothing; were a kind to add a rough transmission, its directions
//     would be the bounce's alone, counted, not lost;
//   - step 6 hands on (handed_on()) the density p_b it drew a direction
//     with, or counted_in_full where the direction leaves on wo's other
//     side, which step 5 does not draw;
//   - step 3 asks p_l of the light the bounce struck from the ray's own
//     origin, the point the bounce left from.
//
// So for a direction wi that both can draw at a surface, wi leaves alike
// with wo (leave_alike()), and the two points are leaving_point(surface,
// wo) and leaving_point(surface, wi): the same expression on the same
// position and normal, its one branch taken the same way, the same bits.
// Step 5's P p_l is select_light()'s 1 / N times sample_light()'s pdf from
// that point, and step 3's is light_probability()'s 1 / N times
// light_pdf() from that point, each 1 over the cone's solid angle from
// view_light() of the same light at the same point (lights/
// sphere_light.metal.h): the same a. Step 5's p_b is pdf(wo, wi) and step
// 3's is the pdf sample() drew wi with: the same b, to the rounding of a
// conductor's half vector found two ways (contract 2). Then w_l =
// a^2 / (a^2 + b^2) and w_b = b^2 / (b^2 + a^2), which sum to 1. A
// direction only one strategy draws has the other's density 0: outside the
// cone from that point light_pdf() is 0 and w_b = 1; on wo's other side
// step 6 hands on counted_in_full and w_b = 1; step 5's directions all have
// p_b > 0 at a lobe they can be lit through. Every direction is counted
// once in expectation. tests/gpu/path_test.cpp holds the sum from these
// very functions and points, at a firefly millimeters across, near enough
// that the surface's point and the leaving point see different cones.
//
// What is estimated, then, is the light as seen from the point the rays
// leave from, not from the surface's point: the same integrand for both
// strategies, as the bounce always saw it. For a light at height h over a
// surface that is a relative 2 delta / h more of its light directly below
// it, and as much as delta / h less to its side, far from it, delta the
// leaving point's step off the surface: the float rounding of the point
// (surface.metal.h), 15 um on the marbles' table, 0.75 m up, so 1% for a
// firefly resting there with its middle 1.5 mm up, where the 0.1 mm step
// before it made 7% (docs/research/2026-10-11-ray-offset.md). pbrt-v4's
// PathIntegrator draws as this does (SampleLd() moves its light sample's
// point to the ray origin offset toward wo for a reflective BSDF), its
// offset the bound of the float error.
// Considered and not taken: drawing and weighing both from the surface's
// point, the offset only the traversal's. The bounce's ray then sees the
// light from a point other than the one its density is asked of, and the
// two cones differ by a crescent on either side, where one strategy's
// density is 0 and the other's is not: each crescent counted in full
// (7751a65's first version), or, with step 3 asked from the ray's origin
// and step 5 from the point (7751a65), two densities of the same light that
// differ, whose weights do not sum to 1 (the review of 7751a65). Only a
// ray traced from the point itself would make the point's integrand the
// bounce's; surface.metal.h records why rays are not.
//
// emission_pdf's two named values are values of its one purpose, how the
// next surface counts a light (ES.26; ES.45): counted_in_full is 0 and
// not_counted -1, and every density step 6 hands on is above 0 (a sample of
// pdf 0 ends the path), so no density is taken for either. One float and
// two names, rather than a bool beside a float as pbrt-v4 keeps them
// (specularBounce, bsdfPDF), so the loop carries no more across its traces
// than before (GPU.3); that shape was not measured against this one.
//
// The power heuristic, for the strategy that drew a direction with density
// f > 0 beside the other's g >= 0, is f^2 / (f^2 + g^2), in (0, 1].
// Computed from r, the smaller density over the larger, in [0, 1]:
// 1 / (1 + r^2) for the strategy of the larger, r^2 / (1 + r^2) for the
// smaller's; no density is squared, so none a float holds overflows it
// (ES.103). Squared directly, any density past 1.8 x 10^19 made the sum
// infinite and the weight infinity over infinity, NaN; a light's P p_l
// passes that for a cone under some 10^-19 / N sr, as a sphere of 10^-10 m
// seen from a meter, which the scene accepts (review of 7751a65). Defined
// at the edges, not left to a precondition (I.5): g = 0 gives 1 (r = 0); an
// infinite density beside a finite one takes the whole weight (r = 0
// again, a finite number over infinity being 0, under fast math as well,
// as the test shows); equal densities, both infinite included, 1/2. Pure
// (F.8).
//
// What the camera ray starts from, and what is done with L, are the pass's:
// a point in the pixel drawn anew each frame, and the mean folded into the
// accumulated image (metal/film/accumulate.metal.h).
//
// The numbers a path draws are a function of the pixel, the frame's index
// and the dimension, the count of numbers the path drew before
// (metal/sampler/sampler.metal.h): independent between frames, so their mean
// converges, and any frame can be rendered again exactly, alone (principle 2).
// MIS draws no number: each path draws the numbers it drew before MIS, in
// the same order. Its bounces are the rays they were; its shadow rays are
// drawn from the point they leave from (above, "One origin"), just off
// the point they were drawn from before, so they and what they add differ.
//
// A safety stop at 256 surfaces bounds one thread's work, against a path
// that roulette keeps alive for long (each survival is at most 0.95 likely,
// and a survivor's throughput is divided by it, so roulette leaves every
// path's expected contribution unchanged). The stop is the one place the
// estimator drops light: the light that reaches the camera only after more
// than 256 surfaces. Every surface that absorbs some of what it scatters
// shrinks that light by its albedo, so it falls off at least as fast as the
// largest albedo to the 256th power: below 1e-18 of the light for albedos of
// 0.85 and under, as the floor's and the brass's are. Only lossless surfaces
// (smooth glass) do not shrink it, and a ray through a glass sphere leaves
// it within a few surfaces. Stated, not hidden; not zero.
//
// Cost per path: one ray to the next surface and at most one shadow ray per
// surface, so about twice the mean path length, a handful of surfaces in
// practice; 512 rays at the safety stop. MIS adds no ray and draws no
// number. What it adds, per frame of N pixels with H_nd surfaces reached
// after a lobe that is not delta (the camera's ray and a delta lobe's
// already read them before MIS), counted as logical loads (GDSA.6):
//
//   - at each such surface, step 3 reads which light the struck shape is,
//     shape_lights[primitive], 4 bytes: 4 N H_nd bytes a frame, from a
//     small array in the constant address space, so from its cache in the
//     common case, not from DRAM;
//   - at each light struck after such a lobe, its density: the light's
//     record (8 bytes), sphere (16), transform (48) and glow (4), the same
//     addresses step 3's L_e already reads for that light, so the same
//     cache lines; and the cone from the ray's origin (a length, a square
//     root, a cross product, a divide), the selection's 1 / N and the
//     heuristic;
//   - per step 5 that reaches its shadow ray, the BSDF's density (contract
//     2's pdf: a dot product for a Lambert surface; a half vector, D and G1
//     for a conductor; a Fresnel term for a coated one) and the heuristic
//     (a min, a max, a compare, a divide, a few multiplies and an add);
//     per step 5, the point it draws from and the side its direction
//     leaves on (a dot product each; the point's offset_ray(), a few
//     integer adds, compares and selects), T over the sample's distance;
//   - per step 6, the side its direction leaves on (a dot product).
//
// Each is on the lanes that already take step 5, 6 or 3 (GPU.4). Measured
// against the build before it: docs/research/2026-10-11-path-mis.md; the
// offset sized to the point's rounding, docs/research/
// 2026-10-11-ray-offset.md.
//
// The split, for ReSTIR DI. The loop above is one function, walk_path(),
// whose start and stop are template parameters, so the reference and ReSTIR
// share every line of light transport but the direct light at the first
// rough surface (logical-overview.md, principle 3; ES.3), each caller
// compiling the loop for its own start and stop:
//
//   radiance()        <camera, at_end>: the loop above, from the camera's
//                     ray to the path's end. The path pass's.
//   camera_segment()  <camera, at_first_aimed>: steps 1 to 4 and, at
//                     surfaces with no lobe that aims at lights, 6 to 8, from
//                     the camera ray to the first surface that aims at lights
//                     (aims_at_lights), stopping there after step 4: its
//                     Shading, and the path's state, L, beta, eta_scale, the
//                     surface's count, and the numbers' counter, which the
//                     caller holds. Or the path's end before any such surface
//                     (it left the scene, reached a light, or roulette or the
//                     safety stop ended it among glass and mirrors), with L.
//                     Every lobe before that surface is delta, so the segment
//                     counts every light it strikes in full, and emission_pdf
//                     is not part of the path state.
//   continue_path()   <after_step_5, at_end>: from that surface, steps 6 to 8
//                     there, then the loop on from the next surface; step 5
//                     there is the caller's.
//
// Why one loop, specialized, and not two loops called in turn: the path
// pass's loop must keep the state it kept before the split live across its
// ray queries, and no more, since that state is what holds its occupancy
// down (docs/research/2026-10-10-path-kernel-counters.md). Two
// loops called in turn compiled the transport twice into the path kernel,
// and the second carried the first surface's Shading across every later ray
// query; one specialized loop compiles once, the same steps over the same
// state, in locals as before the split, so the path pass keeps its time
// (docs/research/2026-10-11-path-split-cost.md, which also records
// the shapes of the same loop that changed the time or, under fast math,
// the rounding).
//
// What the loop does with the light it finds, and how it draws its numbers
// and ends, is a third template parameter, Gather (decided 2026-10-11, for
// ReSTIR PT: docs/research/2026-10-11-restir-pt-layout.md, "The headers"):
// the same loop, steps 1 to 8, compiled once per caller, never copied (the
// copy Phase A tried cost 19%, docs/research/2026-10-11-path-split-cost.md).
// Its hooks, each called where its step is:
//
//   escaped(scene, L, beta, ray, surface)        step 2, the sky reached
//   crossed(kept)                                 step 1, what the medium kept
//   emitted(scene, L, beta, light, point, ray,    step 3, a light struck
//           emission_pdf, surface, surface_at)
//   continues_at(surface)                         after step 3: whether the
//                                                 surface goes on to step 4
//   reached(here, from, surface)                  after step 4
//   next_event(scene, L, beta, here, surface,     step 5
//              numbers)
//   bsdf_numbers(numbers, surface)                step 6's three numbers
//   scattered(here, sample, surface, from,        after step 6's throughput
//             factor)
//   survives(beta, eta_scale, surface, numbers)   step 7
//
// Radiance, below, is the path tracer's: every hook does what the step did
// before the parameter, in the same order on the same values, so the path
// pass's loop is the loop it was (its image the same to the bit and its
// time the same, measured: docs/research/2026-10-11-restir-pt-phase1.md).
// ReSTIR PT's initial pass gathers a path tree instead, each light found a
// candidate path streamed into its path reservoir with its density, NEE by
// resampled importance sampling, its numbers at fixed dimensions, Russian
// roulette's survival folded into the source density rather than divided
// out (metal/passes/restir_pt/restir_pt.metal.h, TreeGather, which lives
// with the Pass because it composes the Integrator and Sample reuse,
// tools/check_boundaries.sh rule 5). The hooks are inlined into the loop
// (GEN.6, overridden deliberately, as the loop's own), and a gather's state
// is the caller's local (F.17: an in-out parameter), so the path pass
// carries nothing across its ray queries it did not (GPU.3).
//
// The path pass calls radiance() alone. Its image changed with MIS, on
// purpose, by the weights alone, and its expectation did not: the one
// change that has broken its bit-identity with the build before it,
// declared (docs/process/MEMORY.md, 2026-10-11) and held instead by
// converging to the same references (docs/research/2026-10-11-path-mis.md).
// ReSTIR's initial pass calls camera_segment() and writes its result as the
// pixel's surface record and path state (contracts/surface_record.h); the
// shade calls continue_path() on what it reads back, so step 5 is skipped
// there and ReSTIR's direct light is the shade's to add
// (metal/passes/restir_di/restir_di.metal.h, step 6). Its numbers continue
// the path's stream from where the segment left it: the three step 5 would
// have drawn are not drawn, which changes which numbers later steps draw,
// not their independence.
//
// ReSTIR's first surface, q, is principle 7's one exception. The shade's
// direct light there is the reservoir's estimate, its light candidates
// and BSDF candidates (RTXDI's initial sampling) weighed against each
// other by MIS (metal/integrator/target.metal.h), so it alone counts every
// firefly's direct light at q. continue_path() therefore
// counts no light that the bounce from q strikes off a lobe that is not
// delta: step 6 at q hands on not_counted in place of what handed_on()
// gives, decided by the lobe drawn, so a direction on wo's other side is
// not counted at q either (the reservoir weighs every direction f is not 0
// for, on either side). A light the coat's delta lobe reaches from q is
// counted in full, as everywhere. From
// the next surface on, the continuation is the path tracer's, MIS and all,
// being the same loop. So nothing is counted twice and nothing left out:
// at q each firefly once, by ReSTIR; after q, each light path as the path
// tracer counts it.
//
// What the continuation reads of the surface, beyond its Shading's BSDF, wo
// and medium: the point and geometric normal (leaving it), the flags (which
// side it was arrived at from) and interior (the medium a transmission
// enters). The shade rebuilds those from the record and path state, its
// Shading's BSDF, wo and geometric normal from the record unpacked (the
// geometric normal is the shading normal for every surface so far:
// contracts/surface_record.h), so the continuation leaves from the surface
// ReSTIR weighed, its packing's small difference from the reference's
// stated there; the SurfaceInteraction's material, primitive,
// object_position and shading normal it leaves 0, unread by the
// continuation, and that is a contract of continue_path(): it reads no
// other field of `here.surface`, and adding a read of one adds it to
// contract 5's path state. The path pass never packs.


#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "core/contracts/emitter.h"
#include "core/contracts/medium.h"
#include "core/contracts/surface_interaction.h"
#include "metal/acceleration/trace.metal.h"
#include "metal/camera/thin_lens.metal.h"
#include "metal/device/layout.metal.h"
#include "metal/integrator/surface.metal.h"
#include "metal/light_selection/uniform_light.metal.h"
#include "metal/lights/emitter.metal.h"
#include "metal/lights/gradient_sky.metal.h"
#include "metal/materials/bsdf.metal.h"
#include "metal/materials/resolve.metal.h"
#include "metal/media/media.metal.h"
#include "metal/sampler/sampler.metal.h"
#include "metal/scene/scene_block.metal.h"
#include "metal/shapes/shapes.metal.h"

namespace serenity {
namespace shaders {
namespace path {

constant constexpr uint roulette_from = 3;  // the 4th surface, counting from 0
constant constexpr uint safety_stop = 256;
constant constexpr float max_survival = 0.95f;

// The scene (metal/scene/scene_block.metal.h), its lights chosen among
// uniformly.
using Scene = SceneView<UniformLight>;

// emission_pdf's two named values (above): how the next surface counts a
// light it is, where that is not by a density.
constant constexpr float counted_in_full = 0.0f;  // the camera's ray, a delta lobe's, and one step 5 does not draw: w_b = 1
constant constexpr float not_counted = -1.0f;     // ReSTIR's bounce from q off a lobe that is not delta

// The power heuristic (above): the weight of a sample drawn with density
// `drawn` > 0 beside a strategy of density `other` >= 0 for the same
// direction, either of them possibly infinite. In [0, 1], never NaN for
// such densities, and power_heuristic(f, g) + power_heuristic(g, f) is 1
// to rounding (I.7).
inline float power_heuristic(float drawn, float other) {
    const float larger = metal::max(drawn, other);
    const float smaller = metal::min(drawn, other);
    if (smaller == larger) {
        return 0.5f;  // equal densities, both infinite included: an even split
    }
    // larger > smaller >= 0, so r is in [0, 1): 0 where `smaller` is 0 or
    // `larger` infinite.
    const float r = smaller / larger;
    const float r2 = r * r;
    const float of_larger = 1.0f / (1.0f + r2);
    return drawn == larger ? of_larger : r2 * of_larger;
}

// Step 5's w_l: for a light sample along `wi`, drawn with density
// `light_density` (P p_l, above 0), at a surface of `bsdf` seen from `wo`.
inline float next_event_weight(float light_density, serenity::contracts::Bsdf bsdf, float3 wo, float3 wi) {
    return power_heuristic(light_density, bsdf_pdf(bsdf, wo, wi));
}

// Step 3's w_b: for `light`, struck by a ray from `origin` along unit
// `direction` that its surface drew with emission_pdf `emission_pdf`, p_l
// asked from the ray's origin, where step 5 at that surface draws from
// (above, "One origin"). Precondition: emission_pdf is not not_counted.
inline float emission_weight(UniformLight selection, Lights lights, serenity::lights::LightRecord light, float3 origin,
                             float3 direction, float emission_pdf) {
    if (emission_pdf == counted_in_full) {
        return 1.0f;
    }
    return power_heuristic(emission_pdf,
                           light_probability(selection, light) * light_pdf(lights, light, origin, direction));
}

// Where step 5 at `surface`, seen from `wo`, draws its light sample from and
// its shadow ray leaves from: the point a ray toward wo's side leaves from
// (above, "One origin").
inline float3 next_event_origin(serenity::contracts::SurfaceInteraction surface, float3 wo) {
    return leaving_point(surface, wo);
}

// Step 6's emission_pdf for the sample `next` drawn at `here`: how the next
// surface's step 3 counts a light it strikes. counted_in_full for a delta
// lobe's direction, and for one that leaves on wo's other side, which step
// 5 does not draw; else the density it was drawn with (above, "One origin").
inline float handed_on(serenity::contracts::BsdfSample next, Shading here) {
    if ((next.lobe & serenity::contracts::lobe_delta) != 0u ||
        !leave_alike(here.surface, to_float3(next.direction), here.wo)) {
        return counted_in_full;
    }
    return next.pdf;
}

// Step 5 at `here`: what one light, chosen uniformly, sends toward wo along
// one direction drawn toward it, f |cos| L_e T / (P p) x w_l, or 0 where the
// shadow ray to its surface is blocked. Drawn, weighed and its shadow ray
// traced from next_event_origin(). T is over `here`'s medium, which the
// shadow ray crosses, along it (above, step 5). The term is whole before
// the shadow ray's query, so it alone of the step is alive across it.
inline float3 next_event(Scene scene, Shading here, thread PathNumbers& numbers) {
    const SelectedLight chosen = select_light(scene.selection(), next_number(numbers));
    const float3 origin = next_event_origin(here.surface, here.wo);
    const serenity::contracts::LightSample sample =
        sample_light(scene.lights(), chosen.light, origin, next_numbers2(numbers));
    if (sample.pdf <= 0.0f || !metal::any(to_float3(sample.radiance) > 0.0f)) {
        return float3(0.0f);  // no light from it: no shadow ray (step 5)
    }
    const float3 wi = to_float3(sample.direction);
    if (!leave_alike(here.surface, wi, here.wo)) {
        return float3(0.0f);  // on wo's other side: not step 5's to draw (above, "One origin")
    }
    const float3 f = bsdf_evaluate(here.bsdf, here.wo, wi);
    if (!metal::any(f > 0.0f)) {
        return float3(0.0f);
    }
    const float light_density = chosen.probability * sample.pdf;
    const float3 unshadowed = f * metal::abs(metal::dot(wi, to_float3(here.bsdf.normal))) *
                              to_float3(sample.radiance) * transmittance(scene.media(), here.medium, sample.distance) /
                              light_density * next_event_weight(light_density, here.bsdf, here.wo, wi);
    if (occluded(scene.structure, scene.shapes(), metal::raytracing::ray(origin, wi, 0.0f, sample.distance),
                 sample.primitive)) {
        return float3(0.0f);
    }
    return unshadowed;
}

// What the loop carries from one surface to the next (steps 1 to 8): the
// light found, the throughput, eta_scale, how the next surface counts a
// light it is (emission_pdf, above), the medium the ray is in, the ray, and
// where it truly left from (the camera, then each surface's point; the ray
// itself starts just off that surface, so its hit distance is short of the
// stretch by that much, and a medium's stretch is measured from here).
struct Walk {
    float3 L;
    float3 beta;
    float eta_scale;
    float emission_pdf;
    uint medium;
    metal::raytracing::ray along;
    float3 from;
};

// The walk at the camera: nothing found, throughput 1, the camera's own ray,
// which counts a light in full, in air.
inline Walk walk_from(CameraRay camera) {
    return Walk{float3(0.0f),
                float3(1.0f),
                1.0f,
                counted_in_full,
                serenity::contracts::no_medium,
                metal::raytracing::ray(camera.origin, camera.direction, 0.0f, unbounded),
                camera.origin};
}

// The path tracer's gather (above): the light found added to L, its numbers
// drawn in turn from the path's counter, Russian roulette dividing the
// throughput by the survival chance. Stateless.
struct Radiance {
    // Step 2.
    [[gnu::always_inline]] void escaped(Scene scene, thread float3& L, float3 beta, metal::raytracing::ray along,
                                        uint) const {
        L += beta * gradient_sky(scene.sky(), along.direction);
    }

    // Step 1's: nothing beside beta.
    [[gnu::always_inline]] void crossed(float3) const {}

    // Step 3.
    [[gnu::always_inline]] void emitted(Scene scene, thread float3& L, float3 beta, serenity::lights::LightRecord light,
                                        float3 point, metal::raytracing::ray along, float emission_pdf, uint,
                                        serenity::contracts::SurfaceInteraction) const {
        L += beta * light_emitted(scene.lights(), light, point, -along.direction) *
             emission_weight(scene.selection(), scene.lights(), light, along.origin, along.direction, emission_pdf);
    }

    // Every surface goes on: the safety stop is the loop's.
    [[gnu::always_inline]] bool continues_at(uint) const { return true; }

    [[gnu::always_inline]] void reached(Shading, float3, uint) const {}

    // Step 5.
    [[gnu::always_inline]] void next_event(Scene scene, thread float3& L, float3 beta, Shading here, uint,
                                           thread PathNumbers& numbers) const {
        L += beta * path::next_event(scene, here, numbers);
    }

    // Step 6's numbers, the path's next three.
    [[gnu::always_inline]] float3 bsdf_numbers(thread PathNumbers& numbers, uint) const {
        return next_numbers3(numbers);
    }

    [[gnu::always_inline]] void scattered(Shading, serenity::contracts::BsdfSample, uint, float3, float3) const {}

    // Step 7: whether the path goes on; a survivor's throughput divided by
    // its chance.
    [[gnu::always_inline]] bool survives(thread float3& beta, float eta_scale, uint surface,
                                         thread PathNumbers& numbers) const {
        if (surface >= roulette_from) {
            const float energy = metal::max(metal::max(beta.r, beta.g), beta.b) * eta_scale;
            const float q = metal::min(energy, max_survival);
            if (next_number(numbers) >= q) {
                return false;
            }
            beta /= q;
        }
        return true;
    }
};

// Where walk_path() starts and where it stops (above, "The split").
enum class Start {
    camera,       // at step 1, along the walk's ray
    after_step_5  // at step 6, at a surface already resolved, whose step 5 is the caller's
};
enum class Stop {
    at_end,         // where the path ends
    at_first_aimed  // also at the first surface that aims at lights, after its step 4
};

// Where walk_path() stopped, and the walk there.
struct Stopped {
    bool at_surface;  // at a surface that aims at lights (Stop::at_first_aimed only); else the path ended
    Shading here;     // that surface, after step 4; read only where at_surface
    uint surface;     // its count along the path, from 0; read only where at_surface
    Walk walk;        // where the path ended, only L is meaningful
};

// Steps 1 to 8 for each surface from `first`, the count of the surface the
// walk starts at, until the path ends or `stop` says; `at` the surface
// Start::after_step_5 starts at, unread by Start::camera.
//
// One loop, compiled for each caller's start and stop: they are template
// parameters, not arguments, so the path pass's walk, <camera, at_end>,
// holds neither (ES.3; GPU.3). Its state is in locals and its steps in its
// own body, steps 6 to 8 a lambda over those locals (F.50, F.52), both
// forced inline (GEN.6, overridden deliberately): so the path pass's loop
// is the loop before the split, keeping the same state live across its
// ray queries (above, "The split").
template <Start start, Stop stop, typename Gather>
[[gnu::always_inline]] inline Stopped walk_path(Scene scene, Walk walk, Shading at, uint first,
                                                thread PathNumbers& numbers, thread Gather& gather) {
    float3 L = walk.L;
    float3 beta = walk.beta;
    float eta_scale = walk.eta_scale;
    float emission_pdf = walk.emission_pdf;
    uint medium = walk.medium;
    metal::raytracing::ray along = walk.along;
    float3 from = walk.from;
    const auto carried = [&]() __attribute__((always_inline)) {
        return Walk{L, beta, eta_scale, emission_pdf, medium, along, from};
    };

    // Steps 6 to 8 at `here`, the path's surface number `surface`: the walk
    // sent on from it; the lobe step 6 drew there, or 0 where the path ends
    // there (a sample's lobe has a bit of each pair, contract 2).
    const auto scatter = [&](Shading here, uint surface) __attribute__((always_inline)) -> uint {
        // Step 6.
        const serenity::contracts::BsdfSample next =
            bsdf_sample(here.bsdf, here.wo, gather.bsdf_numbers(numbers, surface));
        if (next.pdf <= 0.0f) {
            return 0u;
        }
        const float3 wi = to_float3(next.direction);
        const float3 factor = to_float3(next.value) * metal::abs(metal::dot(wi, to_float3(here.bsdf.normal))) / next.pdf;
        beta *= factor;
        emission_pdf = handed_on(next, here);
        gather.scattered(here, next, surface, from, factor);
        if ((next.lobe & serenity::contracts::lobe_transmission) != 0u) {
            medium = (here.surface.flags & serenity::contracts::arrived_from_outside) != 0u
                         ? here.surface.interior
                         : serenity::contracts::no_medium;
            const float eta = bsdf_eta(here.bsdf, here.wo);
            eta_scale *= eta * eta;
        }

        // Step 7.
        if (!gather.survives(beta, eta_scale, surface, numbers)) {
            return 0u;
        }

        // Step 8.
        from = to_float3(here.surface.position);
        along = leave(here.surface, wi);
        return next.lobe;
    };

    if constexpr (start == Start::after_step_5) {
        const uint lobe = scatter(at, first);
        if (lobe == 0u) {
            return Stopped{false, Shading{}, 0u, carried()};
        }
        // ReSTIR's q: its direct light the shade's alone, every lobe that is
        // not delta's (above, "ReSTIR's first surface").
        if ((lobe & serenity::contracts::lobe_delta) == 0u) {
            emission_pdf = not_counted;
        }
    }
    for (uint surface = start == Start::after_step_5 ? first + 1u : first; surface < safety_stop; ++surface) {
        // Step 1.
        const Hit hit = trace(scene.structure, scene.shapes(), along);

        // Step 2.
        if (!hit.found) {
            gather.escaped(scene, L, beta, along, surface);
            break;
        }
        const float3 point = along.origin + hit.t * along.direction;
        const float3 kept = transmittance(scene.media(), medium, metal::distance(from, point));
        beta *= kept;
        gather.crossed(kept);
        const serenity::contracts::SurfaceInteraction surface_at =
            surface_interaction(scene.shapes(), hit.primitive, point, along.direction);

        // Step 3: a shape's light record is read only where emission counts;
        // only a walk that starts after step 5 ever holds not_counted.
        if (start == Start::camera || emission_pdf != not_counted) {
            const ShapeLight glowing = light_at(scene.lights(), hit.primitive);
            if (glowing.is_light) {
                gather.emitted(scene, L, beta, glowing.light, point, along, emission_pdf, surface, surface_at);
            }
        }
        if (!gather.continues_at(surface)) {
            break;
        }

        // Step 4.
        const Shading here{surface_at, resolve_bsdf(scene.materials(), scene.textures(), surface_at),
                           -along.direction, medium};
        gather.reached(here, from, surface);

        // Step 5, or the stop before it.
        if constexpr (stop == Stop::at_first_aimed) {
            if (serenity::contracts::aims_at_lights(bsdf_lobes(here.bsdf))) {
                return Stopped{true, here, surface, carried()};
            }
        }
        if (serenity::contracts::aims_at_lights(bsdf_lobes(here.bsdf)) && scene.selection().count > 0u) {
            gather.next_event(scene, L, beta, here, surface, numbers);
        }

        // Steps 6 to 8.
        if (scatter(here, surface) == 0u) {
            break;
        }
    }
    return Stopped{false, Shading{}, 0u, carried()};
}

// The radiance arriving along the camera's ray, by the algorithm above;
// `numbers` the path's own. The path pass's.
inline float3 radiance(Scene scene, CameraRay camera, thread PathNumbers& numbers) {
    Radiance gather;
    return walk_path<Start::camera, Stop::at_end>(scene, walk_from(camera), Shading{}, 0u, numbers, gather).walk.L;
}

// ReSTIR's two parts of the same walk (above, "The split").

// Where the camera segment stopped, and the path's state there.
struct Segment {
    bool at_surface;   // false: the path ended before a surface that aims at lights; only L is meaningful
    Shading here;      // the surface, after step 4
    float3 L;          // the light found on the way
    float3 beta;       // throughput to the surface
    float eta_scale;   // step 6's
    uint surface;      // the surface's count along the path, from 0
};

// The walk from the camera's ray to the first surface that aims at lights,
// stopped there after step 4; or to the path's end before any.
inline Segment camera_segment(Scene scene, CameraRay camera, thread PathNumbers& numbers) {
    Radiance gather;
    const Stopped stopped =
        walk_path<Start::camera, Stop::at_first_aimed>(scene, walk_from(camera), Shading{}, 0u, numbers, gather);
    return Segment{stopped.at_surface, stopped.here,           stopped.walk.L,
                   stopped.walk.beta,  stopped.walk.eta_scale, stopped.surface};
}

// The radiance the path gathers from `segment`'s surface on, L included:
// the walk from step 6 there, step 5 there the caller's. Precondition:
// segment.at_surface. Reads of segment.here.surface only its position,
// geometric normal, flags and interior (above).
inline float3 continue_path(Scene scene, Segment segment, thread PathNumbers& numbers) {
    const Walk walk{segment.L,
                    segment.beta,
                    segment.eta_scale,
                    counted_in_full,  // unread: step 6 at the segment's surface sets it before the next surface reads it
                    segment.here.medium,
                    metal::raytracing::ray(),  // unread: step 8 at the segment's surface sets it
                    float3(0.0f)};             // unread: step 8 at the segment's surface sets it
    Radiance gather;
    return walk_path<Start::after_step_5, Stop::at_end>(scene, walk, segment.here, segment.surface, numbers, gather)
        .walk.L;
}

}  // namespace path
}  // namespace shaders
}  // namespace serenity
