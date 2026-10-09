#pragma once

// Axis: Integrator (the naive path tracer).
//
// The radiance arriving along one camera ray, by the textbook path tracer
// with next event estimation: pbrt-v4's SimplePathIntegrator (section 13.3),
// which counts each light path one way and weighs no two strategies against
// each other, as logical-overview.md's principle 7 requires of every
// estimator here. Milestone 1's baseline, run once per pixel per frame by the
// path pass (metal/passes/path/path.h); its mean over frames converges to
// the scene's full light transport, so it is what every later estimator is
// measured against.
//
// The algorithm, whose steps the code below carries by number:
//
//   L = 0, the radiance found; beta = 1, the path's throughput; the ray from
//   the camera; and "counts emission", true for the camera's own ray.
//   For each surface, until Russian roulette ends the path (step 7):
//
//   Step 1  Trace: the nearest surface along the ray.
//   Step 2  Escape: if there is none, L += beta x sky(direction), and stop.
//           The sky is not aimed at, so this is the one way it is counted.
//   Step 3  Emission: if the surface glows, then if the path counts emission,
//           L += beta x L_e; stop either way, a light scatters nothing. The
//           path counts emission on the camera's own ray and after a delta
//           lobe, which no shadow ray could have reached the light through
//           (glass blocks shadow rays). After any other bounce the light was
//           counted at the surface before, by step 5, so it adds nothing
//           here (principle 7). Light that glass focuses onto a rough surface
//           is therefore counted here, the one way it can be, until manifold
//           next event estimation takes it over.
//   Step 4  Resolve the surface's BSDF (contract 2).
//   Step 5  Next event estimation, where the BSDF has a lobe that is not
//           delta (aims_at_lights): choose one light uniformly, P = 1 / N
//           (light_selection/uniform_light.metal.h); draw a direction toward
//           it, pdf p (contract 3); trace one shadow ray, short of the
//           light's surface; if nothing blocks it,
//             L += beta x f(wo, wi) |cos(wi)| x L_e / (P x p).
//           The cost per surface is the same for any number of lights
//           (principle 9).
//   Step 6  Sample the BSDF: wi, f and pdf (contract 2). If pdf is 0, stop.
//           beta *= f |cos(wi)| / pdf. The path counts emission at the next
//           surface only if this lobe was delta.
//   Step 7  Russian roulette, from the 4th surface: survive with
//           q = min(the largest channel of beta, 0.95), else stop;
//           beta /= q, so the mean is unchanged. This, not a depth limit,
//           ends paths: every path length keeps a chance, so the mean
//           converges to the full light transport (Veach 1997, 2.4).
//   Step 8  Continue: the ray leaves the surface along wi.
//
// What the camera ray starts from, and what is done with L, are the pass's:
// a point in the pixel drawn anew each frame, and the mean folded into the
// accumulated image (metal/film/accumulate.metal.h).
//
// The numbers a path draws are a function of the pixel, the frame's index
// and the dimension, the count of numbers the path drew before
// (metal/sampler/sampler.metal.h): independent between frames, so their mean
// converges, and any frame can be rendered again exactly, alone (principle 2).
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
// practice; 512 rays at the safety stop.
