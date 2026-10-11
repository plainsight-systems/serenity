# Work Queue

This file tracks active and accepted work.

## Active

- **The first frames.** Step (a), done: a frame's whole path, end to end. A
  frame graph file names the passes, the Metal 4 backend records it with two
  frames in flight, and the test pattern reaches an SDL3 window at the
  display's resolution, or PNG files from the headless renderer. Step (b),
  done: the first scene, read from a file, ray traced in hardware: a soft
  brass sphere on a checkerboard at night, lit by two fireflies, through the
  preview pass (direct light with soft shadows, sky light, GGX metal, glass
  without caustics, 4x anti-aliasing), with each frame's GPU time in the
  window's title. 78 ms at the display's 3456 x 2234 on the M3 Max, 20 ms at
  half that each way: rendering below the display and upscaling with MetalFX
  is what brings it to 60 frames a second.

## Accepted

- **Repository skeleton** (2026-10-08). Process documents, the build, the
  toolchain pin and its check, the boundary check, the Metal backend's
  device and library loading.

- **Milestone 1, the naive path tracer.** First slice, done: a still
  scene converging over frames. The textbook path tracer with next event
  estimation (pbrt-v4's SimplePathIntegrator, principle 7: each light path
  counted one way), uniform light selection, Russian roulette, on contracts
  1 to 3 (surface interaction, BSDF, emitter), into an accumulated image
  whose pixels keep their own sample counts; headless --write and --time
  for references. Checked against closed forms: furnaces, an integrating
  sphere, lit floors. The preview reads its lights through the emitter
  contract too.

## Next, in order

1. Milestone 1, the rest: the reference render and its error measure.
   Done: the marbles themselves (porcelain under a clear coat, glass tinted
   by the first medium, cat's-eyes with swirled cores, a steel bearing,
   thirty-six marbles at real scale through a macro lens, seen from above,
   lit by 616 fireflies in two swarms circling among them, the scene written
   by scripts/make_marbles.py; scenes/marbles.toml);
   moving fireflies (the wander; shapes as a geometry and a transform; one
   level of placed boxes rebuilt each frame, docs/research/2026-10-09-
   acceleration-structure.md); fireflies in flight and blinking (episodes
   checked clear at load, glows on the GPU); tone mapping with bloom (a
   radiance image between passes, display and tone_map as the presenting
   passes, Jimenez's bloom pyramid, Khronos PBR Neutral), so a firefly's
   flash shows on the firefly; the dark opening (a black sky; fireflies
   that wake one by one, one swarm perched on the table and the marbles
   that rises, the other held above that comes down; a sleeping light's
   shadow ray not traced, as pbrt-v4 does; a swarm's firefly whose flight
   is refused by chance drawn again; docs/research/2026-10-10-dark-
   opening.md). The baseline the estimators are measured against
   (2026-10-11, superseding the textbook kernel's 29.90 ms at 1a617ab, by
   the user's direction: MEMORY.md): the path tracer with MIS on the
   marbles at t = 120, every firefly awake, 30.04 ms at 3456 x 2234, 31.95
   ms at t = 8, by the method the dark opening's note keeps; error x time
   at 1920 x 1080 about 0.018 at t = 120 and 0.04 at t = 8
   (docs/research/2026-10-11-path-mis.md). The reference and
   its error, built (docs/research/2026-10-10-reference.md): the headless
   renderer's --format pfm, sample indices held under 2^32,
   serenity-measure, and make reference and make convergence. Left: the
   references at t = 120 and t = 8 rendered, their floors, the
   two-reference check, and the naive estimator's convergence at both
   times, into that note's Results.

Then, each in its turn: ReSTIR DI (Phase A done 2026-10-10, branch
restir-di-phase-a: resampled importance sampling with visibility reuse,
graphs/restir_ris.toml; its split of the path loop made one loop again,
the path pass's time and bits kept, docs/research/2026-10-11-path-split-
cost.md; the review of f86e7ae answered 2026-10-11 on branch
phase-a-review: the reuse settings Sample reuse's contract, the light
transport the Integrator's, each candidate's light placed once, the packed
Bsdf's range refused by name, the RIS graph 2.7% faster at t = 120,
phase-a-guidelines-pass.md; MIS in the shared path tracer done
2026-10-11 on branch path-mis, docs/research/2026-10-11-path-mis.md, the
continuation's spikes gone; left: Phase B, the BSDF candidate (for the
first surface's spikes, 2026-10-11-restir-ris-results.md) and the
temporal and spatial merges,
docs/research/2026-10-10-restir-di-layout.md); ReSTIR GI; caustics by manifold next
event estimation; reservoir reuse as kernels; a denoiser of its own against
Open Image Denoise and MetalFX; the Vulkan backend on the AMD machine.

## Parking Lot

- **Whether MoltenVK supports Vulkan ray tracing.** Believed not; verify
  before the Vulkan backend, since it decides whether that backend could also
  run on the Mac.
