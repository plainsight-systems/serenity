# Work Queue

This file tracks active and accepted work.

## Active

- **Formatting and static checks** (2026-10-19, branch `tooling`, by the
  user's decisions): every source in Google style (one formatting-only
  commit), clang-tidy's five families clean over `src/` and `tests/`
  (of 759 findings, the tests' literals aside, some 630 fixed, a commit
  per check or group, and some 130 the deviations in `workflow.md`), `make format`, `format-check` and `tidy` in `make check`,
  clangd in VS Code on save, LLVM 21.1.8 pinned. Every graph's output the
  same to the bit. Not merged: three deviations outside Metal await the
  user (`workflow.md`).

- **Milestone 5 site update** (2026-10-22, published by the user's
  approval): the homepage plays the Milestone 5 film rendered on the RTX
  3090 Ti, 45 s at 60 fps (the user's choice over 30 s at 30 fps: the last
  15 s the full swarm), encoded at crf 16 with peaks capped, 1080p60 at
  about 8 Mb/s and 540p60 at about 2; its caption gives the live 174 fps
  and the film's encoded 60 fps. The fifth article, staged from
  docs/writeups/milestone-5.md, carries the Milestone 4 versus Milestone 5
  comparison (30 fps, Milestone 4's film existing at 30 fps only) and its
  chart, with a phone version 420 px wide; Writing, About and the feed
  cover five milestones. Every film and still on R2 checked against the
  local bytes; the superseded 30 fps Milestone 5 film deleted from R2 and
  media/. Site tests pass (10).

- **Milestone 4 site update** (2026-10-19, published by the user's approval):
  homepage, Writing, About, article registry/media and feed updated for the
  closed fourth milestone, as for Milestone 3. The homepage plays the
  Milestone 4 film; the article carries the Milestone 3 versus Milestone 4
  comparison and its six SVG diagrams, each with a phone version drawn for
  widths up to 900 px (the user's choice over a sideways scroll). The
  film captions give the renderer's live rate (57 fps) apart from the
  film's encoded 30 fps. Films already on R2; nothing uploaded here. Build
  and ten site tests pass; the article and homepage fit a 390-pixel
  viewport. Approved by the user and published 2026-10-19.

- **The denoiser, Milestone 3** (docs/research/2026-10-14-denoiser-training-design.md).
  Phase 1 built 2026-10-14 (docs/research/2026-10-14-denoiser-features.md): the
  `features` pass (contract 17: albedo and normal through glass and mirrors by
  primary surface replacement, depth, NRD/DLSS motion vectors), graphs/features.toml
  and graphs/restir_pt_features.toml, serenity-headless --features, and
  scripts/make_marbles.py --seed N [--out FILE]. Phase 2 rendered 2026-10-14
  (docs/research/2026-10-14-denoiser-dataset.md): denoiser-v1 by
  scripts/make_denoiser_dataset.py, 400 frames in 50 sequences, 4.0 h, 32.3 GB,
  at ../serenity-data/denoiser-v1 beside the checkout. Phase 3 trained
  2026-10-15 (docs/research/2026-10-15-denoiser-training.md): `training/`
  (its own pinned environment, a new dependency, declared there), two sizes
  (small 294k parameters, wide 543k), tested once on the real marbles:
  displayed error 0.29 to 0.39 of the input's, below Open Image Denoise's;
  both exported to ../serenity-data/denoiser-v1-model. Decided by the user
  2026-10-15 after watching both live: the wide network is the demo's
  (graphs/restir_pt_denoised.toml; the small one stays as
  graphs/restir_pt_denoised_small.toml, removed 2026-10-17 with its
  weights and training/config.json), and no retrain for now, the
  improvement being visible; the heavier temporal weight (the table flickers
  36 to 47% more than the targets) waits for the next training round.
  Phase 4 built
  2026-10-15 (docs/research/2026-10-15-denoiser-live.md): the `denoise` pass
  on our own Metal kernels (contract 18), two resolutions in one frame
  (frame::light_size), graphs/restir_pt_denoised.toml with the weights as a
  setting, the match against PyTorch in ctest. The milestone's own guidelines
  and performance pass done 2026-10-15
  (docs/research/2026-10-15-milestone-3-guidelines-pass.md): both corpora
  rule by rule over its code, every valid finding fixed, every graph the
  same to the bit. Its documentation pass done 2026-10-15: the
  architecture documents, MEMORY.md, this queue, the research index, the
  training README and the front page brought up to Milestones 2 and 3.
  Write-up revised 2026-10-16 using Andrew's voice guide:
  docs/writeups/milestone-3.md, accepted 2026-10-16; main fast-forwarded to
  `denoiser` at the close (workflow.md).

## Accepted

- **Milestone 3 site update** (2026-10-16): homepage, Writing, About,
  article registry/media and feed updated for the accepted third milestone.
  Six videos and two posters uploaded to R2 and verified by full public
  readback; original files preserved. Site build and ten tests pass. Both
  new players load and play from R2; the article fits a 390-pixel viewport.
  Andrew approved the preview and requested commit/push for publication
  through the main-branch Pages workflow.

- **Milestone 2: ReSTIR, closed 2026-10-13** (docs/writeups/milestone-2.md;
  main fast-forwarded to `restir-di`). ReSTIR DI, Phases A to C
  (2026-10-10/11): resampled importance sampling with visibility reuse,
  RTXDI's BSDF candidate, the temporal and spatial merges in basic mode,
  graphs/restir_ris.toml, restir_temporal.toml, restir_spatial.toml and
  restir.toml (docs/research/2026-10-11-restir-ris-results.md,
  2026-10-11-restir-temporal.md, 2026-10-11-restir-spatial.md); on the way,
  the path loop made
  one loop again (2026-10-11-path-split-cost.md), MIS in the path tracer,
  the new time baseline (2026-10-11-path-mis.md), and the ray offset sized
  to the point's rounding (2026-10-11-ray-offset.md). ReSTIR DI lost to
  the MIS path tracer at equal time on the marbles, whose light arrives
  mostly through glass and off mirrors, so the milestone moved to ReSTIR
  PT, in place of the ReSTIR GI first planned
  (2026-10-11-restir-pt-layout.md): Phase 1, initial sampling, the hybrid
  shift and paired spatial reuse (2026-10-11-restir-pt-phase1.md); Phase 2
  by the user's direction, temporal reuse by Talbot MIS, duplication maps,
  a G-buffer of primary hits and light tiles, graphs/restir_pt.toml
  (2026-10-11-restir-pt-phase2.md). Manifold NEE for glass (2026-10-12/13,
  by the user's direction): set in graphs/restir_pt.toml by the
  equal-time comparison (glass and metal displayed error -16% at t = 120,
  -5% at t = 8), opt-in on the path tracer (graphs/path_mnee.toml), not in
  graphs/path.toml, which stays the reference (2026-10-12-mnee.md). The
  references at t = 120 and t = 8 rendered again after the ray offset,
  2026-10-13 (that note, "Against the 2026-10-13 references"). The
  convergence runs were not repeated on them: Milestone 2 compared
  estimators at equal frame time instead. ReSTIR DI, its graphs and
  graphs/path_mnee.toml removed 2026-10-17 (MEMORY.md, "Decided by the
  user 2026-10-17"); the tag `milestone-2` keeps them.

- **Publication feed for the Plainsight hub** (2026-10-13). Implemented
  `/writing/index.xml` from staged public articles, following Andrew's approval
  of automatic aggregation. Production build, ten site tests and independent
  publication QA pass. Draft/future exclusion, complete archive, XML escaping
  and deployment subpaths checked. The main-branch Pages workflow publishes it.

- **Milestone 2 site and README update** (2026-10-13). Current homepage film,
  article publication, both milestone entries, per-article previews, archived
  milestone-one film, and current run instructions approved by Andrew. Six
  new R2 videos uploaded and verified byte for byte. Site build and nine
  tests pass; desktop/narrow-screen review and both new film playback checks
  passed. Andrew requested commit/push; main publishes through GitHub Pages.

- **Milestone 2 writeup** (2026-10-13). Andrew approved the voice revision
  and committed it with the renderer work to `main`. The narrative follows
  retaining useful work, scheduling it, and preparing results for the next
  stage, with Charlotte’s KV cache, kernel fusion, and normalization woven
  through the decisions. The table reports displayed MSE at similar frame
  costs. The site update is accepted above.

- **Project site published** (2026-10-12). [PR #1](https://github.com/plainsight-systems/serenity/pull/1)
  merged by the user's direction. Pages deployment passed; the site is live at
  https://serenity.plainsight-systems.com/ with HTTPS enforced. Home, Writing,
  Milestone 1, About, and custom 404 match the checked production build;
  HTTP redirects to HTTPS and the R2 film plays at 1920 × 1080. Eight tests
  pass in GitHub CI. Public state remains milestone 1; ReSTIR stays separate.
  The header also links to GitHub after About, by user request.
  See [site-plan.md](site-plan.md) and `web/README.md`.

- **Repository skeleton** (2026-10-08). Process documents, the build, the
  toolchain pin and its check, the boundary check, the Metal backend's
  device and library loading.

- **Milestone 1, the naive path tracer, closed 2026-10-10**
  (docs/writeups/milestone-1.md). The first frames: a frame graph file
  names the passes, the Metal 4 backend records it with two frames in
  flight, to an SDL3 window at the display's resolution or to files from
  the headless renderer; the first scene, a brass sphere lit by two
  fireflies, through the preview pass (removed 2026-10-17, with the
  display pass that presented it). The textbook path tracer with next
  event estimation (pbrt-v4's SimplePathIntegrator), uniform light
  selection, Russian roulette, on contracts 1 to 3, into an accumulated
  image whose pixels keep their own sample counts; checked against closed
  forms (furnaces, an integrating sphere, lit floors). The marbles
  (scenes/marbles.toml, written by scripts/make_marbles.py): thirty-six
  marbles at real scale through a macro lens, lit by 616 fireflies in two
  swarms; moving fireflies, one level of placed boxes rebuilt each frame
  (2026-10-09-acceleration-structure.md); fireflies in flight and
  blinking; tone mapping with bloom; the dark opening
  (2026-10-10-dark-opening.md). The reference and its error
  (2026-10-10-reference.md): the headless renderer's PFM,
  serenity-measure, make reference and make convergence, the references
  at t = 120 and t = 8 with their floors, the two-reference check and the
  naive estimator's convergence. The time baseline, superseded 2026-10-11
  by the MIS path tracer (MEMORY.md): 30.04 ms at 3456 x 2234 on the
  marbles at t = 120 (2026-10-11-path-mis.md).

## Next, in order

0. **Done 2026-10-16.** Before Milestone 4 starts (the user, 2026-10-15):
   a codebase-wide pass, both C++ guideline corpora rule by rule over everything since Milestone
   1 (ReSTIR DI and PT, MNEE included), as Milestone 3's close did for its
   own code. Source done 2026-10-16
   (docs/research/2026-10-16-codebase-guidelines-pass.md): the core, the
   Metal host and the shaders reviewed and every finding decided, two
   shader defects fixed (a replayed shift's out-of-bounds shape read; MNEE's
   list overflow reported through a NaN), every graph the same to the bit.
   Tests done 2026-10-16 (that note, "Area D"): the tests' own review, every
   finding decided under the user's direction that tests guard invariants,
   not coverage; shared fixtures, failures reported rather than undefined,
   every graph's record() shown allocation-free, coverage-only tests
   removed; one src defect found and fixed (prepare_output's refusal).

0a. **Done 2026-10-17, on branch `cleanup`.** The codebase cleanup by the
   user's rules of 2026-10-17 (MEMORY.md, "Decided by the user
   2026-10-17"): ReSTIR DI removed, superseded by ReSTIR PT; the small
   denoiser network removed; checkerboard rendering no longer a setting
   (a denoised graph always traces its light checkerboarded, a graph
   without the denoiser every pixel); the path tracer's MNEE and the
   `mnee` key removed (ReSTIR PT always runs MNEE); the preview and
   display passes removed. Tests reworked into unit tests of units of
   code plus a handful of integration tests that render frames end to
   end; logic once checked by rendering moved into functions tested
   directly (src/headless/options.h, src/measure/commands.h). The graphs
   left: features, path, restir_pt, restir_pt_denoised (the demo),
   restir_pt_features and test_pattern.

0b. **Open, for the user's decision (found 2026-10-17, during the
   cleanup).** The ReSTIR PT graphs' headless PFM is not bit-reproducible
   run to run, even at the cleanup's starting commit c2dbd1f: a few
   pixels differ by about 1 ulp (at 480 x 270, 4 of 129,600 pixels, the
   largest 1.1e-7 relative), and the denoised graph's PNGs can differ by
   a level. The cause: MNEE's solve adds three or more requests' light to
   a pixel by float atomics, in the order the GPU schedules them
   (src/metal/passes/mnee/mnee.h, "Determinism", which had recorded it as
   measured reproducible on 2026-10-16). So byte identity holds for the
   path and features graphs, and for ReSTIR PT's only as a match at the
   noise's level. Folding MNEE's light in a fixed order would make it hold
   by construction; whether to is the user's call.

1. **Closed 2026-10-19, tagged `milestone-4`.** Milestone 4, making the frame fast
   (docs/research/2026-10-17-milestone-4-plan.md; docs/writeups/milestone-4.md,
   for the user's review before the tag): async compute overlap, ray
   masks and checkerboard rendering of the light, kept by the user; the
   split scene skipped on Metal, kept for the Vulkan and Xbox milestones.
   The network retrained on checkerboard input on the user's RTX 3090 Ti,
   new inputs over denoiser-v1's targets, kept by the user 2026-10-18.
   The frame before the production sweep: 16.7 ms at 1920 x 1080 (60
   fps), 56.1 ms in the window (17.8 fps); after it, 17.5 ms (57 fps) and
   58.5 ms (17.1 fps), accepted by the user.

   **Reopened 2026-10-18 by the production sweep, then resumed and
   closed out by the user's decisions.** A sweep of the whole frame
   against shipped engines (docs/research/2026-10-18-production-sweep.md)
   added four steps to Milestone 4, chosen by the user: depth of field
   after the denoiser (a pinhole trace; Jimenez 2014's gather, at half
   resolution as shipped games do), Halton(2,3) jitter with its offset as
   two network inputs, light tiles drawn by power (RTXDI), and frame
   pacing by the display link (30 to 60 frames a second preferring 60,
   behind a backend-neutral frame source). **Correction:** the earlier
   "retrain on checkerboard input" was not: the data-set input graph had
   no denoise pass, so its light was traced every pixel. Since the sweep,
   a graph with ReSTIR PT and the features pass is the network's input
   and traces as the demo does, checkerboard and fill included, proved to
   the bit by tests/gpu/features_test.cpp. denoiser-v2 rendered whole on
   the Mac (3.55 h, the user's last data-set render on this Mac: any
   later one waits for the RTX 3090 renderer), the network trained on it
   on the RTX 3090 Ti (3.5 h), tested against the data set's own targets
   (no reference renders, the user's decision), kept by the user
   2026-10-19.

2. **Closed 2026-10-22, tagged `milestone-5`.** Milestone 5, the frame on an NVIDIA GPU
   (docs/writeups/milestone-5.md): the frame on CUDA and OptiX on the RTX
   3090 Ti, then designed for its hardware
   (docs/research/2026-10-20-nvidia-hardware-design.md,
   docs/research/2026-10-21-tracing-divergence.md): 89 fps (11.2 ms) at the
   port, 173.9 fps (5.75 ms) at its close at 1920 x 1080, and 25 to 50.1 fps
   (39.6 to 19.96 ms) at 3456 x 2234; films of the opening rendered on the
   3090 (media/2026-10-22-*, not yet on R2). Open, the user's: f16
   accumulation in the denoiser, an MNEE iteration cap, split-K in the
   denoiser (each a picture change). The write-up revised 2026-10-22 against
   `andrew-voice` and Milestone 4: decisions and consequences lead the
   narrative, with the timing inventory left to the chart and research.
   Shortened again at Andrew's request to about 1,100 words, and
   accepted by him 2026-10-22. Next: Milestone 6 (below); then HIP with HIPRT on Strix
   Halo.
   The plan the milestone followed, from 2026-10-19: **the other backends, before any kernel
   tuning,** since the reuse kernels' tiling, layout and fusion depend on
   the hardware. Decided by the user 2026-10-19: each vendor's own API,
   no portable layer (MEMORY.md, "Metal now, Vulkan later", superseded).
   First CUDA with OptiX on the RTX 3090 Ti, headless first so data sets
   render there (the user's rule: no more data-set renders on the Mac);
   then its window; then HIP with HIPRT on AMD's Strix Halo; then
   Direct3D 12 on the Xbox Series X (where the split scene and console
   async compute return). The CUDA backend's plan,
   `docs/research/2026-10-19-cuda-backend.md`: Phases 0 to 5 done
   2026-10-20 (the reference graph, the data set's input graph and the
   demo graph render on the RTX 3090 Ti, the same pictures as the Mac's;
   the demo's frame 11.2 ms at 1920 x 1080 and 39.6 ms at 3456 x 2234,
   the Mac's 17.37 and 58.30). Phase 6 done 2026-10-20: the data-set
   script runs on the 3090 (trialled, resumed and verified; the whole plan
   about 2.6 h there), so the next data set renders there. The 3090's
   window done 2026-10-20 (`docs/research/2026-10-20-cuda-window.md`:
   SDL3 on X11, Vulkan presenting CUDA's frames, `make window`; the demo
   live at 60 fps at 1920 x 1080, 14 ms of GPU a frame); on main
   2026-10-20, untagged. Next (the user, 2026-10-20): the frame designed
   for the 3090's hardware (`docs/research/2026-10-20-nvidia-hardware-design.md`:
   the scene's shapes as triangles on the RT cores, one instance level,
   fused passes, the denoiser on the tensor cores' own shape, RTXDI's
   reservoir layout), which closes Milestone 5; then HIP with HIPRT on
   Strix Halo.

3. **Closed 2026-10-27, tagged `milestone-6`.** Milestone 6, the frame
   designed for Apple silicon (the user, 2026-10-22; docs/writeups/milestone-6.md).
   Changes 1 to 5 built (`docs/research/2026-10-23-apple-silicon-design.md`:
   62.6 fps at 1920 x 1080, 17.6 at 3456 x 2234), then the network on the
   Neural Engine (2026-10-27, `docs/research/2026-10-26-neural-engine-denoiser.md`,
   section 5): 84.2 fps (11.87 ms) and 25.1 fps (39.89 ms); MetalFX
   Denoising measured and not kept. The 1080p frame is its light half
   (11.27 of 11.89 ms); three frames in flight not taken. Closed by the
   user 2026-10-27 as the limit for this frame on the M3 Max's ray
   tracing, Cycles' wavefront left untried on the M3. The write-up, with
   diagrams of the hand-off and the MetalFX comparison, accepted by
   Andrew 2026-10-27. The films and the site published 2026-10-27 by the
   user's approval: the homepage plays Milestone 6's opening rendered on
   the M3 Max (45 s at 60 fps, crf 16 capped, about 7 Mb/s at 1080p60);
   the sixth article carries Milestone 4 beside Milestone 6 on the same
   M3 Max (30 s at 30 fps, labelled 57 and 84 fps) and its diagrams with
   phone versions; About, Writing and the feed cover six milestones. The
   8 new objects on R2 checked against the local bytes; Milestone 5's
   homepage film stays on R2, unlinked. Next: HIP with HIPRT on Strix
   Halo.
   The plan as it was written: The M3 Max runs
   the demo at 57 fps at 1920 x 1080 and 17 at the display's 3456 x 2234,
   against the project's goal of 60 at the display's size; Milestone 4
   applied engines' standard optimizations there, and no design for the
   M3's own hardware has been made, as Milestone 5 made one for the 3090.
   The last breakdown (`docs/research/2026-10-17-milestone-4-plan.md`,
   before Milestone 4, 77 ms at 3456 x 2234): the light's passes about 44
   ms, the ray-tracing unit about 90% busy, the arithmetic 12 to 15%,
   occupancy 16 to 22%; the denoiser 26 ms, arithmetic-bound at 97%. The
   order, as the 3090's:

   0. **The profile, when the user says** (on AC power, the GPU idle, no
      window running): the demo's frame time headless (`--times`, frames
      7000 to 7599) at 1920 x 1080 and 3456 x 2234; each pass's GPU time
      (per-dispatch timestamps); the Performance Limiters counters per
      pass (the saved Instruments template through `xctrace`); each
      pipeline's registers and occupancy as the compiler and the GPU
      report them.
   1. **The design, written before code**
      (`docs/research/2026-10-23-apple-silicon-design.md`): the M3 Max's
      hardware (ray-tracing unit, Dynamic Caching's on-chip memory,
      SIMD groups of 32, 16-bit arithmetic and registers, simdgroup
      matrices); Apple's own guidance (WWDC's M3 profiling and ray
      tracing sessions, Metal 4's) and production on Apple silicon
      (Blender Cycles' Metal backend, developed with Apple; Unreal's
      Metal RHI; Apple's samples), checked first; each change's hardware
      reason and its expected gain stated before it is measured.
   2. **The changes, in the order the profile ranks them; the candidates
      now, each from the 3090's design where it applies:**
      - *Every shape as triangles on the ray-tracing unit, again.* The
        earlier test (`docs/research/2026-10-09-acceleration-structure.md`)
        was the brass scene's 4 shapes in the path tracer, where the
        instance level cost more than the shader's sphere test; the
        marbles' 661 spheres each leave the ray-tracing unit for an
        intersection function, on a unit already the bottleneck. Measured
        in the layouts that answer it: the 3090's (a mesh a level, every
        shape an instance); the scene split (the still marbles' meshes
        placed in one bottom level built once, the fireflies' in another
        rebuilt or refit each frame, a few instances); one flat level of
        placed triangles. The hit point the triangle's (Falcor's), the
        meshes' bound a thirtieth of a pixel, as on CUDA.
      - *MNEE's walk in rounds* (the 3090's: its solve 2.6 to 1.0 ms).
      - *Both reuses over lists of pairs* (ReSTIR PT Enhanced 6.2.2).
      - *Occupancy:* each pipeline's registers against Dynamic Caching;
        16-bit values where they hold the same answer (Apple's guidance);
        threadgroup sizes.
      - *The denoiser:* its simdgroup-matrix shape and tile copies, half
        inputs with float sums as now; Metal 4's tensor operations
        considered; f16 sums stay the user's call.
      - *RTXDI's reservoir layout* (16 x 16 blocks; the sizes are the
        core's already) and the shade in the spatial merge, as on CUDA.
   3. Each change its own commit: the demo's frame time before and after
      at both sizes, Metal's 94 outputs the same to the bit where the
      picture should not change, a changed picture shown side by side,
      quality trades the user's. The gates once, at the end; then the
      write-up, the films (rate and length from the live rate and the
      scene), `milestone-6`, main and the site.

4. Reservoir reuse as kernels, tuned per target: tiling, layout and
   fusion of the merge passes, what each costs and where the bytes go, on
   each backend.

Not scheduled: the denoiser's next training round (the user, 2026-10-15:
no retrain for now). Its planned investment is heavier sampling of the
targets; with it, the temporal weight measured, a longer cycle, and, if
wanted, earlier opening sequences in the data set (Parking Lot:
"Training", "Data set").

## Parking Lot

- Ray offset: one cat's-eye rim still moves ~4% between the chapter's step and one 4x smaller (pbrt-v4 bounds would shrink it).
- Ray offset: step 5's T is over the shadow ray, short of the surface's point by one step (only for a light in the surface's medium).
- Ray offset: +0.3 ms is `offset_ray()` itself; GPU counters for why a cheaper float form was no faster.
- Ray offset: ReSTIR PT's visibility offsets along the primary hit's quantized normal (primary_hit.h); no test aims at it.
- Ray offset: a sphere ~100 m+ in radius whose surface passes near the origin re-hits (fixed 15 um step there); no scene has one.
- ReSTIR temporal: a fixed-order ring around the reprojected pixel moved a grazing floor's mean +5.6% (measured on ReSTIR DI, removed 2026-10-17; the ring order, reuse/temporal.metal.h, is ReSTIR PT's now); a frame-drawn start fixed it; the theory says unbiased either way: why.
- Path MIS asks a light's density of the direction (light_pdf); on a firefly's rim that test rounds out, as ReSTIR DI's BSDF candidate found (light_point_pdf); check whether the path tracer counts rim hits twice.
- ReSTIR PT: roulette's survival folded into the source density has no test of its own; the marbles' accumulated mean holds it (0.186420 against the reference's 0.186422).
- ReSTIR PT: the replay's indirect grid is ordered by a dispatch-to-dispatch barrier; confirm Metal 4 counts the indirect-argument read in that stage.
- ReSTIR PT: a reconnection segment is dimmed by y_{k-1}'s medium, the cached suffix by x's; they differ only for a rough surface inside an absorbing medium (no scene has one).
- ReSTIR PT: eq. 3's signs read as (+, -, +) from the extracted text, the one reading that meets the paper's footnote (sigma 0.8, one shuffle); check against the PDF.
- ReSTIR PT: path_merge's weight functions take four floats in a row (I.24); a struct if they grow.
- ReSTIR PT: the tree's gather costs the initial pass 14 ms over the path pass at 3456 x 2234 with one NEE candidate (44 against 30 ms); GPU counters for where (the reservoir's stores, the criteria's footprints, the shape points' encoding).
- ReSTIR PT: a shift's shadow ray is most of its cost (the spatial pass 30 ms, 13.7 of it the reconnections' visibility, 5.7 the replays); the RT unit on bounding-box shapes, counters.
- ReSTIR PT: the duplication map costs 4.1 ms at 3456 x 2234 (3%); 32-bit keys in two arrays were slower (130.4 against 128.7 ms); a count shared along rows.
- ReSTIR PT: the temporal shift of a pixel's path into last frame's domain traces this frame's structure (a firefly that crossed the ray last frame and not this one is missed); keep last frame's structure and trace it, as RTXDI's previous-frame BVH (restir_pt.metal.h, "What is approximate"). The design, first written for ReSTIR DI's raytraced mode (2026-10-16-codebase-guidelines-pass.md, G1): a still scene's one structure is last frame's too; a moving scene keeps frames_in_flight + 1 structures, three: frame i builds structure i mod 3 (its index's, not its slot's, since a readback takes a slot) and traces (i - 1) mod 3 as last frame's; before rebuilding one it waits for the last submission that traced it (GPU.7, MEM.4). The cost is one more structure and its scratch, made at construction, no extra build.
- ReSTIR PT: a reconnection keeps its suffix after x_k as its source frame lit it (contract 15's cached radiance, as the 2022 code); a moved firefly's light lingers after x_k until resampled away, bounded by the cap.
- ReSTIR PT: the temporal reuse's previous position under a moving primary surface has no test; the marbles' surfaces are still.
- ReSTIR PT: dual motion vectors (Enhanced §6.4) for disocclusions under camera motion; the marbles' camera is fixed.
- MNEE: from the straight seed Newton diverges over most of a ball's caustic rim (18 of 65 floor points converge; -6.5% of the light in a glass sphere's shadow; glass and metal's mean 0.94 of the reference at t = 120 in ReSTIR PT's graph, 0.98 without MNEE); Cycles' algorithm, a seed nearer the rim's path if it shows.
- MNEE: G is not clamped (Cycles' clamp at 2 is in meters); caustic folds give fireflies (glass and metal's linear MSE at t = 8 3.7 -> 13.3 with MNEE, displayed level); a clamp relative to the straight segment's G if they show.
- MNEE: conductors are not receivers (near-mirror lobes with no MIS); their caustics through glass stay the path tracer's.
- Features: motion is the primary surface's; what glass and mirrors show behind or in them moves differently (production's specular motion vectors, NRD's virtual motion under PSR): not given.
- Features: one ray a pixel a frame, as the noisy input's own; the 1920 x 1080 target's albedo is not averaged over its 4,096 samples (Open Image Denoise: aux images "should use the same pixel reconstruction filter as the beauty"): average it if the network's output shows edge or depth-of-field artifacts.
- Features: glass beyond the first hit follows one branch by Fresnel's larger weight; a marble seen through another at near-grazing angles snaps between branches (Open Image Denoise suggests albedo 1 there if it is noisy).
- Features: the pass costs ReSTIR PT 1.2 ms at 960 x 540 (12.5 -> 13.7, +10%); written by the initial pass from its primary hit, it would save the camera ray; Phase 4 measures it in the denoised frame.
- Data set: no training sequence starts before 3 s (the opening's draws, 2.97 to 6.97 s); the test's t = 2 is earlier than any; add earlier sequences if Phase 3's validation shows the first fireflies waking poorly.
- Data set: 4,096-sample targets still sparkle inside glass and cat's-eye cores (path tracer, no MNEE); denoise or deepen those targets if the network learns the speckle. Heavier targets are the next training round's planned investment (the user, 2026-10-15); at denoiser-v1 the targets' own error is a tenth of the network's (2026-10-15-denoiser-training.md).
- Denoiser: the full-size albedo retraces the features kernel at the output size each frame (Phase 4's note gives its cost); an albedo-only variant saves writes, not rays; reusing the light's features upsampled would not be the training's input.
- Denoiser: half-precision activations change 0.7 to 0.9% of displayed pixels by one level (up to 9 in a few) against PyTorch's float32 on real frames; float activations if a difference is ever seen.
- Denoiser: the export's SHA256SUMS is not checked at load (sizes, shapes, constants and finite values are).
- Data set: the test inputs use frame indices 0 to 7,207; Phase 4's 65,536-sample references for scoring must start past them (make reference batches from index 0).
- Training: the table flickers 36 to 47% more than the targets once every firefly is awake (and on validation); the temporal term is under 3% of the loss at Chaitanya et al.'s 0.125. Measure 0.125 x 2, x 4, x 8: validation flicker against validation error (2026-10-15-denoiser-training.md). Decided by the user 2026-10-15: no retrain now; it waits for the next training round.
- Training: validation still fell about 0.0005 per 50 epochs through the anneal; a 600-epoch cycle to see if it keeps falling.
- Training: Thomas et al. 2022's loss weights and sub-pixel handling unread (the PDF could not be fetched); check them against loss.py and model.py.
- Training: MPS runs from one seed agree to four figures, not to the bit.
- ReSTIR PT's radiance is not bit-reproducible on the GPU: reruns of a frame differ in 1 to 9 pixels by up to 2.4e-7 (features exact). An order-dependent sum found 2026-10-17, most likely this one: MNEE's float atomics (Next, 0b, for the user's decision).
- Denoiser: the design's comparison with MetalFX's denoiser, "if available", was not made; Open Image Denoise was the yardstick. Whether it is still wanted is the user's call.
