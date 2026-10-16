# Work Queue

This file tracks active and accepted work.

## Active

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
  graphs/restir_pt_denoised_small.toml), and no retrain for now, the
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
  same to the bit. Next: the milestone's close.

- **ReSTIR PT** (docs/research/2026-10-11-restir-pt-layout.md). Phase 1
  built 2026-10-11 (initial sampling, the hybrid shift, paired spatial
  reuse): docs/research/2026-10-11-restir-pt-phase1.md, its gate not met.
  Phase 2 built 2026-10-11 by the user's direction (temporal reuse by
  Talbot MIS, the duplication map, a G-buffer of primary hits, light tiles;
  graphs/restir_pt.toml is the whole of Enhanced at its defaults):
  docs/research/2026-10-11-restir-pt-phase2.md. Manifold NEE done
  (2026-10-12/13, by the user's direction; branch mnee-wip): behind a
  graph's `mnee`; graphs/restir_pt.toml sets it, by the equal-time
  comparison (glass and metal displayed error -16% at t = 120, -5% at
  t = 8); graphs/path.toml does not and stays the reference:
  docs/research/2026-10-12-mnee.md. Then Phase 3's references.

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

0. Before Milestone 4 starts (the user, 2026-10-15): a codebase-wide pass,
   both C++ guideline corpora rule by rule over everything since Milestone
   1 (ReSTIR DI and PT, MNEE included), as Milestone 3's close did for its
   own code.

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
continuation's spikes gone; the review of 7751a65 answered 2026-10-11 on
branch mis-review: both strategies drawn and weighed from the point rays
leave from, the power heuristic without overflow, the probes' layouts one
header per probe, that note's "Review of 7751a65"; the ray offset sized to
the point's rounding (Wachter and Binder, hit points put back on their
shapes) done 2026-10-11, docs/research/2026-10-11-ray-offset.md, the
references at t = 120 and t = 8 to render again once, when ReSTIR DI
is finished (not before), with the convergence runs after them; Phase
B done 2026-10-11 on branch restir-di: RTXDI's BSDF candidate, weighed
against the light candidates by MIS, and the temporal merge in basic
mode, graphs/restir_temporal.toml, docs/research/2026-10-11-restir-
temporal.md; Phase C done 2026-10-11 on branch restir-di: the spatial
merge, RTXDI's k = 1 within 32 pixels, basic mode and raytraced alone,
graphs/restir.toml (ReSTIR DI whole) and graphs/restir_spatial.toml,
docs/research/2026-10-11-restir-spatial.md; left: the references
rendered again and the convergence runs); ReSTIR GI; caustics by manifold next
event estimation; reservoir reuse as kernels; a denoiser of its own against
Open Image Denoise and MetalFX; the Vulkan backend on the AMD machine.

## Parking Lot

- **Whether MoltenVK supports Vulkan ray tracing.** Believed not; verify
  before the Vulkan backend, since it decides whether that backend could also
  run on the Mac.
- Ray offset: one cat's-eye rim still moves ~4% between the chapter's step and one 4x smaller (pbrt-v4 bounds would shrink it).
- Ray offset: step 5's T is over the shadow ray, short of the surface's point by one step (only for a light in the surface's medium).
- Ray offset: +0.3 ms is `offset_ray()` itself; GPU counters for why a cheaper float form was no faster.
- Ray offset: ReSTIR's visibility offsets along the record's 16-bit-quantized normal; no test aims at it.
- Ray offset: a sphere ~100 m+ in radius whose surface passes near the origin re-hits (fixed 15 um step there); no scene has one.
- ReSTIR temporal: raytraced mode is refused with restir_temporal; it needs last frame's structure kept (scene_acceleration.h, Keep::last_too).
- ReSTIR temporal: a fixed-order ring around the reprojected pixel moved a grazing floor's mean +5.6%; a frame-drawn start fixed it; the theory says unbiased either way: why.
- ReSTIR temporal: basic mode's edge bias at a grazing view, a box's shadow edge -7% to +10% by row, the lower floor -3% (restir-temporal.md).
- ReSTIR temporal: the boiling filter (RTXDI's default, 0.2) still deferred.
- Path MIS asks a light's density of the direction (light_pdf); on a firefly's rim that test rounds out, as ReSTIR's BSDF candidate found (light_point_pdf); check whether the path tracer counts rim hits twice.
- ReSTIR's BSDF candidate costs 7 ms at 3456 x 2234 (+19% on the RIS graph); its ray and 8 BSDF densities are most of it.
- ReSTIR DI loses to the MIS path tracer at equal time on the marbles (MSE x 1.4 to 1.7 at 1 s): measure the direct light's error alone, and at the traced resolution.
- ReSTIR spatial: graphs/restir.toml in basic mode is -1.45% against the reference's mean at t = 120 (-0.33% at t = 8), each merge alone within 0.16%; check against raytraced once last frame's structure is kept (restir-spatial.md).
- ReSTIR spatial: the pass costs 3.4 ms at 3456 x 2234; GPU counters against its counted traffic (gathers within the radius; a tiled order if line-bound).
- bsdf_candidates = 0 is Phase A but for rim points, now weighed where Phase A's direction test dropped them; not bit-identical.
- ReSTIR PT: roulette's survival folded into the source density has no test of its own; the marbles' accumulated mean holds it (0.186420 against the reference's 0.186422).
- ReSTIR PT: the replay's indirect grid is ordered by a dispatch-to-dispatch barrier; confirm Metal 4 counts the indirect-argument read in that stage.
- ReSTIR PT: a reconnection segment is dimmed by y_{k-1}'s medium, the cached suffix by x's; they differ only for a rough surface inside an absorbing medium (no scene has one).
- ReSTIR PT: eq. 3's signs read as (+, -, +) from the extracted text, the one reading that meets the paper's footnote (sigma 0.8, one shuffle); check against the PDF.
- ReSTIR PT: path_merge's weight functions take four floats in a row (I.24); a struct if they grow.
- ReSTIR PT: the tree's gather costs the initial pass 14 ms over the path pass at 3456 x 2234 with one NEE candidate (44 against 30 ms); GPU counters for where (the reservoir's stores, the criteria's footprints, the shape points' encoding).
- ReSTIR PT: a shift's shadow ray is most of its cost (the spatial pass 30 ms, 13.7 of it the reconnections' visibility, 5.7 the replays); the RT unit on bounding-box shapes, counters.
- ReSTIR PT: the duplication map costs 4.1 ms at 3456 x 2234 (3%); 32-bit keys in two arrays were slower (130.4 against 128.7 ms); a count shared along rows.
- ReSTIR PT: the temporal shift of a pixel's path into last frame's domain traces this frame's structure (a firefly that crossed the ray last frame and not this one is missed); keep last frame's structure (scene_acceleration.h, Keep::last_too) and trace it.
- ReSTIR PT: a reconnection keeps its suffix after x_k as its source frame lit it (contract 15's cached radiance, as the 2022 code); a moved firefly's light lingers after x_k until resampled away, bounded by the cap.
- ReSTIR PT: the temporal reuse's previous position under a moving primary surface has no test; the marbles' surfaces are still.
- ReSTIR PT: dual motion vectors (Enhanced §6.4) for disocclusions under camera motion; the marbles' camera is fixed.
- MNEE: from the straight seed Newton diverges over most of a ball's caustic rim (18 of 65 floor points converge; -6.5% of the light in a glass sphere's shadow; glass and metal's mean 0.94 of the reference at t = 120 in ReSTIR PT's graph, 0.98 without MNEE); Cycles' algorithm, a seed nearer the rim's path if it shows.
- MNEE: G is not clamped (Cycles' clamp at 2 is in meters); caustic folds give fireflies (glass and metal's linear MSE at t = 8 3.7 -> 13.3 with MNEE, displayed level); a clamp relative to the straight segment's G if they show.
- MNEE: `shift_path()` checks `settings.mnee` at run time: +1.2% to ReSTIR PT without MNEE at 3456 x 2234; a template parameter, as the tree has.
- MNEE: the MNEE pass costs the path graph +28 ms at 3456 x 2234 (30.4 -> 58.7); GPU counters on mnee_solve (iterations, registers).
- MNEE: the path tracer + MNEE at 1 sample loses to the path tracer at 2 at equal time (displayed 0.82 to 1.01); graphs/path_mnee.toml stays opt-in.
- MNEE: conductors are not receivers (near-mirror lobes with no MIS); their caustics through glass stay the path tracer's.
- Features: motion is the primary surface's; what glass and mirrors show behind or in them moves differently (production's specular motion vectors, NRD's virtual motion under PSR): not given.
- Features: one ray a pixel a frame, as the noisy input's own; the 1920 x 1080 target's albedo is not averaged over its 4,096 samples (Open Image Denoise: aux images "should use the same pixel reconstruction filter as the beauty"): average it if the network's output shows edge or depth-of-field artifacts.
- Features: glass beyond the first hit follows one branch by Fresnel's larger weight; a marble seen through another at near-grazing angles snaps between branches (Open Image Denoise suggests albedo 1 there if it is noisy).
- Features: the pass costs ReSTIR PT 1.2 ms at 960 x 540 (12.5 -> 13.7, +10%); written by the initial pass from its primary hit, it would save the camera ray; Phase 4 measures it in the denoised frame.
- Features: the window passes the frame before but no window graph has the features pass yet.
- Data set: no training sequence starts before 3 s (the opening's draws, 2.97 to 6.97 s); the test's t = 2 is earlier than any; add earlier sequences if Phase 3's validation shows the first fireflies waking poorly.
- Data set: 4,096-sample targets still sparkle inside glass and cat's-eye cores (path tracer, no MNEE); denoise or deepen those targets if the network learns the speckle.
- Denoiser: the full-size albedo retraces the features kernel at the output size each frame (Phase 4's note gives its cost); an albedo-only variant saves writes, not rays; reusing the light's features upsampled would not be the training's input.
- Denoiser: half-precision activations change 0.7 to 0.9% of displayed pixels by one level (up to 9 in a few) against PyTorch's float32 on real frames; float activations if a difference is ever seen.
- Denoiser: the export's SHA256SUMS is not checked at load (sizes, shapes, constants and finite values are).
- Data set: the test inputs use frame indices 0 to 7,207; Phase 4's 65,536-sample references for scoring must start past them (make reference batches from index 0).
- Training: the table flickers 36 to 47% more than the targets once every firefly is awake (and on validation); the temporal term is under 3% of the loss at Chaitanya et al.'s 0.125. Measure 0.125 x 2, x 4, x 8: validation flicker against validation error; the user decides (2026-10-15-denoiser-training.md).
- Training: validation still fell about 0.0005 per 50 epochs through the anneal; a 600-epoch cycle to see if it keeps falling.
- Training: the network reads view-space normals (data.py); Phase 4 transforms the features pass's world normal by the camera's axes, or the features pass writes view space.
- Training: small or wide for Phase 4 by their Metal cost (small 1 to 9% higher displayed error, 56% of the U-Net's multiply-adds; PyTorch fp16 U-Net 7.2 against 9.6 ms); both exported.
- Training: Thomas et al. 2022's loss weights and sub-pixel handling unread (the PDF could not be fetched); check them against loss.py and model.py.
- Training: MPS runs from one seed agree to four figures, not to the bit.
- ReSTIR PT's radiance is not bit-reproducible on the GPU: reruns of a frame differ in 1 to 9 pixels by up to 2.4e-7 (features exact); find the order-dependent sum.
