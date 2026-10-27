# Project Memory

This file is the canonical entry point for durable project context.

## Product Identity

- **Product:** Serenity. Named 2026-09-30.
- **Operating brand:** None. Internal R&D under the parent entity.
- **Parent entity:** Plainsight Systems LLC
- **Repository:** <https://github.com/plainsight-systems/serenity> (public).

## Purpose

A real-time path tracer: simple procedural geometry lit by hundreds to
thousands of moving lights, with the lighting correct as everything moves.
Many-light sampling and the reuse of whole light paths by ReSTIR PT
(ReSTIR PT Enhanced, Lin, Kettunen and Wyman 2026), caustics through
glass by manifold next event estimation, a denoiser of its own compared
against Intel Open Image Denoise, and a reference render at thousands of
samples per pixel to compare against. No art assets: the light
carries the image.

The project's brief is internal planning and is not in this repository.

## Inherited Governance

Canonical list, public URLs, and status: `inherited.md`.

- `engineering_philosophies.md`
- `product_memory_workflow.md`
- `repo_creation_runbook.md`

## C++ Governance

This product is C++-dominant and performance-sensitive. All four C++ docs are
inherited and both gates bind:

- `cpp_architecture_playbook.md`
- `cpp_architecture_review.md`
- `cpp_performance_playbook.md`
- `cpp_performance_review.md`

## Locked Decisions

Decided 2026-10-08, at the repository's creation:

- **C++20, native, on the development machine first.** One target until the
  second backend exists: an Apple M3 Max, through Metal, on the toolchain
  pinned in `cmake/toolchain.json`. Every figure names the machine and build
  it came from; nothing is claimed for hardware it did not run on.
- **Metal now, Vulkan later, as two thin backends with no shared GPU
  interface.** `src/metal/` now; a Vulkan backend for the AMD machine
  (Strix Halo) later, beside it. What the backends share is the
  platform-neutral core: the scene, the sampling math, and the CPU
  computations the GPU tests check against. Shaders are written per backend.
  Alternative considered: Unreal's model, a render hardware interface over
  every API with shaders written once in HLSL and converted for each
  (checked against Epic's documentation, 2026-10-08). Rejected for this
  project: it pays off across many platforms and thousands of shaders, and
  here there are two machines and a handful of kernels whose per-vendor
  differences are the subject. Revisit when the Vulkan backend exists and
  the cost of the duplication can be measured.
  Superseded by the user 2026-10-19, at the start of the backends: **each
  vendor's own API, with no portable layer over vendors.** Metal on the
  Apple M3 Max; CUDA with OptiX on the NVIDIA RTX 3090 Ti; HIP with HIPRT
  on AMD's Strix Halo; Direct3D 12 on the Xbox Series X. Shaders and
  kernels are written in each API's own language, with no one-source layer
  (Slang, HLSL cross-compiled, Metal Shader Converter). The reason: each
  backend's kernels are tuned for its hardware next, and a layer that
  emits code for several vendors (a shader language, or a portable API
  such as Vulkan) stands between the tuning and the hardware. CUDA reaches
  PTX and the tensor cores' matrix instructions, OptiX the RT cores, and
  CUDA is where attention kernels are written, which makes the demo's
  parallel literal on that machine. A window on the 3090 needs a thin
  bridge to a graphics API for presentation only; headless needs none.
  Vulkan is not a target. The guard across backends is the core's CPU
  computations the GPU tests check each backend against, not output
  matched to the bit between machines.
- **Metal 4** for the host API and the shading language (`-std=metal4.0`),
  not Metal 3. Its explicit command allocation, residency and barriers are
  close to Vulkan's model, which the second backend will need anyway; its
  machine-learning encoder and in-shader tensors are where the denoiser's
  inference can run inside the frame; and MetalFX's denoised upscaler is
  there as a second baseline beside Open Image Denoise. Amended 2026-10-15
  (Milestone 3, Phase 4): the denoiser's convolutions run as our own
  compute kernels on simdgroup matrices inside the frame's encoder, not
  through the machine-learning encoder, whose networks are packages built
  offline with fixed input sizes and whose kernels on the M3 Max we could
  neither tune nor match to PyTorch by construction
  (`research/2026-10-15-denoiser-live.md`).
- **The core is platform-neutral; only `src/metal/` uses Metal's host API.**
  Enforced by `tools/check_boundaries.sh`, whose rules are proven to fire by
  `tests/test_check_boundaries.sh`.
- **The toolchain is pinned by version and checked when the build is
  configured.** Metal cannot run in a container, so it cannot be pinned by
  an image as Charlotte pinned emsdk. `cmake/toolchain.json` records Xcode,
  the macOS SDK the build compiles against (the presets name it,
  `CMAKE_OSX_SYSROOT`), the Metal compiler, Apple clang as each of the C,
  C++, Objective-C and Objective-C++ compilers CMake resolved (Objective-C++
  since 2026-10-18, for the display link, `src/metal/device/
  display_link.mm`), and CMake itself;
  configuring fails on any mismatch. Moving the pin is a commit of its own, with the tests rerun.
- **Shaders are compiled at build time with pinned flags and compiled into
  the executable** (`cmake/MetalLibrary.cmake`). Never compiled from source
  at run time, never loaded from a path.
- **Correctness is shown on the GPU, the way the work will be shown.**
  ReSTIR against a reference rendered at thousands of samples per pixel: the
  error falls to zero as samples grow. The reservoir merge against softmax:
  fed attention scores, it picks each key with probability exp(score) / sum.
  No CPU copies of kernels, and no limits on shader math for a CPU to check
  against; shaders compile with Metal's default math. Supersedes, the same
  day, a contract for bit-equal CPU and GPU math: it served only tests, cost
  the renderer's math, and showed nothing the goal needs. What it found is
  kept in `research/2026-10-08-metal-math-modes.md`.
- **The target is the look of *Marbles at Night*:** full path tracing of
  glass and polished metal lit by thousands of moving fireflies, with
  indirect light and caustics. ReSTIR DI for direct light, ReSTIR GI for
  indirect, manifold next event estimation for caustics, each in the
  configuration production ships, judged against one reference by error x
  time. Amended 2026-10-10, by the user's direction: an IC demo performs as
  real-world software does, so each estimator's default is a shipping
  system's (ReSTIR DI's is RTXDI's, `basic` bias correction among it), and
  its unbiased, textbook-exact form is a setting, kept to check it against
  the reference (`research/2026-10-10-restir-di-layout.md`). Superseding
  "all unbiased against one reference". The window shows the display's full resolution at
  60 fps, traced lower and upscaled with MetalFX
  (`../architecture/logical-overview.md`). Amended by what Milestones 2
  and 3 built: ReSTIR PT (Enhanced, at its defaults) took the place of
  ReSTIR GI, reusing direct and indirect light as whole paths
  (`research/2026-10-11-restir-pt-layout.md`); and the upscale is the
  denoiser's own, 2x each way, not MetalFX's (below, "Milestone 3").
  Amended 2026-10-17, by the user's direction: ReSTIR DI is removed,
  superseded by ReSTIR PT, and its modes (`basic`, `raytraced`) with it;
  the estimators are the path tracer (the reference), ReSTIR PT with
  manifold next event estimation, and the denoised pipeline (below,
  "Decided by the user 2026-10-17").
- **A window and a headless renderer arrive together**, in the first
  rendering change. The headless renderer writes frames at a fixed timestep;
  the window shows the same frames live.
- **The design of a change lives in its file headers, from the start.** No
  packets. Carried over from Charlotte's decision of 2026-10-02, along with
  designing optimizations in rather than deferring them, and deriving a data
  path's cost before building it (Charlotte, 2026-10-03). `workflow.md`
  states each.
- **The architecture documents come before rendering code**: logical
  overview, then change axes, then file mapping (`../architecture/`).

Amended 2026-10-11, by the user's direction:

- **The path tracer weighs light sampling against BSDF sampling by multiple
  importance sampling**, as production path tracers do (pbrt-v4's
  PathIntegrator rather than its SimplePathIntegrator; Falcor). At a surface
  with a lobe that is not delta, a firefly's light is found by next event
  estimation and by a BSDF-sampled ray striking it, each weighted by the
  power heuristic; after a delta lobe and on the camera's own ray, in full;
  the sky one way, as before; ReSTIR DI's first surface the exception, its
  direct light the reservoir's alone (the exception removed with ReSTIR
  DI, 2026-10-17). The reason: light sampling alone, at
  the marbles' near-mirror steel (GGX alpha 10^-3), returns rare samples of
  10^5 whose variance grows as 1 / alpha^2
  (`research/2026-10-11-restir-ris-results.md`). Supersedes two earlier
  decisions:
  - **principle 7's one-way counting**, "every light path is counted once,
    no path found two ways" (`../architecture/logical-overview.md`, which
    states the new principle 7);
  - **the time baseline as "the textbook kernel"**
    (`research/2026-10-10-corpus-sweep.md`, "Decision";
    `../writeups/milestone-1.md`, which records why it was kept so at the
    time): the baseline every estimator is measured against is now the MIS
    path tracer, measured again (`research/2026-10-11-path-mis.md`).

  The path pass's images change; their expectation does not. Its
  bit-identity with the build before it, held through every change until
  now, is broken by this change alone, on purpose, and held instead by
  converging to the same references, which stay valid.

## Milestone 3: the denoiser

Decided with the user 2026-10-14 (`research/2026-10-14-denoiser-training-design.md`):

- **A kernel-predicting network that also upscales 2x each way.** Per
  output pixel, scores over 3 x 3 neighbors of the noisy lighting at
  spacings 1, 2 and 4, softmax-normalized, and a blend with its last
  output moved by the motion vectors: the scores computed fresh from each
  frame, as attention computes its weights. Upscaling from the start: the
  light is traced at half the output each way, which replaces MetalFX's
  upscaler in the locked decision above.
- **PyTorch for training** (Open Image Denoise's toolkit is PyTorch); MLX
  stays the alternative, since one Metal kernel could then serve training
  and the renderer. The toolkit is `training/`, outside the C++ build, with
  its own pinned environment (Python 3.12.15, `uv.lock` with hashes, torch
  2.14.1): a dependency declared as one, which nothing the renderer builds
  or runs reads (`research/2026-10-15-denoiser-training.md`).
- **The real marbles are never trained on.** They are the test, scored
  once after training; training is on scenes generated by
  `scripts/make_marbles.py --seed` (101 to 140, never 7) and the brass
  sphere scenes; validation (4 of the generated scenes) selected the best
  checkpoint during fixed 300-epoch runs and informed the learning rate.
- **Data and models live beside the checkout, never in it:**
  `../serenity-data/denoiser-v1` (the data set, 32.3 GB, a manifest with
  every file's SHA-256), `denoiser-v1-warmup`, `denoiser-v1-runs` and
  `denoiser-v1-model/{wide,small}` (the exported networks the graphs name;
  since 2026-10-17 the renderer names `wide` alone, below).

Decided by the user 2026-10-15:

- **The targets' sample count.** denoiser-v1 trains on its 4,096-sample
  targets; heavier sampling of the targets is the planned investment for
  the next training round.
- **The network runs on our own Metal kernels** (the Metal 4 amendment
  above; `research/2026-10-15-denoiser-live.md`).
- **The wide network is the demo's** (543k parameters,
  `graphs/restir_pt_denoised.toml`), chosen after watching both live; the
  small one (294k) stays, as `graphs/restir_pt_denoised_small.toml`.
  Amended 2026-10-17: the small one is removed (the graph, its pinned
  files in `scripts/fetch_denoiser_weights.sh`, `training/config.json`);
  the wide network (`training/config-wide.json`) is the demo's, and `make
  weights` fetches it alone (below, "Decided by the user 2026-10-17").
- **No retrain for now:** the improvement is visible. The heavier
  temporal weight (the table flickers 36 to 47% more than the targets)
  waits for the next training round, measured then (QUEUE.md).
- **A codebase-wide guidelines pass before Milestone 4:** both C++ corpora
  rule by rule over everything since Milestone 1, as Milestone 3's close
  did for its own code (`research/2026-10-15-milestone-3-guidelines-pass.md`).

Decided with the user 2026-10-14: reservoir reuse as kernels, the
optimization milestone (what each pass costs and where the bytes go),
waits until the full pipeline exists.

- **Milestone 3 editorial revision, 2026-10-16:**
  `docs/writeups/milestone-3.md` now follows building, training and running
  the reconstruction model inside the renderer. The held-out scenes support
  its quality claims; memory reuse explains the kernel improvements. The
  table reports displayed MSE, with frame costs separate. Revised at Andrew's
  request and accepted by him 2026-10-16, with the milestone's close.

## Project Site Direction

Clarified by the user 2026-10-12:

- The initial site published **milestone 1**. On 2026-10-13 Andrew accepted
  the milestone-two writeup, committed the renderer and article to `main`,
  and requested the site and README advance to **milestone 2**. This explicit
  publication decision supersedes the initial branch-only ReSTIR status.
- Add a GitHub Pages project site with Charlotte and Cradle as references.
  The Hugo site was implemented on 2026-10-12. Pages is configured for
  `serenity.plainsight-systems.com`, with the domain verified and HTTPS enforced.
- Selected media will be hosted on Cloudflare R2 through a custom media
  domain, with GitHub Pages hosting the site (agreed 2026-10-12). Preserve
  full-quality renders and originals, with smaller playback options for
  slower connections. The user created `plainsight-public-media`, connected to
  `media.plainsight-systems.com`. The four public URLs are verified and set in
  `web/data/media.json`; normal builds no longer need local media.
  All four selected files are at the bucket root. The master was uploaded
  through the bucket-scoped local `serenity-r2` profile on 2026-10-12;
  a complete readback matched the local SHA-256 (416,036,234 bytes).
- Include the original single brass sphere as a "Where we started" image.
  Local candidates are `media/2026-10-08-brass_sphere-preview-3456x2234.png`
  and `media/2026-10-09-brass_sphere-path-1024spp-3456x2234.png`; `media/`
  is ignored by Git, so publication needs an explicit asset preparation step.
- The plan is [site-plan.md](site-plan.md); the mockup was approved for implementation on 2026-10-12. The production
  source is `web/`, built and checked by `scripts/site.py`.
- Shared visual direction (2026-10-12): match current Cradle and Plainsight
  Systems, including full-width navy header and footer, ivory surfaces, Plex
  typography, navigation styling, and a parent-project link. Keep Serenity's
  own illustration and renderer presentation within that family.
- Plainsight Systems is the hub and project sites are spokes: no "Other work"
  or sibling-project lists on Serenity. Link to the hub from About, as Cradle
  does, and retain the shared parent-project footer link.
- The footer follows Cradle's exact layout, attribution, and license links,
  with Serenity as its name. Public prose follows the `andrew-voice` skill and
  its approved Cradle calibration (confirmed 2026-10-12).
- Use more screen width throughout: one fluid container with Cradle's
  responsive gutters, and no separate narrow About column (2026-10-12).
- Four content pages are implemented: Home, Writing (milestones index),
  Milestone 1, and About, plus a 404. The milestone article is staged from
  `docs/writeups/milestone-1.md`, rewritten with `andrew-voice`. Historical
  results remain October 10 results; the inference connection and reference
  comparisons no longer claim equivalence or proof of unbiasedness.
- Editorial correction (2026-10-12): public writeups explain the reasoning for
  readers; omit opaque commit hashes and bookkeeping from the narrative. The
  milestone article defines MSE and derives error × time with an equal-time
  numerical example, interpreting the actual results and the measure’s limits.
- `web/data/media.json` centralizes publication media. `--local-media` copies
  only the named film and brass image into ignored preview output. A public
  build requires their configured HTTPS URLs and contains no local media.
  Public delivery checks passed (sizes, MIME types, byte ranges). GitHub Pages
  workflow is prepared: PRs validate; only main deploys. Actions, Python, and
  Hugo are pinned; CI needs no R2 credentials.
  [PR #1](https://github.com/plainsight-systems/serenity/pull/1) was merged
  and deployed on 2026-10-12 without ReSTIR changes. The live site is
  https://serenity.plainsight-systems.com/. All four pages and the custom 404
  match the validated build; HTTP redirects to HTTPS, and the R2 film plays
  at 1920 × 1080 on the live domain.
- Navigation amendment (2026-10-12): by user request, GitHub now appears
  after About in the header as well as in the footer, linking to the Serenity
  repository. Homepage build/source actions remain omitted.

- Milestone-two editorial pass (2026-10-13): `docs/writeups/milestone-2.md`
  revised with `andrew-voice`, accepted by Andrew and merged to `main`.
  Explain decisions and mechanisms in connected prose. Displayed
  error uses a 2,048-sample reference; linear error uses 65,536 samples.
  Andrew's follow-up: keep this a project account, not a rendering tutorial.
  Organize around the image, decisions, and measured outcomes; omit unrelated
  test evidence. The table reports displayed MSE at approximately equal frame
  cost, not MSE × time. Explain both the cost comparison and the change from
  linear to displayed error. Use a concrete example for the attention link.
  Further direction: develop the broader inference pipeline connection,
  including KV-cache reuse, GPU work scheduling, and conditioning intermediate
  results. Weave those connections through the decisions rather than append
  an attention comparison or recite rendering measurements. RMSNorm/QKNorm
  control scale; denoising reconstructs a signal from noisy samples. Explain
  their roles without claiming identical operations or repeatedly fencing off
  the analogy.

- Milestone-two site update (2026-10-13): homepage features the ReSTIR PT +
  MNEE film; Writing lists milestone 2 before milestone 1. Both articles use
  their accepted source Markdown. Milestone 2 adds the side-by-side film,
  and milestone 1 retains its original film and preview. README and About
  explain the current renderer, measured ~40 ms at 1080p on M3 Max, and
  denoising/upscaling as next work. Andrew approved the site and requested
  commit/push for publication through the main-branch Pages workflow. The
  original navy framing, width, navigation and footer stay.
- R2 now also holds six dated October 13 videos: playback, small, and master
  versions of `marbles-opening-restir-pt-mnee` and
  `marbles-opening-path-vs-restir-pt`. Each is selected explicitly in
  `web/data/media.json`. New objects use immutable one-year caching and
  video/mp4; no existing objects were replaced. Comparison playback is
  1920 × 540 despite its filename; the master is 3840 × 1080. Masters are
  the original local files, without another encoding pass. All six public
  downloads match the local SHA-256 hashes; MIME types, lengths, and cache
  headers checked. Playback-file byte ranges and both new browser players
  work. The 892 MB comparison master bypasses the standard CDN cache; its
  initial range responses varied between full 200 and partial 206 responses,
  while the full download matched. It stays a separate original-file link.
  Site build, nine tests, local links, and desktop/narrow-screen review pass.

## Publication feed

- Milestone-four site update (2026-10-19, published): the homepage
  selects the Milestone 4 denoised opening; Writing, About and the RSS feed
  include milestone 4, whose article carries the Milestone 3 versus
  Milestone 4 comparison (labels measured on the fully lit scene, t = 120 s).
  Earlier articles keep their films. The accepted article is staged
  unchanged; its SVG diagrams render through `render-image.html`, each
  with a `-narrow.svg` version drawn for phones and shown below 900 px (a
  diagram without one falls back to a sideways scroll at its drawn size). Prose gives frames a
  second, not milliseconds (the user's direction). The eight 2026-10-19
  objects were uploaded and verified by the user before this update.

- Milestone-three site update (2026-10-16): Andrew requested publication
  preparation after the renderer and accepted article reached main. The
  homepage now selects the denoised opening; Writing and the RSS feed include
  milestone 3, with its ReSTIR-versus-denoised comparison. About reflects the
  trained model, measured frame costs and tracing/reuse as next work. Earlier
  articles retain their films. The accepted article is staged unchanged.
- R2 holds all eight October 15 milestone-three media files: playback, small,
  master and PNG poster for both `marbles-opening-restir-pt-denoised` and
  `marbles-opening-restir-pt-vs-denoised`. Public readback SHA-256 matches all
  local files; MIME types, lengths, immutable cache headers and all six video
  byte ranges pass. Denoised playback/master are 1920 × 1080; comparison
  playback is 1920 × 540 and master 3840 × 1080. Originals are unchanged.
  Local verification records are in `.cache/site-media/milestone-3/`.
  Andrew approved the local preview and requested commit/push for publication
  through the main-branch Pages workflow.

- On 2026-10-13 Andrew approved automatic aggregation of project writing by
  the Plainsight Systems hub. `/writing/index.xml` derives complete published
  article metadata from the existing staged Writing pages, with no body text,
  internal notes, drafts or future articles. RSS is advertised in the HTML head.
  Tests exercise XML escaping, more than twenty entries, preview exclusions and
  deployment subpaths. This adds no renderer changes or separate article list.
  Publication uses the main-branch Pages workflow; the hub reads the public
  feed and retains its previous article list during a source outage.

## Research

`../research/README.md` is the index.

## Active Workflow Pointers

- Queue: `QUEUE.md`
- Workflow: `workflow.md`

## How this project works (2026-10-17, the user)

Serenity is a tech demo, not a research project. The repository's
`CLAUDE.md`, "How this project works", is the project's constitution and
overrides the global operator profile and the inherited governance where
they conflict: production practice by default; the design order
(architecture documents, then headers with both C++ guideline corpora,
then code); one measurement, the demo graph's frame time; an image change
shown before and after for the user to judge; byte identity as the guard
where the image should not change; tests that guard invariants and
contracts, not coverage; valid findings fixed together; quality trades the
user's call; short reports. Its default agent brief is the starting point
for every delegated task.

## Decided by the user 2026-10-17

- **A decision removes the alternative.** Superseded passes, graphs,
  modes, settings and weights are deleted, not kept "just in case".
  Source control and the milestone tags (`milestone-1`, `milestone-2`,
  `milestone-3`, and `milestone-N` at every future close, `workflow.md`)
  are the archive.
- **Past milestones' films, images and numbers are kept as made,** and
  never rendered again to compare against.
- **Unit tests test units of code.** Tests that run the program are
  integration tests, and only a small handful; logic checked by rendering
  moves into functions tested directly.
- **Abandoned code is not carried forward.**

The codebase cleanup on branch `cleanup` (2026-10-17) applied them.
Removed: ReSTIR DI, superseded by ReSTIR PT (its passes, graphs and
renderer family, contract 14 and contract 5's surface record, the
`raytraced` mode); the small denoiser network
(`graphs/restir_pt_denoised_small.toml`, its pinned weights,
`training/config.json`); the checkerboard switch (a denoised graph's
light is always traced checkerboarded, and a graph without the denoise
pass traces every pixel); the path tracer's opt-in MNEE and the `mnee`
key (ReSTIR PT always runs MNEE; the path tracer never does, and stays
the reference and the training targets' renderer); the preview pass, and
the display pass that presented it (`tone_map` is the one presenting
pass). The tests were reworked into unit tests plus a handful of
integration tests that render frames end to end; logic once checked by
rendering moved into functions tested directly (`src/headless/options.h`,
`src/measure/commands.h`), and no test depends on what `graphs/` or
`scenes/` hold beyond every file there reading.

## Milestone 4: making the frame fast (2026-10-17/18)

`research/2026-10-17-milestone-4-plan.md`; `writeups/milestone-4.md`.

- **Kept by the user:** async compute overlap (a denoised frame's light
  and image halves on two queues), ray masks (visibility rays trace a
  structure of the surfaces alone; no firefly casts a shadow), and
  checkerboard rendering of the light (how every denoised graph traces).
  The split scene is skipped on Metal, kept for the Vulkan and Xbox
  milestones.
- **The network retrained on denoiser-v2** (2026-10-19, kept by the
  user): the whole data set rendered again on the Mac (3.55 h), sharp,
  checkerboarded and jittered as the demo traces, proved to the bit by
  tests/gpu/features_test.cpp; trained on the RTX 3090 Ti (3.5 h);
  exported to `../serenity-data/denoiser-v2-model/wide`, which the demo
  graph names and `make weights` fetches. Corrects an earlier claim: the
  2026-10-18 "checkerboard retrain" (denoiser-v1-checkerboard) was not
  on checkerboard input, the input graph having traced every pixel.
- **The production sweep** (`research/2026-10-18-production-sweep.md`,
  kept by the user): depth of field after the denoiser (half resolution),
  Halton jitter as two network inputs, light tiles drawn by power, frame
  pacing by the display link behind a backend-neutral frame source (30 to
  60 frames a second). **The user's rule: denoiser-v2 is the last
  data-set render on the Mac; any later one waits for the RTX 3090
  renderer.**
- **Training runs on either GPU:** `train.py --device cuda|mps`, by
  default the machine's; the data is held on the GPU when it fits beside
  training's own memory, measured at the start, and in host memory when
  not. The RTX 3090 Ti (`ssh ai`, the checkout and environment under
  `/data/serenity`) is where training runs; the data set is rendered on
  the Mac and copied over.
- **The final frame:** 17.5 ms at 1920 x 1080 (57 fps), 58.5 ms at the
  window's 3456 x 2234 (17.1 fps), accepted by the user 2026-10-19 with
  depth of field after the denoiser (before the sweep: 16.7 and 56.1).

## The production sweep (Milestone 4, 2026-10-18)

`research/2026-10-18-production-sweep.md`. Three production steps the
frame lacked, chosen by the user from a sweep against shipped engines:

- **Depth of field after the denoiser** (principle 12): a denoised graph
  traces a pinhole (`frame::traced_inputs`) and its `depth_of_field`
  pass, directly after the denoise pass, blurs the network's output by
  the thin lens's circle of confusion, Jimenez's SIGGRAPH 2014 method
  (contract 21). The path tracer, the reference, keeps the lens in its
  rays, as Unreal's path tracer does. The data set is rendered through a
  pinhole (its scenes written without their lens).
- **Camera jitter**: every ReSTIR PT frame's camera rays through one
  Halton(2, 3) offset of 32 phases (`core/sampling/jitter.h`), given to
  the network as two input channels (contract 18: 30 inputs); motion
  vectors and the full-size albedo unjittered.
- **Light tiles drawn by power**, RTXDI's presampling: a power pyramid a
  frame, every ReSTIR PT density that power over the total.
- **The frame:** 17.37 ms at 1920 x 1080 (57.6 fps), 58.30 ms in the
  window (17.2 fps), measured with the demo's network padded to the new
  inputs.
- **The network must be retrained**: every export has 28 inputs and the
  renderer refuses it by name. The data set to render is denoiser-v2
  (`scripts/make_denoiser_dataset.py --out ../serenity-data/denoiser-v2`),
  inputs and targets both new, about 3.1 h (the estimate); the last data
  set rendered on this Mac, later ones on the RTX 3090's renderer; tested
  against its own targets, no reference rendered again (the user,
  2026-10-18).
- **The training data is what the network sees** (principle 13, the
  user's decision 2026-10-18): every graph with ReSTIR PT and the features
  pass, the data set's input graph among them, traces its light as the
  denoised graph does, checkerboarded and filled, pinhole, jittered; a test
  holds the input graph's light and features to the demo's hand-off, to
  the bit. denoiser-v1-checkerboard's inputs, found traced at every pixel,
  were not.

## Formatting and static checks (decided by the user 2026-10-19)

- **Google style, unchanged** (`.clang-format`), and clang-tidy's
  `google-*`, `cppcoreguidelines-*`, `bugprone-*`, `performance-*` and
  `clang-analyzer-*`, every finding an error; `make check` runs both
  (`make format-check`, `make tidy`). LLVM 21.1.8, pinned apart from the
  build's tools (`cmake/toolchain.json`, "$lint").
- **Deviations only where Metal requires them** (shared GPU layouts, the
  binding indices, metal-cpp interop), and one more by the user: tests keep
  their literal expected values (`tests/.clang-tidy`). The table, with the
  three outside Metal still the user's to confirm, is in `workflow.md`,
  "Formatting and static checks".
- **clangd in VS Code, on save** (`.clangd`, `.vscode/`), formatting with
  the same clang-format and checking with the same clang-tidy, the
  analyzer aside.

## The RTX 3090 Ti's own design (decided by the user 2026-10-21)

- **Milestone 5 editorial pass (2026-10-22):** Andrew requested his voice
  guide and Milestone 4 as the reference, with particular attention to
  keeping the article engaging. The revised writeup follows visible
  problems, decisions and consequences rather than a pass-by-pass timing
  inventory. Scheduling gains and the slower wavefront experiment stay
  together; the chart carries the detailed performance progression.
  Andrew found the first revision too long. A second pass cut it from
  about 1,700 to 1,100 words, retaining the geometry, scheduling and
  precision decisions and the inference connection; secondary experiments
  remain in the linked research notes.
  Accepted by Andrew 2026-10-22; Milestone 5 closed.

- **CUDA builds with `--use_fast_math`**, as Blender Cycles and pbrt-v4
  build their GPU kernels (and as Metal's fast math mode is): the
  backend's earlier IEEE default superseded. Every GPU test's comparison
  with the CPU holds under it, its tolerance unchanged. The demo 129.7 to
  163.7 fps at 1920 x 1080 (`docs/research/2026-10-21-tracing-divergence.md`,
  section 8.6).
- **ReSTIR PT's initial pass stays one OptiX program on Ampere.** Cycles'
  wavefront was built and measured, and lost (140.8 against 163.7 fps);
  not kept (its branch deleted by the user 2026-10-22; the design and
  measurements in the tracing doc's section 8), to be built again on
  hardware with shader execution reordering. Both shifts run one thread a
  pair of a list (ReSTIR PT Enhanced's pair lists).
- **Optimizations follow production whole, with parameters from the
  hardware and the programs' own properties,** never fitted to one scene's
  statistics; scenes confirm a design, they do not choose its numbers.

## Milestone 6: Apple silicon (decided by the user 2026-10-27)

- **The denoiser stays our trained network,** not MetalFX Denoising.
  MetalFX was measured in the frame with every input Apple lists, glass
  and mirrors described by primary surface replacement (NVIDIA's DLSS-RR
  sample): faster at every size (77.3 against 63.0 fps at 1920 x 1080;
  40.0 at 3x against 17.5 at 3456 x 2234) but it loses the caustics in
  the glass and the fireflies' pinpoints in the mirrors
  (`research/2026-10-27-metalfx-denoising.md`). Its prototype was never
  committed, and was deleted 2026-10-27.
- **The network runs on the Neural Engine, beside the GPU** (kept by the
  user 2026-10-27; `research/2026-10-26-neural-engine-denoiser.md`,
  section 5): the image half split around a Core ML prediction
  (`Submission::hand_off()`), the input and features shared in place
  through IOSurfaces, a ladder of 121 enumerated shapes the light is padded
  to. 63.3 to 84.2 fps at 1920 x 1080, 17.7 to 25.1 fps at 3456 x 2234,
  the picture the GPU's within 4 levels at the demo's sizes. Not Apple's
  rendering guidance (networks on the GPU): the user's call, on the
  numbers. Metal 4's machine-learning route removed. Between rungs (window
  sizes whose light is not a rung) the extra padding changes the network's
  output near the bottom and right edges.
