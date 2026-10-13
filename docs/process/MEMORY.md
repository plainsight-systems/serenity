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
Many-light sampling by ReSTIR DI (Bitterli et al. 2020), a denoiser of its own
compared against Intel Open Image Denoise, and a reference render at
thousands of samples per pixel to compare against. No art assets: the light
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
- **Metal 4** for the host API and the shading language (`-std=metal4.0`),
  not Metal 3. Its explicit command allocation, residency and barriers are
  close to Vulkan's model, which the second backend will need anyway; its
  machine-learning encoder and in-shader tensors are where the denoiser's
  inference can run inside the frame; and MetalFX's denoised upscaler is
  there as a second baseline beside Open Image Denoise.
- **The core is platform-neutral; only `src/metal/` uses Metal's host API.**
  Enforced by `tools/check_boundaries.sh`, whose rules are proven to fire by
  `tests/test_check_boundaries.sh`.
- **The toolchain is pinned by version and checked when the build is
  configured.** Metal cannot run in a container, so it cannot be pinned by
  an image as Charlotte pinned emsdk. `cmake/toolchain.json` records Xcode,
  the macOS SDK the build compiles against (the presets name it,
  `CMAKE_OSX_SYSROOT`), the Metal compiler, Apple clang as each of the C,
  C++ and Objective-C compilers CMake resolved, and CMake itself;
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
  (`../architecture/logical-overview.md`).
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
  direct light the reservoir's alone. The reason: light sampling alone, at
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

## Project Site Direction

Clarified by the user 2026-10-12:

- The public project is officially at **milestone 1**. ReSTIR is in progress
  on a branch; implementation notes in the queue do not promote it to a
  completed public milestone.
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

## Research

`../research/README.md` is the index.

## Active Workflow Pointers

- Queue: `QUEUE.md`
- Workflow: `workflow.md`
