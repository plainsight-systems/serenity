# Project Memory

This file is the canonical entry point for durable project context.

## Product Identity

- **Product:** Serenity. Named 2026-09-30.
- **Operating brand:** None. Internal R&D under the parent entity.
- **Parent entity:** Plainsight Systems LLC
- **Repository:** local only; not yet on GitHub.

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
- **The core is platform-neutral; only `src/metal/` uses Metal's host API.**
  Enforced by `tools/check_boundaries.sh`, whose rules are proven to fire by
  `tests/test_check_boundaries.sh`.
- **The toolchain is pinned by version and checked when the build is
  configured.** Metal cannot run in a container, so it cannot be pinned by
  an image as Charlotte pinned emsdk. `cmake/toolchain.json` records Xcode,
  the SDK, the Metal compiler and Apple clang; configuring fails on any
  mismatch. Moving the pin is a commit of its own, with the tests rerun.
- **Shaders are compiled at build time with pinned flags and compiled into
  the executable** (`cmake/MetalLibrary.cmake`). Never compiled from source
  at run time, never loaded from a path.
- **Math shared by the CPU and the GPU gives equal bits, under a stated
  contract** (`src/core/portable_math.h`): shaders built with `safe` and
  `precise` math and contraction off, C++ with contraction off, only
  IEEE-exact operations, and no subnormals, which the GPU flushes. Measured
  in `research/2026-10-08-metal-math-modes.md` and held by
  `tests/gpu/portable_math_test.cpp`. A kernel that wants Metal's fast math
  opts out explicitly, as a labelled and measured optimization.
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

## Research

`../research/README.md` is the index.

## Active Workflow Pointers

- Queue: `QUEUE.md`
- Workflow: `workflow.md`
