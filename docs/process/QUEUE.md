# Work Queue

This file tracks active and accepted work.

## Active

- **Repository skeleton.** Process documents, the build, the toolchain pin
  and its check, the boundary check, the Metal backend's device and library
  loading, and the portable-math contract with its tests. No rendering code.

## Next, in order

1. `docs/architecture/logical-overview.md`: what the renderer does, as
   phases, responsibilities and principles.
2. `docs/architecture/change-axes.md`: the reasons a file changes.
3. `docs/architecture/file-mapping.md`: modules, axes and contracts.
4. The first rendering change, designed in its headers: a path tracer over
   procedural geometry with many moving lights, sampled naively, shown in a
   window and written to frames by a headless renderer.

Later, each in its turn: ReSTIR DI with spatial and temporal reuse; a
denoiser of its own against Intel Open Image Denoise; the Vulkan backend on
the AMD machine; ReSTIR GI.

## Accepted

- None yet.

## Parking Lot

- **CI.** Not set up. Unknown whether GitHub's macOS runners, which are
  virtual machines, expose Metal ray tracing; find out before relying on the
  GPU tests there. Without it, CI could run only the core's tests and the
  structural checks.
- **Metal 3 or Metal 4 for the host API.** The skeleton uses Metal 3 objects
  and compiles shaders as `-std=metal3.2`. Decide in the first rendering
  change, with the language version beside it.
- **The cost of `safe` and `precise` math** in the renderer's kernels: not
  measured (`research/2026-10-08-metal-math-modes.md`).
- **Whether MoltenVK supports Vulkan ray tracing.** Believed not; verify
  before the Vulkan backend, since it decides whether that backend could also
  run on the Mac.
- **A sanitizer configuration.** Charlotte found AddressSanitizer's runtime
  hanging at start-up on macOS and ran it on Linux only; there is no Linux
  build here yet.
