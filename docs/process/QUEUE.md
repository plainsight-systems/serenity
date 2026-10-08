# Work Queue

This file tracks active and accepted work.

## Active

- **Repository skeleton.** Process documents, the build, the toolchain pin
  and its check, the boundary check, the Metal backend's device and library
  loading, with its tests. No rendering code.

## Next, in order

1. `docs/architecture/change-axes.md`: the reasons a file changes.
2. `docs/architecture/file-mapping.md`: modules, axes and contracts.
3. The first frames: a full path tracer over procedural geometry with many
   moving fireflies, sampled naively, shown in a window and written to frames
   by a headless renderer.

Then, each in its turn: ReSTIR DI; ReSTIR GI; caustics by manifold next
event estimation; reservoir reuse as kernels; a denoiser of its own against
Open Image Denoise and MetalFX; the Vulkan backend on the AMD machine.

## Accepted

- None yet.

## Parking Lot

- **Whether MoltenVK supports Vulkan ray tracing.** Believed not; verify
  before the Vulkan backend, since it decides whether that backend could also
  run on the Mac.
