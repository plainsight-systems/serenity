# Work Queue

This file tracks active and accepted work.

## Active

- **Repository skeleton.** Process documents, the build, the toolchain pin
  and its check, the boundary check, the Metal backend's device and library
  loading, with its tests. No rendering code.

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

- **Whether MoltenVK supports Vulkan ray tracing.** Believed not; verify
  before the Vulkan backend, since it decides whether that backend could also
  run on the Mac.
