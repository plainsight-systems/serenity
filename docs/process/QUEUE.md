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

## Next, in order

1. A full path tracer over procedural geometry with many moving fireflies,
   sampled naively, in the window and headless. It brings the BSDF contract
   (contract 2), through which each material kind is evaluated and sampled,
   and with it the material switch leaves the preview's direct integrator
   (metal/integrator/direct.metal.h), whose light loop moves onto light
   records and the emitter (contract 3).

Then, each in its turn: ReSTIR DI; ReSTIR GI; caustics by manifold next
event estimation; reservoir reuse as kernels; a denoiser of its own against
Open Image Denoise and MetalFX; the Vulkan backend on the AMD machine.

## Parking Lot

- **Whether MoltenVK supports Vulkan ray tracing.** Believed not; verify
  before the Vulkan backend, since it decides whether that backend could also
  run on the Mac.
