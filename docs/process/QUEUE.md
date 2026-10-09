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

1. Milestone 1, the rest: fireflies that move (in progress: a wander about
   an anchor, closed in time; the structure rebuilt each frame; a sphere
   light read from its sphere; accumulation of one instant at a time;
   headless --samples for clean movies); fireflies that blink; procedural
   geometry toward marbles on a table with many fireflies, flying free; the
   reference render and its error measure.

Then, each in its turn: ReSTIR DI; ReSTIR GI; caustics by manifold next
event estimation; reservoir reuse as kernels; a denoiser of its own against
Open Image Denoise and MetalFX; the Vulkan backend on the AMD machine.

## Parking Lot

- **Whether MoltenVK supports Vulkan ray tracing.** Believed not; verify
  before the Vulkan backend, since it decides whether that backend could also
  run on the Mac.
