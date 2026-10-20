# The CUDA and OptiX backend, on the RTX 3090 Ti

2026-10-19. The plan for the second backend, written before its headers,
as the design order asks. Decided by the user the same day: each vendor's
own API and no portable layer (`../process/MEMORY.md`, "Metal now, Vulkan
later", superseded), so on NVIDIA's RTX 3090 Ti the backend is CUDA, with
OptiX for the rays.

## The goal

`serenity-headless` renders any frame graph on the RTX 3090 Ti, so the
denoiser's data sets render there (the user's rule: no more data-set
renders on the Mac) and `scripts/make_denoiser_dataset.py` runs there
unchanged in what it asks. Then the demo graph's frame time on that
machine. The window comes after, as its own step.

## The machine, checked 2026-10-19

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 Ti, Ampere, sm_86 |
| OS | Ubuntu 24.04.5 LTS |
| Driver | 595.91.07; its OptiX runtime (`libnvoptix.so.595.91.07`) is 9.1.0 |
| CUDA toolkit | 13.2 (`nvcc` V13.2.86) |
| Host compilers | clang 18.1.3, g++ 13.3.0 |
| CMake | 3.28.3 |
| OptiX headers | 9.1.0, `NVIDIA/optix-dev` at `f1f6dd8` (NVIDIA's header-only distribution; its SDK licence, binary redistribution only, so fetched at configure time and never committed) |

The core, the headless renderer's and the measuring program's sources
already compile there under both compilers with the project's warnings: no
error but the fetched headers' absence, and one `-Wshadow` from g++
(`src/headless/options.cpp:127`), not a warning under the pinned clang.

## Facts the design rests on

From the OptiX 9.1 headers and NVIDIA's forum answers (staff), and Blender
Cycles' OptiX device (`main`, read 2026-10-19):

1. **OptiX traces only inside its own programs.** `optixTrace` and
   `optixTraverse` are available in ray generation, closest-hit, miss and
   callable programs, which `optixLaunch` starts; there is no ray query
   from a plain CUDA kernel. `optixTraverse` returns a hit object without
   running a closest-hit or miss program, and the ray generation program
   reads it: the nearest thing to Metal's `intersection_query`, and what
   Cycles' shadow rays do. `optixReorder` (shader execution reordering)
   sits on the hit object; whether Ampere regroups threads for it is
   unconfirmed (one 2026 preprint says from Ada).
2. **Cycles splits the work along that line.** Only the kernels that trace
   are OptiX launches, one ray generation program each (closest, shadow,
   MNEE in its own module, and so on); everything else is a plain CUDA
   kernel on the same stream. Its programs are compiled at build time by
   `nvcc` to PTX.
3. **Custom primitives run on the SMs, and so do built-in spheres.** Only
   triangles are intersected by Ampere's RT cores (NVIDIA staff, 2023).
   A box as twelve triangles would put boxes on them; a sphere has no such
   form.
4. **A mask is an instance's.** OptiX tests `ray mask & instance mask`, 8
   bits, as Metal does, so masks need an instance level.
5. **Tensor cores:** `mma.sync` and WMMA take half inputs with float
   accumulation on sm_80 and later, so on sm_86.
6. **Timing:** CUDA events on the stream, resolution about 0.5 us;
   `optixLaunch` is ordered on the stream it is given, as a kernel is.

## The design

**One backend, `src/cuda/`, mirroring `src/metal/`'s families.** The same
directories (acceleration, camera, device, film, frame, integrator, lights,
materials, math, media, passes, reuse, sampler, scene, shapes, textures),
the same files where the job is the same, each written for this hardware.
It decides how; the core still decides what (principle 10).

**Two kinds of pass.**

| A pass that | is | its code |
|---|---|---|
| traces rays: path, ReSTIR PT's initial, temporal and spatial, MNEE, features | an OptiX launch: a ray generation program that is the pass's body, traversing with `optixTraverse` and reading the hit object itself, with no closest-hit or miss program; the shapes' intersection programs | `passes/<pass>/<pass>.cu`, compiled to PTX, one module and one pipeline per pass |
| traces none: ReSTIR PT's shade and duplication, denoise, tone map, depth of field, checkerboard fill, test pattern | a CUDA kernel | `passes/<pass>/<pass>.cu`, compiled to the machine's code (sm_86) |

Which is which is read from the Metal passes' includes (2026-10-19): the
six that trace reach `acceleration/trace.metal.h` through the integrator;
shade and duplication include ReSTIR PT's header but trace nothing, and
the denoiser's full-size albedo is the features pass run again, not the
denoiser's own tracing. Every pass runs on the frame's stream in the
schedule's order, as Metal's encoder does.

The headers say `optixTraverse` traverses and fills the hit object
without running closest-hit or miss; they do not say in words that a
custom primitive's intersection program runs during it. It must, for a
traversal to hit one at all, and Cycles' shadow rays rely on it. Proven
on this machine before Phase 3's headers ("Phase 3's design").

**What it ports first, and what is tuning later.** The first port is
Metal's design carried over: shapes as bounding boxes with their own
intersection programs, the two structures for the two questions of
principle 11 (every shape, the surfaces alone, each a geometry structure
traced by its own handle), the same pass boundaries and layouts. Boxes as
triangles on the RT cores, an instance level with mask bits, reordering,
fused passes: each is this machine's tuning, measured one at a time after
the port runs, not a decision of the port.

**Device code includes the core's shared layouts only.** A shared layout
is a core header that includes `core/contracts/shared_layout.h`: the
contracts, the kinds' records (lights', materials', media's, shapes',
textures', passes'), and their encodings (the `constexpr` packing
functions beside them), never math. Metal's shaders already include
nothing else from the core (checked 2026-10-19). CUDA is C++, so a kernel
could include the core's sampling or material code; it does not. The GPU
tests check each backend against the core's CPU computations, and a
kernel compiled from the same source would only check that it compiles
twice. Enforced by `tools/check_boundaries.sh` for both backends' device
code.

**Host code is C++ compiled by the host compiler; `nvcc` compiles device
code only.** The host uses the CUDA driver API (`cuda.h`), loading the
kernels' fatbin and the programs' PTX at start-up, as the Metal backend
loads its library and as Cycles does; OptiX through its function table
(`optix_stubs.h`). So the host code's warnings, formatting and static
checks are the project's, and `nvcc`'s flags reach only device code.

**The headless seam.** `src/headless/main.cpp` names Metal's device,
submission, target and renderer today. As the window's program does
(`core/presentation/frame_source.h`), the program splits: what it decides
(the plan, the files, the refusals, the failure counts' handling) stays in
`src/headless/`, backend-neutral; each backend gives an offline renderer
behind one interface in the core (render a frame's samples, wait, read
back the image, the accumulation, the features, the failure counts), and
only `main.cpp` composes the one this build has. The command line and
what it writes do not change: the data-set script drives it. On Metal
this is a refactor, guarded by every graph's output being the same to the
bit.

**The build.** CMake selects the backend by platform: macOS builds
Metal, Linux builds CUDA, and either refuses what it cannot build. The
presets keep their names (`native-debug`, `native-release`,
`native-sanitize`) with a condition on the host, so the script's
`cmake --preset native-release` is the same command on both.
`cmake/toolchain.json` gains a Linux section, checked by the same
script: clang 18.1.3 as the host compiler (one compiler family across
both machines; `nvcc -ccbin` the same), `nvcc` 13.2.86, CMake 3.28.3, the
driver 595.91.07 (it is the OptiX runtime), and the OptiX headers' commit.
CUDA 13.2 takes both clang 18 and g++ 13 as host compiler (a kernel built
with each, 2026-10-19).

**The lint.** Formatting needs no headers, so `make format-check` checks
every file, `src/cuda/` included, on the Mac with the pinned LLVM 21.1.8.
The static checks need each file's compile line: `src/cuda/`'s host code
is checked on the 3090, which needs LLVM 21.1.8 there (Ubuntu's archive
stops at 20; LLVM's own apt repository has it), a dependency for the
user's yes in Phase 1.

**The data-set script.** Three things in it assume macOS: the GPU's
utilization read from `ioreg`, the Metal compiler's flags in the
renderer's record, and the deployment target. Each becomes the backend's
own (`nvidia-smi`; `nvcc`'s flags; none), named in the manifest.

**Tests.** The core's tests run on Linux as they are. The GPU tests are
per backend: `tests/gpu/` stays Metal's, and `tests/cuda/` checks the
same invariants on this one: the contracts' layouts as the device sees
them, each family's kernels against the core's CPU computations (shapes,
BSDFs, camera, lights, sampling), the denoiser against PyTorch's fixture,
no allocation per frame, the same command writing the same files, and the
network's input graph rendering the demo's light to the bit
(`features_test.cpp`'s check). Metal's argument-table, library and
device tests have no counterpart. No test matches the 3090's output to the Mac's.

**The measurement.** The demo graph's frame time at 1920 x 1080 headless,
by CUDA events around a frame's work on its stream, with nothing else on
the GPU (training stopped). Per-pass events when a change targets a pass.

## What carries over, and what does not

Mapped file by file over `src/metal/` (2026-10-19): 9,857 lines of host
code and 10,019 of shaders (30% of either comments). Line counts in mixed
files are estimates.

**Host code.**

| Class | Lines | What | On CUDA |
|---|---|---|---|
| Not Metal at all | ~1,700 | the scene block's layout (`scene/scene_block.h`) and its entries table; the denoiser's network plan (sizes, levels, strides, history parity: `frame/denoise_images.cpp`); the renderer's frame plan and every per-frame constants block it fills (`FrameConstants`, `PathReuseConstants`, `DenoiseConstants`, checkerboard, depth of field); the scene and frame refusals (`check_time`, `check_packable`, the light-tile, MNEE and glower limits); the image and array sizes per configuration (ReSTIR PT's, bloom's, depth of field's, duplicated in two files today, MNEE's capacity); "keep the history on a copy, remake on a new size", written four times; the acceleration build's plan (members, the surfaces, the moving boxes' pairing); the failure counts' bookkeeping; the headless seam's types (`FeatureImages::Read`, `NetworkInput`) | **moves into `src/core/` first**, so both backends use one copy |
| Metal's API | ~2,600 | device, library and pipelines, buffers, readback, encoders, barriers, the acceleration build's calls, every pass's dispatch | rewritten: `CUcontext`, `CUstream`, `cuMemAlloc`, `cuModuleLoadData`, `cuLaunchKernel`; `optixAccelBuild`, modules, program groups, pipelines, `optixLaunch`; events for the two queues' ordering |
| Metal only | ~900 | residency sets, argument tables and their binding indices, the indirect grids, the display link and presenter | none; each pass's bindings become its launch parameters; the window's comes later |
| the rest of the mixed files | ~4,600 | the Metal halves of the files above | rewritten with them |

**Shaders.**

| Class | Lines | What | On CUDA |
|---|---|---|---|
| Line for line | ~7,960 (79%) | the integrator, shifts, NEE, MNEE's solve, materials, lights, sampler, hashing, warps, media, textures, reuse, shapes' intersection math, camera, shade, fill, duplication, tone map, depth of field | translated: address spaces, `as_type`, `simd_*` to warp intrinsics, `half`, and the vector types, which CUDA lacks (operators, swizzles, `dot`, `normalize`: 88 swizzles, 118 `dot`; `src/cuda/math/` first) |
| Restructured | ~890 (9%) | `acceleration/trace.metal.h`'s candidate loops (175); MNEE's segment seeding (~80); the denoiser's convolutions on `simdgroup_float8x8` and its threadgroup memory (~310); the six tracing passes' entry points (~230); the indirect grids (~40); the scene's structure views (~30) | the loops become intersection programs (and any-hit for MNEE's seeding, which collects up to four crossings and aborts on glass that is not a sphere); trace and occluded keep their signatures as wrappers over `optixTraverse`; the denoiser on `mma.sync` (WMMA's smallest half tile, 16 x 16 x 16, would not keep contract 18's blocks of 8 channels) |
| Layouts | 2,522 | `core/contracts/` | included as they are, with a `__CUDACC__` branch in `shared_layout.h` |

**Not in Metal's design, needed here.**

- **No indirect launch.** The replays and MNEE's solve take their grid
  sizes from GPU memory; neither a CUDA kernel nor `optixLaunch` can. They
  launch at capacity and return past the count (the guard exists), or the
  count is read back; measured when they are built.
- **Memory is not shared.** Per-frame arrays, constants, counters and
  MNEE's list header are written by the CPU in place on Apple's unified
  memory. On the discrete 3090 each is an upload on the frame's stream,
  or mapped pinned memory; chosen per array, by what it costs.
- **Math mode.** Metal compiles with `-fmetal-math-mode=fast` (no
  infinities or NaNs assumed). CUDA's default is IEEE division and square
  root, denormals kept, contraction on. The CUDA backend keeps the default:
  `-ftz` would break the BSDF packing's subnormal bound (contract), and
  the GPU tests compare with the CPU, which is IEEE.

**To verify as each step comes:** warp intrinsics inside ray generation
programs (MNEE's and the replays' appends, the non-finite count use them;
OptiX forbids shared memory there), checked for the count before Phase 3
("Phase 3's design"), the appends' warp-wide prefix still to check in
Phase 4; any-hit called once per primitive for
MNEE's seeding (`OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL`); a
`constexpr` array indexed at run time in `contracts/denoise.h`
(`spacing_pixels`) in device code: device code may index a host constant
only in a constant expression, so a table is copied at compile time from
the core's into a `static constexpr __device__` constant (checked on the
3090, with and without `-rdc`; the noise's gradients in Phase 3, the
pairing's tables and the spacings when their passes come).

**Tests.** Every invariant `tests/gpu/` checks carries but three: the
argument tables, the library and the device's Apple family. The encoding
tests carry (they test contract layouts, not Metal's). Their probe kernels'
layouts (`tests/gpu/kernels/`, ~950 lines) are shared; the harness is per
backend.

## Phase 3's design, checked 2026-10-19

Written before Phase 3's headers. What it rests on was checked on the RTX
3090 Ti first, with a probe outside the repository (two spheres as custom
primitives, a ray generation program, OptiX 9.1's headers at `f1f6dd8`),
and in NVIDIA's OptiX 9.1 Programming Guide (December 2025, its sections
named below), against Blender Cycles' OptiX device where it does the same
job.

**Proven on the machine.**

- An intersection program runs during `optixTraverse` over a single
  geometry structure of custom primitives, from a ray generation program
  with no closest-hit program and a miss group with no program: the
  nearest of two spheres, its primitive index, its t and its hit record's
  data read back from the hit object; with `TERMINATE_ON_FIRST_HIT`, the
  shadow ray's answer. So the design above holds as written: each pass that
  traces is its ray generation program, reading hit objects itself.
- A payload value reaches the intersection program
  (`optixGetPayload_0`, available in IS): the shape a shadow ray ignores.
- A warp-aggregated atomic in the ray generation program
  (`__activemask`, `__ballot_sync`, one lane adding the count) counts
  exactly. The lane is read from `%laneid`: in an OptiX program
  `threadIdx` is the launch index (the guide's table of PTX special
  registers, 6.2), which the probe's one-dimensional launch matched by
  chance. The guide (6.2, "Programming model") forbids shared memory and
  warp- or block-wide synchronization, and allows vote and ballot for an
  algorithm independent of which threads share a warp, naming
  warp-aggregated atomic adds; the failure counts are that.
- The core's shared layouts all compile in device code under `nvcc` with
  the project's flags, but for their `constexpr` functions, which device
  code may call only under nvcc's experimental `--expt-relaxed-constexpr`.
  So `core/contracts/shared_layout.h` gains `SERENITY_SHARED_FUNCTION`,
  `__host__ __device__` under `__CUDACC__` and nothing elsewhere, on each
  of those functions (ES.33: one prefixed name; P.11: written in the one
  header that holds the languages' differences), the way CUTLASS marks its
  functions, rather than a flag NVIDIA marks experimental. And its
  `Array<T>` (`core/scene/scene_block.h`) is a pointer under `__CUDACC__`,
  as under Metal, where the host keeps the 64-bit address. Neither changes
  what Metal or the host compiles.

**The programs' input: OptiX-IR.** The guide (6.1, "Program input")
recommends OptiX-IR over PTX when `nvcc` compiles the programs, for
optimizations PTX does not carry (its 9.x notes on live state and
`__restrict__`); Cycles compiles PTX, for the range of toolkits and
architectures it supports, which this backend does not. Flags, from the
guide's list: `-optix-ir -m64 -rdc=true -arch=compute_86 -std=c++20 -O3`,
the pinned host compiler, warnings as errors, and no `--use_fast_math`:
the guide recommends it for speed and names the exception, precision,
which the tests' comparisons with the CPU need (the plan's "Math mode").
They live in `cmake/CudaModules.cmake` beside the kernels', in a second
function. OptiX compiles the IR to the machine's code when the module is
made, at start-up: unlike the kernels' fatbin, the code that runs is
OptiX's, fixed by the pinned driver (595.91.07, its OptiX 9.1.0). OptiX's
compile cache (guide 4.2) stays at its default, on, as Cycles leaves it: it
keeps OptiX's compile of the same input, so what runs is the same.

**Tracing.** Metal's one level, carried over:

- Two geometry structures of custom primitives, the core's plan
  (`core/shapes/structure_plan.h`): everything, box i shape i, and the
  occluders with their table. The pipeline allows a single geometry
  structure as the traversable (`ALLOW_SINGLE_GAS`); no instance level,
  so no masks, as on Metal, and a ray traces the structure its mask names.
  The occluders with no surface hold nothing: no structure, and a ray
  carrying them finds nothing, as Metal's `empty`.
- One hit group, the shapes' intersection program, and one hit record a
  structure: record 0 for everything, its table none, record 1 for the
  occluders, its table the occluders' shapes. The ray generation program
  passes the record of the structure it traces as the SBT offset, and the
  intersection program names a box's shape through its record's table
  (`optixGetSbtDataPointer`), as Metal's `MaskedStructure::shape` does.
  The SDK's own pattern for per-geometry data; the alternative, the table
  in the launch parameters and the mask in a payload, carries a mask the
  record already names.
- The intersection program is the shape's exact test, its ray clipped to
  the nearest so far (`optixGetRayTmax`, which OptiX narrows as hits are
  reported), reporting t. Any-hit disabled on the geometry and on the ray.
  The nearest hit: `DISABLE_ANYHIT`. Occluded: `DISABLE_ANYHIT |
  TERMINATE_ON_FIRST_HIT`, the shape it ignores in payload 0 (none: all
  bits set). Trace depth 1; the pipeline's stack sizes from
  `optix_stack_size.h`'s helpers, for that depth and a single structure.
- The boxes are the core's `shapes::Bounds`, two packed triples of
  floats, which is `OptixAabb`'s layout, asserted beside the build.
  Build flags none, as Metal builds with its defaults; no compaction. A
  still structure is built once, at construction, and waited for; a
  moving one rebuilt each frame on the frame's stream, a rebuild not an
  update, as Metal (refits decay; principle 9). Choosing flags, compaction
  and boxes as triangles are this machine's tuning, after the port.

**Per-frame data on a discrete GPU.** Apple's memory is the GPU's: Metal's
CPU writes a frame's arrays where its shaders read them, so it keeps a
copy a frame in flight. Here each frame's data (the launch parameters, the
moving boxes, the shapes' transforms, the lights' glows) is staged by the
CPU in pinned host memory, one copy a frame slot, written only once the
slot's last frame has settled (`cuda/device/frame_sequence.h`), and
uploaded on the frame's stream into one device copy (GPU.7). The stream
orders frame n + 1's upload after frame n's launches, so one device copy
is enough. The scene's still arrays are uploaded once, at construction.
Nothing is allocated per frame (MEM.9).

**Launch parameters.** A pass that traces is an `optixLaunch` over the
frame's pixels; its parameters are one struct in OptiX's launch-parameter
variable, which OptiX reads from constant memory, the space Metal's
`constant` buffers are in: the frame's constants and the camera by value,
and the addresses of the scene's block, the transforms, the glows, the
accumulated image and the failure counter, and the two structures'
handles. The struct is the pass's own, one header compiled by both
compilers.

**Images.** The accumulated image is pitched linear memory, RGBA32Float,
which the path program writes through its address and the tone map reads
through a texture object over it, filtered: CUDA filters 32-bit floats,
with interpolation weights held to 8 fractional bits (the CUDA
Programming Guide's texture-fetching appendix). The bloom pyramid's
levels are CUDA arrays of half floats, written through surface objects and
read through filtering texture objects, at the core's sizes
(`core/passes/bloom.h`). Every texture object samples clamped to the edge
with normalized coordinates, as Metal's sampler.

**The failure count.** A counter a frame slot in device memory, zeroed
on the stream as the frame begins and copied at its commit into pinned
host memory before the frame's event; the host collects a slot's count
once the frame has completed, as Metal's `FailureCount` does.

**Device code.** The path slice of `src/metal/`'s shaders translated line
for line into `.cuh` files under the same families: `math/` first, the
vector types' operators and functions CUDA lacks, under Metal's names
(`dot`, `normalize`, `mix`, `saturate`, `select`) so the translation stays
reviewable against its source; then the sampler and hashing, warps,
transforms, shapes and their intersection program, textures, materials,
media, lights, uniform light selection, the camera, the film and the path
integrator. Not the shifts, MNEE, ReSTIR PT's reuse or the features
(Phase 4).

**Tests.** As the user's rule has them: the GPU runs only GPU code.
Probes run each family's device code against the core's CPU computations,
as `tests/gpu/` does on Metal, sharing its probes' layouts; and the one
integration test, the headless program on the 3090, renders the reference
graph at a small size: written, every sample finite, and the same command
writing the same bytes. No test matches the 3090's frames to the Mac's:
the two GPUs' arithmetic differs (Metal's fast math, CUDA's IEEE), so the
same scene is shown from each, side by side, for the user to look at.

**Checked against Cycles** (`intern/cycles/device/optix/device_impl.cpp`,
`kernel/device/optix/kernel.cu`, `main`, read 2026-10-19). The same: each
kernel that traces is a ray generation program (closest, shadow,
subsurface, volume stack, dedicated light), trace depth 1, stack sizes
set from the program groups' own, the launch parameters one struct in
constant memory (`kernel_params`), the compile cache left at OptiX's
default, the work on the device's stream. Not taken: PTX (above); its one
intersect pipeline holding every tracing kernel, where this backend makes
one pipeline a pass, as Metal makes one a kernel (merging them is tuning);
and its instance level, which it needs for meshes and motion and this
scene does not have yet.

## Phases

0. **What is not Metal's moves into the core,** on the Mac, before any
   CUDA: the list above, each move a refactor with every graph's output
   the same to the bit. **Done 2026-10-19** (branch `cuda`, 315ef86 to
   2aa9c2c; file-mapping.md, "What every backend computes the same, in
   the core"): the scene's block and arrays, each frame's constants
   blocks, the refusals, the U-Net's dispatches, the depth of field's,
   bloom's and ReSTIR PT's sizes, and the structures' plan: `src/metal/`
   437 lines smaller (649 out, 212 in, mostly its calls into the core),
   the core 1,120 larger: eight new units and two grids added to the depth
   of field's, with their prose, and unit tests for the
   refusals, the U-Net's wiring and the structures' plan; rule 7 of the
   boundary check (shaders include shared layouts only). Every graph's
   94 output files the same to the bit and the refusals' words the same;
   three presets at 0 warnings, ctest and `make check` green. Left in
   Metal, each for its reason: the history joined on a copy, the failure
   counts, the submission's protocol, MNEE's capacity. The demo graph's
   frame interval, before (0633a78) and after (2aa9c2c), the marbles at
   t = 120, AC power and an idle GPU, 3 rounds each, interleaved
   (serenity-scratch/phase0/): 1920 x 1080, 17.40 -> 17.45 ms (57.5 ->
   57.3 fps); 3456 x 2234, 58.08 -> 58.14 ms (17.2 fps both; the medians
   of the rounds, one of the before's at 59.49). Each round 0.05 ms
   slower after, 0.3%, the before always run first; not separated
   further.
1. **The core on Linux.** CMake's per-platform selection, the presets,
   the toolchain's Linux section, the fetched dependencies, the shadow
   warning; the core's tests green there, `make check`'s rules extended.
   **Done 2026-10-19:** the Mac-only settings (SDK, deployment target,
   the Objective-C compilers and flags) moved from the presets into
   `CMakeLists.txt`, so one preset name configures on both machines;
   Linux builds the core, the options libraries, the measuring program
   and the tests that need no GPU, and refuses any platform but the two.
   `cmake/toolchain.json` one set of pins per platform, the checker and
   its proof by platform; LLVM 21.1.8's own release on the RTX 3090 Ti's
   machine (its SHA-256 GitHub's digest). On the Mac every compile line of
   the three presets the same as before (350 each), so its output too. On
   the RTX 3090 Ti: the three presets at 0 warnings under clang 18, ctest's
   six tests green under each (UBSan's included), `make format-check` and
   `make tidy` clean. g++'s `-Wshadow` was not one under the pinned clang.
   The review script's test, which needs the codex CLI, left `make check`
   for `make review-check` (the user: our review workflow's, not every
   developer's); it fails by name without the CLI.
2. **The device and the headless seam.** Context, stream, memory,
   readback, the offline target; the headless program split, Metal byte
   identical; the test pattern rendered on the 3090 to a file. The seam is
   one interface and one factory declared in the core, defined by each
   backend's library and chosen by what the headless program links, so
   `src/headless/` names no backend (file-mapping.md, "The headless
   renderer behind an offline renderer"); the alternative, a preprocessor
   switch in its `main.cpp`, was not taken. Two commits, two intents: the
   seam on Metal, then the CUDA layer. OptiX waits for Phase 3.
   **Done 2026-10-19.** The seam (fa3d737): `core/presentation/
   offline_renderer.h`, the run in `headless/run.{h,cpp}`, Metal's side in
   `metal/frame/offline_frames.{h,cpp}`, rule 8 of the boundary check;
   every graph's 94 files the same to the bit, the refusals' words the
   same. The CUDA layer: `src/cuda/` (the driver API's device, modules,
   memory, submission and offscreen target; the test pattern; a renderer
   that refuses by name every pass it does not build yet; the offline
   renderer and its factory), `cmake/CudaModules.cmake` (nvcc's flags in
   one place, sm_86 machine code alone, no fast math) and the one
   bytes-to-header script both backends' libraries use; `nvcc` 13.2.86 and
   the driver 595.91.07 pinned; rules 3b and 7 of the boundary check over
   `src/cuda/`. `tests/cuda`: unit tests of the backend's own logic, which
   run no GPU (the submission's bookkeeping, `cuda/device/
   frame_sequence.h`, apart from the driver; which passes it builds,
   `unbuilt_pass`); and one integration test on the RTX 3090 Ti,
   `serenity-headless` writing the pattern as PNGs, each the CPU's (red
   and green to the bit, blue within a level), refusing a graph it cannot
   render yet, and naming the machine. A first version had more tests
   that ran the GPU to check logic (the protocol, the refusal) or repeated
   the integration test (the pattern against the CPU, rendered twice); by
   the user's rule (unit tests test units) the logic moved into units and
   the repeats were removed. The `headless` integration test renders the path
   tracer, so it joins Linux's tests in Phase 3.
3. **The scene and the path tracer.** Scene buffers, the two structures,
   the shapes' intersection programs, the materials, lights and sampler
   with their CPU checks, the path pass and its accumulation, and the
   tone map: the reference graph (`graphs/path.toml`) on the 3090. Its
   design: "Phase 3's design", above. The tone map moved here from Phase
   5: the reference graph ends in it, and it is the data-set script's
   target graph, so without it the phase renders nothing anyone asked
   for.
   **Done 2026-10-19.** The core: `SERENITY_SHARED_FUNCTION` on the
   shared layouts' 28 functions, the scene block's arrays pointers under
   `nvcc`; Metal's output the same to the bit (94 files against 4a021ec)
   and its refusals' words the same. The build: OptiX's headers fetched
   at `f1f6dd8`, the programs compiled to OptiX-IR by a second function
   in `cmake/CudaModules.cmake`. The backend: OptiX's context; a pass's
   pipeline (module, three program groups, stack sizes, binding table);
   pinned memory and the frame upload; the pitched and level images; the
   scene's buffers, transforms and glows; the two structures, whose hit
   records carry the table and the shapes' arrays, so one intersection
   program serves every pass; the failure count; the accumulated image
   and the bloom pyramid; the path pass, a ray generation program, and
   the tone map, four kernels; the renderer building them. Device code:
   31 of Metal's headers, the path slice, translated line for line (a
   script for the mechanical pass, then by hand), on `cuda/math/
   vector.cuh`'s operators under Metal's names. What the translation
   found: `nvcc` does not warn of an enumerator a switch leaves out (its
   front end has no `-Wswitch`), so each kind dispatch traps after its
   switch (`cuda/device/layout.cuh`, `unhandled_kind`) where Metal refuses
   at compile time, and the three switches whose cases broke now return;
   a core constant table read at run time is copied to device memory at
   compile time (the noise's gradients); a `float3` constant becomes
   three floats. Tests: the probe tests of eight families (sampler,
   camera, shapes, textures, lights, film, encodings, BSDFs) are one
   source for both backends, each backend's runner behind
   `<probe_runner.h>` (SF.12; a quoted include found Metal's beside the
   shared helper in either build), and pass on the 3090 against the
   core's computations; the integration test renders the reference graph
   on a moving scene and writes the same bytes twice; `tests/
   test_headless.sh` runs on both platforms, so the failure count's path
   (a scene whose light overflows) is checked on the 3090 too. The
   probes of the families that trace (the structures, leaving a surface,
   the path loop) need OptiX launches in their harness and come with
   Phase 4's. Gates: both machines' three presets at 0 warnings, ctest
   green under each, `make check` green (clang-tidy over the new host
   code on the 3090). The image: the reference graph at 640 x 360, 256
   samples, t = 120, on both machines, side by side: the same picture,
   each channel's mean within 0.01 of a level, the noise the same grain
   (the same numbers drawn); the marbles' accumulated radiance as PFM,
   mean 0.188579 on both. Not timed: the demo graph comes in Phase 5.
   For tuning, after the port: the scene's arrays from global memory
   (Metal reads them from its constant space), the structures' build
   flags and compaction, boxes as triangles on the RT cores, one
   pipeline for every tracing pass.
4. **ReSTIR PT and MNEE.** Initial, temporal, spatial, shade, duplication,
   MNEE's work list; the features pass.
5. **The denoised frame.** The denoiser on the tensor cores (matched to
   PyTorch), the fill, depth of field: the demo graph, its frame time.
6. **The data set on the 3090.** The script run there, its manifest
   naming the machine.

Each phase ends with its gates once; the report gives the frame time where
there is a frame.

## Rollback

The backend is a directory and a branch of the build. `src/metal/`
changes only by what moves into the core (Phase 0) and its headless half
moving behind the seam, each a commit of its own guarded by Metal's
output being the same to the bit; those stand on their own whether or
not the CUDA backend stays. Removing `src/cuda/` and the Linux branch of
the build returns the project to one backend.

## Not decided here

- The window on the 3090: a thin graphics API for presentation alone
  (CUDA interop), its own step after Phase 6.
- AMD's Strix Halo (HIP, HIPRT) and the Xbox Series X (Direct3D 12):
  later milestones. Whether hardware ray tracing is reachable in the
  retail console's developer mode, or only through the GDK on a
  development kit, is to be checked before that milestone.
- MNEE's float atomics make ReSTIR PT's radiance order-dependent on this
  GPU as on Metal (QUEUE.md, Next 0b, the user's call); the features and
  path graphs are exact.
