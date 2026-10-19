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
(`src/headless/options.cpp:127`).

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
| traces none: ReSTIR PT's shade and duplication, denoise, tone map, depth of field, checkerboard fill, accumulation, test pattern | a CUDA kernel | `passes/<pass>/<pass>.cu`, compiled to the machine's code (sm_86) |

Which is which is read from the Metal passes' includes (2026-10-19): the
six that trace reach `acceleration/trace.metal.h` through the integrator;
shade and duplication include ReSTIR PT's header but trace nothing, and
the denoiser's full-size albedo is the features pass run again, not the
denoiser's own tracing. Every pass runs on the frame's stream in the
schedule's order, as Metal's encoder does.

The headers say `optixTraverse` traverses and fills the hit object
without running closest-hit or miss; they do not say in words that a
custom primitive's intersection program runs during it. It must, for a
traversal to hit one at all, and Cycles' shadow rays rely on it; Phase 2's
first launch proves it on this machine.

**What it ports first, and what is tuning later.** The first port is
Metal's design carried over: shapes as bounding boxes with their own
intersection programs, the two structures for the two questions of
principle 11 (every shape, the surfaces alone, each a geometry structure
traced by its own handle), the same pass boundaries and layouts. Boxes as
triangles on the RT cores, an instance level with mask bits, reordering,
fused passes: each is this machine's tuning, measured one at a time after
the port runs, not a decision of the port.

**Device code includes `core/contracts/` only.** CUDA is C++, so a kernel
could include the core's sampling or material code; it does not. The
GPU tests check each backend against the core's CPU computations, and a
kernel compiled from the same source would only check that it compiles
twice. As on Metal (file-mapping.md, "The contracts": layouts only, never
math), enforced by `tools/check_boundaries.sh`.

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
user's yes in Phase 0.

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
(`features_test.cpp`'s check). Metal's argument-table and encoding tests
have no counterpart. No test matches the 3090's output to the Mac's.

**The measurement.** The demo graph's frame time at 1920 x 1080 headless,
by CUDA events around a frame's work on its stream, with nothing else on
the GPU (training stopped). Per-pass events when a change targets a pass.

## Phases

0. **The core on Linux.** CMake's per-platform selection, the presets,
   the toolchain's Linux section, the fetched dependencies, the shadow
   warning; the core's tests green there, `make check`'s rules extended.
1. **The device and the headless seam.** Context, stream, memory,
   readback, the offline target; the headless program split, Metal byte
   identical; the test pattern rendered on the 3090 to a file.
2. **The scene and the path tracer.** Scene buffers, the two structures,
   the shapes' intersection programs, the materials, lights and sampler
   with their CPU checks, the path pass and the accumulation: the
   reference graph on the 3090.
3. **ReSTIR PT and MNEE.** Initial, temporal, spatial, shade, duplication,
   MNEE's work list; the features pass.
4. **The denoised frame.** The denoiser on the tensor cores (matched to
   PyTorch), the fill, depth of field, tone mapping: the demo graph, its
   frame time.
5. **The data set on the 3090.** The script run there, its manifest
   naming the machine.

Each phase ends with its gates once; the report gives the frame time where
there is a frame.

## Rollback

The backend is a directory and a branch of the build. Nothing in
`src/metal/` changes but its headless half moving behind the seam, which
Metal's byte identity guards; nothing in `src/core/` changes but the seam's
interface and what Linux's compilers find. Removing `src/cuda/` and the
Linux branch of the build returns the project to one backend.

## Not decided here

- The window on the 3090: a thin graphics API for presentation alone
  (CUDA interop), its own step after Phase 5.
- AMD's Strix Halo (HIP, HIPRT) and the Xbox Series X (Direct3D 12):
  later milestones. Whether hardware ray tracing is reachable in the
  retail console's developer mode, or only through the GDK on a
  development kit, is to be checked before that milestone.
- MNEE's float atomics make ReSTIR PT's radiance order-dependent on this
  GPU as on Metal (QUEUE.md, Next 0b, the user's call); the features and
  path graphs are exact.
