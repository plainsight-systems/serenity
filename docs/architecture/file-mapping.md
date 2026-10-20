# File mapping

How the boxes in [`logical-overview.md`](logical-overview.md) become files,
applying the axes in [`change-axes.md`](change-axes.md): the algorithm laid
over the axes, which gives the file families.

It fixes the layers, the families and the direction of dependencies, and
it names the contracts between families. It was written before the code,
when the directories were empty; it does not define the contracts or list
the kinds: each contract is designed in its own header before the code that
uses it, and each kind arrived with the change that first needed it.

## Layers

- **The core, `src/core/`,** is platform-neutral C++: the scene, its kinds'
  parameters, animation, the frame's schedule, film, output and measurement.
  It uses no GPU or windowing API.
- **The Metal backend, `src/metal/`,** is the only code that uses Metal: the
  device, acceleration structures, and every shader with the host code that
  launches it. **The CUDA backend, `src/cuda/`,** beside it, is the only
  code that uses CUDA and OptiX, on NVIDIA's RTX 3090 Ti, mirroring
  `src/metal/`'s families (`docs/research/2026-10-19-cuda-backend.md`).
  Each vendor has its own API and no layer spans them
  (`../process/MEMORY.md`).
- **The app, `src/app/`, and the headless renderer, `src/headless/`,** are
  the two ways to run a frame: in a window, when the backend's frame
  source asks for one, or to files, with a fixed step.
- **The measuring program, `src/measure/`,** makes a reference from the
  headless renderer's files and judges images against it, on the CPU, with
  the core alone.
- **Scene descriptions, `scenes/`,** and **frame graphs, `graphs/`,** are
  data, at the repository root, in separate files: what is rendered, and
  which passes render it, so one scene runs under any graph.
- **The denoiser's training, `training/`,** is outside the build: a Python
  toolkit, PyTorch, with its own pinned environment (`training/uv.lock`).
  No CMake target, test or graph depends on it. What it hands the renderer
  is a file: the trained network, exported as weights and their layout,
  which the core reads and checks at start-up.
- **Data and models live beside the checkout,** never in it:
  `../serenity-data/` holds the denoiser's data set (`denoiser-v1`, 32.3
  GB, rendered by `scripts/make_denoiser_dataset.py` with a manifest of
  every file's checksum), the training runs, and the exported network
  (`denoiser-v2-model/wide`), which the denoised graph names by a path
  relative to `graphs/`.

Dependencies point one way: the app and the headless renderer depend on the
backend and the core, the measuring program on the core alone, the backend
on the core, and the core on nothing platform-specific. Nothing in
`src/core/` uses a platform API or includes from a backend or a program,
and the measuring program includes from no backend;
`tools/check_boundaries.sh` enforces both, and `make check` proves each
rule fires.

## Composition

No family knows a feature. A feature is a composition of kinds from several
families, made in the scene description:

- **A firefly** is a sphere *shape*, a sphere *light*, a flight-path
  *animation*, and its color, intensity and path in the scene's data.
- **A glass marble** is a sphere *shape* wearing a dielectric *material*.
- **The table** is a box *shape* wearing a rough *material* whose color comes
  from a wood *texture*.
- **ReSTIR PT** is the *integrator*'s paths, their direct light taken at
  every vertex, reused whole through *sample reuse*, which the
  *integrator* shifts from one pixel or frame to another.
- **Manifold next event estimation** is the *integrator*'s strategy for
  light through glass, solved by a *pass* of its own from a work list.
- **The denoiser** is a *pass* reading the radiance and the *film*'s
  features, its trained weights data read by the core, its history kept by
  the *frame graph*.
- **The reference** is the same integrator with reuse switched off,
  accumulating on the *film*.

Adding a firefly to the scene, or a new kind of thing that glows, changes
data or adds a kind; it does not change a family that already exists.

## The families

| Directory | Axis | Holds |
|---|---|---|
| `src/core/contracts/` | contract | the interfaces between axes (below) |
| `src/core/scene/` | Scene content | reading a scene description into its kinds |
| `src/core/animation/` | Animation | each motion kind, placing an object or light at *t*, and each glow kind, setting a light's brightness at *t* |
| `src/core/camera/` | Camera | each camera kind's parameters, and its pose at *t* |
| `src/core/shapes/` | Shape | each shape kind's parameters, bounds and geometry for the acceleration build |
| `src/core/materials/` | Material | each material kind's parameters |
| `src/core/textures/` | Texture | each texture kind's parameters |
| `src/core/lights/` | Light | each light kind's parameters |
| `src/core/media/` | Medium | each medium kind's parameters |
| `src/core/film/` | Film | what a pixel accumulates and outputs |
| `src/core/sampling/` | Sampler | static sampling data the CPU builds: ReSTIR PT's pairing textures |
| `src/core/passes/` | Pass | each pass's settings as a frame graph writes them, and what the tone map computes |
| `src/core/denoise/` | Pass | the denoiser's trained network, read from its export and checked against contract 18 |
| `src/core/frame/` | Frame graph | the passes, their order, the images between them, the history kept across frames, and the light and output sizes; reading frame graph files |
| `src/core/output/` | Output | each file format's writer |
| `src/core/measurement/` | Measurement | error against the reference, and timing reports |
| `src/metal/device/` | GPU backend | the device, libraries and pipelines, resources, encoding and synchronization |
| `src/metal/scene/` | Scene content | which of the scene's arrays the GPU holds, and their addresses |
| `src/metal/acceleration/` | Acceleration | building and updating the acceleration structures |
| `src/metal/shapes/` | Shape | each shape kind's intersection and point sampling |
| `src/metal/materials/` | Material | each material kind's BSDF |
| `src/metal/textures/` | Texture | each texture kind's evaluation |
| `src/metal/lights/` | Light | each light kind's emission and sampling |
| `src/metal/media/` | Medium | each medium kind's transmittance |
| `src/metal/camera/` | Camera | each camera kind's ray generation |
| `src/metal/math/` | shared shader math | orthonormal frames, and warps from the unit square to the disk and the hemisphere: mappings every family uses, owned by none |
| `src/metal/film/` | Film | what a pixel accumulates: folding a frame into the mean |
| `src/metal/sampler/` | Sampler | deriving random numbers from pixel, frame and purpose |
| `src/metal/light_selection/` | Light selection | each strategy for choosing a light |
| `src/metal/integrator/` | Integrator | the path loop, its strategies, and the target function |
| `src/metal/reuse/` | Sample reuse | the reservoir, the merge rule, and temporal and spatial reuse |
| `src/metal/passes/` | Pass | each pass, its shader and its launcher together |
| `src/metal/frame/` | Frame graph | running the core's schedule on Metal |
| `src/core/presentation/` | Presentation | the seams between the programs and a backend: the window's frame source and its client; the headless renderer's offline renderer |
| `src/app/` | Presentation | the macOS window, its event loop, and each frame's inputs, at the instant the backend's frame source gives |
| `src/headless/` | Presentation | rendering frames to files at a fixed step, on whichever backend the build links, naming none |
| `src/measure/` | Measurement | the measuring program's command line: a reference made from files, and images judged against it |
| `scenes/` | Scene content | scene descriptions, as data |
| `graphs/` | Frame graph | frame graphs, as data |
| `training/` | outside the build | the denoiser's training toolkit: data loading, the network, the loss, training, the test, the export, and the check of the renderer's pass against PyTorch |

A kind lives in two halves with the same axis and the same name: its
parameters and anything the CPU needs in the core (a sphere's bounds, for
the acceleration build), and its shader in the backend (the sphere's
intersection). The halves change together when the kind changes, and are
kept as a pair, the way a shader and its launcher are.

## The contracts

Each contract has one owner and is read by the families listed; families
depend on contracts, not on each other.

| # | Contract | Owned by | Read by |
|---|---|---|---|
| 1 | **Surface interaction**: position, in the world and the shape's own coordinates, normals, the side arrived from, material, the medium inside; (u, v) with the first texture that needs them | Shape | Material, Texture, Integrator, Sample reuse |
| 2 | **BSDF**: a material resolved at a surface, and evaluate, sample, pdf and lobes on it; and its packed form, 16 bytes, for ReSTIR PT's G-buffer of primary hits | Material | Integrator, Sample reuse |
| 3 | **Emitter**: a direction toward the light drawn from numbers, the radiance along it, the distance to it, the pdf; the radiance it emits; which light, if any, a shape is; and its presampled form, ReSTIR PT's light tiles (`light_tile.h`, 32 bytes an entry), drawn by power from the frame's power pyramid, whose layout it holds | Light | Light selection, Integrator, Sample reuse |
| 4 | **Light sample**: which light and which point on it, in its own coordinates (`light_sample.h`: the light's record index with a valid bit and the point in two 16-bit numbers, equal-area octahedral for a sphere, 8 bytes, as RTXDI stores them), and what that point gives a surface | Light | Sample reuse, Integrator |
| 5 | **Film outputs**: radiance, normal, depth, motion per pixel; and ReSTIR PT's G-buffer of primary hits, which every shift starts from (`primary_hit.h`, 48 bytes). ReSTIR DI's surface record and path state (`surface_record.h`) were removed with ReSTIR DI, 2026-10-17 | Film | Pass, Sample reuse, Integrator (the shift) |
| 6 | **Frame images**: the images between passes and the history across frames; first, the accumulated image | Frame graph | every Pass |
| 7 | **Frame constants**: the frame's time and index, the size of the image written, the frames the accumulated image holds, and the camera's jitter where the frame takes it | Frame graph | every Pass |
| 8 | **Camera**: where the frame is seen from, and that view framed for an image | Camera | Scene content, Frame graph, every Pass that generates rays |
| 9 | **Texture reference**: which texture a parameter takes its value from, or none | Texture | Material |
| 10 | **Transform**: where an instance of a geometry is, from its own coordinates to the world's | Shape | Animation, Light, Acceleration |
| 11 | **Obstacles**: what a motion must keep clear of, as distances and touches against the still shapes | Animation | Scene content (answers) |
| 12 | **Medium**: what fills a shape's inside, and the share of light a stretch through it keeps | Medium | Integrator; named by Shape |
| 13 | **Linear image**: a rendered image's radiance on the CPU, three floats a pixel, before tone mapping | Film | Output, Measurement |
| 14 | **Reservoir**: ReSTIR DI's light reservoir (`reservoir.h`, RTXDI's packed reservoir); removed with ReSTIR DI, 2026-10-17 | (none) | (none) |
| 15 | **Path reservoir**: ReSTIR PT's one path a pixel by its numbers' stream and its reconnection vertex (a primitive and a point in its own coordinates), its F, contribution weight, count and cached Jacobian (`path_reservoir.h`, 64 bytes, ReSTIR PT Enhanced's Algorithm 1); its settings beside it (`path_reuse_settings.h`) | Sample reuse | Pass, Integrator (the shift), Frame graph |
| 16 | **MNEE's work list**: a request a pixel ReSTIR PT's path loop leaves for the manifold solve where a light's straight segment crosses glass (the receiver, the light sample, the crossings, the weight; 128 bytes), and the list's header: its count, capacity and the solve's indirect grid (`mnee_request.h`) | Integrator | Pass (the MNEE pass), Frame graph |
| 17 | **The denoiser's features**: albedo, world normal, view depth and 2.5D motion a pixel, along the light's own camera rays, through glass and mirrors by primary surface replacement (`features.h`) | Film | Pass (the denoiser), the headless renderer's `--features` |
| 18 | **The denoiser's network as it runs**: its constants (scale 2, 30 input channels, the frame's jitter among them, 124 scores, spacings 1, 2 and 4, the albedo floor, the PU transfer), the threadgroup shapes host and shaders share, and its per-frame constants (`denoise.h`) | Pass | the core's loader, the Metal denoise pass, host and shaders |
| 19 | **Ray mask**: the classes of shape, a light or a surface, as bits, and the two masks a ray carries, everything or the occluders (the surfaces), with which purpose of ray carries which (`ray_mask.h`) | Acceleration | Integrator, Pass (the features) |
| 20 | **Checkerboard**: which pixels a frame of the network's input traces, the packed slot of a traced pixel's reservoir, the traced grid, the frame before's traced pixel beside one, and the fill's constants (`checkerboard.h`) | Frame graph | Sample reuse, Pass (ReSTIR PT's and the fill) |
| 21 | **Depth of field**: the circle of confusion's scale, focus and largest radius for a frame, the half-resolution grid and its tiles, the gather's rings (`depth_of_field.h`) | Pass | the Metal depth-of-field pass, host and shaders; the frame graph, which fills its constants from the camera's lens |

Each is designed on its own, header first, before the code that uses it.
Contract 7 was the first written: `src/core/contracts/frame_constants.h`.
Contracts 8 and 9 came with the first scene: `camera.h` and
`texture_reference.h` beside it. Contracts 1 and 2 came with the path
tracer: `surface_interaction.h` and `bsdf.h`, the shared layouts, with the
functions' semantics stated there and written by each backend; contract 3,
`emitter.h`, with them. Contract 6 starts as one image, the accumulated
mean a converging pass folds its frames into (`metal/frame/accumulation.h`),
and grows into the images between passes when a pass first reads another's.
Contract 13, `linear_image.h`, came with the reference: the radiance Film
hands to Output's files and Measurement's errors. Contracts 4, 5 and 14
came with ReSTIR DI: `light_sample.h`, `surface_record.h` and
`reservoir.h`, with `reuse_constants.h`, Sample reuse's per-frame values,
beside them; contract 6 grew by ReSTIR's records and reservoirs, ping-ponged
across frames (`metal/frame/frame_images.h`); ReSTIR DI's removal
(2026-10-17) took contract 14, contract 5's surface record and
`reuse_constants.h` with it, leaving contract 4 to ReSTIR PT's
reconnections. Contract 15 came with ReSTIR
PT, with contract 3's light tiles and contract 5's G-buffer of primary hits;
contract 16 with manifold next event estimation; contracts 17 and 18 with
the denoiser, the frame graph keeping the feature images and the
denoiser's history (`metal/frame/feature_images.h`, `denoise_images.h`), and
contract 7 growing by a second ring of constants: a denoised frame's
passes before the denoiser run at the light size, the rest at the output's.
Contract 19, `ray_mask.h`, came with Milestone 4's ray masks, and
contract 20, `checkerboard.h`, with its checkerboard rendering; contract
21, `depth_of_field.h`, with the lens applied after the denoiser.
The camera is one contract for every camera, as Falcor's `CameraData` is,
since a frame has one camera and cameras differ by parameters, not code.

**How a layout crosses into a shader.** The data a contract passes to the
GPU (a material's parameters, a light sample, a primary hit) has one
definition, in a core header that Metal's shading language can also
include: plain fixed-size fields, nothing only the host has. The host writes
the bytes the shader reads, with no second definition to drift. This shares
layouts only, never math: a shared layout is a header that includes
`core/contracts/shared_layout.h` (the contracts, the kinds' records, and
the encodings beside them), and a shader includes nothing else from the
core. CUDA's device code includes the same headers, and only these: CUDA
is C++ and could include the core's math too, which would leave the GPU
tests checking one source compiled twice. `tools/check_boundaries.sh`
keeps both backends' device code to them.

## The algorithm over the families

![Each logical box mapped to the file families that implement it, by layer](file-mapping.svg)

The per-pixel work, from primary visibility to tone mapping, lives in the
backend; the core holds what is rendered, how it moves, and the schedule:
the core decides what a frame computes and the backend computes it. Light is the
widest box, and it is the split the change axes insist on: the integrator,
light selection, sample reuse, and the lights, materials and textures it
evaluates, each a family of its own.

## Two queues for a denoised frame

The frame's two halves (`logical-overview.md`, "The frame"; the axes in
`change-axes.md`) land in the families that already own each concern; no
file is new:

| What | Family | File |
|---|---|---|
| the second queue, each frame slot's second command allocator and buffer, the event the light half signals and the image half waits for, a two-part submission's commit and settling | GPU backend | `src/metal/device/submission.{h,cpp}` |
| where the frame splits, the barriers on each side, the hand-off images kept one set per frame slot (the features at both sizes, and the light's radiance, which the checkerboard's fill writes), and how many parts a frame has | Frame graph | `src/metal/frame/renderer.{h,cpp}` |
| the per-slot radiance hand-off | Frame graph | `src/metal/frame/denoise_images.{h,cpp}`, beside the denoiser's other memory |
| asking the renderer for its parts and beginning the submission with them | Presentation | the window through `render_to_window()` (since 2026-10-18 from Metal's frame source, `src/metal/frame/display_frames.cpp`), `src/headless/main.cpp` through `render_to_offscreen()` (both in `renderer.cpp`), unchanged themselves |
| a frame's time when frames overlap: from when it began, or the frame before it finished if later, to when it finished | Measurement | `src/core/measurement/frame_times.{h,cpp}` |

The passes (`src/metal/passes/`) do not change: the denoise and tone map
passes record into the image half's encoder as they recorded into the
frame's one.

## Ray masks

Which shapes each ray finds (`logical-overview.md`, principle 11; the axes
in `change-axes.md`) adds one contract and one shared layout, and changes
the families that already own each part:

| What | Family | File |
|---|---|---|
| the classes of shape and the two masks, as bits a two-level structure's instances would carry; which ray purpose carries which (contract 19) | Acceleration (contract) | `src/core/contracts/ray_mask.h`, new |
| a structure for each mask, every shape and the surfaces alone, each built once when nothing in it moves and per frame slot when something does; the surfaces' shape table; the block the shaders bind | Acceleration | `src/metal/acceleration/scene_acceleration.{h,cpp}` |
| the block's shared layout: each structure as a shader binds it, and the table | Acceleration | `src/metal/acceleration/structure_block.h`, new |
| a query by mask: the nearest shape, whether any blocks, and the candidates' shapes for a query of the integrator's own | Acceleration | `src/metal/acceleration/trace.metal.h` |
| each ray's mask, at each site that traces (ReSTIR PT's passes ask through `nee.metal.h` and `mnee.metal.h`, unchanged themselves) | Integrator | `src/metal/integrator/{path,nee,shift,mnee}.metal.h` |
| the features' camera rays: everything | Pass | `src/metal/passes/features/features.metal` |
| the scene's view holding the block, and the kernels binding it where they bound the structure | Scene content, Pass | `src/metal/scene/scene_block.metal.h`, each pass's `.metal`, `src/metal/passes/scene_binding.h`, `src/metal/frame/frame_resources.h` |
| handing the acceleration the shapes that are lights, as it is handed the ones that move | Frame graph | `src/metal/frame/renderer.cpp` |

## Checkerboard rendering

Tracing half the light's pixels a frame, as every denoised frame does
(`logical-overview.md`, "The frame"; the axes in `change-axes.md`), adds
one contract and one pass, and changes the families that own each part:

| What | Family | File |
|---|---|---|
| which pixels frame *n* traces, the packed slot two pixels of a row share, the grid the light's kernels dispatch over, the frame before's pixel beside one, the fill's constants (contract 20) | Frame graph (contract) | `src/core/contracts/checkerboard.h`, new |
| a graph's light checkerboarded exactly when it is the network's input (ReSTIR PT with the features pass; since the production sweep, before it the denoise pass), and its spatial reuse's sigma one the traced lattice can pair at | Frame graph | `src/core/frame/schedule.{h,cpp}` |
| whether the light is checkerboarded, as ReSTIR PT's kernels read it | Sample reuse (contract 15's sibling) | `src/core/contracts/path_reuse_settings.h` |
| each kernel's pixel and slot through the contract; the spatial partner over the traced lattice; the temporal input in the other field | Pass, Sample reuse | `src/metal/passes/restir_pt/restir_pt.metal.h`, each `src/metal/passes/restir_pt_*/*.metal`, `src/metal/reuse/pairing.metal.h` |
| the light's kernels dispatched over the traced grid | Pass | each `src/metal/passes/restir_pt*/*.cpp`, `src/metal/frame/frame_resources.h` |
| the fill: the hand-off written from the traced half and the frame before's | Pass | `src/metal/passes/checkerboard_fill/`, new |
| the fill's history, every pixel's handed-off light and depth, two by the frame's parity | Frame graph | `src/metal/frame/checkerboard_images.{h,cpp}`, new |
| the fill writing the hand-off, the frame before's slot, the reuse textures over the lattice | Frame graph | `src/metal/frame/renderer.{h,cpp}` |

Contract 20 is the one place the field and the packing are written, read
alike by the host's dispatches and every kernel, so no two can disagree
about which pixel a thread or a slot is.

## Frame pacing

The window paced by its backend behind a frame source
(`logical-overview.md`, "The frame", "Pacing"; the axes in
`change-axes.md`), Metal's through a display link, adds the seam, the
Metal source, the first Objective-C++ in the program, and changes the
families that own each part:

| What | Family | File |
|---|---|---|
| the seam: a frame source, which owns the loop, and its client, which handles the window's events and gives each frame's inputs; a frame rate range; what a shown frame reports | Presentation (contract) | `src/core/presentation/frame_source.h`, new |
| the window's side of it: the size wanted, the frame's inputs from the instant it will be shown, the histories, the frame times, the title | Presentation | `src/app/window_frames.{h,cpp}`, new, backend-neutral |
| composing a backend's source with the window, and nothing of how it paces | Presentation | `src/app/main.cpp` |
| the frame's time, from the source's instants, counted from the first | Presentation | `src/app/clock.{h,cpp}` |
| waiting for events at most a given time | Presentation | `src/app/window.{h,cpp}` |
| Metal's frame source: the loop, waiting in the window's events while the display link calls frames; the size followed and a drawable of the size before skipped; the frame rendered into the drawable; a frame's failure kept and rethrown from the loop; which submissions are frames | Frame graph | `src/metal/frame/display_frames.{h,cpp}`, new |
| the display link on the presenter's layer: its rate and latency, the main run loop it is on, the drawable and the instant it hands over, through a callback that must not throw | GPU backend | `src/metal/device/display_link.{h,mm}`, new: the header C++, the implementation Objective-C++, the one file that names Core Animation's Objective-C types |
| the layer, its drawable size set again when something else (SDL) changed it; whether a drawable is the size it renders at | GPU backend | `src/metal/device/presenter.{h,cpp}` |
| a frame rendered into a drawable, refused if the drawable is not the presenter's size | Frame graph | `src/metal/frame/renderer.{h,cpp}`, `render_to_window()` |
| what a frame's GPU time is when the display paces frames | Measurement | `src/core/measurement/frame_times.h` (its header) |
| the app's files but `main.cpp` free of any backend, and all of `src/app/` of the display link | build | `tools/check_boundaries.sh`, rule 6, and its proof |
| Objective-C++ in the build, its compiler pinned with the others | build | `CMakeLists.txt`, `CMakePresets.json`, `cmake/toolchain.json`, `tools/check_toolchain.py` |

The Objective-C++ stays in one file of the backend, and its header in
the backend too: the app includes the Metal source's header, from
`main.cpp` alone, and sees neither Objective-C nor the display link. The
CUDA backend's source, when its window comes, sits beside it, as
`src/cuda/frame/`, implementing the same interface with its own loop.

## Depth of field after the denoiser

The lens applied after the network (`logical-overview.md`, principle 12;
the axes in `change-axes.md`) adds one contract and one pass, and changes
the families that own each part:

| What | Family | File |
|---|---|---|
| the depth-of-field pass's kind and its place, directly after the denoise pass and before the presenting pass; a denoised graph's light traced through the pinhole of its camera (`traced_inputs`) | Frame graph | `src/core/frame/schedule.{h,cpp}` |
| the circle of confusion a thin lens gives a depth, in pixels of an image | Camera | `src/core/camera/thin_lens.{h,cpp}` |
| the pass's constants, grid, tiles and rings (contract 21) | Pass (contract) | `src/core/contracts/depth_of_field.h`, new |
| the blur: presort to half resolution, tiles of the largest near radius, their dilation, the gather, the median, the composite | Pass | `src/metal/passes/depth_of_field/`, new |
| its half-resolution images and tile buffers | Frame graph | `src/metal/frame/depth_of_field_images.{h,cpp}`, new |
| the camera traced through a pinhole in a denoised graph; the pass recorded in the image half with the full-size depth and its constants | Frame graph | `src/metal/frame/renderer.{h,cpp}`, `src/metal/frame/frame_resources.h` |
| the data set's scenes written without a lens | outside the build | `scripts/make_denoiser_dataset.py` |

## The camera's jitter

A fixed low-discrepancy sequence in place of each pixel's random point
for ReSTIR PT's camera rays, given to the network (the axes in
`change-axes.md`):

| What | Family | File |
|---|---|---|
| the sequence, its phases and frame *n*'s offset, constexpr | Sampler | `src/core/sampling/jitter.h`, new |
| which graphs' frames take it (every ReSTIR PT graph's light and features) | Frame graph | `src/core/frame/schedule.{h,cpp}` |
| the offset and whether the frame takes it, in the frame's constants | Frame graph (contract 7) | `src/core/contracts/frame_constants.h` |
| the camera ray's point in the pixel: the frame's jitter or the pixel's own | Camera | `src/metal/camera/thin_lens.metal.h` |
| the jitter written into the light half's ring, not the output's | Frame graph | `src/metal/frame/renderer.cpp` |
| two input channels of the network: the contract, the loader's check, the kernel, the PyTorch model, its data and its export | Pass, outside the build | `src/core/contracts/denoise.h`, `src/core/denoise/network.cpp`, `src/metal/passes/denoise/denoise.metal`, `training/{data,model,export,denoise_check}.py` |
| each frame's jitter in the data set's manifest | outside the build | `scripts/make_denoiser_dataset.py` |

## Light tiles drawn by power

RTXDI's local-light presampling in place of a uniform draw (the axes in
`change-axes.md`):

| What | Family | File |
|---|---|---|
| the power pyramid's layout: its levels, where each lies, its threadgroup | Light (contract 3) | `src/core/contracts/light_tile.h` |
| a light's power, by kind | Light | `src/metal/lights/emitter.metal.h` |
| the pyramid's build, a draw by descending it, a light's probability | Light selection | `src/metal/light_selection/power_light.metal.h`, new |
| the pyramid built and the tiles drawn from it each frame; every density ReSTIR PT weighs a light by asked of it, this frame's and, for the temporal reuse's last frame, last frame's | Pass, Integrator, Scene content | `src/metal/passes/restir_pt/restir_pt.metal.h`, `src/metal/passes/restir_pt_initial/`, `src/metal/passes/restir_pt_temporal/restir_pt_temporal.metal`, `src/metal/passes/restir_pt_spatial/restir_pt_spatial.metal`, `src/metal/integrator/{nee,shift}.metal.h`, `src/metal/scene/scene_block.metal.h` |
| the two pyramids, this frame's and last frame's by parity | Frame graph | `src/metal/frame/frame_images.{h,cpp}`, `src/metal/frame/renderer.cpp`, `src/metal/frame/frame_resources.h`, `src/metal/passes/bindings.h` |

## The network's input, one way

The light the network reads traced alike wherever it is made
(`logical-overview.md`, principle 13):

| What | Family | File |
|---|---|---|
| a graph's light is the network's input exactly when it has ReSTIR PT's passes and the features pass; such a graph traces checkerboarded, through a pinhole and at the jitter | Frame graph | `src/core/frame/schedule.{h,cpp}` (`feeds_network`, `traced_inputs`, `camera_jitter`) |
| the fill's own history, so both graphs ask it alike | Frame graph | `src/metal/frame/checkerboard_images.{h,cpp}` |
| the fill and its output in a graph without the network: the light the presenting pass shows and the headless renderer writes; a readback of the light the network is handed, and its jitter | Frame graph | `src/metal/frame/renderer.{h,cpp}` |
| the test: the input graph's light and features are the denoised graph's, to the bit | Frame graph (test) | `tests/gpu/features_test.cpp` |

## What every backend computes the same, in the core

Before the CUDA backend (`docs/research/2026-10-19-cuda-backend.md`,
Phase 0), what `src/metal/` computed without Metal moves into the core, so
the second backend does not write it again. Each move keeps Metal's output
the same to the bit and its refusals' words the same; the Metal files keep
how they allocate, bind and record.

| What | Axis | Was | Now |
|---|---|---|---|
| the scene's block, the 19 addresses its shaders read the scene through | Scene content | `src/metal/scene/scene_block.h` | `src/core/scene/scene_block.h` (a shared layout; how Metal's shaders read it, in the constant address space, stays in `scene_block.metal.h`) |
| the list of the scene's arrays, each with its field of the block | Scene content | `src/metal/scene/scene_buffers.cpp` | `src/core/scene/scene_arrays.{h,cpp}` |
| each frame's constants blocks, filled from the frame's inputs: the frame constants, the features', ReSTIR PT's reuse, the denoiser's, the fill's; the grid ReSTIR PT's kernels run over | Frame graph | `src/metal/frame/renderer.cpp` | `src/core/frame/frame_constants.{h,cpp}` |
| what a scene or a frame may not be, refused before a frame begins: a time past float's range; a ReSTIR PT scene's media, material values, shapes and lights past what its G-buffer, light tiles and MNEE's requests pack; a glowing light that is not one; a frame before whose camera cannot be framed | Frame graph | `src/metal/frame/renderer.cpp` | `src/core/frame/refusals.{h,cpp}`, each a reason or none, as `frame::invalid` gives; the backend throws its own error with it |
| the denoiser's network as dispatches: each level's size and channels, which activations each convolution and pool reads and writes, the padded grid | Pass (denoise) | `src/metal/frame/denoise_images.cpp` | `src/core/denoise/unet.{h,cpp}`; each activation's memory (its pitch, stride and slack) and each dispatch's tiling stay the backend's |
| the depth of field's half size and its tile grid, written twice | Pass (depth of field) | `src/metal/frame/depth_of_field_images.cpp`, `src/metal/passes/depth_of_field/depth_of_field.cpp` | `src/core/passes/depth_of_field.{h,cpp}` |
| the bloom pyramid's level sizes | Pass (tone map) | `src/metal/frame/frame_images.cpp` | `src/core/passes/bloom.h` |
| ReSTIR PT's arrays and their sizes for each kind of reuse | Sample reuse | `src/metal/frame/frame_images.cpp` | `src/core/passes/restir_pt_arrays.{h,cpp}` |
| the acceleration structures' plan: which shapes each holds, which boxes move and their shapes' bounds, every box at rest, the occluders' table; a frame's moving boxes placed | Acceleration | `src/metal/acceleration/scene_acceleration.cpp` | `src/core/shapes/structure_plan.{h,cpp}`; the builds stay the backend's |
| device code includes the core's shared layouts only | build | none | `tools/check_boundaries.sh`, a rule and its proof |

Left in the backend, each for a reason: the history joined on a copy and
kept once the backend's memory exists (the rule is already the core's,
`frame::ReuseHistory`; what is left is the backend's error and its order of
allocation); the failure counts' bookkeeping and the submission's
protocol (each is how Metal's unified memory and events work; CUDA's
differ); MNEE's capacity (one request a pixel, contract 16 states it).

## The headless renderer behind an offline renderer, and the CUDA device

The CUDA backend's Phase 2 (`docs/research/2026-10-19-cuda-backend.md`):
the headless program renders through a seam both backends implement, and
the CUDA backend's device layer renders its first pass, the test pattern,
on the RTX 3090 Ti, to a file.

**The seam.** What the headless program asks of a backend, from what it
does today: render a frame's sample, wait for it, read back the displayed
image, the accumulated image or the feature images, the failure counts,
and wait for everything at the end. One interface in the core, as the
window's frame source is, and one factory declaration beside it that each
backend's library defines; the headless program's executable links the
backend its platform builds, so no file of `src/headless/` names one, not
even `main.cpp`. The alternative, a preprocessor switch in `main.cpp` on
the platform, is a compile-time switch where the link already is one.
Nothing in a frame's work changes: the window's loop and the demo graph do
not pass through it, so it is guarded by the output being the same to the
bit and the refusals' words, not by frame time.

| What | Axis | Where |
|---|---|---|
| the offline renderer: render a sample, wait, read back the displayed, accumulated and feature images, the failure counts, finish; the feature images read back; the factory each backend defines | Presentation | `src/core/presentation/offline_renderer.h`, new |
| one run: the plan's frames rendered, checked and written through the offline renderer, what it prints | Presentation | `src/headless/run.{h,cpp}`, from `main.cpp` |
| the command line parsed, the graph and scene loaded, the plan made, the run composed with the factory's renderer | Presentation | `src/headless/main.cpp` |
| Metal's offline renderer: its device, submission, target and renderer, and the factory | GPU backend | `src/metal/frame/offline_frames.{h,cpp}`, new |
| no file of `src/headless/` includes a backend | build | `tools/check_boundaries.sh`, a rule and its proof |

**The CUDA device layer.** The driver API (`cuda.h`), as Blender
Cycles' CUDA device: the host code C++ compiled by the pinned clang, the
kernels compiled by `nvcc` at build time into a fatbin for sm_86, embedded
in the program as Metal's library is and loaded at start-up. Device code
is kernels only (`.cu`); every launch is `cuLaunchKernel` from host code,
its parameters by value where they are small (the frame's constants, 32
bytes): no ring until a pass needs device-resident constants (OptiX's
launch parameters, Phase 3). OptiX arrives with the first launch that
traces, Phase 3. The backend's renderer refuses, by name and at
construction, every pass kind it does not build yet.

| What | Axis | Where |
|---|---|---|
| the backend's error, from a driver call's result | GPU backend | `src/cuda/device/error.{h,cpp}` |
| the device and its primary context; the machine it names | GPU backend | `src/cuda/device/device.{h,cpp}` |
| the kernels' module, loaded from the embedded fatbin; a kernel by name, launched | GPU backend | `src/cuda/device/module.{h,cpp}` |
| device memory, owned | GPU backend | `src/cuda/device/buffer.{h,cpp}` |
| the stream, frames in flight by event, waiting, the lifetime rule | GPU backend | `src/cuda/device/submission.{h,cpp}` |
| the submission's bookkeeping apart from the driver: sequences, slots, what is settled, the misuses refused | GPU backend | `src/cuda/device/frame_sequence.{h,cpp}` |
| the offscreen target and its readback | GPU backend | `src/cuda/device/offscreen.{h,cpp}` |
| the test pattern, the kernel and its launch | Pass | `src/cuda/passes/test_pattern/test_pattern.{cu,h,cpp}` |
| the schedule carried out, pass by pass (Phase 3 to 4 refused each kind not built yet by name, `unbuilt_pass`; removed in Phase 5, when every kind is built) | Frame graph | `src/cuda/frame/renderer.{h,cpp}` |
| CUDA's offline renderer and the factory | GPU backend | `src/cuda/frame/offline_frames.{h,cpp}` |
| the kernels compiled by `nvcc` with flags pinned in one place, embedded; the bytes-to-header script both backends' libraries use | build | `cmake/CudaModules.cmake`, new; `cmake/embed_bytes_script.cmake`, from `embed_metallib_script.cmake` |
| `nvcc` and the driver pinned | build | `cmake/toolchain.json`, "linux"; `tools/check_toolchain.py` |
| only `src/cuda/` uses CUDA's host API; its kernels include the core's shared layouts only | build | `tools/check_boundaries.sh`, rules 3 and 7 and their proofs |
| unit tests with no GPU: the submission's bookkeeping, which passes the backend builds; one integration test on the GPU: the headless program on the test pattern, against the CPU's | tests | `tests/cuda/`, new |

## The path tracer on CUDA and OptiX

The CUDA backend's Phase 3 (`docs/research/2026-10-19-cuda-backend.md`,
"Phase 3's design"): the reference graph, the path pass and the tone map,
on the RTX 3090 Ti. The path pass is an OptiX launch whose ray generation
program is the pass, tracing two geometry structures of custom primitives
through the shapes' intersection program and reading the hit objects
itself; the tone map is CUDA kernels. The shading code is Metal's,
translated line for line into `.cuh` files under the same families; the
core's shared layouts are included as they are, with the CUDA branch of
`shared_layout.h`.

| What | Axis | Where |
|---|---|---|
| a shared layout's function callable from CUDA device code; the scene block's arrays as pointers there | contracts | `src/core/contracts/shared_layout.h`, `src/core/scene/scene_block.h` and the shared layouts' `constexpr` functions |
| OptiX's headers fetched at a pinned commit; the programs compiled to OptiX-IR with flags pinned in one place, embedded | build | `CMakeLists.txt`, `cmake/toolchain.json`, `cmake/CudaModules.cmake` |
| OptiX's function table and its device context, over the CUDA context | GPU backend | `src/cuda/device/optix_context.{h,cpp}`, new |
| a pass's OptiX pipeline: its module from the embedded IR, its program groups, its stack sizes, its shader binding table, its launch | GPU backend | `src/cuda/device/pipeline.{h,cpp}`, new |
| pinned host memory, owned | GPU backend | `src/cuda/device/buffer.{h,cpp}` |
| a frame's data staged a slot in pinned memory and uploaded on the frame's stream into one device copy | GPU backend | `src/cuda/device/frame_upload.{h,cpp}`, new |
| the accumulated image's pitched memory and its filtering texture; the bloom levels' arrays, surfaces and textures | GPU backend | `src/cuda/device/images.{h,cpp}`, new |
| the scene's still arrays and its block, uploaded once | Scene content | `src/cuda/scene/scene_buffers.{h,cpp}`, new |
| the shapes' transforms and the lights' glows a frame | Scene content | `src/cuda/scene/shape_transforms.{h,cpp}`, `light_glows.{h,cpp}`, new |
| the two structures built, a moving one each frame; what a pass traces, by mask | Acceleration | `src/cuda/acceleration/scene_acceleration.{h,cpp}`, `structure_block.h`, new |
| the samples not finite, counted a slot and collected | Film | `src/cuda/film/failure_count.{h,cpp}`, new |
| the accumulated image, by the core's history rule | Frame graph | `src/cuda/frame/accumulation.{h,cpp}`, new |
| the bloom pyramid at the core's sizes | Frame graph | `src/cuda/frame/frame_images.{h,cpp}`, new |
| the path pass: its ray generation program, its launch parameters, its launch | Pass | `src/cuda/passes/path/path.{cu,h,cpp}`, `path_launch.h`, new |
| the tone map: its four kernels and their launches | Pass | `src/cuda/passes/tone_map/tone_map.{cu,h,cpp}`, new |
| the schedule carried out with the path pass and the tone map; the accumulated image and the failure count read back | Frame graph | `src/cuda/frame/renderer.{h,cpp}`, `offline_frames.{h,cpp}` |
| device code: the vector types' operators and functions, under Metal's names; hashing, warps, transforms, sRGB; packed layouts read | GPU backend | `src/cuda/math/*.cuh`, `src/cuda/device/layout.cuh`, new |
| device code: the sampler, the camera, the shapes and their intersection program, textures, materials, media, lights, uniform light selection, the scene's view, the film, the path integrator, tracing | each its family | `src/cuda/<family>/*.cuh`, from `src/metal/<family>/*.metal.h` |
| the shapes' intersection program, one for every pass, its shapes and table from its structure's hit record; trace() and occluded() over optixTraverse | Acceleration | `src/cuda/acceleration/intersection.cuh`, `trace.cuh`, new |
| a kind no case of a dispatch handles, refused when device code runs (nvcc does not warn of an enumerator a switch leaves out, where Metal's -Wswitch refuses it when it compiles) | GPU backend | `src/cuda/device/layout.cuh`, `unhandled_kind` |
| the probe tests of eight families (sampler, camera, shapes, textures, lights, film, encodings, BSDFs), one source for both backends, each backend's runner behind `<probe_runner.h>` (SF.12); CUDA's runner and probe kernels | tests | `tests/gpu/*_test.cpp`, `tests/gpu/support/probe_runner.h` (`ProbeScene`), `tests/cuda/support/probe_runner.h`, `tests/cuda/kernels/`, new |
| the headless program rendering the reference graph, the same bytes twice; the headless shell test on both platforms | tests | `tests/cuda/headless_test.cpp`, `tests/CMakeLists.txt` |

## ReSTIR PT, MNEE and the features on CUDA

The CUDA backend's Phase 4 (`docs/research/2026-10-19-cuda-backend.md`,
"Phase 4's design"): the data set's input graph, ReSTIR PT's five passes,
MNEE's solve, the features, the checkerboard's fill and the tone map, on
the RTX 3090 Ti. Each pass's kernels that trace are OptiX programs, those
that do not CUDA kernels; every pass reads the frame's one launch
parameters, Metal's frame resources as one uploaded struct.

| What | Axis | Where |
|---|---|---|
| the frame's launch parameters: its constants, camera and the address of every array its passes read or write, uploaded once a frame; the path pass's from Phase 3 folded in | Frame graph | `src/cuda/frame/frame_launch.h`, new; `src/cuda/passes/path/path_launch.h`, removed |
| a pipeline's hit groups, one an intersection program, and a hit record each structure and program | GPU backend | `src/cuda/device/pipeline.{h,cpp}` |
| appending to a list, one atomic a warp, the lanes' places from ballots; the lane and the lanes below it, from PTX | GPU backend | `src/cuda/device/append.cuh`, `lanes.cuh`, new |
| a kernel's launch shape, its grid for a count of threads | GPU backend | `src/cuda/device/module.{h,cpp}` |
| the passes' one OptiX module: the launch parameters' declaration and both intersection programs; a pass's pipeline over the scene's hit records | Frame graph | `src/cuda/frame/traced_module.cuh`, `traced_program.{h,cpp}`, new |
| MNEE's seed as an intersection program, filling the ray's seed through a payload pointer | Integrator | `src/cuda/integrator/mnee_seed.cuh`, new |
| the reuse textures' seed, one constant both backends read | Sample reuse | `src/core/sampling/pairing.h`; `src/metal/frame/renderer.cpp` |
| device code: NEE, MNEE, the shifts, reconnection, the reuse family, selection by power, shape points, ReSTIR PT's pass header | each its family | `src/cuda/<family>/*.cuh`, from `src/metal/<family>/*.metal.h` |
| ReSTIR PT's arrays at the core's sizes, ping-ponged by the frame's parity, by the core's reuse history | Frame graph | `src/cuda/frame/path_reuse_images.{h,cpp}`, new |
| MNEE's work list and its header | Frame graph | `src/cuda/frame/mnee_list.{h,cpp}`, new |
| the feature images, contract 17's formats, read back widened | Frame graph | `src/cuda/frame/feature_images.{h,cpp}`, new |
| the fill's history and image | Frame graph | `src/cuda/frame/checkerboard_images.{h,cpp}`, new |
| each pass: its programs (`<pass>.cu`), its kernels (`<pass>_kernels.cu`), its launches | Pass | `src/cuda/passes/{restir_pt_initial,restir_pt_temporal,restir_pt_spatial,restir_pt_shade,restir_pt_duplication,mnee,features,checkerboard_fill}/`, new; `src/cuda/passes/restir_pt/` (the shift and replay sequence both reuses record) |
| the schedule carried out with ReSTIR PT's, MNEE's, the features' and the fill's families; last frame's transforms and glows; the feature images and the lost MNEE requests read back | Frame graph | `src/cuda/frame/renderer.{h,cpp}`, `offline_frames.{h,cpp}` |
| the probe tests of the families that trace (acceleration, leave, path, MNEE, scene block), one source for both backends; each backend's traced-probe and accumulating harness behind one name, found by the include path | tests | `tests/gpu/*_test.cpp`; `tests/{gpu,cuda}/support/{traced_probe,accumulating_rig}.h` |
| CUDA's probes for those families: the tracing ones as ray generation programs of one OptiX module over positional bindings, the others kernels | tests | `tests/cuda/kernels/{traced_probes.cu,probe_launch.h,path_probe.cu,mnee_probe.cu,scene_block_probe.cu}`, new |
| the headless shell: the input graph's frames the same bytes twice | tests | `tests/cuda/headless_test.cpp` |

## The denoised frame on CUDA

The CUDA backend's Phase 5 (`docs/research/2026-10-19-cuda-backend.md`,
"Phase 5's design"): the demo graph on the RTX 3090 Ti, the network on
the tensor cores, the depth of field, the frame's two halves on two
streams, and the frame time the headless program reports.

| What | Axis | Where |
|---|---|---|
| a frame begun in one part or two, the second on its own stream after the light half's hand-off event; each frame's GPU instants on one timeline | GPU backend | `src/cuda/device/submission.{h,cpp}`, `frame_sequence.{h,cpp}` |
| the tensor cores' product: fragments loaded from shared memory and multiplied, half in, float sums | GPU backend | `src/cuda/device/matrix.cuh`, new |
| the denoiser: the depth median, the input, the convolutions, the pools and the output, and the weights in its layouts | Pass | `src/cuda/passes/denoise/`, new |
| the depth of field's kernels, line for line | Pass | `src/cuda/passes/depth_of_field/`, new |
| what is uploaded or zeroed once, at construction or a new size, waited for, as the legacy stream's copies are not ordered before the frames' streams | GPU backend | `src/cuda/device/buffer.{h,cpp}` (upload_now, zero_now) |
| Metal's vector names the depth of field needs; a launch's third dimension, the convolutions' output chunks | GPU backend | `src/cuda/math/vector.cuh`, `src/cuda/device/module.{h,cpp}` |
| the denoiser's activations, history, median and hand-off; the depth of field's images | Frame graph | `src/cuda/frame/denoise_images.{h,cpp}`, `depth_of_field_images.{h,cpp}`, new |
| the output's launch struct beside the light's; the hand-off ringed by the slot (the light-size and full-size features a set a slot, the fill writing the slot's hand-off and keeping no image of its own); the full-size albedo; the two halves recorded; the network's input read back | Frame graph | `src/cuda/frame/renderer.{h,cpp}`, `checkerboard_images.{h,cpp}` |
| Metal's offscreen frame returning what its begin() settled, for the seam's settled samples | Frame graph | `src/metal/frame/renderer.{h,cpp}` (render_offscreen_frame) |
| the instants of every settled frame, through the seam, from both backends | Presentation | `src/core/presentation/offline_renderer.h`; `src/{metal,cuda}/frame/offline_frames.{h,cpp}` |
| `--times`: the frame-time summary each second of GPU time, the frames back to back, a frame waited for only to be read back | Presentation (headless) | `src/headless/{options,run,main}`, `times.{h,cpp}`, new |
| the summary's instants: the backend's one timeline, host time on Metal, the GPU's on CUDA | Measurement | `src/core/measurement/frame_times.h` |
| the denoiser against PyTorch's fixture, the hand-off, the input graph to the demo's hand-off, one source for both backends, each backend's runner and rigs behind one name; the parts' bookkeeping; the times' parse and line; the denoised pipeline's same bytes twice | tests | `tests/gpu/{denoise,features}_test.cpp`; `tests/{gpu,cuda}/support/{denoise_runner,hand_off_rig,frame_rig}.h`, `tests/gpu/support/denoise_fixture.h`; `tests/cuda/kernels/delay.cu`; `tests/cuda/frame_sequence_test.cpp`, `tests/options_test.cpp`, `tests/cuda/headless_test.cpp` |

## The window on the RTX 3090 Ti

`serenity`, the window, on Linux (`docs/research/2026-10-20-cuda-window.md`):
the same program and its window through SDL3 on X11, the CUDA backend's
frames presented through Vulkan, which copies them to the screen and
nothing else.

| What | Axis | Where |
|---|---|---|
| the window's size from the command line, `--size` | Presentation | `src/app/options.{h,cpp}` |
| the window on X11: made for Vulkan, its surface's instance extensions and the surface by opaque handles, as Metal's layer is | Presentation | `src/app/window.{h,cpp}` |
| the window composed with the platform's backend | Presentation | `src/app/main.cpp` |
| Vulkan for presenting: the CUDA device's Vulkan device, the swapchain, a frame slot's buffer shared with CUDA, the copy and the blit, the timeline semaphores both sides wait on | GPU backend | `src/cuda/present/presenter.{h,cpp}`, new |
| the swapchain's choices of what the surface offers: format, present mode, extent, images | GPU backend | `src/cuda/present/swapchain_choice.{h,cpp}`, new |
| CUDA's frame source: the loop of polled events, acquire, render, present | Frame graph | `src/cuda/frame/swapchain_frames.{h,cpp}`, new |
| a frame recorded into a target given by address and size; a frame rendered into the window's slot | Frame graph | `src/cuda/frame/renderer.{h,cpp}`, `src/cuda/device/offscreen.h` |
| SDL3 and Vulkan on Linux, the window's build there; Vulkan pinned | Build | `CMakeLists.txt`, `cmake/toolchain.json`, `tools/check_toolchain.py` |
| the window launched on the 3090's monitor with an X server of its own | Build | `Makefile` (`window`) |
| the swapchain's choices and `--size`, units (the presenter itself needs the X server: checked by the window's run) | tests | `tests/cuda/swapchain_choice_test.cpp`, `tests/options_test.cpp` |

## Not here yet

- **HIP with HIPRT** on AMD's Strix Halo and **Direct3D 12** on the Xbox
  Series X, later milestones, each its own backend.
