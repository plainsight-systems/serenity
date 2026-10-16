<p align="center">
  <img src="docs/assets/serenity-readme.png" alt="An original screenprint-style illustration of glass and brass spheres on a table at night, lit by fireflies that look like render noise on the left and settle into a calm, clean scene on the right, where a flock of them forms the outline of a small ship." width="100%" />
</p>

# Serenity

Serenity is a GPU path tracer I’m building to work through techniques that
also matter to inference: processing work in small pieces, keeping a small
amount of running state, combining partial results, and reusing earlier work.
The scene gives me something visible to judge that work by: 36 glass,
porcelain, and metal marbles on a walnut table, lit by 616 moving fireflies.

It runs natively on Apple silicon through Metal 4, in C++20. The goal is a
clean moving image at the display’s full resolution and 60 frames a second,
with the light traced at a lower resolution and brought up to the display’s.

[![The marbles rendered with ReSTIR PT and MNEE, ten seconds into the opening](docs/writeups/images/marbles-restir-pt-mnee-10s.jpg)](https://serenity.plainsight-systems.com/#film)

**Current state: milestone 3, at its close.** The renderer now traces the
light at a quarter of the output’s pixels and hands it to a small neural
network I trained for it. For every output pixel, the network decides how
much each nearby noisy pixel should count, and the pixel becomes their
weighted average. That removes most of the noise ReSTIR leaves and doubles
the resolution each way. I tested it once on the marbles, which it never saw
in training. Measured after tone mapping against 4,096-sample path-traced
frames, the wider of the two networks leaves 0.29 to 0.39 of the raw input’s
error, 18 to 38% less than Intel’s Open Image Denoise run on each frame. On
my M3 Max a denoised 1920 × 1080 frame takes 23.5 ms (about 43 frames a
second) and the window’s 3456 × 2234 takes 77 ms (about 13 frames a
second), so 60 frames a second at full resolution is still ahead. Once
every firefly is awake, the table’s light also changes more from frame to
frame than the path-traced frames do (36 to 47% more), which I’ve left for
the next training round.

Milestone 2 built the light the network starts from. ReSTIR PT reuses whole
light paths across neighboring pixels and from the previous frame, and
manifold next event estimation (MNEE) finds the light that the glass marbles
focus onto the table. At 1920 × 1080 a frame of the fully lit scene takes
39.8 ms (about 25 frames a second), about what five path-traced samples
cost (40.6 ms), and at that cost it roughly halves the error left after
tone mapping.

[Milestone 2: reusing work with ReSTIR](docs/writeups/milestone-2.md) follows
the decisions, measurements, and connections to inference.
Milestone 3’s write-up comes with the milestone’s close; until then the
denoiser’s record is in its research notes: the
[design](docs/research/2026-10-14-denoiser-training-design.md), the
[training](docs/research/2026-10-15-denoiser-training.md) and the
[network in the renderer](docs/research/2026-10-15-denoiser-live.md).
[Milestone 1](docs/writeups/milestone-1.md) records the original baseline.
The design is in [`docs/architecture/`](docs/architecture/README.md).

## Project site

The [project site](https://serenity.plainsight-systems.com/) includes the renderer,
milestone writing, and project background.
To preview it with the hosted media:

```sh
python3 scripts/site.py serve --port 1314
```

Open <http://127.0.0.1:1314/>. Requirements, content editing, verification, and
media configuration are in [web/README.md](web/README.md). Media is hosted on
R2; GitHub Pages serves the site.

## Why I built it

Lighting a scene with thousands of moving lights is a sampling problem. You
can’t trace a ray to every light for every pixel, every frame, so you pick a
few, and the picture is only as good as the picking. ReSTIR (Bitterli et al.,
2020) is the algorithm that made this work in real time: each pixel keeps a
tiny reservoir of light samples and shares it with its neighbors and with the
previous frame. NVIDIA’s *Marbles at Night* demo showed what that looks like,
and it’s the spirit of this scene.

What interests me is underneath. A frame has a hard deadline, about 16 ms
at 60 frames a second.
Meeting it means streaming through work in on-chip memory, carrying a few
bytes of state instead of everything you’ve seen, and merging that state
across neighbors and across time. Those are the same problems I work on in
inference, met from the other side. Serenity is where I get to work on them
with a picture at the end.

So far the renderer has the MIS path tracer as its baseline, ReSTIR DI,
ReSTIR PT with spatial and temporal reuse, MNEE for caustics, and a denoiser
of its own. Reusing direct light alone didn’t pay off in this scene at equal
time, because most of what the camera sees arrives through glass or off
metal, so the work moved to reusing whole paths.

The denoiser is where the connection to attention is closest. For each
output pixel the network produces a score for every noisy pixel around it,
the scores go through a softmax, and the output is the weighted average of
those pixels. As in attention, what the network learned is how to compute
the scores; the scores themselves come from the frame in front of it, which
is why it works on a scene it never saw. The correspondence has limits: the
neighborhood is a small, fixed window (3 × 3 at three spacings), and the
values being averaged are the noisy colors themselves, not learned
projections. I trained it in PyTorch on scenes generated for the purpose and
run it every frame in the renderer’s own Metal compute kernels, which match
PyTorch to a few parts in ten thousand.

Next I want to make the whole frame fast, treating the reuse passes as
kernels and measuring where the time and the bytes go. After that, a Vulkan
backend will let me run the same scene on AMD’s Strix Halo.

Serenity is R&D under Plainsight Systems LLC.

### Why Serenity

In path tracing, the bright stray pixels that light up a noisy render are
called fireflies. Half the work of a renderer like this one is getting rid of
them, by sampling better and then by denoising. In this scene they're also,
literally, the lights.

So the project is named for what's left when the fireflies are gone: a calm,
clean frame. Noise in, serenity out.

If the name also brings to mind a small ship that kept flying, that's not an
accident, and any Browncoat reading this caught it a paragraph ago. It's why
the metal spheres are polished: they had to be shiny.

## Run it locally

You need an Apple silicon Mac, Xcode 26.2 with its Metal toolchain, CMake
4.2.3, and Python 3.9 or later. Apple's M3 and later trace rays in hardware; earlier
Apple silicon runs the same code in software.

```sh
git clone --recurse-submodules https://github.com/plainsight-systems/serenity.git
cd serenity
make test     # builds, then runs the tests, the GPU tests among them
make check    # structural rules, and tests proving each check fires
make run      # the window, running graphs/test_pattern.toml
make headless # the same frames, as 60 PNGs in frames/
```

The current renderer is ReSTIR PT and MNEE, denoised and upscaled by the
network: the light is traced at half the window’s resolution each way, and
the network brings it to every pixel. It reads the trained network from
`../serenity-data/denoiser-v1-model/wide`, beside the checkout. The weights
aren’t committed; `make weights` downloads both networks from the project’s
media host and checks every file against its pinned checksum.

```bash
make weights
make run GRAPH=graphs/restir_pt_denoised.toml SCENE=scenes/marbles.toml
```

To make them yourself, [`training/README.md`](training/README.md) covers the
whole cycle, from rendering the data set (four hours on my M3 Max) and
training (about six and eight hours for the two networks) to the export.
`graphs/restir_pt_denoised_small.toml` runs the smaller network, a little
faster with slightly higher error.

The same light without the denoiser, at half the window’s resolution each
way, stretched to the window:

```bash
make run GRAPH=graphs/restir_pt.toml SCENE=scenes/marbles.toml SCALE=0.5
```

To compare with the MIS path tracer:

```bash
make run GRAPH=graphs/path.toml SCENE=scenes/marbles.toml SCALE=0.5
```

`graphs/restir.toml` runs ReSTIR DI. `graphs/path_mnee.toml` enables the
optional glass solver on the plain path tracer; it stays off in the baseline
because more ordinary samples worked better at equal time.
`graphs/restir_pt_features.toml` and `graphs/features.toml` write the
denoiser's inputs for training, through the headless renderer’s
`--features`.

`serenity-headless` writes into its `--out` directory only if it is new or
empty, and refuses one that holds anything, so no earlier run's frame sits
beside this run's; `make headless` clears its own frames from `frames/`
first.

`make run GRAPH=graphs/other.toml` runs another frame graph, and `SCENE=` gives
it a scene. The marbles are written by `scripts/make_marbles.py`; edit the
scene there. The first lit scene, a soft brass sphere on a checkerboard at
night lit by two fireflies:

```bash
make run GRAPH=graphs/preview.toml SCENE=scenes/brass_sphere.toml
```

The path tracer converges while the image holds still:

```bash
make run GRAPH=graphs/path.toml SCENE=scenes/brass_sphere.toml
```

The fireflies awake, each drifting about where it hung. Through the path
tracer the window starts over every frame while they move, so it shows one
noisy sample of each instant, without accumulating earlier instants:

```bash
make run GRAPH=graphs/path.toml SCENE=scenes/brass_sphere_wander.toml
```

`SCALE=0.5` renders at half the window's resolution each way, for speed.
Escape or closing the window ends it. `make movie` with the same `GRAPH` and
`SCENE` renders ten seconds headless, `SAMPLES` samples of each frame's
instant (64 unless given), and encodes them with ffmpeg into `media/`, which
git ignores.

A reference image, and the path tracer’s convergence toward it:

```bash
make reference SCENE=scenes/marbles.toml TIME=120    # some ten minutes of GPU
make convergence SCENE=scenes/marbles.toml TIME=120  # error, and error x time
```

Both write into `.cache/references/`: the reference as a PFM, and a CSV of
mean squared error and error x time at 1, 2, 4 ... 16,384 samples.
`serenity-measure` does the arithmetic, on the CPU, from the headless
renderer's float images.

The build is pinned to one toolchain. Metal can't run in a container, so the
versions of Xcode, the SDK the build compiles against, the Metal compiler,
Apple's clang (as the C, C++ and Objective-C compiler) and CMake are recorded
in [`cmake/toolchain.json`](cmake/toolchain.json), and configuring the build
stops if yours differ. Shaders are compiled when the project builds and
compiled into the binary. The first build fetches metal-cpp, SDL3, toml++,
stb_image_write and doctest, each pinned by commit or by checksum.

The denoiser’s training is optional and pinned separately. `training/` is
its own Python environment, Python 3.12.15 and PyTorch 2.14.1 through
[uv](https://docs.astral.sh/uv/), every package locked by version and hash
in `training/uv.lock`. Nothing the renderer builds or runs depends on it;
you need it only to make or change the network. Its data and models live
in `../serenity-data/`, beside the checkout, and are never committed.

## Find your way around

| Path | What it holds |
|---|---|
| `src/core/` | the platform-neutral core: what is rendered, how it moves, the frame's schedule, film and output. No GPU or windowing API |
| `src/metal/` | the Metal backend, the only code that touches Metal: the device, and every shader |
| `src/app/`, `src/headless/` | the two ways to run a frame: in a window, or to files |
| `src/measure/` | the measuring program: a reference made from files, and images judged against it |
| `scenes/` | scene descriptions, as data: what is rendered |
| `graphs/` | frame graphs, as data: which passes a frame runs |
| `cmake/` | the toolchain pin, and how shaders are compiled into the binary |
| `tests/` | tests; `tests/gpu/` runs on this machine's GPU |
| `tools/` | the toolchain and boundary checks |
| `scripts/` | scene generation, the denoiser's data set, the project-site build, and independent review of a commit |
| `training/` | the denoiser's training toolkit, in PyTorch: its own pinned environment, outside the C++ build |
| `web/` | the Hugo site’s templates, styles, and hosted-media manifest |
| `docs/architecture/` | the design, written before the code it describes: start with the [logical overview](docs/architecture/logical-overview.md) |
| `docs/research/` | measurements and investigations, each dated, with the machine and toolchain it ran on |
| `docs/writeups/` | what each milestone taught me, for the interested engineer |
| `docs/process/` | decisions and the work queue |

The rule that shapes the layout: the core never knows which GPU it runs on.
Only `src/metal/` uses Metal, and the Vulkan backend will sit beside it, so
both machines render the same scene from the same core. `make check` enforces
it.

## Credits

- ReSTIR: Benedikt Bitterli, Chris Wyman, Matt Pharr, Peter Shirley, Aaron
  Lefohn and Wojciech Jarosz, "Spatiotemporal reservoir resampling for
  real-time ray tracing with dynamic direct lighting," SIGGRAPH 2020.
- *Marbles at Night*, NVIDIA's 2020 RTX demo, for showing what many moving
  lights on simple geometry can look like.
- pbrt-v4, by Matt Pharr, Wenzel Jakob and Greg Humphreys: the MIS path
  tracer follows its PathIntegrator.
- Lin, Kettunen, and Wyman’s *ReSTIR PT Enhanced*, for the path-reuse
  implementation.
- Blender Cycles’ manifold next event estimation, for the approach to
  finding caustic paths through glass.
- Intel’s Open Image Denoise, for the training’s loss, augmentation and
  schedule, and as the denoiser’s comparison; Thomas et al. (Intel, 2022),
  for denoising and upscaling in one kernel-predicting network; Chaitanya
  et al. (NVIDIA, 2017), for the temporal loss; NVIDIA’s NRD, for the
  motion-vector and albedo conventions.

## License

Code is licensed under the Apache License 2.0; documentation, prose and
original visual assets under CC BY 4.0. See [`LICENSE`](LICENSE) for the
split, and [`NOTICE`](NOTICE) for third-party components and their licenses.
