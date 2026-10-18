# Logical overview

What the renderer does, and the principles that decide how it does it.

The target is the look of NVIDIA's *Marbles at Night*: glass and polished
metal on a table at night, lit by hundreds to thousands of moving fireflies,
fully path traced: reflections, refractions, light bouncing between surfaces,
and caustics focused through the glass.

This is the **logical** view: phases, responsibilities and principles. It
names no files, no types, no interfaces and no GPU API. The rules that turn
these boxes into files are in [`change-axes.md`](change-axes.md), and the
resulting arrangement in [`file-mapping.md`](file-mapping.md). Every claim here can be checked
against a rendered frame or against the reference.

## The flow

![Serenity's logical flow: startup, every frame, and the light at each rough surface expanded](logical-overview.svg)

Shading is the same in every figure: **gray** is a fixed step, the same for
every scene and estimator; **teal** is an estimator, swappable per frame;
**purple** is scene data and per-frame inputs; **coral** is state kept from
one frame to the next.

**Startup** runs once. The scene description is numbers: where the spheres
and the table are, what they are made of, how many fireflies there are and
how each one moves. It is built into geometry and the acceleration
structures rays are traced against, and the state every frame reads and
writes is allocated.

**Every frame** is a function of its inputs: the time *t*, the frame's index
and the camera.

- **Animate** places every firefly where it is at *t*, sets how brightly
  each glows at *t*, and updates the acceleration structures to match. The
  fireflies are small glowing spheres: geometry that rays can hit, and
  lights that the estimators sample. Rays that only test whether a light
  is visible skip them (principle 11).
- **Primary visibility** traces a ray from the camera through each pixel:
  in ReSTIR PT's frames, the light the network reads, through the same
  point of every pixel, the frame's **jitter**, a fixed low-discrepancy
  sequence (Halton in bases 2 and 3) of 32 phases, as temporal upscalers
  take it (DLSS's guide: about 8 times the square of the upscale ratio);
  in the path tracer's, through a point of its own in each pixel, drawn
  at random, the box filter a reference converges to. At
  glass it reflects or refracts, chosen by the Fresnel term; at polished metal
  it reflects; it stops at the first rough surface, or when it leaves the
  scene. What it finds is the pixel's **surface record**: position, normal,
  material, the throughput of the glass and metal it passed through, and
  where that surface was in the previous frame. A firefly struck on the way
  is light seen directly, kept for Resolve.
- **Light** estimates what arrives at that rough surface, in three parts.
  The estimator is chosen by the frame graph:
  - **Direct** light from every firefly. The path tracer, for each sample,
    picks a light, picks a point on it and traces a shadow ray; it weighs
    what it finds against the bounced ray, which counts a firefly it
    strikes with the complementary weight (principle 7). ReSTIR PT takes
    it at every vertex of its paths, by resampling candidate lights from
    the frame's light tiles, and reuses it with the indirect light, as one
    path (below). The tiles are drawn each frame in proportion to each
    firefly's power as it glows at *t*, as RTXDI presamples its local
    lights from a power PDF and its mip chain: a firefly still asleep is
    never drawn, a bright one often, and every candidate's density, and the
    density MIS weighs a firefly struck by a bounced ray by, is that same
    power over the frame's total. ReSTIR DI (Bitterli et al. 2020, as RTXDI ships it),
    which reused the direct light alone, lost to the path tracer at equal
    time on the marbles and was removed 2026-10-17, superseded by ReSTIR
    PT.
  - **Indirect** light: the path bounces on from the rough surface, through
    glass, off metal and off other rough surfaces, taking direct light at
    each rough surface it reaches. **ReSTIR PT** (Lin, Kettunen and
    Wyman's *ReSTIR PT Enhanced*, 2026, at its defaults) reuses these
    paths whole, direct and indirect light together, across pixels and
    frames: one path reservoir a pixel, shifted to a neighbor's or last
    frame's surface by reconnecting where the surfaces are rough enough
    and replaying where they are not, so its reuse reaches through glass
    and off mirrors, where most of the marbles' light arrives
    (`docs/research/2026-10-11-restir-pt-layout.md`). It
    took the place planned for ReSTIR GI (Ouyang et al. 2021), and it is
    the renderer's estimator since Milestone 2.
  - **Caustics**: firefly light focused onto a rough surface through glass.
    A shadow ray cannot bend through a sphere, so these paths are found by
    manifold next event estimation (Hanika, Droske and Fascione 2015, as
    Blender Cycles builds it), which solves for the path through the glass
    directly: refraction through up to four glass surfaces, solved from a
    work list by a pass of its own. ReSTIR PT always runs it, solved per
    pixel and not reused; the path tracer never does: it is the reference,
    and on it, at equal time, more ordinary samples did better
    (`docs/research/2026-10-12-mnee.md`).
    Caustics reflected off metal, under 1% of the error where they fall,
    are left to the bounced rays.
- **Resolve** adds the fireflies seen directly to the light, giving the
  pixel's radiance.
- **Denoise** removes the noise that remains, with a network trained
  for this renderer (Milestone 3,
  `docs/research/2026-10-14-denoiser-training-design.md`). It reads the
  light divided by the surface's color, so it filters lighting and not
  texture, and the **features** beside it: the surface's color, normal,
  depth and motion along the same camera rays, through glass and mirrors
  to the surface seen in them. Per output pixel it predicts weights over
  three 3 x 3 neighborhoods of the noisy light, at spacings of 1, 2 and 4
  pixels, softmax-normalized, and a blend with its own last output moved
  by the motion vectors, and the frame's jitter, as DLSS, TSR and FSR take
  it, so it knows where in each pixel its light was traced (the motion
  vectors stay unjittered, NRD's and DLSS's convention): a
  kernel-predicting network, its weights computed
  fresh from every frame as attention computes its weights from the tokens
  in front of it. Then the surface's color is multiplied back.
- **Upscale** brings the image to the output's resolution, and the same
  network does it: its kernels are predicted at twice the light's
  resolution each way, so rays are traced at a quarter of the output's
  pixels, as *Marbles at Night* traced below its output resolution and
  upscaled with DLSS. A graph without the denoiser renders at the window's
  size, or a fraction of it that the window stretches.
- **Depth of field**, in a denoised frame, is the lens, applied after the
  network, at the output's resolution: every step before it traces a
  pinhole, sharp at every depth, and this step blurs each pixel by the
  circle of confusion the scene's thin lens gives its depth, the same
  radius and plane of focus the lens rays had, so the look is the lens's.
  It is where production puts it (Unreal's post-process order, temporal
  upscalers "between Depth of Field and Motion Blur"; DLSS and NRD take
  inputs without lens effects): through a lens, every ray of a pixel
  starts at a random point on the lens, so at a defocused edge the
  features flip between the near marble and the table behind from pixel
  to pixel, and the network keeps that speckle. The blur follows Jimenez's
  *Next Generation Post Processing in Call of Duty: Advanced Warfare*
  (SIGGRAPH 2014): scatter as gather at half the output's resolution,
  the near and far fields kept apart so that a defocused marble in front
  bleeds over the sharp table behind it, as a lens shows it, while the
  blurred background never spreads over what is in focus in front of it.
  A graph without the denoiser traces the lens itself (principle 12).
- **Tone map** turns linear radiance into display values, and the frame is
  **presented** in the window or **written** by the headless renderer.

The **reference** is the same light transport without reuse: the path
tracer, the same bounces, time frozen, and thousands of samples
accumulated over frames, at full resolution and with no denoiser or
upscaler, through the scene's lens, as Unreal's path tracer renders its
depth of field in the rays and its real-time renderer after them. Every
other image is judged against it. The denoiser was trained against
references of the same kind, on scenes other than the one it is shown on,
through a pinhole: the network's output is sharp at every depth, and so
are its targets (principle 12).

The **state kept between frames** is drawn beside the merges rather than as a
step, because that is what it is: ReSTIR PT's reservoirs and surface
records of this frame and the last (a path reservoir and a primary hit a
pixel), and a map of how many neighbors hold the same path, which caps
how much history the next frame keeps. The merges are the only light
steps that read them across pixels or across frames. The denoiser keeps
one thing of its own: its last output, which the next frame blends in;
and the checkerboard's fill reads the light it handed off the frame
before ("The frame").

## Principles

1. **Time is an input.** A frame is a function of the scene description, *t*,
   the frame's index and the camera. The window passes the time it measured
   and the headless renderer a fixed step, but nothing that renders reads a
   clock. The same inputs give the same image in both.

2. **Randomness is a function of pixel, frame and purpose.** Every random
   number is derived from where and when it is used. There is no shared
   generator state, so any frame can be rendered again exactly, alone. The
   camera's jitter is a function of the frame alone: its index's phase in
   a fixed sequence.

3. **The reference shares the scene and the light transport, never the
   reuse.** ReSTIR PT runs Enhanced's defaults, whose merges are weighted
   by pairwise and Talbot MIS; its means are checked region by region
   against the reference (`docs/research/2026-10-11-restir-pt-phase1.md`,
   `docs/research/2026-10-11-restir-pt-phase2.md`), and manifold next
   event estimation's bias, dark over a glass sphere's shadow, is measured
   there too (`docs/research/2026-10-12-mnee.md`). Because the reference
   does none of its resampling, a bug in it cannot pass by matching
   itself.

4. **Estimators are interchangeable.** The path tracer (the reference),
   ReSTIR PT with manifold next event estimation (with temporal reuse,
   spatial reuse or both), and the denoised pipeline (ReSTIR PT's light,
   denoised and upscaled) take the same scene and estimate the same image.
   Any two can be run on the same frame and compared side by side.

5. **Of the light's steps, only the merges look beyond their own pixel and
   frame,** and what they see is reservoirs and surface records. Every other
   light step works on one pixel alone. Reuse across pixels and time lives
   in one place, and that is where the kernel work concentrates. The image
   steps after Resolve look at neighborhoods by design: the denoiser, which
   also reads its own last output, the depth of field's blur, and the tone
   map's bloom. So does the
   checkerboard's fill ("The frame"), which reads the frame before's light
   and the traced pixels around the ones it fills: it reconstructs an
   image, it reuses no sample.

6. **A reservoir holds a path by identity, not by position.** ReSTIR PT's
   path reservoir keeps the random numbers that drew its path, and its
   reconnection vertex as a shape and a point in that shape's own
   coordinates, and the path is evaluated again wherever it is reused. The
   fireflies move between frames, so a position kept from last frame is
   wrong in this one.

7. **Direct light at a rough surface is found two ways and weighed; every
   other light path is counted one way.** In the path tracer, a firefly's
   light reaches a surface with a lobe that is not delta two ways: sampled
   from it in a straight line by next event estimation, and found by the ray
   the surface's BSDF sends on when that ray strikes the firefly. Each
   strategy's sample is weighted by the power heuristic (Veach 1997, chapter
   9): its density squared over the sum of both densities squared. The two
   weights for any one direction sum to one, so the light is counted once
   in expectation. This is how production path tracers light glossy
   surfaces (pbrt-v4's PathIntegrator, Falcor). Light sampling alone spends
   its samples where a near-mirror lobe gives nothing and, rarely, lands one
   in the lobe, returning a spike that grows as 1 / alpha^2
   (`docs/research/2026-10-11-restir-ris-results.md`); the BSDF's density
   matches such a lobe. A firefly struck by the camera's own path, or after
   a delta lobe, is counted in full, since no shadow ray could have reached
   it that way. The path tracer, the reference, runs no manifold next
   event estimation, so it counts the light glass and metal focus onto a
   rough surface the one way it can: a bounced ray that reaches a firefly
   through a delta lobe, which a shadow ray, blocked by the glass, cannot.
   In ReSTIR PT, which always runs manifold next event estimation, it takes
   over the paths through glass from a diffuse surface, and the bounced ray
   stops counting those. The sky is not aimed at, and is counted one way,
   by the rays that escape to it. Amended 2026-10-11 by
   the user's direction, superseding "every light path is counted once, no
   path found two ways" (`docs/process/MEMORY.md`).

8. **The scene is data.** Positions, materials, firefly count, color,
   intensity and motion are numbers in the description. Changing the scene
   never changes code.

9. **A pixel's cost does not grow with the number of lights.** ReSTIR weighs
   a fixed number of candidates per pixel whether there are a hundred
   fireflies or ten thousand. What grows with the count is Animate and
   whatever is built over the lights each frame, and nothing built over them
   outlives the frame, because they move. This binds the estimators.

10. **The core decides what is computed; a backend decides how.** Both
    backends render the same scene from the same description, and each is
    judged against the reference.

11. **A ray finds what its question needs, and a firefly casts no
    shadow.** Every shape is in one of two classes: a light (a firefly) or
    a surface (everything else). A ray that asks *what it reaches*, the
    camera's, a bounced ray, a replayed path, the features', finds both,
    since it must be able to strike a firefly: that is how MIS's BSDF
    strategy finds a light, and how the camera sees one. A ray that asks
    *whether a chosen light is visible*, a shadow ray, a reconnection's
    check between two vertices, manifold next event estimation's segments,
    finds surfaces only. Each estimator asks the same question the same
    way wherever it asks it, so a shift tests a path's visibility as the
    path was sampled. This is production's per-object "casts shadows"
    switch, kept by an instance mask (DXR's and Vulkan's mask bits,
    Unreal's ray-traced shadows), applied to every light: of the
    scene's 653 shapes, 616 are fireflies 1.5 mm across, which every
    visibility ray would otherwise search among for the shadows of
    fireflies on each other and on the surfaces, too small to see.
    The image changes by exactly those shadows, measured before the
    choice was kept (Milestone 4's Step 2,
    `docs/research/2026-10-17-milestone-4-plan.md`), and every estimator
    and the reference make it alike, so they still compare. A scene whose
    lights are large enough for their shadows to matter would give the
    class to the scene's data, per shape, as Unreal's primitives carry it;
    no scene here has one.

12. **The lens is traced by the reference and applied by the denoised
    frame after the network.** A graph that denoises traces its light,
    its features and its full-size albedo through a pinhole, the scene's
    camera with its lens taken off, and its depth-of-field step then
    blurs the network's output by the circle of confusion of that same
    lens; every other graph traces the scene's lens in its rays, the path
    tracer, the reference, among them. So the two meet where they are
    compared: the same lens, its radius and plane of focus, one exactly
    in the rays and one approximately in the image, as Unreal's path
    tracer and its real-time renderer differ. The denoiser's training
    data is rendered through a pinhole, inputs and targets alike: its
    scenes are written without a lens.

## The design space

What varies, and which principle handles it.

| What varies | Examples | Handled as |
|---|---|---|
| Number of fireflies | hundreds to thousands | scene data (8); no per-pixel cost (9); none in a visibility ray's way (11) |
| Firefly motion, color, intensity | per firefly | scene data, evaluated at *t* (1, 8) |
| Geometry | spheres, the table | scene data (8) |
| Material | rough, glass, polished metal | rough surfaces take light; glass and metal are passed through, by the camera's path and by caustic paths |
| Light path | direct, bounced, through glass | direct light at a rough surface two ways, weighed by MIS; every other path one way (7) |
| Camera | still or moving | a frame input (1) |
| Lens | a pinhole, or a thin lens of a radius and a plane of focus | scene data; traced in the rays without the denoiser, applied after it with (12) |
| Time step | measured, or fixed | a frame input (1) |
| Estimator | the path tracer, the reference; ReSTIR PT with manifold next event estimation, temporal, spatial or both; the denoised pipeline | interchangeable (4) |
| Reuse settings | M candidates, k neighbors, history length | settings of the estimator, not code |
| Samples per pixel | one a frame, or thousands for the reference | a setting; the reference is the light transport without reuse (3) |
| Point in the pixel | the frame's jitter, or each pixel's own | the jitter in ReSTIR PT's frames, the light the network reads, and given to it; each pixel's own random point in the path tracer's, the reference's box filter (2) |
| Pixels traced a frame | every one, or half in an alternating checkerboard | half in a denoised graph, the other half the frame before's light, moved ("The frame"); every one in a graph without the denoiser |
| Resolution | traced, and output | with the denoiser, traced at half the output each way and upscaled by it; without, traced at the window's size or a fraction the window stretches; anything, headless |
| Denoiser | none, or our own network | a stage after Resolve, its weights data; judged against the reference, and against Open Image Denoise offline |
| Backend | Metal, Vulkan | the core is shared (10) |

Two consequences are worth stating outright:

- **The light count reaches only Animate and the light sampling.** Primary
  visibility, the merges and shading each touch one light per pixel, or a
  fixed number. That is the claim the path tracer's and ReSTIR PT's images
  make side by side: as fireflies are added, the path tracer's image gets
  noisier and ReSTIR PT's does not.
- **Unbiasedness costs rays.** Merging a reservoir from another frame or
  another pixel is unbiased only if the merge accounts for whether that
  reservoir's path could have been found here, and finding out takes
  rays. Production's direct-light reuse (RTXDI's default) does without
  them and accepts a small darkening at shadow edges. ReSTIR PT's merges
  trace them: each shift of a path to another pixel traces its
  reconnection's visibility or replays the path through glass and
  mirrors, and that tracing is most of what the reuse costs.

## Reservoirs are state keyed on surfaces

A reservoir belongs to a surface, not to a pixel. When the camera or a sphere
moves, the surface a pixel saw last frame is somewhere else on screen, so the
temporal merge looks up last frame's reservoir where this surface was, using
the surface record's previous position. Where there was no such surface,
because it has just come into view, there is nothing to merge and the pixel
starts again from its candidates.

Reuse never decides correctness here, only noise. Unbiased merges weight
every input by whether its sample could have been chosen at this surface, so
a reservoir from a dissimilar surface is weighted down, not trusted. The
surface records exist to choose good neighbors, not to keep the answer right.

The history a reservoir carries is capped. Lights move, and a sample that
was the best choice many frames ago may no longer be; the cap bounds how long
an old choice persists, which trades noise for how quickly the lighting
follows a moving firefly.

## The frame

The window shows the display's full resolution, 3456 x 2234 on the
development machine, and the target is 60 frames a second: 16.7 ms for
every step. Rays are traced at a lower resolution and upscaled. With the
denoiser a frame has two sizes: every step up to Resolve runs at the
**light size**, the output's halved each way and rounded up, and the
denoiser, depth of field and the tone map at the **output size**. In the window the light
is traced at 1728 x 1117 and the network brings it to every pixel.

**A denoised frame runs in two halves, on two queues** (Milestone 4,
`docs/research/2026-10-17-milestone-4-plan.md`). The **light half** is
everything that traces rays: Animate, the light, the reservoirs' merges,
the features at the light size, and the surface's color at the output
size, which the denoiser multiplies back. The **image half** is the rest:
Denoise and Upscale, Depth of field, Tone map, at the output size. The light half is
bound by the ray tracing hardware and leaves the arithmetic units nearly
idle; the image half is the reverse. So frame N's
image half runs beside frame N + 1's light half, and a frame is shown
about as often as the slower half takes, not their sum.

- **What crosses between them** is what the denoiser reads of the light
  half: the frame's radiance and its features. Each frame in flight has
  its own copy of these, so frame N + 1's light half never writes what
  frame N's image half is still reading.
- **What stays in order** is each half's own history. The light half's
  reservoirs and accumulated image pass from frame to frame on its queue,
  the denoiser's last output on the other; frame N's image half starts
  only once frame N's light half has finished.
- **The cost is one frame of latency.** A frame is still shown only once
  both its halves have run, so from its light half's start to its showing
  takes what it did; but frames now leave the light half faster than one
  frame's two halves take, so one more frame is in flight between being
  traced and being shown. The image of each frame is the same as when the
  halves run in turn, bit for bit; only when it is shown changes.

A graph without the denoiser has one half and runs on one queue, as
before.

**Checkerboard rendering of the light** (Milestone 4's Step 4,
`docs/research/2026-10-17-milestone-4-plan.md`) is how a denoised frame
traces its light, not a setting; a graph without the denoiser traces
every pixel. Each frame traces the light at half the light size's
pixels, a checkerboard whose two fields alternate frame by frame, so
every pixel is traced every other frame; the other half of the image is
the frame before's light, moved by the motion vectors, as console
checkerboard rendering reconstructs the pixels it did not shade. It is RTXDI's checkerboard mode
for ReSTIR PT:

- **What runs on the half**: ReSTIR PT's initial sampling (and with it the
  caustics' requests, so their solve), the temporal and spatial reuse, the
  shade and the duplication map. Their reservoirs, primary hits and light
  buffer are kept packed, two pixels of a row to one slot, as RTXDI keeps
  its reservoirs, so the traced half lies side by side in memory.
- **Where reuse looks**: a spatial neighbor is a pixel traced this frame,
  and the temporal input a pixel traced the frame before, the other field,
  as RTXDI moves each to the field that holds a reservoir.
- **What stays whole**: the features at the light size, the denoiser's
  guide images, traced at every pixel every frame, as RTXDI's G-buffer and
  a console's depth and ID buffers are: they are what tells a moved pixel
  from a revealed one, and what the network edges on.
- **The fill**: at the end of the light half, writing the hand-off of the
  light's radiance, each pixel not traced this frame takes the light
  the frame before traced where its surface was, found by its motion and
  kept only where the depth seen there is the depth the motion predicts;
  where none is (a surface just revealed, a history's first frame), the
  traced pixels beside it, kept by the same test. The traced pixels go
  through as they are. The fill writes the hand-off, never the accumulated
  image, so a still scene's mean is each pixel's own traced frames.

It changes what the denoiser sees, which was trained on light traced at
every pixel: a half of each frame a frame old. The user kept it
(2026-10-17), from the denoised frames before and after; whether the
network is retrained on checkerboarded light is the user's call, and it
has not been.

The target is not met yet. The denoised frame takes 56 ms at the
window's size and 16.7 ms at 1920 x 1080 (Milestone 4's Step 4,
`docs/research/2026-10-17-milestone-4-plan.md`); ReSTIR PT alone at the
window's full size takes about 150. Making the frame fit the budget is
later work. The headless renderer is not held to the budget: footage is
written at any resolution and any number of samples.

## Platform constraints

**Rays are traced in hardware.** The development machine is an Apple M3 Max,
whose GPU traces rays in hardware, through Metal 4. The second backend
targets an AMD Strix Halo through Vulkan's ray tracing. Neither is assumed to
behave like the other; each is measured on its own.

**A mask needs an instance level, and the scene has none.** Metal, like
DXR and Vulkan, tests a ray's mask against an instance's, never against a
primitive's, and the scene's structure is one level of boxes, because on
this GPU the instance level cost each ray more than it saved
(`docs/research/2026-10-09-acceleration-structure.md`). So principle 11's
two questions are kept, on Metal today, as two structures: every shape,
and the surfaces alone, each ray tracing the one its question names. A
two-level structure keeps the same two questions as mask bits on its
instances; which a ray asks does not change.

**Little is kept alive across a ray query.** On the M3 Max the naive path
tracer's one kernel holds every path's state, and its ray queries' scratch,
across each query. That state fills the L1 cache, the GPU's occupancy
manager runs a quarter of the threads it could, and the ray tracing unit,
active 88% of the time, waits on too few threads to hide its latency
(`docs/research/2026-10-10-path-kernel-counters.md`). So every kernel after
it traces with little alive: one ray query at a time, and what a later step
needs passed between kernels rather than held across the query. ReSTIR
PT's passes are built this way from the start. The naive path
tracer stays one kernel: it is the baseline for time as well as for noise,
and its counters are what the later kernels are measured against. On the
Strix Halo this is measured again, not assumed.

**The network runs on the renderer's own kernels.** The denoiser's
convolutions are compute kernels of the renderer's own, on the GPU's
matrix units, inside the frame's own submission, not a machine-learning
runtime's: Metal 4's machine-learning encoder runs networks built offline
for fixed input sizes, on kernels that could neither be tuned nor matched
to the training's arithmetic (`docs/research/2026-10-15-denoiser-live.md`).
Its weights are read from a file at start-up; the kernels match PyTorch's
outputs to 2.2 x 10^-4 relative on a test fixture.

**No rendering dependencies.** No engine, no ReSTIR library, no denoising
or inference library inside the renderer. Open Image Denoise is a baseline
the renderer's own denoiser is compared with, offline, not a part of it.
The training is outside the renderer: a Python toolkit with its own pinned
environment (PyTorch), which nothing the renderer builds or runs depends
on. MetalFX's upscaler, once planned as the one exception, is not used:
the network upscales.

**No asset files.** Every shape is procedural and every number is in the
scene description; there is nothing to download and nothing an artist made.
