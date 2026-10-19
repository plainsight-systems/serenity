# Change axes

One translation unit, one reason to change. This document names the reasons,
because without that list the rule cannot be applied: every split sounds
arguable, and the argument is settled by taste.

It governs how [`logical-overview.md`](logical-overview.md)'s boxes become
files. It does not list the files; that is `file-mapping.md`.

The axes are those of a real-time path tracer, not of this one scene. They
follow the interfaces of pbrt-v4 (shapes, cameras, samplers, materials,
textures, lights, integrators, film) and the real-time structure of NVIDIA's
Falcor (a render graph of passes over a GPU API layer). Serenity's features
appear below only as examples: a firefly, ReSTIR and the caustics are each a
combination of axes, and none is an axis of its own.

## The axes

**What is rendered**

| Axis | Triggered by | May change |
|---|---|---|
| **Shape** | a new kind of geometry | that kind's bounds, intersection, surface interaction, and sampling a point on it |
| **Material** | a new scattering model | that model's BSDF: evaluate, sample, pdf |
| **Texture** | a new spatially varying pattern | that pattern's evaluation at a surface interaction |
| **Light** | a new kind of emitter | that kind's emission, sampling, and pdf |
| **Medium** | a new participating medium | that medium's density, scattering and sampling |
| **Camera** | a new projection or lens | ray generation for that model |
| **Animation** | a new kind of change over time: a motion, or a glow | how that kind places an object or light, or sets a light's brightness, at time *t* |
| **Scene content** | the scene is art-directed | values only: what is where, made of what, moving how |

**How light is computed**

| Axis | Triggered by | May change |
|---|---|---|
| **Acceleration** | how rays find geometry is built or updated | the acceleration structures' layout, build and update |
| **Sampler** | a new sample sequence | how a random number is derived from pixel, frame and purpose |
| **Light selection** | a new way to choose which light to sample | that strategy and its probabilities |
| **Integrator** | what light paths the image includes, or how they are found | the path loop, its strategies, and the target function |
| **Sample reuse** | how samples are resampled and shared across pixels and time | the reservoir, how two merge, and their weights |

**The image and the frame**

| Axis | Triggered by | May change |
|---|---|---|
| **Film** | what each pixel accumulates and outputs | accumulation, and the per-pixel outputs: radiance, and the denoiser's features (albedo, normal, depth, motion) |
| **Pass** | a pass is written or changed | that pass only: a denoiser, an upscaler, a tone mapper |
| **Frame graph** | passes are added, reordered or rewired | the passes' order, the images between them, the history kept across frames |

**Platform**

| Axis | Triggered by | May change |
|---|---|---|
| **GPU backend** | the GPU API | device, resources, pipelines, command encoding, synchronization |
| **Presentation** | the window, display or input | the window, its event loop, the frame asked for and its time, the frame source's interface |
| **Output** | an image or video format | that format's writer |
| **Measurement** | a new metric or timing | that metric |

Every file maps to exactly one row. Two rows means it splits. Two files that
always change together means they merge, unless they cannot, for a reason
recorded below.

## The kinds are open sets

Shape, material, texture, light, medium and camera are each a set of kinds,
and the set grows. A new kind is a new implementation behind its axis's
interface, never a branch inside an existing kind, and it does not belong to
the scene that first needed it. A sphere light is a light kind that any scene
can use; glass is a material any shape can wear.

Pass is a family in the same way. Each pass is its own reason to change:
retraining the denoiser touches no tone mapper, and swapping the tone curve
touches no denoiser.

The test is mechanical: **adding a kind touches no other kind**, and nothing
outside its axis but the place that lists which kinds exist.

## The contracts between axes

What keeps the kinds independent is that the axes meet only at interfaces.
Each interface has one owner; everything else reads it.

| Contract | Owned by | Read by |
|---|---|---|
| **Surface interaction**: position, normal, surface coordinates, material | Shape | Material, Texture, Integrator |
| **BSDF**: evaluate, sample, pdf | Material | Integrator, Sample reuse |
| **Emitter**: emission, sample a point, pdf | Light | Light selection, Integrator, Sample reuse |
| **Light sample**: which light and which point on it, in its own coordinates | Light | Sample reuse, Integrator |
| **Film outputs**: radiance, albedo, normal, depth, motion per pixel | Film | Pass, Sample reuse |
| **Images** between passes, and the history kept across frames | Frame graph | every Pass |
| **Ray mask**: the classes of shape (lights, surfaces), and which a ray finds | Acceleration | Integrator, Pass (the features) |
| **Checkerboard**: which pixels a frame traces, and where their arrays keep them | Frame graph | Sample reuse, Pass |

If any of these lived inside the code that uses it, a change to it would
touch every user at once, and the axes would silently become one.

Three shared things are owned by the axis whose reason they change for, not
by their users:

| Shared thing | If it lived inside its users | Belongs to |
|---|---|---|
| deriving a random number | every stage would carry its own, and none could be replayed | **Sampler** |
| the target function: a sample's contribution, ignoring visibility | light selection, every merge and shading would each carry a copy | **Integrator**: it defines what "contribution" means |
| the reservoir and the merge rule | every reuse, direct and indirect, temporal and spatial, would change with it | **Sample reuse** |

The last is the one worth insisting on. ReSTIR PT picks among different
samples, a light among next event estimation's candidates, a path in the
path tree, last frame's path and a neighbor's, but by the same rule
wherever it picks. Kept in one place, the rule is one unit that can be
studied, measured and optimized on its own, which is what the kernel work
is about.

## Mechanism is code; values are data

Animation is the motions an object *can* make; scene content is the motions
*these* objects do make. The same holds across the table: a material kind
against a sphere's color and roughness, a sample-reuse rule against how many
candidates and neighbors it uses, a denoiser against its trained weights, a
tone mapper against its exposure. Keeping them apart means art-directing the
scene, tuning the estimator or retraining the denoiser never looks like a
code change. The denoiser's weights are a file the frame graph names, read
at start-up; the toolkit that trains them is outside the build, and nothing
the renderer builds depends on it.

## Every shader-side axis exists once per backend

The core decides what is computed, and each backend decides how, in its own
shaders. So shape, material, texture, light, medium, camera, sampler, light
selection, integrator, sample reuse and every pass are each implemented once
per backend: Metal's, CUDA's, and each later vendor's.

That makes each of them a family over backends as well as over kinds. The
test extends: **optimizing a kind on one backend touches no other backend.**
A change to *what* a kind computes touches each backend's implementation of
it. That duplication is the cost of thin backends, each in its vendor's
own API, without a shared interface, accepted in `../process/MEMORY.md`:
each backend's kernels are tuned for its hardware, and no layer stands
between them.

## The logical boxes, by axis

The boxes of [`logical-overview.md`](logical-overview.md) that are already one
axis:

| Box | Axis |
|---|---|
| Scene description | Scene content |
| Allocate frame state | Frame graph |
| Denoise and Upscale, Depth of field, Tone map | Pass: denoising and upscaling one pass (below), the depth of field and the tone map each their own |
| Reservoirs | Sample reuse |

The boxes that are compound:

| Box | Axes it mixes | Splits into |
|---|---|---|
| **Build geometry** | Shape, Acceleration, GPU backend | each shape kind's geometry · the acceleration structures · the API calls that build them |
| **Frame inputs** | Camera, Animation, Presentation | the camera at *t* · the time: the display's, for the window; a fixed step, headless |
| **Animate** | Animation, Acceleration | placing each object and light, and setting each light's brightness, at *t* · updating the structures to match |
| **Primary visibility** | Camera, Shape, Material, Integrator, Film | generating the ray · intersecting it · passing through glass and metal · writing the surface record |
| **Light** | Integrator, Light selection, Sample reuse, Light, Material | the path loop and its strategies · choosing lights · reusing samples · each emitter · each BSDF |
| **Resolve** | Integrator, Film | adding light seen directly · accumulating the pixel |
| **Surface records** | Film, Frame graph | what a record holds · keeping this frame's and last frame's |
| **Present or write** | Presentation, Output | the window · each file format |

**Light** is the split worth insisting on. Fused, the path loop, the choice of
lights, the reservoirs and every BSDF would be one unit, and changing the
merge rule would mean re-verifying the materials. Split, each of the
milestones touches its own axis: ReSTIR PT is sample reuse (its shift of
a path to another pixel is the integrator's), caustics are the
integrator, a new material is a material.

## Cases that look wrong and are not

**A kernel is one responsibility expressed as two files.** A shader and the
host code that launches it change together: retile the shader and its
dispatch moves with it. The strict rule says merge them; they are different
languages, so the coupling is a language artifact, not a design choice. They
stay a co-located pair.

**The reference is not an integrator of its own.** It is the same integrator
with sample reuse switched off, time frozen, and the film accumulating over
thousands of frames: settings, not code. Giving it its own code would let it
drift from what it is the reference for.

**A light samples its shape.** A sphere light picks a point on its sphere by
asking the sphere. Which light to sample is light selection's business;
where on it is the shape's. A new selection strategy never touches sphere
sampling.

**The first medium touches the integrator too.** A path through a medium
is a different light transport, so the first medium changes the integrator
once, to step through media. After that, a new medium is a kind like any
other. The first is the absorbing medium inside tinted glass (contract 12,
`src/core/contracts/medium.h`): the integrator now carries the medium a path
is in and dims each stretch by it. The first medium that scatters will
change it once more, to sample where along a stretch light scatters.

**The denoiser and the upscaler are one pass.** The network predicts its
kernels at the output's resolution from the light at half of it: denoising
and upscaling are one set of weights, trained together, and no change to
one leaves the other alone, so by the decisive question below they are one
unit. The other passes stay apart: retuning the network touches no tone
mapper. Two of its parts live elsewhere: reading the weights, in the core
and still on the Pass axis, so every backend runs the same numbers; and the
full-size albedo the output is multiplied by, which is the features pass's
kernel run again at the output's size.

**How frames are paced is the backend's; what a frame is, the
window's; a frame source is the seam.** Pacing the window to the display
(`logical-overview.md`, "The frame", "Pacing") is two axes, and the
boundary between them is an interface in the core, so that neither side
is shaped by the other:

- **Presentation** (`src/app/`): what a frame is when one is asked for:
  its inputs, the instant it will be shown among them; the size the
  window wants; the histories; the title; the window's events. It
  implements the frame source's client and never knows how frames are
  paced: not whether they are called from a run loop while it waits for
  events, nor what they are rendered into.
- **GPU backend**: the frame source, which owns the loop: when frames
  come, what they render into (Metal's drawable, a swapchain's image),
  the rate and latency, how long the window may wait for events between
  them, and what a frame failing means for the loop. Metal's is a
  display link on the layer, which calls frames from Core Animation's
  run loop, through a callback that must not throw, so the source keeps
  a frame's failure and ends the run with it from its own loop; a
  CUDA or Direct3D 12 source would wait on its swapchain and present
  in a loop of its own.

The interface (`src/core/presentation/frame_source.h`) names neither
side's types: an instant, a size, the frame's inputs, and what a shown
frame reports for the title (its GPU time once settled, the bug
counters). Only the program's `main` composes a backend's source with
the window; `make check` keeps every other file of `src/app/` free of
any backend, and all of it free of the display link.

**Running the denoised frame on two queues is two axes, and neither is a
pass's.** Which work goes on which queue, and the hand-off between them
(`logical-overview.md`, "The frame"), split in two:

- **GPU backend**: the second queue, its command buffers, the events that
  order one queue's work after the other's, and when a frame's slot is
  free again. It knows a submission has one part or two, never what is
  in them.
- **Frame graph**: where the frame splits (at the denoiser), which images
  cross the split and so are kept once per frame in flight, and the
  barriers on each side. It knows the schedule's passes, never how a
  queue is made or waited on.

Neither leaks further. The core's schedule says nothing of queues: it says
what a frame computes, and a backend that cannot overlap (or CUDA, with
its streams) runs the same schedule its own way. No
pass knows which queue it is on: each records into the encoder it is
given, after the barriers the frame graph records, as before. The window
and the headless renderer ask the renderer how many parts its frames have
and hand that to the submission; they learn nothing else of it.

Three things in it look wrong and are not. **The features are kept twice
while nothing about them changed**: the second copy is the hand-off's, the
frame graph's (contract 6), not Film's, whose contract is what a pixel's
features are. **The light's radiance is not handed off as the features
are**: the features are written whole each
frame, so a second set is enough, but the accumulated image carries
history across frames on the light's queue (a still scene's mean), so the
denoiser gets an image of it per frame in flight instead, which the
checkerboard's fill writes whole (below). **The full-size albedo, the denoiser's own
input, runs in the light half**: its kernel traces rays, and beside the
light half's tracing, in the image half, the two shared the ray tracing
hardware and it took four times as long; split by the hardware each
uses rather than by which pass reads what, the frame is shown sooner
(Milestone 4's Step 1 measures both).

**A ray mask is two axes' business, and neither is the scene's.** Which
shapes a ray finds (`logical-overview.md`, principle 11) splits in two:

- **Acceleration** owns the mask: the classes a shape can be in (a light,
  a surface), which class each shape is in, learned from Light's own
  answer to whether a shape is a light (contract 3), and the guarantee
  that a ray carrying a mask finds exactly the shapes of its classes, at
  the distances and as the shapes it would have found them unmasked. How
  that is kept, one structure per mask on today's single level or mask
  bits on a two-level structure's instances, is Acceleration's alone and
  changes nothing outside it.
- **Integrator** (and the features pass, which traces the camera's rays)
  chooses each ray's mask by the question the ray asks: what it reaches,
  or whether a chosen light is visible.

The test holds: splitting the scene into two levels (Milestone 4's Step
3) touches no estimator, and an estimator asking a new question picks a
mask without touching the structure. It looks wrong that **the scene has
no say** in which shapes cast shadows, as Unreal's primitives each have:
every light here is a firefly-sized sphere, so the class follows from
being a light; a light whose shadow should show is the change that moves
the class into the scene's data, as a per-shape value.

**Checkerboard rendering is three axes' business, and none is the
estimator's.** Tracing half the light's pixels a frame, as every
denoised frame does (`logical-overview.md`, "The frame"), splits in
three:

- **Frame graph** owns which pixels a frame traces and where their
  arrays keep them: the field rule (a pixel is traced in a frame of the
  same parity as its coordinates' sum), the packed slot two pixels of a
  row share, and the grid the light's kernels dispatch over, as one
  contract (contract 20) every pass reads alike; and the fill, which
  writes the hand-off, since what the image half reads of the light
  half's is the frame graph's (contract 6).
- **Sample reuse** asks it where a neighbor and last frame's input may
  be: a spatial partner among this frame's traced pixels (the pairing
  taken over their lattice, so such a graph's sigma must be one the
  lattice can pair at), a temporal input among the frame before's. The
  merges themselves, their weights and the shifts do not change.
- **Pass**: the fill's kernel, which reads the features and the frame
  before's hand-off and writes this frame's.

It looks wrong that **the choice is the frame graph's and not
[restir_pt]'s**, since only ReSTIR PT's passes read it: the fill and the
hand-off are the network's input, which ReSTIR PT's settings know
nothing of, so a graph traces its light checkerboarded exactly when its
light is the network's input, ReSTIR PT with the features pass
(`logical-overview.md`, principle 13), the denoised graph and the data
set's input graph alike, and the renderer copies that into the path
reuse settings. It looks wrong too that **a graph without the network
runs the fill**: the input graph's light is the network's input, and the
fill is how that light is made; the fill writes the light it hands on,
to the network where there is one, and to the presenting pass and the
headless renderer's files where there is not. And that **the
features stay whole** while the light is halved: they are Film's guide
images for the denoiser, and the fill's evidence for whether a moved
pixel is the same surface; halved, both would guess at edges.

**Depth of field is three axes' business, and the lens is still one
camera's.** Applying the lens after the denoiser
(`logical-overview.md`, principle 12) splits in three:

- **Camera** owns the lens: its radius and plane of focus, the rays
  through it, and the circle of confusion it gives a point at a depth
  (the thin lens's, `src/core/camera/thin_lens.h`), so the rays a graph
  traces through the lens and the blur a graph applies for it are one
  lens's, and a test holds them to each other.
- **Pass**: the blur, which reads the network's output, the full-size
  depth and that circle of confusion (contract 21), and knows nothing of
  how its input was made.
- **Frame graph** decides which graphs trace the lens and which apply it
  after the network: a denoised graph's light, features and full-size
  albedo are traced through the camera with its lens taken off, and its
  depth of field follows the denoise pass.

It looks wrong that **a graph changes the camera it is given**, since the
camera is a frame input (principle 1). The camera is unchanged: the frame
graph chooses where its lens is applied, in the rays or after the
network, as it chooses at what size the light is traced; the lens's
numbers reach both places from the one camera.

**The camera's jitter is three axes' business.** The point in the pixel
every camera ray of a ReSTIR PT frame takes (`logical-overview.md`,
"Primary visibility") splits in three:

- **Sampler** owns the sequence: Halton in bases 2 and 3, its phase count,
  and the offset of frame *n* (`src/core/sampling/jitter.h`), a function
  of the frame's index alone, as every number the sampler gives is of
  where and when it is used.
- **Frame graph** decides which frames take it, a ReSTIR PT graph's light
  and its features, never the full-size albedo, which is at the output's
  resolution as the network's output is and so unjittered, and carries it
  to the shaders in the frame's constants (contract 7), which the camera's
  ray generation reads.
- **Pass**: the denoiser reads it as two input channels (contract 18),
  trained so.

It looks wrong that **the jitter is not the camera's**, since DLSS's
guide puts it in the projection matrix. It is a sample position, not a
property of the lens or the view: the camera is the same for the
path tracer, which draws its own point in every pixel, and the motion
vectors and every reprojection go through the camera without it, which
is NRD's and DLSS's convention for them.

**Drawing lights by power is two axes' business.** RTXDI's presampling
by power (`logical-overview.md`, "Light") splits in two:

- **Light** owns a light's power: what a kind emits in all, a sphere's
  4 pi r^2 x pi x its radiance's luminance (RTXDI's `GetPower`), at its
  glow this frame (`metal/lights/emitter.metal.h`), so a new kind of
  light brings its own.
- **Light selection** owns the distribution: the pyramid of the lights'
  powers, built each frame, a light drawn from it by descending it, and
  the probability of any one light, which every density ReSTIR PT weighs
  by asks (`metal/light_selection/power_light.metal.h`). The tiles
  (contract 3's presampled form) hold its draws.

The path tracer keeps the uniform selection: it is the reference, its
noise the baseline the estimators are measured against, and a
selection of its own changes nothing ReSTIR PT computes.

**Scene content and animation are two files, not one.** Both describe how the
fireflies move, which makes merging them tempting. But animation changes when
a new kind of motion is written, and scene content when the scene is
directed. Merged, retuning a flight path would look like a code change.

## How to apply this

Before adding to a file, ask which row of the table would cause this line to
change. If it is not the row that file already owns, it belongs somewhere
else.

Before creating a file, name its row. A file whose row cannot be named is
either a utility bucket or a responsibility nobody has articulated, and both
tend to accumulate.

The decisive question for a split is not "are these different concerns",
which is unfalsifiable, but "can I name a change that touches one and not
the other." If not, they are one file. And when the only evidence that two
things change together is that the examples at hand happened to differ in
both, look for an example where they do not.
