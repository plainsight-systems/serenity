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
| **Animation** | a new kind of motion over time | how that kind places an object or light at time *t* |
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
| **Film** | what each pixel accumulates and outputs | accumulation, and the per-pixel outputs: radiance, normal, depth, motion |
| **Pass** | a pass is written or changed | that pass only: a denoiser, an upscaler, a tone mapper |
| **Frame graph** | passes are added, reordered or rewired | the passes' order, the images between them, the history kept across frames |

**Platform**

| Axis | Triggered by | May change |
|---|---|---|
| **GPU backend** | the GPU API | device, resources, pipelines, command encoding, synchronization |
| **Presentation** | the window, display or input | the window, its event loop, the measured clock |
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
retuning the denoiser touches no upscaler, and swapping the tone curve
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
| **Film outputs**: radiance, normal, depth, motion per pixel | Film | Pass, Sample reuse |
| **Images** between passes, and the history kept across frames | Frame graph | every Pass |

If any of these lived inside the code that uses it, a change to it would
touch every user at once, and the axes would silently become one.

Three shared things are owned by the axis whose reason they change for, not
by their users:

| Shared thing | If it lived inside its users | Belongs to |
|---|---|---|
| deriving a random number | every stage would carry its own, and none could be replayed | **Sampler** |
| the target function: a sample's contribution, ignoring visibility | light selection, every merge and shading would each carry a copy | **Integrator**: it defines what "contribution" means |
| the reservoir and the merge rule | every reuse, direct and indirect, temporal and spatial, would change with it | **Sample reuse** |

The last is the one worth insisting on. ReSTIR DI and ReSTIR GI reuse
different samples but merge them by the same rule. Kept in one place, the
rule is one unit that can be studied, measured and optimized on its own,
which is what the kernel work is about.

## Mechanism is code; values are data

Animation is the motions an object *can* make; scene content is the motions
*these* objects do make. The same holds across the table: a material kind
against a sphere's color and roughness, a sample-reuse rule against how many
candidates and neighbors it uses, a denoiser against its trained weights, a
tone mapper against its exposure. Keeping them apart means art-directing the
scene, tuning the estimator or retraining the denoiser never looks like a
code change.

## Every shader-side axis exists once per backend

The core decides what is computed, and each backend decides how, in its own
shaders. So shape, material, texture, light, medium, camera, sampler, light
selection, integrator, sample reuse and every pass are each implemented once
per backend: today for Metal, later for Vulkan too.

That makes each of them a family over backends as well as over kinds. The
test extends: **optimizing a kind on one backend touches no other backend.**
A change to *what* a kind computes touches each backend's implementation of
it. That duplication is the cost of two thin backends without a shared
interface, accepted in `../process/MEMORY.md`.

## The logical boxes, by axis

The boxes of [`logical-overview.md`](logical-overview.md) that are already one
axis:

| Box | Axis |
|---|---|
| Scene description | Scene content |
| Allocate frame state | Frame graph |
| Denoise, Upscale, Tone map | Pass, each its own |
| Reservoirs | Sample reuse |

The boxes that are compound:

| Box | Axes it mixes | Splits into |
|---|---|---|
| **Build geometry** | Shape, Acceleration, GPU backend | each shape kind's geometry · the acceleration structures · the API calls that build them |
| **Frame inputs** | Camera, Animation, Presentation | the camera at *t* · the time step: measured by the window, fixed when headless |
| **Animate** | Animation, Acceleration | placing each object and light at *t* · updating the structures to match |
| **Primary visibility** | Camera, Shape, Material, Integrator, Film | generating the ray · intersecting it · passing through glass and metal · writing the surface record |
| **Light** | Integrator, Light selection, Sample reuse, Light, Material | the path loop and its strategies · choosing lights · reusing samples · each emitter · each BSDF |
| **Resolve** | Integrator, Film | adding light seen directly · accumulating the pixel |
| **Surface records** | Film, Frame graph | what a record holds · keeping this frame's and last frame's |
| **Present or write** | Presentation, Output | the window · each file format |

**Light** is the split worth insisting on. Fused, the path loop, the choice of
lights, the reservoirs and every BSDF would be one unit, and changing the
merge rule would mean re-verifying the materials. Split, each of the
milestones touches its own axis: ReSTIR DI and GI are sample reuse,
caustics are the integrator, a new material is a material.

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

**The first medium will touch the integrator too.** A path through a medium
is a different light transport, so the first medium changes the integrator
once, to step through media. After that, a new medium is a kind like any
other. Medium has no implementation yet; the axis exists because the reason
to change exists.

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
