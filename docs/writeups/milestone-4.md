# Milestone 4: making the frame fast

![The marbles twenty seconds into the opening, the swarm perched among them, traced in a checkerboard and denoised by the retrained network](images/marbles-m4-20s.jpg)

*Twenty seconds into the opening, at 60 frames a second on my M3 Max.*

## Where Milestone 3 left the frame

[Milestone 3](milestone-3.md) finished the pipeline. Every frame now goes
from moving fireflies to a finished image:

![Milestone 3's frame: the light traced at a quarter of the pixels (scene, trace paths, caustics, reuse in time, reuse in space, shade), then the image at every pixel (surface features, denoise and upscale, tone map, screen)](images/pipeline-m3.svg)

The light is traced at a quarter of the output's pixels. ReSTIR PT traces
a few light paths for each pixel, manifold next event estimation (MNEE)
finds the light that the glass marbles focus onto the table, and each
pixel then borrows good paths from its own previous frame and from its
neighbors. The result is noisy. The renderer also records what each pixel
shows: its surface color, orientation, depth and motion. The network I
trained in Milestone 3 uses those to decide how much each noisy neighbor
should count, removes the noise, and fills in the full resolution. A tone
map turns the light into display colors.

That frame looked right, but it ran at about 43 frames a second at
1920 × 1080 and about 13 at the full size of my laptop's display.

## What this milestone set out to do

Once a renderer produces the right image, the next work is the same in
most engines. They keep what the pipeline computes and change how much of
it is computed each frame and how well the GPU is kept busy while doing
it. This milestone applied four of those standard optimizations to the
Milestone 3 pipeline, kept three of them, and retrained the network for
the input that changed:

![Milestone 4's frame: the same stages, the light of frame N + 1 running at the same time as the image of frame N; shadow rays skipping the fireflies in the scene, path tracing, caustics and both reuse steps; those stages and shading running on half the pixels each frame, with a new fill stage for the untraced half; and the network retrained](images/pipeline-m4.svg)

## The optimizations

### Running the light and the image at the same time

A GPU has different kinds of hardware: units that test rays against the
scene, units that do arithmetic, and memory. Most passes lean on one of
them and leave the others waiting. Engines, consoles above all, put
independent work on a second queue so it can run in the gaps. This is
usually called async compute.

Serenity's frame splits that way. Tracing the light keeps the ray-tracing
hardware busy, and the network is almost entirely arithmetic. So the
network now works on one frame while the light of the next frame is
traced, each frame's data kept separate so neither overwrites what the
other is reading. The image is unchanged; the price is a little more
latency.

It helped, but much less than the GPU's counters suggested. Side by side,
the two halves slowed each other down. My reading, not yet confirmed, is
that the arithmetic units looked idle without being free: each core's
threads, registers and on-chip memory are one pool, and the ray-tracing
work holds its share of that pool while it waits for the ray-tracing
unit. The same question comes up in inference whenever two kernels are
overlapped to fill each other's gaps. A utilization counter shows what is
idle, not what is available.

### Shadow rays that skip what cannot cast a shadow

Engines tag geometry by kind and let each ray skip the kinds it does not
care about. Particles, foliage and small effects are commonly left out of
shadow rays, which only need to know whether anything blocks the light.

In Serenity, 616 of the scene's 653 objects are fireflies a millimeter or
two across. A firefly's own shadow is almost never visible, but every
shadow ray was still being tested against every firefly. Shadow rays now
see only the 37 surfaces: the table and the marbles. Metal's version of
this feature works on a layout Serenity's scene doesn't use, and adopting
that layout made every ray slower, so I built a second ray-tracing
structure holding only the surfaces, once at start-up. The image changes
only where a firefly resting on the table used to shade the wood beside it.

The usual companion to this is splitting the scene into a still part and
a moving part, so only what moves is rebuilt each frame. Here rebuilding
the fireflies was already nearly free, and the two-level layout it needs
is the one that slowed every ray on Metal, so I skipped it. It returns
with the Vulkan and Xbox backends, where that layout is the hardware's
normal case.

### Tracing half the pixels each frame

Checkerboard rendering traces half the pixels each frame, alternating like
the squares of a checkerboard, and rebuilds the other half from the frame
before. Consoles used it to reach 4K, and NVIDIA's RTXDI has a checkerboard
mode for ray-traced lighting. I read RTXDI's source and followed it.

The light is the expensive half of Serenity's frame, so it is traced in a
checkerboard while the surface features stay complete. A new fill stage
rebuilds each untraced pixel: from the light the previous frame traced
where the pixel's motion says its surface was, if the depth there agrees,
and otherwise from its traced neighbors. My first attempt reused the whole
previous image, filled pixels included, and the untraced half drifted
darker; taking only pixels that were actually traced fixed it. The image
came out a little smoother and a little softer.

### Teaching the network the new input

The network had learned from complete frames of light, and checkerboard
changed what it sees, so I retrained it. The targets it learns toward come
from the path tracer, which checkerboard does not change, so only the
noisy inputs had to be rendered again.

This was the first training run on my RTX 3090. Training now runs on
either GPU, and keeps its data on the card when it fits or in system
memory when it doesn't, as on the 3090. The retrained network is about as
accurate on the held-out marbles as the original was on complete frames.

## Where it stands

The fully lit scene now runs at 60 frames a second at 1920 × 1080 on my
M3 Max, up from 43, and about 18 at my laptop display's full size, up
from 13. The largest gain came from tracing half the pixels; the
smallest, running the two halves together, taught me the most.

The network and the reuse steps are now the larger costs. The reuse steps
merge reservoirs much as attention merges partial softmax results, and so
far I have treated them as passes rather than as kernels. Their tiling,
data layout and fusion are the questions that brought me to Serenity, and
they come next. After that, the Vulkan backend will put the same scene on
my RTX 3090 and AMD's Strix Halo.
