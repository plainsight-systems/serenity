# Milestone 1: a path tracer that tells the truth about itself

![The marbles, 10 seconds into the opening: fireflies resting among them](images/marbles-opening-10s.jpg)

*Ten seconds into the opening: 36 marbles on a walnut table, lit only by
fireflies. Naive path tracer, 64 samples a pixel, 1920 x 1080.*

Serenity is a real-time path tracer on Apple's Metal 4, written in C++20.
Its purpose is a parallel I want to make concrete: the pieces of a modern
real-time renderer have counterparts in the pieces of transformer attention.
ReSTIR's reservoir merge is a streaming softmax, and choosing which of 616
lights to sample at a pixel is choosing which keys a query attends to.

Milestone 1 builds the thing every later optimization is measured against:
the textbook path tracer, a scene hard enough to stress it, a trustworthy
frame time, a reference image, and a single number for error. It ended here:

| | |
|---|---|
| Frame, 3456 x 2234, every firefly awake | 29.90 ms |
| Frame, the opening, no firefly awake yet | 19.10 ms |
| Reference image, 65,536 samples a pixel at 1920 x 1080 | 9.5 minutes |
| Error x time, the naive estimator | 1.3 to 3 MSE-seconds |

Most of what I learned getting there was about measurement and correctness
rather than rendering.

## The scene is the spec

I chose the scene before any optimization, because the scene decides which
optimizations matter. It is 36 marbles at real scale through a macro lens:
clear and tinted glass, cat's-eyes with swirled cores, porcelain under a
clear coat, and brass, steel and copper. Each material exercises a different
kind of light transport. The only light comes from 616 fireflies in two
swarms, flying closed-form loops that are checked clear of every surface
when the scene loads.

The scene opens in the dark. Fireflies perched on the marbles wake one at a
time and rise, while a second swarm comes down from above. That started as
an art decision, and it turned out to be the most useful performance case in
the scene. In the first seconds almost every light is dark, and the naive
estimator picks one of the 616 uniformly at random, so nearly every pick is
wasted. That is precisely the noise ReSTIR exists to remove.

It also exposed a gap in my kernel. It traced a shadow ray toward whichever
light it picked, even a dark one, whose contribution is zero no matter what
the ray finds. pbrt-v4, the implementation I follow, tests the light
sample's radiance before tracing anything. Adding the same test changed no
pixel and made the dark frames far cheaper:

| Time into the opening | Without the test | With it |
|---|---|---|
| 0.5 s, no light awake | 32.35 ms | 19.16 ms |
| 8 s, a few awake | 32.45 ms | 24.34 ms |
| 120 s, all awake | 29.89 ms | 29.77 ms |

When my output diverges from the reference implementation, I now check the
reference before I call the difference a design choice.

## Measure before you redesign

Twice I adopted a design from a production renderer before I had measured my
own bottleneck, and both times it made the frame slower.

The first was Apple's primitive data, which stores per-primitive records
inside the acceleration structure. Apple reports 10 to 16% faster ray
tracing from it. In my scene it cost 0.2 to 0.3 ms a frame. Apple's gain came
from replacing reads of large per-triangle buffers; my shapes' data was a
few hundred records already in Metal's constant address space, which was
fast to begin with.

The second was Unreal's two-level acceleration structure, where each mesh
gets a bottom-level structure that is built once and instanced. For my exact
spheres it took 18.1 ms against 8.2 ms for a single level of placed boxes.
Unreal's design pays off for meshes of thousands of triangles worth building
once; a sphere has nothing worth keeping.

A third change, moving every scene array behind one bindless block, first
made the five-sphere frame 9.5% slower (16.18 ms against 14.77). Two things
were wrong. Views of the scene built once and carried through the path kept
their pointers in registers for its whole length; building them where they
are used saved 0.6 ms. And the block lived in `device` memory where Apple's
own bindless example uses `constant`; switching saved another 0.5 ms and left
the block within 2% of the original.

What finally explained the frame was a hardware counter recording, which I
should have made first:

| Counter | Marbles |
|---|---|
| Ray-tracing unit active | 93.5% |
| Kernel occupancy | 22.3% |
| L1 eviction rate | 91% |
| L1 filled by ray-tracing scratch / registers | 48.7% / 23% |
| ALU utilization | 13.9% |

The frame is bound by ray traversal at low occupancy. The cache is full of
per-thread state kept alive across ray queries, so the GPU's occupancy
manager runs about a quarter of the threads it could, too few to hide the
traversal's latency. Arithmetic is barely used and the scene data hardly
matters. The fix is to carry less state across each query, or to split the
kernel at its ray queries (wavefront path tracing). I deferred it on purpose:
the baseline should stay the textbook kernel, so that later gains are
measured against something honest. The broader takeaway is to copy a
production design's reason, not its shape, and to record counters before
redesigning anything.

## Bugs that look like working code

I ran a rule-by-rule review of the whole codebase against the C++ Core
Guidelines and a set of C++ performance guidelines. It produced 419 findings,
and the ones that mattered had one thing in common: the renderer produced
plausible images with every one of them present.

A `double` was cast to `int64` before its range check, which is undefined
behavior reachable from numbers in a scene file. Rays ended at `INFINITY` in
shaders compiled with fast math, which assumes infinities don't exist. On an
error mid-run, GPU buffers were released while two frames still used them,
because the object that waits for the GPU was destroyed last. A scene file
whose read failed was parsed as an empty scene, because the failure was
checked on the wrong stream. Russian roulette was ending paths inside glass
about twice as often as their energy warranted, a real bias in the
estimator; fixing it cost 0.9 ms a frame and made the image correct. Each
pixel's random numbers came from a 32-bit key shared by thousands of pixel
pairs every frame. And five tests could not fail at all. I fixed each one
and proved it by breaking the code it guards and watching the test go red.

Two more surfaced later, when I tested inputs I hadn't chosen. Loading the
scene with 30 other seeds for one swarm, 14 refused to load: a single unlucky
firefly whose flight couldn't be drawn clear refused the whole scene. Six of
those failures predated my changes; the scene had only ever loaded because
its seeds happened to work. A swarm now redraws a refused firefly from its
next seed, and any firefly whose first draw works stays bit-for-bit the
same. Separately, the shaders keyed random numbers by the low 32 bits of a
sample's index while the batch renderer accepted 64-bit indices, so a batch
started past 2^32 would have silently reused another batch's random numbers.
That is now refused before anything renders.

"It renders" is not evidence of correctness. A test only counts once I've
seen it fail, and the inputs I didn't pick (seeds, sizes, indices) are where
the remaining bugs were.

## Performance numbers you can trust

My first time baseline, 28.44 ms, came from a measurement harness I never
committed, so I couldn't reproduce it. I rebuilt the measurement with its
method written down next to the result: two frames in flight as the window
runs them, 200 frames, the median of frames 21 to 199, and the GPU idle
before each run. On the old code the rebuilt method gives 28.60 ms, within
0.6% of the lost figure, which is close enough to trust both.

The dark opening added a wrinkle: the same scene costs 19 ms or 30 ms
depending on how many fireflies are awake, so every frame time now names the
moment it was measured at, and the baseline is t = 120 seconds. Earlier, a
window I'd left running made every measurement read twice as slow. It looked
like a regression until a bisect showed the baseline commit had "regressed"
too, and I now confirm the GPU is idle before any timing run.

Two other measurements changed my mind. Removing the barrier between frames
looked like free overlap, but the frames contended and went from 27.7 ms to
between 67 and 79 ms. And scene loading fell from 459 ms to 124 ms once a
profile showed a single distance query taking 92% of it, because every still
shape was being re-placed on every query instead of once at load.

A performance number without its method is an anecdote. The method belongs
in the repository, next to the number.

## What "correct" means for a renderer

The reference image is the same path tracer with time frozen and many
samples. Building it raised two problems I hadn't anticipated.

The first was precision. The GPU keeps each pixel's running mean in 32-bit
floats, and at 65,536 samples any sample within 0.4% of the mean no longer
moves it. So the reference is 64 batches of 1,024 samples, each read back and
combined on the CPU in double precision, since Metal has no doubles.

The second was the shape of the noise. Two references rendered from
different samples agree on mean brightness to within 0.08%, so the estimator
is unbiased. But 95% of the error between them sits in 0.01% of the pixels:
about 200 pixels where a few enormous samples dominate even at 65,536
samples. A single image's error is therefore an unreliable number, and my
convergence curves jumped tenfold between doublings.

I started designing around that with repeated runs, a second reference and
trimmed metrics, then stepped back. What I need is to compare estimators, and
for that I use one number: mean squared error multiplied by the time the
samples took. It is the inverse of Monte Carlo efficiency, and the closest
thing rendering has to bits per byte for language models. It ignores the
incidental details (how many samples, what each costs), lower is better, and
for an unbiased estimator it holds roughly steady as samples grow. It also
states ReSTIR's whole claim directly: less error in the same time.

| Samples, naive, t = 120 | MSE | MSE x seconds |
|---|---|---|
| 1,024 | 0.211 | 2.0 |
| 2,048 | 0.0707 | 1.35 |
| 4,096 | 0.0335 | 1.28 |
| 16,384 | 0.0211 | 3.2 |

## What carries forward

Early on I built a contract that made CPU and GPU math bit-identical, so
tests could check shaders against C++. I dropped it the same day, because it
constrained the renderer's math in order to serve the tests. Correctness is
shown on the GPU instead, against closed-form answers and the reference
image, which is also how the work is shown to anyone else.

I've also kept every rejected alternative in the repository with its
measurement. That record is how I know which ideas were tried and why they
lost, not just which ones won.

## Next

ReSTIR DI, measured against 29.90 ms a frame and 1.3 to 3 MSE-seconds. The
dark opening, with a handful of the 616 lights awake, is where it should show
most clearly, and its reservoir merge is where the attention parallel
begins.

*The measurements behind every number here are in
[docs/research/](../research/), one note per question.*
