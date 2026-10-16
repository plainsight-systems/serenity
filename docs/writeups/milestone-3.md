# Milestone 3: turning samples into an image

![The marbles twenty seconds into the opening, the swarm perched among them and the first fireflies rising](images/marbles-denoised-20s.jpg)

*Twenty seconds into the opening: ReSTIR PT and manifold next event
estimation traced at 960 × 540, denoised and upscaled to 1920 × 1080 by the
network described here.*

[Milestone 2](milestone-2.md) left me with better samples and a frame that
was still visibly noisy. ReSTIR PT roughly halved the displayed error for
the cost of five path-traced samples, but half of a noisy image is still
noisy, and the fireflies move, so it never settles. Production renderers
that use ReSTIR run a denoiser after it, and treat the samples as evidence
for a reconstruction that produces the picture.

I wanted that reconstruction to be my own, for two reasons. Tracing the
light at a lower resolution and bringing it up to the display's is how a
frame like this one gets anywhere near real time, so the same pass should
also upscale. And of everything in the renderer, a denoiser is where the
connection to attention is most direct, because it decides how much each
nearby value should count and then takes their weighted average.

## Learning how to weigh the neighbors

The network I built predicts kernels. For every output pixel it looks at
the noisy pixels around the matching input pixel, gives each of them a
score, and passes the scores through a softmax. The output is the weighted
average of those pixels' light. It does this at three spacings, 3 × 3
neighborhoods at one, two and four pixels apart, so it can reach further
across flat surfaces without reading every pixel in between. It also
learns how much of its own previous frame to blend in, moved by the motion
vectors to where each pixel's surface was. Each of the four output pixels
over an input pixel gets its own kernel, which is how it doubles the
resolution each way.

A small example shows the correspondence. Suppose a pixel on the table has
three neighbors: one on the table, one on the edge of a marble, and one
with a stray bright sample. If the network scores them 2, 1 and −1, the
softmax turns those scores into weights of about 0.71, 0.26 and 0.04, and
the pixel takes mostly the table's light, a little of the edge's, and
almost none of the outlier's. Attention does the same arithmetic with
queries and keys. As in attention, the network has not learned the weights
themselves. It has learned how to compute them from what is in front of
it: the noisy light, the surface color, the normal, the depth and the
motion of each pixel. That is why it can work on a scene it has never
seen.

The correspondence has limits. The neighborhood is small and fixed rather
than the whole sequence, and the values being averaged are the noisy
colors themselves, not learned projections of them. What carries over is
the structure: a score from features, a normalization, and a weighted sum
over a window.

## What it learned from

The question I cared about most was whether it would learn the task or
learn the scene. The network should learn what noise looks like next to an
edge, or behind glass, not what the marbles happen to look like. So it
never saw the marbles in training. I gave the scene generator a seed and
rendered 40 other layouts, with different marbles, swarms and camera
positions, plus the brass-sphere scenes, and held the real marbles back
as the test. Validation on four of the generated scenes decided when to
stop, and the marbles were scored once, after training ended.

The inputs are what the renderer will actually produce: ReSTIR PT's own
output at 960 × 540. Reuse correlates neighboring pixels and consecutive
frames, so its noise is different from a path tracer's, and a denoiser
trained on the wrong noise would be solving a different problem. Each
input frame comes with its surface features, taken through glass and
mirrors to the first rough surface behind them, as Intel's Open Image
Denoise and NVIDIA's NRD both prescribe. The noisy light is divided by
the surface color before the network sees it and multiplied back after,
so the wood grain and the swirls in the cat's-eyes don't depend on the
network keeping them. That step is close to what normalization does in a
model: it takes a factor the network shouldn't have to learn out of the
values it works on.

The targets are the same instants rendered with 4,096 samples a pixel.
Rendering all 400 frames of the data set took about four hours on my M3
Max. At that sample count most of the frame is clean, but the inside of
the glass and the cat's-eye cores still speckle. I kept those targets for
this first round, and the test let me check what it cost: against
65,536-sample references, the targets' own error is about a tenth of the
network's, in the glass as much as anywhere. For now the network's own
error is what limits the result.

I trained two sizes in PyTorch, with Open Image Denoise's loss and
schedule and a temporal term meant to discourage flicker: a small network
of 294,000 parameters and a wider one of 543,000 at Open Image Denoise's
own scale. Training and validation loss stayed together for all 300
epochs, so neither was memorizing its scenes. The wider one was 1 to 9%
better, and that is the one the demo uses.

## Running it inside the frame

Running the network inside the renderer meant writing it as Metal compute
kernels in the same encoder as the rest of the frame. Apple's Metal 4 has
an encoder for machine-learning models, but it would have meant building
each network into a Core ML package offline, rebuilding the pipeline when
the window changes size, and running shaders I couldn't inspect or check
against PyTorch. MPSGraph records into its own command buffers. The
shipping real-time upscalers and denoisers I could find out about run
their networks as the vendor's own compute shaders in the engine's command
stream, which is what this does.

The first version of the denoise pass took 162 ms at the window's size.
Most of that time went to reading the same input pixels and weights from
device memory over and over. Staging each block of input channels and
their weights in threadgroup memory once, and having the threads that need
them read from there, brought it to 57 ms. Keeping the head's scores on
chip instead of writing them out, storing the weights in half precision,
and letting each decoder read the layer below it where it already lies,
instead of copying it into a concatenated buffer, brought it to 30 ms
while the GPU was still shared with training, and 26 ms once it wasn't.
The same sequence shows up in attention kernels: most of the gain comes
from where the data sits when the arithmetic needs it.

The kernels match PyTorch closely enough that, on real frames, 99.1 to
99.3% of the displayed pixels are identical and nearly all the rest differ
by one level of eight-bit color. The difference comes from half precision,
and a test in the build fails if a single weight is shifted by 0.1.

## What the frame looks like now

![ReSTIR PT and MNEE (left) and the same light denoised (right), ten seconds into the opening](images/marbles-restir-pt-vs-denoised-10s.jpg)

*Ten seconds into the opening. Left: Milestone 2's ReSTIR PT and MNEE at
1920 × 1080, 39.8 ms (about 25 frames a second). Right: the same light
traced at 960 × 540 and denoised to 1920 × 1080, 23.5 ms (about 43 frames
a second).*

On the marbles, which it never saw, the network leaves 0.29 to 0.39 of the
raw input's displayed error across the seven test sequences, measured
against the 4,096-sample targets after tone mapping. That is 18 to 38%
less than Open Image Denoise run on each frame.

| Region, all lights awake (t = 120 s) | ReSTIR PT, upscaled | Open Image Denoise | The network |
|---|---|---|---|
| Whole frame | 0.00462 | 0.00221 | 0.00163 |
| Glass and metal | 0.0280 | 0.0151 | 0.0095 |

*Displayed MSE against the 65,536-sample reference. Open Image Denoise runs
on each 960 × 540 frame, then the result is upscaled.*

Because the light is traced at a quarter of the pixels, the denoised frame
is also cheaper than the raw one it replaces: 23.5 ms at 1920 × 1080 against
Milestone 2's 39.8 ms. Most of what I hoped for is visible. The porcelain
reads as porcelain, the swirls and the wood grain stay sharp, and the
frame no longer boils from one frame to the next.

Some things are not fixed. Once every firefly is awake, the light on the
table changes 36 to 47% more from frame to frame than in the path-traced
targets, a slow shimmer that a single frame doesn't show. The temporal
term turned out to be under 3% of the training loss, so the network was
hardly asked to prevent it; giving that term more weight is the first
thing for the next training round. The large out-of-focus marbles in the
foreground have ragged edges, because each pixel's features come from a
single ray through the lens. And the fireflies seen through glass and in
the metal marbles, points smaller than a pixel at the input's resolution,
come out as soft glows rather than sharp glints.

## Next

At the window's 3456 × 2234, close to 4K, the frame takes 77 ms, about 13
frames a second. The network is 26 ms of that, a full-resolution pass for
the surface color another 9 ms, and tracing the light at the lower
resolution 41.5 ms. Even a free denoiser would leave the frame at about 24
frames a second, so the next milestone is the light itself: the reuse
passes treated as kernels, measured for where the time and the bytes go,
with the same questions I would ask of an attention kernel that reads too
much of its cache.
