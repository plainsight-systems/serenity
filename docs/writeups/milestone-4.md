# Milestone 4: making the frame fast

![The marbles twenty seconds into the opening, the swarm perched among them, traced in a checkerboard and denoised by the retrained network](images/marbles-m4-20s.jpg)

*Twenty seconds into the opening. Half the light's pixels are traced each
frame, in an alternating checkerboard, at 960 × 540; the retrained network
denoises and upscales them to 1920 × 1080. The frame takes 16.7 ms.*

[Milestone 3](milestone-3.md) ended with a clean image that was still too
slow. A 1920 × 1080 frame took 23.5 ms, about 43 frames a second, and the
window's 3456 × 2234 took 77 ms, about 13. This milestone was about the
time itself: finding where a frame goes, then applying the optimizations
that shipped engines use, one at a time, measuring each against the frame
before it.

At 1920 × 1080 the frame now takes 16.7 ms, 60 frames a second. The
window's size takes 56 ms, about 18. The largest gain came from tracing
less. The most instructive step gained the least: running two blocks of
work side by side showed how little of the GPU's apparently idle hardware
was free to use.

## Where the time went

Before changing anything I took per-pass GPU timestamps and the GPU's
performance-limiter counters at the window's size, with every firefly
awake. Two blocks made up most of the frame. Tracing and reusing the light
took about 41 ms, and the hardware ray-tracing unit was about 90% busy
throughout. The arithmetic units sat at 12 to 15%, and only a fifth of
each core's thread capacity was occupied. The denoiser took 26 ms and was
the opposite: arithmetic 97 to 98% busy, the ray-tracing unit idle.

Run one after the other, each block leaves the other's hardware waiting.
That suggested an order. Overlap the two blocks first, then make the light
cheaper, since it was the larger block and the one an overlap would leave
exposed.

## Running the two halves at once

I split the frame into two halves. The light half traces and reuses the
light and writes the denoiser's inputs; the image half runs the network
and the tone map. They go on two command queues, and the image half of a
frame waits for the light half of the same frame through a shared event,
so frame N's network can run while frame N + 1 traces. Everything the
image half reads from the light half has a copy for each frame in flight,
so the next frame never overwrites what the current one is still reading.
The image is unchanged, bit for bit. The cost is latency: a frame's work
now spans two frame intervals on the GPU. Since two frames were already in
flight, the delay from a frame's inputs to its image grew only from 147 to
153 ms at the window's size.

The counters had promised a lot. If the two halves really used different
hardware, the frame would shrink toward the longer half alone, about
50 ms. The timestamps showed them overlapping almost completely, and the
frame went from 76.8 to 73.8 ms. That is 4% at the window's size and 10%
at 1920 × 1080.

Both halves had slowed down. Running beside the light, the network
reached 43% of its own speed. The light ran at 67% of its own speed
beside the network. Small passes suffered most: a 0.55 ms depth median filter
took 8.9 ms when it was queued behind the light's long-running
threadgroups. My explanation, which I have not confirmed with counters
taken during the overlap, is that a core's threads, registers and
threadgroup memory are one pool. The ray-traced kernels hold their share
of that pool while they wait on the ray-tracing unit, so the arithmetic
units look idle but there is little room left to put work on them.

This is the part of the milestone closest to inference. The same question
comes up there whenever one kernel's work is overlapped with
another's, such as compute beside communication, or two models sharing a
device. A utilization counter says what is idle, not what is free to take.
I kept the overlap because it helps and costs only latency, but it did not
change where the time goes.

## Not tracing what cannot cast a shadow

616 of the scene's 653 objects are fireflies about 1.5 mm across. Every
shadow ray, asking whether a chosen light is visible from a point, was
also tested against all of them. In this scene a firefly's own shadow is
almost never visible, so those tests were nearly all wasted.

The usual way to skip them is a ray mask, which DirectX ray tracing and
Vulkan both provide. Metal also has masks, but only on instances, and
Serenity's scene is a single structure with no instances. I tried the
literal version first: one structure for the surfaces, one for the
fireflies, each an instance with its own mask under a top level. It
rendered the same image and made the frame slower, 87 ms against 75,
because every ray, the camera's included, now had to cross the extra level.

The version I kept builds a second structure holding only the 37
surfaces, once at start-up since none of them moves. Shadow rays and the
reuse passes' visibility checks trace it; everything else traces the full
scene. The reuse passes' shifts, where a visibility ray is most of the
tracing, became 13 to 15% faster, and the frame 3 to 4%. The image changes only where a firefly used to shade the
wood beside it, which I looked at and accepted.

That measurement also settled the next planned step. Splitting the scene
into a static structure and a moving one is the standard layout, but its
point is to avoid rebuilding what does not move, and rebuilding the
fireflies here already costs almost nothing. It is the same two-level
layout that made the instance masks slower, so I skipped it on Metal. On
the Vulkan and Xbox backends, where two-level structures are the
hardware's normal case, I expect to come back to it.

## Tracing half the pixels

With the light still the longer half, the remaining option was to trace
less of it. Checkerboard rendering traces half the pixels each frame: the
pixels where x + y is even on one frame and odd on the next. I followed
NVIDIA's RTXDI, which has a checkerboard mode, reading its source before
building mine. Every pass in the light half runs one thread per traced
pixel, the reservoirs are packed two pixels to a slot, and each reuse
step looks for its candidates in the pixels the relevant frame actually
traced. The surface features stay complete every frame, because the
network uses them as guides and the next step needs them as evidence.

That step is the fill. Each untraced pixel takes the light the previous
frame traced where its motion vector says the surface was, provided the
depth there matches what the motion predicts. Otherwise it borrows from
the traced pixels beside it at the same depth. My first attempt reprojected
the whole previous image, as temporal antialiasing does. That let filled
pixels feed the next fill, and the untraced half drifted 8% darker than
the traced half. Taking only pixels the previous frame actually traced
brought that to 2.8%, most of it in fireflies seen directly, which are a
pixel or two across and move.

The light half's passes now take 56% of their earlier time at the
window's size. At 1920 × 1080 the frame went from 20.7 to 16.7 ms, and at
the window's size from 71.7 to 56.0 ms. The image was slightly smoother,
with less speckle on the table, and slightly softer: the striped marbles'
edges and the sparkle in the steel marble were less crisp.

## Teaching the network the new input

The network had been trained on complete frames of light, so I retrained
it on checkerboard input. The training targets come from the path tracer,
which checkerboard does not touch, so I compared one regenerated target
with the stored one. The ray masks had changed it by 0.1% in the mean,
which is far below the targets' own noise, so I kept the existing targets
and rendered only the new inputs. That took half an hour on the Mac
instead of four hours for the whole data set.

This was also the first training run on my RTX 3090. The training set is
about 24 GB once loaded, which fits beside the network in the Mac's shared
memory but not on a 24 GB card. Training now measures its own working
memory with one throwaway step when it starts, then keeps the data on the
GPU if both fit and in host memory if not, copying each batch across.
On the 3090 the batches came from host memory and the GPU still stayed
92% busy. The run took 3.5 hours. The original took 8.3 on the Mac, though
it shared the GPU with a second training run.

On the held-out marbles, the retrained network on checkerboard input
leaves 0.29 to 0.39 of the raw input's error, within about 2% of the
original network on complete frames in every sequence and region. It
still has 18 to 37% less error than Open Image Denoise. Side by side with
the original network on checkerboard input, the difference is small, a
little smoother on the table and the porcelain. The table's slow shimmer
is unchanged, still about 44% more frame-to-frame variation than the
targets once every firefly is awake. That remains for a training round
aimed at it.

## Where the frame stands

| Output | Milestone 3 | Milestone 4 |
|---|---|---|
| 1920 × 1080 | 23.5 ms, 43 fps | 16.7 ms, 60 fps |
| 3456 × 2234 | 77 ms, 13 fps | 56.1 ms, 18 fps |

*The fully lit scene (t = 120 s) on my M3 Max on AC power, frame interval
with two frames in flight, as the window runs. The
[measurement notes](../research/2026-10-17-milestone-4-plan.md) have every
step's numbers and the passes behind them.*

At the window's size, the light half takes about 32 ms alone and the image
half about 28, and overlapped they share the GPU in 56. The network is
still arithmetic-bound at 26 ms, and producing the full-resolution surface
color it multiplies back in takes another 8.5 ms. Both now sit beside a
light half that has room left to shrink.

That is where the next milestone starts. The reuse passes merge
reservoirs much as attention merges partial softmax results, and I have
treated them as passes rather than as kernels. Tiling, data layout and
fusing the merges are the questions that brought me to Serenity, and they
are what I want to measure next. After that, the Vulkan backend will put
the same scene on the RTX 3090 and the Strix Halo.
