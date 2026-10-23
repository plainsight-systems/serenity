# The frame on the M3 Max's hardware

Status: the profile, 2026-10-23 (the plan's step 0, docs/process/QUEUE.md,
Milestone 6). The design follows in this note, from Apple's guidance and
production on Apple silicon, before any header or code, as the RTX 3090
Ti's did (`2026-10-20-nvidia-hardware-design.md`). Milestone 4 applied
engines' standard optimizations on this machine; nothing yet is designed
for its own hardware.

## 1. The machine

| | |
|---|---|
| GPU | Apple M3 Max, 40 GPU cores |
| OS, tools | macOS 26.6.2 (25G83), Xcode 26.2, `xctrace` 26.0 (17C52) |
| Power | AC, the GPU idle before each run (no window, wallpaper, video or Photos analysis running) |
| Threads | SIMD groups of 32 (`threadExecutionWidth`); every pipeline reports 1,024 threads a threadgroup (`maxTotalThreadsPerThreadgroup`), the convolutions and the denoiser's output included: with the M3's Dynamic Caching a pipeline's registers no longer bound that number, so occupancy is read from the GPU's counters (section 3), not from the API |

## 2. The frame, and each pass's time

**The frame** (`serenity-headless --times`, the demo graph on the marbles,
frames 7000 to 7599, t = 117 to 127 s, the second of two runs; the first
runs agreed on the mean and had wide ranges):

| Size | Frame |
|---|---|
| 1920 x 1080 | 56.6 fps (17.67 ms) |
| 3456 x 2234 | 15.8 fps (63.21 ms) |

Milestone 4 recorded 57 fps (17.5 ms) and 17.1 fps (58.5 ms) by its own
harness. 1920 x 1080 agrees; at 3456 x 2234 this is 8% slower, not yet
explained (open, section 4).

**The two halves' overlap.** The same frames rendered one at a time, each
waited for before the next is begun (below): 20.0 ms at 1920 x 1080 and
62.1 ms at 3456 x 2234. Overlapping frame N's image half with frame N + 1's
light half saves 2.4 ms at 1920 x 1080 and nothing at 3456 x 2234, where
the overlapped frame is 1 ms longer.

**Each pass's GPU time** (median of 80 frames at t = 120 s after 21
warm-up frames, rendered one at a time; a precise timestamp after every
dispatch, Metal 4's counter heap; the probe is scratch code, never
committed, as Milestone 4's frame-cost map was):

| Block | 1920 x 1080 | 3456 x 2234 | Share at 3456 x 2234 |
|---|---|---|---|
| The denoiser: depth median, input, 11 convolutions, 4 pools, output | 7.77 ms | 26.54 ms | 43% |
| of which the convolutions | 6.08 | 21.24 | 34% |
| ReSTIR PT's initial pass (the power pyramid, the light tiles, the path tree) | 3.36 | 8.95 | 14% |
| The features at the output's size (the full-size albedo) | 2.53 | 8.10 | 13% |
| The spatial reuse: shift, replay, merge | 1.48 | 4.71 | 8% |
| The temporal reuse: shift, replay, merge | 1.27 | 3.69 | 6% |
| The features at the light's size | 1.17 | 2.62 | 4% |
| MNEE's solve | 0.96 | 2.56 | 4% |
| The depth of field | 0.70 | 2.47 | 4% |
| The tone map | 0.31 | 1.06 | 2% |
| The shade, the duplication map, the fill | 0.36 | 1.23 | 2% |
| **The frame, one at a time** | **20.0** | **62.1** | |

## 3. What limits each pass

The GPU's Performance Limiters counters at 3456 x 2234, t = 120 s (the
saved Instruments template through `xctrace`, every dispatch separated by
a one-thread spin so the samples can be cut into dispatches, as Milestone
4's were). Only the larger dispatches; percentages are each counter's mean
inside the dispatch:

| Dispatch | ms | Limited by | Occupancy (target) |
|---|---|---|---|
| the convolutions, each of the 11 | 0.16 to 7.03 | F32 arithmetic 80 to 98%, F16 0% | 20 to 31% (86 to 100%) |
| the denoiser's output | 3.00 | F32 57% | 20% (100%) |
| ReSTIR PT's initial | 8.76 | ray-tracing unit 78%, last-level cache 39% | 15% (30%) |
| the features at the output's size | 7.99 | ray-tracing unit 97% | 22% (24%) |
| the features at the light's size | 2.58 | ray-tracing unit 86% | 20% (30%) |
| the shifts and replays, four | 1.21 to 2.63 | ray-tracing unit 75 to 90% | 16 to 18% (23 to 35%) |
| MNEE's solve | 2.60 | F32 56%, instruction issue 50%, ray-tracing unit 33% | 13% (40%) |
| the merges, the shade, the fill, the denoiser's input | 0.21 to 1.15 | memory: address translation (MMU) 38 to 81%, last-level cache 33 to 61% | 31 to 88% |
| the depth of field, the tone map | 0.19 to 1.03 | instruction issue 33 to 96% | 33 to 99% |

The 1920 x 1080 recording did not save: on this machine `xctrace` saves a
counters recording once or twice after its services start and fails from
then on, idle runs included (exit 138, an empty trace; an Instruments left
running since 2026-10-10 had blocked every save before it was quit). The
display's size is the hard case the measurement rule asks for; its
timestamps at 1920 x 1080 are in section 2.

What the counters say, facts first:

- **Fact:** the network's convolutions run entirely in 32-bit arithmetic
  (the F32 limiter at 80 to 98%, the F16 limiter at 0), at about a quarter
  of the occupancy the GPU's occupancy manager aims for. **Inference:**
  their registers hold them there; half-precision inputs, which the 3090
  runs with 32-bit sums, would move the work to 16-bit arithmetic and
  halve its registers. The largest block, 43% of the frame.
- **Fact:** every pass that traces is bound by the ray-tracing unit (75 to
  97% active), and the occupancy manager holds those passes to a target of
  23 to 40%. **Inference:** what moves them is fewer ray-tracing cycles a
  ray (the shapes: section 4) or fewer rays.
- **Fact:** the full-size albedo, one ray an output pixel only to find the
  first surface that is not glass or a mirror, costs 8.1 ms, nearly the
  path tree's 9.0, with the unit 97% busy. **Inference:** production
  engines find that surface by rasterizing it, and trace only where glass
  or a mirror sends the ray on (primary surface replacement); a lead the
  3090 did not show, where it costs 0.74 ms.
- **Fact:** the small passes that read and write ReSTIR PT's arrays are
  bound by address translation and the last-level cache. **Inference:**
  RTXDI's 16 x 16 reservoir blocks, which the CUDA backend uses and Metal
  does not, are what these passes' locality asks for; they hold 4% of the
  frame together.

## 4. Open

- The 8% at 3456 x 2234 against Milestone 4's recorded frame (section 2).
- Each pipeline's registers, which Metal does not report; Xcode's GPU
  capture and shader profiler do, should the design need them.
- The design itself: Apple's guidance (WWDC's M3 GPU, profiling and ray
  tracing sessions; Metal 4's), Blender Cycles' Metal backend, Unreal's
  Metal RHI and Apple's samples, checked before each change is designed;
  then the changes in the order sections 2 and 3 rank them.
