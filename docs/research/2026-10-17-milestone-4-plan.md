# Milestone 4: making the frame fast

2026-10-17. Four standard optimizations, chosen by the user from the
frame's measured costs: async compute overlap, ray masks, a split scene
structure, and checkerboard rendering of the light. Reservoir reuse as
kernels (tiling, layout and fusion of the merge passes) stays in the queue
as its own later item.

## Where the frame goes

The denoised demo (graphs/restir_pt_denoised.toml, the wide network) in
the 3456 × 2234 window, light traced at 1728 × 1117, marbles at t = 120,
M3 Max on AC power. Per-dispatch GPU timestamps, median of three rounds;
limiters from the GPU's Performance Limiters counters. t = 20 is within 2%;
1920 × 1080 output (23.5 ms) scales the same way. Raw data and scripts:
serenity-scratch/frame-cost-map/ beside the checkout.

| Block | ms | Limited by |
|---|---|---|
| Initial path sampling | 15.6 | ray-tracing unit, 91% active |
| MNEE solve | 4.5 | F32 arithmetic 59%, ray tracing 47% |
| Temporal reuse (shift, replay, merge) | 7.3 | ray tracing 91 to 95% in the shift and replay |
| Spatial reuse (shift, replay, merge) | 9.5 | ray tracing 94% in the shift and replay |
| Shade, duplication map | 1.3 | memory, instruction issue |
| Features at the light's size | 2.7 | ray tracing 84% |
| Features at the output's size (for the full-size albedo) | 8.4 | ray tracing 97% |
| Denoiser | 26.4 | F32 arithmetic 97 to 98% |
| Tone map | 1.0 | |
| **Frame** | **77** | about 13 frames a second |

Two facts shape the plan. The light block is bound by the ray-tracing
unit, about 90% busy, while ALU utilization sits at 12 to 15% and
occupancy at 16 to 22%. The denoiser is the reverse: arithmetic-bound with
the ray-tracing unit idle. Run in sequence, each leaves the other's
hardware idle.

## The four optimizations, in order

1. **Async compute overlap.** Frame N's denoiser runs while frame N + 1
   traces its light, on different hardware, so the frame time moves toward
   the slower of the two instead of their sum. The image is unchanged; the
   cost is one frame of latency. Console engines use this routinely.
2. **Ray masks.** 616 of the scene's 653 objects are fireflies 1.5 mm
   across, and every ray traverses them. Visibility rays (NEE shadow rays,
   the shifts' reconnection checks) skip them through Metal's intersection
   masks, as DXR's instance masks and Vulkan's cull masks are used in
   production. The image changes only by the fireflies' own shadows,
   shown to the user before it is kept.
3. **Split scene structure.** The static marbles and table built once, the
   moving fireflies in their own structure, a top level over both, so only
   what moves is rebuilt and rays that only need occluders never touch the
   swarm. The standard two-level layout.
4. **Checkerboard rendering of the light.** Each frame traces half the
   pixels, alternating, and temporal reuse fills the other half, as console
   checkerboard rendering and RTXDI's checkerboard mode do. It changes the
   image and what the denoiser sees: the network was trained on full light
   frames. Built last, judged with the user on the denoised image; if the
   network needs retraining on checkerboard input, the pipeline from
   Milestone 3 does it (data about 4 h on the Mac; training on the RTX 3090,
   below).

Each change is measured as it lands (per-dispatch timestamps, frame time
at both sizes), kept only if faster, and checked for an unchanged image
except where the change is meant to alter it (2's firefly shadows, 4),
which the user sees first.

## Training on the RTX 3090

Any retraining runs on the user's RTX 3090 Linux machine over SSH, not on
the Mac: it frees the Mac's GPU and should train faster than the M3 Max's
`mps` backend. The data set is rendered on the Mac (the renderer is
Metal) and copied over. This needs the training toolkit's environment to
resolve for Linux with CUDA (its lockfile pinned for both), a declared
dependency change made when the retrain is decided, not before.
