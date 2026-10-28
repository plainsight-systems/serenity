---
title: Why Serenity?
description: Why I’m building Serenity, and the GPU questions connecting rendering to inference.
layout: about
---

I’m building Serenity to explore GPU performance from the rendering side. Moving light gives me a way to investigate techniques that also matter to transformer inference: processing data in small pieces, keeping a small amount of running state, and combining partial results or reusing earlier work.

The scene is deliberately simple: glass, porcelain, and metal marbles on a walnut table, lit by moving fireflies. That leaves the lighting to carry the image, and gives me something visible to inspect as I work on the kernels underneath.

## Where it is now

Milestone 6 designs the frame again for my M3 Max, which now runs it at 84 frames a second at 1920 × 1080, up from 57. The production designs for Apple silicon, Blender Cycles’ tracing and Apple’s own guidance, bought a few percent each. Most of the gain came from moving the denoiser’s network off the GPU to the Neural Engine, where it runs beside the GPU while the next frame’s light is traced. Milestone 5 had done the same for my RTX 3090 Ti, through CUDA and OptiX, taking it from 89 frames a second to 174.

Next is AMD’s Strix Halo through HIP and HIPRT, then Direct3D 12 on the Xbox Series X. After that the reuse steps can be tuned as kernels on each GPU, the question that started this.

The [sixth milestone]({{< relref "/writing/milestone-6" >}}) follows the M3 Max design and the move to the Neural Engine. The [fifth]({{< relref "/writing/milestone-5" >}}) follows the port to the 3090 and the choices that doubled it. The [fourth]({{< relref "/writing/milestone-4" >}}) follows each optimization on the Mac and what it bought; the [third]({{< relref "/writing/milestone-3" >}}) follows the model from training data to the Metal kernels that run it; the [second]({{< relref "/writing/milestone-2" >}}) covers path reuse and light focused through glass; the [first]({{< relref "/writing/milestone-1" >}}) records the baseline.

## Why the name

In path tracing, the bright stray pixels in a noisy render are called fireflies. Here they are also, literally, the lights. Serenity is named for the calm, clean frame I’m working toward.

If the name brings to mind a small ship that kept flying, that’s not an accident. It’s why the metal spheres are polished: they had to be shiny.

## Part of Plainsight Systems

I’m a systems architect and builder. At [Plainsight Systems](https://plainsight-systems.com/about/), I work through questions that interest me by building things I can inspect and measure. Serenity lets me approach the GPU problems I encounter in inference through rendering, where the consequences of a change are visible in the image as well as the frame time.

The [Plainsight Systems site](https://plainsight-systems.com/) brings that work together, with the other projects, research, and writing.
