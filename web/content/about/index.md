---
title: Why Serenity?
description: Why I’m building Serenity, and the GPU questions connecting rendering to inference.
layout: about
---

I’m building Serenity to explore GPU performance from the rendering side. Moving light gives me a way to investigate techniques that also matter to transformer inference: processing data in small pieces, keeping a small amount of running state, and combining partial results or reusing earlier work.

The scene is deliberately simple: glass, porcelain, and metal marbles on a walnut table, lit by moving fireflies. That leaves the lighting to carry the image, and gives me something visible to inspect as I work on the kernels underneath.

## Where it is now

Milestone 4 makes the frame fast. It applies the optimizations real-time renderers use once the image is right: the light and the network run at the same time on two GPU queues, shadow rays skip the 616 fireflies, and each frame traces half the light’s pixels in an alternating checkerboard. Checking every stage against shipped engines then moved depth of field after the network, which cleaned up the near marbles, and I retrained the network for the new input, for the first time on my RTX 3090.

The fully lit scene now runs at 57 frames a second at 1920 × 1080 on my M3 Max, up from 43, and about 17 at the display’s full size. The network and the reuse steps are now the larger costs. How to tune the reuse steps as kernels depends on the GPU, so next Serenity leaves the Mac: a Vulkan backend for my RTX 3090 and AMD’s Strix Halo, and Direct3D 12 for the Xbox Series X.

The [fourth milestone]({{< relref "/writing/milestone-4" >}}) follows each optimization and what it bought. The [third]({{< relref "/writing/milestone-3" >}}) follows the model from training data to the Metal kernels that run it; the [second]({{< relref "/writing/milestone-2" >}}) covers path reuse and light focused through glass; the [first]({{< relref "/writing/milestone-1" >}}) records the baseline.

## Why the name

In path tracing, the bright stray pixels in a noisy render are called fireflies. Here they are also, literally, the lights. Serenity is named for the calm, clean frame I’m working toward.

If the name brings to mind a small ship that kept flying, that’s not an accident. It’s why the metal spheres are polished: they had to be shiny.

## Part of Plainsight Systems

I’m a systems architect and builder. At [Plainsight Systems](https://plainsight-systems.com/about/), I work through questions that interest me by building things I can inspect and measure. Serenity lets me approach the GPU problems I encounter in inference through rendering, where the consequences of a change are visible in the image as well as the frame time.

The [Plainsight Systems site](https://plainsight-systems.com/) brings that work together, with the other projects, research, and writing.
