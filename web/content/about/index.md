---
title: Why Serenity?
description: Why I’m building Serenity, and the GPU questions connecting rendering to inference.
layout: about
---

I’m building Serenity to explore GPU performance from the rendering side. Moving light gives me a way to investigate techniques that also matter to transformer inference: processing data in small pieces, keeping a small amount of running state, and combining partial results or reusing earlier work.

The scene is deliberately simple: glass, porcelain, and metal marbles on a walnut table, lit by moving fireflies. That leaves the lighting to carry the image, and gives me something visible to inspect as I work on the kernels underneath.

## Where it is now

Milestone 2 adds ReSTIR path reuse and a search for light focused through glass. Reusing earlier work makes much of the displayed image cleaner for a similar frame cost, especially the table and porcelain. The glass remains the harder part.

The current renderer takes around 40 ms at 1920 × 1080 on my M3 Max. Next are denoising and tracing at a lower resolution with upscaling, working toward a clean image at the display’s full resolution and 60 frames a second.

The [second milestone]({{< relref "/writing/milestone-2" >}}) follows those decisions and their connections to inference. The [first milestone]({{< relref "/writing/milestone-1" >}}) records how I built the baseline.

## Why the name

In path tracing, the bright stray pixels in a noisy render are called fireflies. Here they are also, literally, the lights. Serenity is named for the calm, clean frame I’m working toward.

If the name brings to mind a small ship that kept flying, that’s not an accident. It’s why the metal spheres are polished: they had to be shiny.

## Part of Plainsight Systems

I’m a systems architect and builder. At [Plainsight Systems](https://plainsight-systems.com/about/), I work through questions that interest me by building things I can inspect and measure. Serenity lets me approach the GPU problems I encounter in inference through rendering, where the consequences of a change are visible in the image as well as the frame time.

The [Plainsight Systems site](https://plainsight-systems.com/) brings that work together, with the other projects, research, and writing.
