---
title: Why Serenity?
description: Why I’m building Serenity, and the GPU questions connecting rendering to inference.
layout: about
---

I’m building Serenity to explore GPU performance from the rendering side. Moving light gives me a way to investigate techniques that also matter to transformer inference: processing data in small pieces, keeping a small amount of running state, and combining partial results or reusing earlier work.

The scene is deliberately simple: glass, porcelain, and metal marbles on a walnut table, lit by moving fireflies. That leaves the lighting to carry the image, and gives me something visible to inspect as I work on the kernels underneath.

## Where it is now

Milestone 3 adds a neural network I trained to denoise and upscale the light from ReSTIR. It learns how much nearby samples and earlier frames should contribute to each output pixel. Tracing at half the width and height leaves fewer pixels to render, while the reconstruction produces a much cleaner image.

A fully lit 1920 × 1080 frame now takes about 23.5 ms on my M3 Max, compared with 39.8 ms before reconstruction. At the window’s 3456 × 2234 it takes about 77 ms. The table still shimmers and small reflections lose detail. Next I’m working on the cost of tracing and reuse, toward a clean moving image at the display’s full resolution and 60 frames a second.

The [third milestone]({{< relref "/writing/milestone-3" >}}) follows the model from training data to the Metal kernels that run it. The [second milestone]({{< relref "/writing/milestone-2" >}}) covers path reuse and light focused through glass; the [first]({{< relref "/writing/milestone-1" >}}) records the baseline.

## Why the name

In path tracing, the bright stray pixels in a noisy render are called fireflies. Here they are also, literally, the lights. Serenity is named for the calm, clean frame I’m working toward.

If the name brings to mind a small ship that kept flying, that’s not an accident. It’s why the metal spheres are polished: they had to be shiny.

## Part of Plainsight Systems

I’m a systems architect and builder. At [Plainsight Systems](https://plainsight-systems.com/about/), I work through questions that interest me by building things I can inspect and measure. Serenity lets me approach the GPU problems I encounter in inference through rendering, where the consequences of a change are visible in the image as well as the frame time.

The [Plainsight Systems site](https://plainsight-systems.com/) brings that work together, with the other projects, research, and writing.
