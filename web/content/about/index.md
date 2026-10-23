---
title: Why Serenity?
description: Why I’m building Serenity, and the GPU questions connecting rendering to inference.
layout: about
---

I’m building Serenity to explore GPU performance from the rendering side. Moving light gives me a way to investigate techniques that also matter to transformer inference: processing data in small pieces, keeping a small amount of running state, and combining partial results or reusing earlier work.

The scene is deliberately simple: glass, porcelain, and metal marbles on a walnut table, lit by moving fireflies. That leaves the lighting to carry the image, and gives me something visible to inspect as I work on the kernels underneath.

## Where it is now

Milestone 5 moves the frame to my RTX 3090 Ti, through CUDA and OptiX rather than a portable layer, so I can work with NVIDIA’s own tools and instructions. The first port ran at 89 frames a second at 1920 × 1080. Arranging its work for the card brought it to 174: the spheres became triangle meshes for the ray-tracing cores, unfinished light searches are regrouped so threads don’t sit idle, the GPU code uses fast math, and the network runs in the tensor cores’ own shape. On my M3 Max, Milestone 4 runs the same frame at 57.

Next is a milestone designed for the M3 Max: what held up on the 3090, redesigned from Apple’s own guidance. Then AMD’s Strix Halo through HIP and HIPRT, and Direct3D 12 on the Xbox Series X. After that the reuse steps can be tuned as kernels on each GPU, the question that started this.

The [fifth milestone]({{< relref "/writing/milestone-5" >}}) follows the port and the choices that doubled it. The [fourth]({{< relref "/writing/milestone-4" >}}) follows each optimization on the Mac and what it bought; the [third]({{< relref "/writing/milestone-3" >}}) follows the model from training data to the Metal kernels that run it; the [second]({{< relref "/writing/milestone-2" >}}) covers path reuse and light focused through glass; the [first]({{< relref "/writing/milestone-1" >}}) records the baseline.

## Why the name

In path tracing, the bright stray pixels in a noisy render are called fireflies. Here they are also, literally, the lights. Serenity is named for the calm, clean frame I’m working toward.

If the name brings to mind a small ship that kept flying, that’s not an accident. It’s why the metal spheres are polished: they had to be shiny.

## Part of Plainsight Systems

I’m a systems architect and builder. At [Plainsight Systems](https://plainsight-systems.com/about/), I work through questions that interest me by building things I can inspect and measure. Serenity lets me approach the GPU problems I encounter in inference through rendering, where the consequences of a change are visible in the image as well as the frame time.

The [Plainsight Systems site](https://plainsight-systems.com/) brings that work together, with the other projects, research, and writing.
