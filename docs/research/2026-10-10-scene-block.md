# Binding the scene: one block, against Apple, Falcor and Unreal

*2026-10-10. Apple M3 Max, macOS 26.6.2, Xcode 26.2, Metal compiler
32023.864, `-std=metal4.0`. Release builds.*

## Question

The review of df9d622 asked that passes bind the scene, not each of its
kinds: before, every pass bound each of 19 arrays at a slot of its own, and
each new material, texture, medium or light kind changed every pass. The
block that replaced them (`src/metal/scene/scene_block.h`, e851c53) made the
path tracer about 5% slower. What do production renderers do on Metal, what
did the block do differently, and what does each difference cost?

## Method

The path tracer (`graphs/path.toml`) at 3456 x 2234 on two scenes: five
spheres (the five-sphere marbles of a987908: wood table, glass, brass, no
medium, no coat) and `scenes/marbles.toml`. 200 frames committed back to back
with frames in flight, each settled frame's command-buffer GPU start to end,
the median of frames 21 to 199; each variant run twice, the runs within
0.03 ms of each other. No serenity window running. The variants were
working builds; the baselines were built in a worktree at b3977c6, before
the block. Frame times only: Instruments' Metal System Trace crashed
recording from the command line (exit 139, an empty trace) with Developer
Mode off, so no counters attribute the costs further (GPU.10).

## What production renderers do

| System | How shaders reach the scene's arrays | Source |
|---|---|---|
| Apple's bindless scene | A `Scene` struct of 64-bit GPU addresses the host writes, bound once as `constant Scene&`; its arrays are `constant` pointers (`constant Mesh* meshes; constant Material* materials;`). | WWDC22 10101 |
| Apple's shader guidance | The `constant` address space is for data read many times by many threads; a fixed-size struct passed by reference in `constant` can be preloaded into constant registers. | WWDC16 606 |
| Apple's ray tracing guidance | Per-primitive data stored in the acceleration structure and read with one load (`get_candidate_primitive_data()`), instead of chains of buffer lookups in the intersection code: 10 to 16% in Apple's test, and fewer live buffer pointers (register pressure). | WWDC22 10105 |
| Falcor | `ParameterBlock<Scene> gScene` holds the scene's buffers; its hottest, `worldMatrices`, `inverseTransposeWorldMatrices` and `geometryInstances`, are `[root]`: bound as D3D12 root descriptors, outside the block's descriptor table. | `Scene/Scene.slang`; `Core/Program/ProgramReflection.h` (`isRootDescriptor`) |
| Unreal on Metal | Shaders are written once in HLSL for D3D12 and cross-compiled: HLSL to SPIR-V to MSL through ShaderConductor and SPIRV-Cross, and, experimentally since UE 5.4, DXIL to Metal IR through Apple's Metal Shader Converter. Under the converter every resource is reached indirectly through one top-level argument buffer per pipeline, laid out from the D3D12 root signature: root constants and raw-buffer addresses inline, descriptor tables as further tables of entries. SPIRV-Cross puts argument buffers in the `constant` address space unless told otherwise (`--msl-device-argument-buffer`, for buffers over 64 KB). Scene data itself is GPUScene: flat primitive and instance buffers indexed by ID. Hardware ray tracing (Lumen, the path tracer) is not supported on macOS. | Epic, "Bringing Unreal Engine on macOS up to feature parity" (2025); Epic's `FShaderConductorContext` API page; Apple WWDC23 10124; the `spirv-cross` manual; Epic's hardware specifications page |

Not verified: Unreal's own Metal binding code. Its source is in Epic's
private GitHub repository, and the Metal Shader Converter manual, which
gives the converter's address spaces, is a developer download. Whether
Unreal enables SPIRV-Cross's argument buffers on Metal, and in which
address space the converter reads a structured buffer, are open.

So all three put the scene behind one block of addresses, as serenity
does; Unreal on Metal goes further, reaching every resource indirectly.
The block differed from Apple's in its address space, from Falcor's in
binding nothing hot outside it, and from Apple's ray tracing guidance, as
the per-slot bindings did before it, in chaining lookups per candidate hit.

## Results

| Variant | Five spheres | Marbles |
|---|---|---|
| Each array at a slot of its own, `device` (b3977c6) | 14.77 ms | 10.99 ms |
| Each array at a slot of its own, `constant` | 14.80 | 11.00 |
| The block, `device` pointers, views built once per thread and carried through the path (e851c53 as first written) | 16.18 | — |
| The same, without the medium fixes of e851c53 | 16.09 | — |
| The block, views built where they are used (e851c53) | 15.57 | 11.64 |
| The same, the shape records and boxes read through the block inside the intersection loop | 15.63 | — |
| **The block, its arrays `constant` (adopted)** | **15.07** | **11.23** |
| The same, the sky and light counts held in the block by value | 15.06 | 11.24 |
| The same, the frame's transforms and glows also `constant` (kept, for one rule) | 15.07 | 11.24 |
| The same, the shape records and boxes also bound directly beside the block, as Falcor's `[root]` | 15.08 | 11.23 |

## Findings

- A `device` pointer loaded from the block cost what a bound one did not;
  the same pointer in the `constant` address space costs what a bound one
  does. Per-slot bindings measure the same in either address space, so the
  address space mattered only for the block. Apple's own bindless example
  uses `constant`; the block had not.
- Views carried through the path held their pointers in registers for its
  whole length; built at each use, they are reloaded from the block. That
  saved 0.6 ms, the register cost Apple names in WWDC22 10105.
- Binding the hottest arrays directly saved nothing: on Metal a pointer read
  from a `constant` block costs as a bound one does. Falcor's `[root]` saves
  a D3D12 descriptor-table fetch, which Metal does not have.
- Holding the sky and light counts in the block by value, where Metal could
  preload them, saved nothing: each is read once per bounce.
- What is left: 0.27 ms on five spheres over the per-slot bindings, of which
  the medium fixes of e851c53 are 0.07–0.09 ms (they are not in the b3977c6
  baseline); the block itself about 0.2 ms, under 2%. The marbles, 0.24 ms.

## Decision

Adopted: the block, its arrays and the frame's transforms and glows in the
`constant` address space, views built where they are used
(`scene_block.h`, `scene_block.metal.h`), 24e78a3.

Open:

- Per-primitive data in the acceleration structure, Apple's WWDC22 10105
  recommendation and Unreal's GPUScene shape: one load per candidate hit
  instead of the shape record, its box and its transform. It targets a
  cost the per-slot bindings had too, and the moving fireflies' structure
  is rebuilt each frame (2026-10-09-acceleration-structure.md), which a
  design for it must account for.
- Counters, to attribute the remaining 0.2 ms, once Instruments can record.

## Sources

- Apple, WWDC22 10101, [Go bindless with Metal 3](https://developer.apple.com/videos/play/wwdc2022/10101/)
- Apple, WWDC22 10105, [Maximize your Metal ray tracing performance](https://developer.apple.com/videos/play/wwdc2022/10105/)
- Apple, WWDC16 606, [Advanced Metal Shader Optimization](https://developer.apple.com/videos/play/wwdc2016/606/)
- Apple, WWDC23 10124, [Bring your game to Mac, Part 2: Compile your shaders](https://developer.apple.com/videos/play/wwdc2023/10124/)
- Apple, [Metal shader converter](https://developer.apple.com/metal/shader-converter/)
- NVIDIA, Falcor, [Scene.slang](https://github.com/NVIDIAGameWorks/Falcor/blob/master/Source/Falcor/Scene/Scene.slang) and [ProgramReflection.h](https://github.com/NVIDIAGameWorks/Falcor/blob/master/Source/Falcor/Core/Program/ProgramReflection.h)
- Epic, [Bringing Unreal Engine on macOS up to feature parity with Windows](https://www.unrealengine.com/tech-blog/bringing-unreal-engine-on-macos-up-to-feature-parity-with-windowsprogress-report) (2025)
- Epic, [FShaderConductorContext](https://dev.epicgames.com/documentation/unreal-engine/API/Developer/ShaderCompilerCommon/FShaderConductorContext)
- Epic, [Hardware and software specifications](https://dev.epicgames.com/documentation/en-us/unreal-engine/hardware-and-software-specifications-for-unreal-engine)
- Epic, [ERHIBindlessConfiguration](https://dev.epicgames.com/documentation/unreal-engine/API/Runtime/RHI/ERHIBindlessConfiguration)
- Khronos, [spirv-cross manual](https://manpages.debian.org/testing/spirv-cross/spirv-cross.1)
- Microsoft, [Using descriptors directly in the root signature](https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-descriptors-directly-in-the-root-signature)
