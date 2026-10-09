# The acceleration structure for moving shapes

*2026-10-09. Apple M3 Max, macOS 26.6.2, Xcode 26.2, Metal compiler
32023.864, `-std=metal4.0`. Release builds.*

## Question

When fireflies move, how should the structure rays are traced against keep
up: as Unreal keeps its ray tracing scene (each geometry's bottom-level
structure built once, every object an instance by its transform, the top
level rebuilt every frame), or otherwise? The decision is
`src/metal/acceleration/scene_acceleration.h`.

## Method

The path tracer (`graphs/path.toml`) on `scenes/brass_sphere.toml`, still,
at 3456 x 2234: the median GPU time of frames 30 to 199 (each frame committed
and waited for, its command buffer's GPU start and end), three rounds per
variant, the variants run one after another. Each variant was a working
build of the renderer; the throwaway ones were scratch code, not committed.

Image quality: `graphs/preview.toml` (deterministic) at 1600 x 1200 on a
close-up of a glass sphere of radius 0.75 over a checkerboard, two fireflies,
each tessellation against the exact sphere, compared in 8-bit display
values.

## Results

| Structure | Frame |
|---|---|
| One level of world-space boxes, exact spheres tested in world space (`main` at bad9bc9) | 7.8 ms |
| One level, exact spheres placed by transforms, tested in object space (adopted) | 8.2 ms |
| One level, the same, before the object-space ray was computed once per candidate | 9.2 ms |
| Every shape's box in one bottom level, under one identity instance | 14.3 ms |
| Triangle spheres (icosphere, 1,280 / 5,120 / 20,480 / 81,920 triangles), one instance per shape, opaque, intersected wholly in hardware | 11.2 / 11.5 / 12.0 / 12.5 ms |
| Exact spheres, a one-box bottom level per geometry, one instance per shape | 18.1 ms |

Ruled out as the cost of the instanced exact spheres: the ray-sphere
arithmetic (the old form measured the same), Metal's per-candidate
object-space ray (`get_candidate_ray_origin`; computing it from the
transform instead was slower), and `max_levels<2>`, which is already the
default for `instancing`.

Moving, adopted structure: the wandering brass scene (two fireflies) takes
8.28 ms a frame against the still scene's 8.17, placing the shapes and
rebuilding the structure together 0.11 ms. A build alone, in a command buffer
of its own: 68 us over 4 shapes, 95 us over 64, 161 us over 1024, 340 us
over 4096.

Tessellated glass against the exact sphere, pixels differing by more than
8/255: 7.9% (L3), 4.0% (L4), 1.9% (L5), 0.9% (L6); the silhouette off by at
most 1.2, 0.3, 0.08 and 0.02 pixels. From L4 on the differences are checker
edges shifting in already-aliased regions; no facet is visible. Triangle
lights shadowed themselves until shadow rays stopped short of the light's
exact surface: a shadow ray starts a hair off the surface it leaves, so it
reaches past the exact surface into the light's tessellation.

## Conclusions

- **Fact:** on this GPU, for boxes whose hits the shader decides, the
  instance level costs more per ray than the whole frame's build of a single
  level saves: 14.3 ms with one instance against 9.2 without, and 18.1 ms
  with one per shape.
- **Fact:** triangle geometry intersected wholly in hardware takes away most
  of that (11.5 ms at 5,120 triangles a sphere, no visible facets), but not
  all: still 3.3 ms over the single level.
- **Inference:** the instance level's cost is in returning each candidate to
  the shader through it; Apple does not document its traversal, so this is
  not confirmed.
- **Inference:** Unreal's design keeps bottom levels because a mesh is
  thousands of triangles worth building once. A shape here is one box; its
  bottom level would hold nothing worth keeping, and a single level of the
  shapes' placed boxes is Unreal's top level with each instance's transform
  folded into its box, rebuilt each frame as Unreal rebuilds its top.

Adopted: one level, exact spheres, rebuilt each frame when shapes move. A
shape that is a mesh, when one comes, is the case for an instance, and is
measured then.
