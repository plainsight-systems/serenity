# Where a ray leaves a surface: an offset sized to the point's rounding

*2026-10-11, from `restir-di` at 084f83c ("base"). Apple M3 Max, the
toolchain pinned in `cmake/toolchain.json`, release preset.*

## The change, and why

Rays left a surface from a point a fixed 0.1 mm off it
(`integrator/surface.metal.h`). Since both MIS strategies see a light from
that point (`2026-10-11-path-mis.md`, "One origin"), blocks beside the
fireflies resting on the table rendered 1 to 2% dark at t = 8.

Now: Wachter and Binder's `offset_ray()` (Ray Tracing Gems, 2019, ch. 6):
256 ulps of each coordinate per unit of normal, a fixed 1/65536 within 1/32
of the origin. 15 um on the marbles' table, 2 mm at 100 m. Minimum
distance stays 0.

Two parts follow from the method's assumptions:

- **Hit points put back on their shapes** (`shapes/shapes.metal.h`). The
  method assumes a point accurate to a few ulps of itself (Falcor's and
  RTXDI's comments: barycentric, "not from the ray t"). Serenity has no
  triangles; its spheres' and boxes' points were `origin + t d`, whose error
  grows with t. A sphere's point is now reprojected along its radius (as
  pbrt-v4 does), a box's snapped to its face plane. Without it the offset
  alone fails the test below.
- **Step 5's T over the shadow ray** from the leaving point (as pbrt-v4 and
  Falcor), not from the surface's point: the old closed form assumed an
  offset along the normal, and the exact length cost some 0.13 ms. Short
  by at most the step (0.2% at the ruby's 140/m), only for a light inside
  the surface's own medium, which no scene has. Step 1 stays exact.

## Production systems, checked 2026-10-11

| System | Constants (band, fixed, ulps) | Minimum distance |
|---|---|---|
| The chapter | 1/32, 1/65536, 256 | 0 |
| Falcor `computeRayOrigin` | 1/16, 3/65536, 768 | 0 |
| RTXDI sample path tracer `AdjustRayOrigin` | Falcor's | 10 um (a TODO says the offset alone does not suffice) |
| RTXDI sample visibility rays | none, from the surface point | 1 mm |
| pbrt-v4 `OffsetRayOrigin` | per-point error bound | 0 |

Chosen: the chapter's constants. Falcor's also pass the test, but the step
is the light's error here, and theirs is three times larger.

Not taken: pbrt-v4's per-point error bounds (tighter, but a float3 more in
contract 1, held in registers across the path's queries); tracing from the
surface point with a minimum distance (drops true hits nearer than it, such
as a firefly touching the table, and does not bound a grazing ray's
self-hit); a fixed or scene-scaled step (the 0.1 mm: too large by the
fireflies, under one ulp 10 km out).

## Tests

`tests/gpu/leave_test.cpp`: rays found as the path tracer finds them, then
16 leaving rays each, out and in, cosine 1 down to 3e-4. Failing rays of
3072 per case (4608 for the firefly):

| Case | Change | Zero step | No reprojection | Fixed 0.1 mm | Falcor's | 1/4 of chapter's |
|---|---|---|---|---|---|---|
| firefly 3 mm on the table | 0 | 2285 | 128 | 0 | 0 | 0 |
| marble 16 mm | 0 | 1517 | 264 | 0 | 0 | 0 |
| table top | 0 | 0 | 0 | 0 | 0 | 0 |
| ground 400 m | 0 | 0 | 72 | 0 | 0 | 0 |
| sphere 2 m at origin | 0 | 805 | 215 | 0 | 0 | 0 |
| box 1 m at origin | 0 | 0 | 323 | 0 | 0 | 0 |
| sphere 2 m 10 km off | 0 | 1536 | 0 | 1536 | 0 | 0 |
| box 1 m 10 km off | 0 | 0 | 0 | 0 | 0 | 0 |
| sphere 2 cm 170 m off | 0 | 1536 | 0 | 0 | 0 | 0 |
| box 1 cm 170 m off | 0 | 0 | 0 | 0 | 0 | 0 |
| sphere 20 m, top at origin | 0 | 986 | 32 | 0 | 0 | 16 |

Boxes pass with zero step only because reprojection puts the point exactly
on the plane; with zero step and no reprojection every case but the far box
fails. Domain limit, kept in the test as its control: a sphere 2 mm across
170 m off is as wide as its step and is entered short.
`shape_test.cpp` checks the reprojection directly.

The Lambert floor under a 3 mm firefly 2 cm up (`path_test.cpp`) now holds
against the closed form at the floor's own point, to 0.5%: −0.11 to +0.09%.
The 0.1 mm step fails it (+0.91% below the firefly); Falcor's gives +0.38%.
The glossy test, the weights probe and its 7751a65 control are unchanged in
outcome. The tinted-bead test moved to z = 5, where the step is 0.12 mm,
so measuring step 1 from the ray's start still fails it (+8%).

## Frame time

Harness as before (3456 x 2234, median of frames 21-199), base and change
alternated, six rounds; the desktop added noise, so the paired median Δ:

| Graph, t | Base | Change | Δ |
|---|---|---|---|
| path, 120 | 30.32 ms | 30.40 ms | +0.12 (noisy; +0.24 to +0.38 in quiet sessions) |
| path, 8 | 31.97 ms | 32.67 ms | +0.51 |
| RIS, 120 | 36.75 ms | 37.38 ms | +0.61 |
| RIS, 8 | 36.86 ms | 37.48 ms | +0.63 |

Most of it is `offset_ray()` itself (+0.3 ms alone); a float form of it was
no cheaper, and the exact T stretch added 0.13 (dropped).

## The means against the references

1024 samples from index 65,536, against the MIS references (7751a65):

| t | Build | Blocks median / p10 / p90 | MSE |
|---|---|---|---|
| 120 | base | 1.0002 / 0.9963 / 1.0046 | 0.002110 |
| 120 | change | 1.0000 / 0.9966 / 1.0058 | 0.002137 |
| 8 | base | 0.9986 / **0.9877** / 1.0045 | 0.003856 |
| 8 | change | 1.0004 / **0.9939** / 1.0139 | 0.004424 |

Against the pre-MIS reference the t = 8 p10 goes 0.9824 -> 0.9931.

The higher p90 and MSE at t = 8 hold over three more windows. They sit at
cat's-eye rims, glass seen at grazing angles. Mean over one rim by step
size: 0.1 mm (reference, base) 0.169; Falcor's 0.210; the chapter's 0.258;
a quarter of it 0.253. The rim converges by 15 um. Both references, rendered
with the 0.1 mm step, are about a third dark there. They are to be rendered
again.

## The references, to render again

From the repository root (a reference is never replaced, hence the move):

```sh
mkdir -p .cache/references/7751a65
mv .cache/references/marbles-t120-1920x1080-64x1024* .cache/references/marbles-t8-1920x1080-64x1024* \
   .cache/references/7751a65/
make reference SCENE=scenes/marbles.toml TIME=120
make reference SCENE=scenes/marbles.toml TIME=8
make convergence SCENE=scenes/marbles.toml TIME=120
make convergence SCENE=scenes/marbles.toml TIME=8
make convergence SCENE=scenes/marbles.toml TIME=120 GRAPH=graphs/restir_ris.toml
make convergence SCENE=scenes/marbles.toml TIME=8 GRAPH=graphs/restir_ris.toml
```

Each reference: 65,536 samples at 1920 x 1080, some 10 minutes. Each
convergence run: 16,384 samples, some 3 minutes.
