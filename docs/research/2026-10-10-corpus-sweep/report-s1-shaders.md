## Shader review: src/metal (2,822 lines) and the shared src/core headers, branch marbles

I changed nothing in the repo. To check two findings I compiled `path.metal` to LLVM IR (`-std=metal4.0`, default fast math), with the output going only to the scratchpad. Rules were looked up in the perf corpus on localhost:7015 (GPU.3, GPU.4, GDSA.3, GDSA.13) and in the cpp-guidelines MCP (ES.20). Both servers answered.

### Findings

**1. High (undefined behaviour, not yet showing as a wrong image): `INFINITY` is passed as `t_max` while every shader compiles with fast math.**
- **Where:**
  - `src/metal/integrator/path.metal.h:156`
  - `src/metal/integrator/direct.metal.h:98` and `:227`
  - also `tests/gpu/kernels/trace_probe.metal:35`
- **Rule:** correctness (LLVM's fast-math semantics), and ES.43's point that code which only works under one optimizer setting is not portable.
- **What is wrong:** `cmake/MetalLibrary.cmake` keeps Metal's default fast math. The IR's function attributes say `"no-infs-fp-math"="true"`, and every comparison is `fcmp fast`. `trace()` is called with `float 0x7FF0000000000000` (+inf), and `intersect_sphere` / `intersect_box` compare against it (`fcmp fast olt float %t, %t_max`). Under `ninf`, an infinite operand makes the result poison, so the backend may legally fold `near < t_max` to anything. `box.metal.h:23-25` already says fast math "does not promise to carry" infinities, but the callers still pass one.
- **Fix:** pass a finite far bound, e.g. `metal::numeric_limits<float>::max()` or a named scene constant, at all four call sites.

**2. Medium (performance on the hot path): the path loop keeps shading state alive across the shadow ray's query.**
- **Where:** `src/metal/integrator/path.metal.h:193-215`
- **Rule:** GPU.3 (occupancy is limited by live registers, so cut the state, don't chase a number), together with the counters note (`2026-10-10-path-kernel-counters.md`) and logical-overview's "Little is kept alive across a ray query".
- **What is wrong:** in the AIR, the `occluded()` call (IR line 550 of `radiance`) is followed by:
  - the |cos| multiply;
  - the medium-record load and `fast_exp` for transmittance;
  - `bsdf_sample()` (IR line 675).

  So the resolved `Bsdf` (kind, colour, alpha, ior, escape, normal), `wo`, `f`, `sample.radiance`, `chosen.probability`, `sample.pdf`, `medium`, `surface_at.flags` and `surface_at.interior` all stay in registers across the second query. The counters blame exactly this kind of state for 25% occupancy.
- **Fix:**
  - (a) Compute the unoccluded contribution `beta*f*|cos|*Le*T/(P*pdf)` before calling `occluded()`, then add it only if the ray is not blocked. One float3 then crosses the query.
  - (b) Do step 6 (`bsdf_sample` and the medium update) before the shadow query, so the `Bsdf` and `wo` die before it.

  Neither change biases the estimate; only the order of the drawn numbers changes. Measure against the same counters.

**3. Low (variance only, no bias): Russian roulette uses the radiance-scaled throughput.**
- **Where:** `src/metal/integrator/path.metal.h:219`
- **Rule:** correctness of the estimator.
- **What is wrong:** `beta` includes the 1/eta_t² factor from `dielectric.metal.h:90`. Inside glass of ior 1.5, `q` drops by about 0.44 even though no energy was lost, so paths inside marbles past the 4th surface are killed about twice as often as they should be. The estimate stays unbiased (`beta /= q`), but variance rises exactly in the marbles. pbrt-v4's PathIntegrator avoids this by tracking `etaScale` and testing `beta*etaScale`.
- **Fix:** keep a scalar `eta_scale`, multiplied by eta_t² on each transmission, and take `q` from `max(beta)*eta_scale`. It needs an eta_t from `BsdfSample` or a lobe-derived value.

**4. Low (dead code with a duplicate formula): two shader functions are never called.**
- **Where:** `src/metal/materials/conductor.metal.h:49-63` (`conductor_reflectance`) and `src/metal/shapes/shapes.metal.h:73-76` (`shape_normal`).
- **Rule:** no specific rule fits; this is the brief's "misleading docs" case.
- **What is wrong:**
  - Neither function is called anywhere in `src/` or `tests/`.
  - `conductor_reflectance` is a second copy of the GGX BRDF in a different form (BRDF times cosine, taking `ConductorData`, Schlick on `dot(v,h)`). Its comment presents it as part of the kind's interface, and it can drift from `conductor_evaluate`, the one actually used.
- **Fix:** delete both.

**5. Low: a struct is returned with fields left unset.**
- **Where:** `src/metal/integrator/direct.metal.h:99-105` (`Reached`).
- **Rule:** ES.20 (always initialize an object).
- **What is wrong:** `r.point` and `r.surface` are left indeterminate when `!hit.found`, and the struct is returned by value. No caller reads them today, but copying indeterminate floats is formally undefined in C++.
- **Fix:** zero-initialize, e.g. `Reached r{}`.

### Checked and fine
- **Non-finite guard:** `finite()` / `isfinite` survives fast math. The AIR lowers it to an exponent-bit integer test (`and 0x7f800000; icmp ne`), so the guard in `accumulate`/`count_non_finite` is not folded away. `simd_sum` is reached by every thread (`path.metal:34-58`).
- **Conductor (VNDF):** the pdf G1·D/(4 n·wo) matches Heitz 2018. The weight comes out to F·G2/G1. `sample_visible_normal` matches the reference, with `max(0, nh.z)` and the `lensq` guard. alpha has a floor of 1e-3, so D stays finite.
- **Dielectric:** reflection weight 1 and refraction weight 1/eta_t², with the side handled correctly (pbrt-v4 radiance mode). Total internal reflection gives pdf 1. `fresnel_reflectance` returns cos_t=0 under TIR, and its denominators are non-zero for ior > 1.
- **Coated:** the base f (1-F_i)(1-F_o)ρ/(π·ior²·((1-ρ)+ρ·escape)) matches Mitsuba's nonlinear plastic. The sampling probability F(cos_o) agrees with `coated_pdf`. The coat's delta weight is 1. Next event estimation (NEE) evaluates only the base and emission is counted after the coat, so light is not counted twice.
- **Lambert:** sample, pdf and evaluate agree. `cosine_direction` clamps its sqrt argument and `basis()` (Duff 2017) is safe at n.z = -1.
- **Sphere light:** `one_minus_cos = sin²/(1+cos)` avoids cancellation, `distance_to_light` uses the cross-product form, sampling is uniform over the cone with pdf 1/Ω, and a point inside the light returns pdf 0.
- **Sphere intersection:** the stable quadratic (Ray Tracing Gems ch. 7) is correct for the half-b form. The box slabs avoid 0·inf via the 1e-20 guard.
- **Light selection:** uniform selection clamps the index, and `count>0` is checked before calling it. NEE's light probability and pdf match `sphere_light_pdf`.
- **Russian roulette:** q = 0 when beta = 0 ends the path with no division. Survivors are divided by q, so the estimate stays unbiased.
- **Medium:** transmittance is measured from the true surface point, not the offset origin. NEE uses the path's medium, which is correct because every non-delta lobe here reflects (wi on wo's side). The shadow ray's `t_max` from the un-offset point is harmless because the light's own shape is ignored.
- **Self-intersection:** an offset of 1e-4 along the geometric normal is about 200 ulps at 5 m and 0.007 in a 3 cm firefly's object space.
- **Shared structs:** all of them are fixed by `static_assert`. `resolve_bsdf`, `sphere_sample_light` and every `*_sample` set every field on every path, so `Bsdf`, `BsdfSample` and `LightSample` are fully initialized.
- **Random numbers (GDSA.3):** addressed by pixel, frame and draw count, with no shared state. Pixels sharing a 32-bit key: 2.8k–17.5k per frame in four frames I computed, against about 6.9k expected by chance at 32 bits. Within one pixel, the frames' numbers cannot repeat (pcg_hash is a bijection).
- **Tone map:** PBR Neutral matches Khronos' reference; `tone_map_up` reads and writes in place on its own pixel only.
- **Branching (GPU.4):** the kind dispatch switches have no default (enforced by `-Werror`).

Scratch files (IR dump and probe scripts) are in `(scratch)/fm/`.