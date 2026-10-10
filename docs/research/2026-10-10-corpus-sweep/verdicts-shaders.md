# Verdicts: shaders (src/metal/**/*.metal, *.metal.h; tests/gpu/kernels)

Branch `worktree-agent-a8451e2ba2eca58b4`, from `marbles` at c0bc7d1. Not
pushed or merged.

| Commit | Subject |
|---|---|
| 9dfb07b | Fix the shaders' correctness findings: a finite far bound, no grazing sample, every field set |
| 519441d | Track the eta scale in the path tracer's roulette, so paths inside glass are not ended twice as often |
| 8282071 | Remove the shaders' dead functions and unused includes, and include what each names |
| 9e5dcc7 | Draw a path's numbers from a 64-bit stream, so no two pixels share them |
| 550e2c4 | Reshape the shaders' interfaces: a ray, a shading point and returned results in place of loose arguments |
| 2137d2f | Name the shaders' remaining constants, one declaration a line, one qualification form |
| 9082306 | Name transmittance()'s medium index `which`, apart from `media` |

## Measurements

The path tracer (graphs/path.toml) on scenes/marbles.toml at 3456 x 2234,
200 frames back to back with frames in flight, each frame's GPU start to
end from Metal 4's commit feedback, median of frames 21-199. One binary per
commit (the metallib is embedded), all six alternated in one session, three
rounds; no serenity window, and each run waited until no other agent's
serenity GPU process was running (a throwaway TIMING case appended to
tests/gpu/frame_images_test.cpp, removed before every commit).

| Build | Median frame (3 rounds) | Change |
|---|---|---|
| c0bc7d1, marbles head | 27.51, 27.51, 27.55 ms | |
| 9dfb07b, correctness | 28.02, 27.87, 27.85 | +0.35 |
| 519441d, eta scale | 29.37, 29.21, 29.30 | +1.40 |
| 9e5dcc7, 64-bit streams (with 8282071) | 28.81, 28.78, 28.77 | -0.50 |
| 550e2c4, interfaces | 28.49, 28.49, 28.55 | -0.30 |
| 2137d2f, naming | 28.51, 28.54, 28.51 | 0 |

Net: 27.5 ms before, 28.5 ms after (+1.0 ms, 3.6%). The eta scale's cost is
the paths inside glass living as long as their energy warrants, which is
the fix: more bounces per path inside the marbles, less noise there; the
estimate's mean is unchanged. An attempt to measure the noise directly (MSE
of 16 and 64 frames against 1024) was dominated by the scene's heavy-tailed
caustic samples and did not decrease with frames, so no variance figure is
claimed.

Spikes, measured the same way and not committed:

- GDSA.6, the path pass's second write (`radiance` beside `accumulated`)
  removed: 28.75 ms -> 28.46 ms (-0.30 ms, 1.0%), three rounds each
  (28.76/28.74/28.77 against 28.46/28.46/28.45).
- P.9, the sphere's inner normalize and the swirl's cos/sin of atan2
  replaced: 28.26-28.61 against 28.35-29.84 under a busier GPU, minimums
  27.89-27.91 against 27.78-27.88: no difference above the noise.

## Table

| report | file:line | rule ID | verdict | what was done or why rejected | commit |
|---|---|---|---|---|---|
| s1 #1 | path.metal.h:156; direct.metal.h:98, :227; tests/gpu/kernels/trace_probe.metal:35 | correctness (fast-math no-infs), ES.43 | fix | Rays end at `unbounded`, metal::numeric_limits<float>::max() (acceleration/trace.metal.h), not INFINITY. | 9dfb07b |
| s1 #2 | path.metal.h:193-215 | GPU.3 | reject | Deferred by the policy: the path's state across the shadow query stays as it is. The later refactor (next_event()) keeps the order of draws and traces exactly. | - |
| s1 #3 | path.metal.h:219 | correctness (estimator variance) | fix | eta_scale per pbrt-v4's PathIntegrator, from bsdf_eta() (bsdf.metal.h; dielectric.metal.h's transmitted_eta()); roulette tests beta x eta_scale. New GPU test (bsdf_eta, weight x eta_t^2 = 1). +1.4 ms, see Measurements. | 519441d |
| s1 #4 | conductor.metal.h:49-63; shapes.metal.h:73-76 | (misleading interface) | fix | conductor_reflectance() and shape_normal() deleted. | 8282071 |
| s1 #5 | direct.metal.h:99-105 | ES.20 | fix | Reached value-initialized; later reach() returns Reached{} where nothing is hit. | 9dfb07b, 550e2c4 |
| s2 | direct.metal.h:182-183 | I.1, I.5 | cross-area | The u = (0, 0) is now the named constant `light_middle`, its comment saying contract 3 does not define what u means for a kind. Stating it (or adding an emitter query for a representative direction) is core/contracts/emitter.h's. | 550e2c4 |
| s2 | path.metal.h:125,136; direct.metal.h:60,110 | ES.3, F.10, NL.7 | fix | One `ray_offset` and one leave() in integrator/surface.metal.h, used by both estimators. | 550e2c4 |
| s2 | direct.metal.h:73-77 | Enum.3 | fix | `constant constexpr uint` purposes and light_purpose(j). | 550e2c4 |
| s2 | box.metal.h:26 | ES.45 | fix | `least_component`. | 550e2c4 |
| s2 | resolve.metal.h:68 | ES.45 | fix | `least_alpha` (the resolve rule is the shader's; core names no alpha floor). | 2137d2f |
| s2 | noise.metal.h:30 `12u` | ES.45 | fix | `gradient_count`, the table's own extent. | 9e5dcc7 |
| s2 | wood.metal.h:24 `>> 8u`, `16777216.0f` | ES.45, NL.11 | fix | unit_float3() (math/hash.metal.h), 2^-24 as 0x1p-24f. | 9e5dcc7 |
| s2 | wood.metal.h:32 `3u`, :39 `seed + 1u`; swirl.metal.h:25 `2u` | ES.45 | cross-area | Tuning numbers core/textures/wood.h and swirl.h own (their step texts spell them); the core should name them. | - |
| s2 | tone_map.metal:55 `0.08f`, `6.25f`, `0.04f` | ES.45 | cross-area | PBR Neutral's F90 and its derived values: core/passes/tone_map.h should name them beside neutral_start. | - |
| s2 | direct.metal.h:239,285,286 `0.5f, 0.5f` | ES.45 | fix | `any_direction`. | 550e2c4 |
| s2 | sphere_light.metal.h:114 `float3(0,1,0)` | ES.45 | fix | No sample is LightSample{}. | 550e2c4 |
| s2 | path.metal:49 `float2(0.5f)` | ES.45 | fix | has_lens(); a pinhole's lens point is unread, 0, said so. | 550e2c4 |
| s2 | preview.metal:57 `i + 2` | ES.45, ES.100 | fix | `lens_shift = 2u`. | 550e2c4 |
| s2 | test_pattern.metal:18 `4.0f` | ES.45 | fix | `pulse_seconds`. | 2137d2f |
| s2 | srgb.metal.h:15 | ES.45 | fix | IEC 61966-2-1 constants named. | 2137d2f |
| s2 | sampler.metal.h:35-37; noise.metal.h:16 | ES.45 | fix | PCG and LCG constants named in math/hash.metal.h. | 9e5dcc7 |
| s2 | direct.metal.h:227 `~0u` | ES.45 | fix | serenity::contracts::no_primitive. | 550e2c4 |
| s2 | direct.metal.h:70; sampler.metal.h:41,52; wood.metal.h:24 | NL.11 | fix | 0x1.fffffep-1f, 0x1p-24f, R2's steps to a float's 9 digits (verified to round to the same floats). | 550e2c4, 9e5dcc7 |
| s2 | out-parameter locals and field-by-field structs (coated, dielectric, conductor, warp, sphere_light, shapes, path, direct, trace, rough, bsdf, resolve, path.metal, preview.metal) | ES.20, ES.22 | fix | Results returned whole (aggregate initialization), `T x{}` where fields follow; resolve sets kind before its switch (a record of no known kind resolves to none); DielectricData temp removed. | 9dfb07b, 550e2c4 |
| s2 | fresnel.metal.h:21; warp.metal.h:38; transform.metal.h:49; shapes:41, box:21, sphere:30; emitter.metal.h:45 | F.20, F.21 | fix | Fresnel{reflectance, cos_t, total_internal}, Tangents, ray, Crossing{found, t}, ShapeLight{is_light, light}. | 550e2c4 |
| s2 | Bsdf, SurfaceInteraction, Reached, Transform, SphereLight, LightView by value | F.16 | reject | F.16 is a cost rule ("a stricter enforcement would depend on the performance characteristics of the target architecture"): every shader function is inline, so a by-value parameter is no copy. In MSL a reference names its address space; `thread const T&` cannot bind the `constant` records (Transform, LightRecord) without a copy into thread memory, which is what by value already says. | - |
| s2 | direct.metal.h:121, :206, :157; trace.metal.h:34, :66; shapes.metal.h:41; box.metal.h:21 | I.23 | fix | metal::raytracing::ray as the bundle; Shading{surface, bsdf, wo, medium}; from_lights 4, shade_reflected 4, shade_scattering 3, trace 3, occluded 4, intersect_shape 3, intersect_box 2. (conductor_f, a helper of the conductor's own, takes 6.) | 550e2c4 |
| s2 | direct.metal.h:157, :122; leave(); thin_lens.metal.h:28 | I.24 | fix | leave(SurfaceInteraction, direction); shade_reflected(Shading, direction); camera_ray(..., uint2 size). | 550e2c4 |
| s2 | direct.metal.h:292, :297-315; dielectric.metal.h:62, :87; transform.metal.h:29,54-55; no-sample builds | ES.3 | fix | `entering` used; one continuation; Fresnel's total_internal flag; transmitted_eta(); transform_column()/transform_transposed(); BsdfSample{} and LightSample{} as the one no-sample. | 519441d, 550e2c4 |
| s2 | transform.metal.h:42-43 | P.9 | fix | 1 / s^2 from the squared column length, no sqrt. Measured with 550e2c4: -0.30 ms. | 550e2c4 |
| s2 | coated.metal.h:55,67,68 | P.9 | fix | F(cos theta_o) once per sample (coated_base()). Measured with 550e2c4. | 550e2c4 |
| s2 | conductor.metal.h:118-125 | P.9 | fix | The sample reuses the half vector it drew and its D (conductor_f/conductor_density). Measured with 550e2c4. | 550e2c4 |
| s2 | conductor.metal.h:45 | P.9 | fix | (1 - cos)^5 as products, not pow. Measured with 550e2c4. | 550e2c4 |
| s2 | direct.metal.h:166 | P.9 | fix | The light record is read only where see_glow (and in the path tracer only where emission counts). Measured with 550e2c4. | 550e2c4 |
| s2 | shapes.metal.h:64+70 | P.9 | reject | Spiked and measured: no difference above the noise (see Measurements); Per.6. The unit normal from sphere_normal() is the clearer contract. | - |
| s2 | swirl.metal.h:20-23 | P.9 | reject | Spiked and measured with the above: no difference; Per.6. atan2 then cos/sin is core/textures/swirl.h's step 2 as written. | - |
| s2 | direct.metal.h:217 | P.9 | reject | The unused component of sample_2d() costs one fract the compiler may drop after inlining; an x-only form would copy sample_2d()'s formula (ES.3). Unmeasured, Per.6. | - |
| s2 | thin_lens.metal.h:41; tone_map.metal:83,126 | P.9, Per.11 | reject | Two normalizes per camera ray (one per pixel per frame) and one exp2 per pixel of the tone map, against paths of several ray queries; precomputing them needs CameraData/ToneMap contract changes (core). Unmeasured, Per.6/GPU.10. | - |
| s2 | noise.metal.h:50-51; tone_map.metal:34-37 | ES.10, NL.21 | fix | One declaration a line. | 2137d2f |
| s2 | tone_map.metal:34-37; conductor.metal.h:39,51-52; sphere.metal.h:43; direct.metal.h:85-87, :297 | NL.19, ES.8 | fix | Taps named by compass point; masked_l/masked_v; root_c/root_a; PixelSamples{xy, position}; the passed direction `d`. | 550e2c4, 2137d2f |
| s2 | media.metal.h:28 | NL.19 | fix | The medium index is `which`, apart from the Media `media`. | 9082306 |
| s2 | wood/swirl `tx`, tone_map `tm`; coated:59, dielectric:59 reflect; thin_lens:36 vs path.metal:49; any/all; sampler `>> 8`; emitter.metal.h include order | NL.8, ES.1 | fix | Full serenity:: qualification everywhere (ToneMap's using-declaration stays: the kernel parameter lines use it); metal::reflect; has_lens(); any(f > 0) in both estimators; unit_float in one place; includes sorted. | 9e5dcc7, 550e2c4, 2137d2f |
| s2 | bsdf_sample/bsdf_evaluate vs sample_light/light_emitted | NL.8 | reject | The names are contract 2's and 3's (core/contracts/bsdf.h, emitter.h name sample_light, emitted, light_at); renaming the shader side alone would set it against the contracts' text. A rename belongs to the contracts first (core). | - |
| s2 | path.metal.h:173, :146; direct.metal.h:260; path.metal:45 | SF.10 | fix | core/lights/light.h, core/contracts/medium.h, sampler.metal.h, core/contracts/transform.h included where named. | 8282071 |
| s2 | path.metal:10-12, preview.metal:10-12; path.metal.h:103-104; sphere.metal.h:23; conductor.metal.h:19 | (include what you use) | fix | Removed. Every header still compiles on its own (SF.11). | 8282071 |
| s2 | dielectric.metal.h:53-56; path.metal.h:142, direct.metal.h:257 | ES.26 | fix | cos_signed and cos_i consts; the loop ray `along`, apart from the camera's ray. | 550e2c4 |
| s2 | wood.metal.h:29; sampler.metal.h:52; direct.metal.h:237; tone_map.metal:60 | Con.4, Con.5 | fix | const r; `r2_step`; ends[] gone; `neutral_span`. | 9e5dcc7, 550e2c4, 2137d2f |
| s2 | direct.metal.h:237-245 | ES.27, ES.71, ES.77 | fix | from_delta_lobe(): two explicit tries, no loop, no break. | 550e2c4 |
| s2 | noise.metal.h:37; wood.metal.h:23 | ES.46 | fix | The range that keeps each cast defined stated beside it, from core/textures/wood.h's bounds. | 2137d2f |
| s2 | test_pattern.metal:16-17; direct.metal.h:238; preview.metal:57 | ES.100, ES.102 | fix | 1u; the e < 2 loop gone; lens_shift is uint. | 550e2c4, 2137d2f |
| s2 | `uint` loop indices throughout | ES.100, ES.102, ES.107 | convention | uint loop indices in shaders. | - |
| s2 | path.metal.h:142-232; direct.metal.h:257-319, :206-249 | F.3 | fix | next_event() (path step 5), from_delta_lobe(), and one continuation in the preview's radiance(). | 550e2c4 |
| s2 | path.metal.h:154-155, 168-172, 226 | NL.1, NL.3 | fix | The in-code step comments are the step numbers and what the header does not say. | 550e2c4 |
| s2 | sphere_light.metal.h:132 | NL.1 | reject | "The same radiance from every point of it, every way" states the sphere's answer to contract 3's emitted(point, direction), which the code (ignoring both) does not say. | - |
| s2 | direct.metal.h:29-30; uniform_light.metal.h:23 | NL.2 | fix | Comments made true: shadow rays are dimmed by the surface's medium; count may be 0, select_light()'s precondition is >= 1. | 9dfb07b |
| s2 | coated.metal.h:61; dielectric.metal.h:81,90 | I.5, I.7 | fix | wo in the surface's plane has no sample (pdf 0). New GPU test, failing before the fix (1048576 non-finite samples for glass and for the coat). | 9dfb07b |
| s2 | sampler.metal.h:62-69 | GDSA.3 | fix | pcg3d(pixel key, frame, dimension): a 64-bit stream per (pixel, frame), a bijection. New GPU test: pixels (2627, 65) and (0, 818) of frame 0, which shared the old key, now differ. Every path-traced image changes; no test compares fixed bytes. Measured -0.50 ms. | 9e5dcc7 |
| s2 | passes/path/path.metal:55-56 | GDSA.6 | cross-area | Measured: dropping the second write saves 0.30 ms of 28.75 (1.0%). Worth taking: the graph's downstream passes would read the accumulated image as the frame's radiance (its rgb is the mean; display and tone map read rgb only). That is the frame graph's and the path pass's bindings (host); the shader then drops its radiance.write. | - |
| s2 | direct.metal.h:209-232, 279 vs 283 | GPU.4 | reject | Unmeasured (Per.6, GPU.10). The per-lobe sample counts are preview.h's estimator; the path tracer is the pass the frame budget is measured on. | - |
| s2 | all files (half precision) | GPU.3, GPU.10 | reject | Unmeasured; the counters (2026-10-10-path-kernel-counters.md) put ALU at 13% and the lever in live state across queries, which is deferred. Per.6. | - |
| s2 | emitter.metal.h:39-50 | CACHE.3, Per.13 | reject | Unmeasured, rated not a priority by the report itself; the lights' tables are in the constant address space (2026-10-10-scene-block.md). Per.6. | - |
| s2 | every *.metal.h:1 | SF.8 | convention | #pragma once. | - |
| s2 | scene_block.metal.h:52; preview.metal:28-29 | NL.17 | fix | Wrapped; the double blank line removed. | 550e2c4, 2137d2f |
| s2 (checked) | core/textures/wood.h:83, swirl.h:44 | ES.45 | cross-area | Their "every tuning number is a named constant" is false while the octave counts and the pores' seed offset are bare in the steps (see the cross-area list). | - |

## Cross-area

For the core agent (src/core):

1. core/contracts/emitter.h (contract 3): state what u = (0, 0) means, or add
   an emitter question for a representative direction toward a light; the
   preview's shade_reflected() relies on the sphere mapping u = (0, 0) to its
   middle (direct.metal.h, `light_middle`). (I.1, I.5)
2. core/textures/wood.h and swirl.h: name the fbm octave counts (wood 3,
   swirl 2) and the pores' seed offset (+1), so their shader halves can use
   the names; until then their "every tuning number is named (ES.45)" claim
   is false. (ES.45)
3. core/passes/tone_map.h: name PBR Neutral's F90 (0.04) and its derived
   0.08 and 6.25 beside neutral_start, for tone_map.metal's step 5. (ES.45)
4. core/contracts/bsdf.h (contract 2): BsdfSample could carry eta (pbrt-v4's
   BSDFSample::eta), so the roulette's eta_t comes with the sample instead of
   bsdf_eta() asking the Bsdf again; the struct has no spare field today (32
   bytes, static_assert). Optional; bsdf_eta() is correct as it is.
5. Optional, unmeasured: CameraData could carry unit right/up for the lens,
   ToneMap the exposure's scale, saving a normalize per camera ray and an
   exp2 per pixel. Not recommended without a measurement (Per.6).

For the host agent (frame graph, pass bindings, build):

6. GDSA.6: let the passes after the path pass read its accumulated image as
   the frame's radiance and drop the path pass's radiance output (measured
   0.30 ms a frame, 1.0%). The shader change (removing radiance.write and the
   texture(1) parameter) follows the binding change and is in the parameter
   lines the binding agent owns.
7. tests/gpu/CMakeLists.txt: the path-numbers probe lives in bsdf_probe.metal
   and its test in bsdf_test.cpp because a new sampler_probe.metal /
   sampler_test.cpp needs a line in the GPU tests' source lists; move them if
   a sampler test file is added.
8. AGENTS.md "Guideline deviations": the convention rows here rely on SF.8
   (#pragma once) and uint loop indices in shaders (ES.100/ES.102/ES.107)
   being recorded there.

## Counts

64 rows (s1's 5 findings, s2's rows split where one row names several
items with different verdicts): fix 46, convention 2, reject 11, cross-area
5.

Tests: `./build/native-release/tests/serenity_tests` passes (1522
assertions); `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1
./build/native-release/tests/gpu/serenity_gpu_tests` passes (78 cases, 3 of
them new: grazing wo, bsdf_eta, path-number streams); native-debug builds
and its core tests pass; every `*.metal.h` compiles on its own with
`-std=metal4.0 -Werror`.
