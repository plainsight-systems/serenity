# Shader review: src/metal/**/*.metal and *.metal.h (marbles branch, read-only)

**What was covered:** I read all 38 files in scope (2,822 lines). The first sweep's findings were left out, including the deferred path state across the shadow query. Both corpora answered:
- **Perf corpus (localhost:7015):** I listed all 6 categories.
- **C++ Core Guidelines MCP:** I listed all 10 categories.

**Compile check:** I compiled each of the 34 `*.metal.h` headers on its own in the scratchpad with `-std=metal4.0 -Werror -I src`. All 34 compiled, so SF.11 holds. Nothing in the repository was edited.

**Guideline IDs cited in shader comments.** There are three, and each says what the comment claims:

| Comment | Cites | Verdict |
|---|---|---|
| path.metal.h:63, divergence at glass/air SIMD groups | GPU.4 | Correct: per-lane divergence |
| swirl.metal.h:6, "No branch" | GPU.4 | Correct: fbm's loop runs a constant 2 times, the same in every lane |
| scene_block.metal.h:17, pointers held in registers across ray queries | GPU.3 | Correct: registers limit occupancy |

The core headers wood.h:83 and swirl.h:44 claim "Every tuning number is a named constant (ES.45)". That claim is false for the shader halves: see row 4.

## Findings

| file:line | rule ID | rule title | violation | fix | severity |
|---|---|---|---|---|---|
| direct.metal.h:182-183 (header :16) | I.1, I.5 | Make interfaces explicit / State preconditions | `sample_light(..., float2(0.0f))` is commented "toward the light's middle, for a sphere". This relies on how the sphere maps u, which contract 3 (core/contracts/emitter.h:18-22) does not state. The header claims the file "names no light kind". A second light kind would silently change the preview. | State in contract 3 what u=(0,0) means, or add an emitter query for a representative direction. | medium |
| path.metal.h:125,136; direct.metal.h:60,110 | ES.3, F.10, NL.7 | Don't repeat yourself / Name reusable operations / Name length matches scope | `offset = 1e-4f` and `leave()` are defined word for word in both integrators, comments included. A namespace-scope constant named only `offset` also collides in meaning with `sample_offset` and sampler's `offset` parameter. | Move both into one shared header and name the constant something like `ray_offset`. | low |
| direct.metal.h:73-77 | Enum.3 (also Enum.7) | Prefer class enums | `enum Purpose : uint` is a plain enum. `purpose_light + j` does arithmetic on it. | Use `constant constexpr uint` constants, or an enum class plus a `light_purpose(j)` helper. | low |
| box.metal.h:26 `1e-20f` twice; resolve.metal.h:68 `1e-3f`; noise.metal.h:30 `12u`; wood.metal.h:24 `>> 8u`, `16777216.0f`; wood.metal.h:32 `3u` octaves, :39 `seed + 1u`; swirl.metal.h:25 `2u` octaves; tone_map.metal:55 `0.08f`, `6.25f`, `0.04f`; direct.metal.h:239,285,286 `0.5f, 0.5f`; sphere_light.metal.h:114 `float3(0,1,0)`; path.metal:49 `float2(0.5f)`; preview.metal:57 `i + 2`; test_pattern.metal:18 `4.0f`; srgb.metal.h:15 sRGB constants; sampler.metal.h:35-37 and noise.metal.h:16 hash constants | ES.45 | Avoid magic constants | Literals without names:<br>- `12u` is the length of `noise_gradients[12]`.<br>- The octave counts and pore seed offset contradict wood.h/swirl.h's "every tuning number is named".<br>- tone_map's F90 = 0.04 is spelled out only in `neutral_start`'s comment, and its derived values 0.08 and 6.25 are bare.<br>- The minimum GGX alpha is named only in a comment. | Use named `constant constexpr` values, kept beside the core constants where the core header owns them. For `12u`, use the array's extent. | style |
| direct.metal.h:227 | ES.45 | Avoid magic constants | `~0u` means "ignore no shape", but core/contracts/emitter.h:68 already defines `no_primitive` for that. | Use `serenity::contracts::no_primitive`. | style |
| direct.metal.h:70; sampler.metal.h:41,52; wood.metal.h:24 | NL.11 | Make literals readable | `0.99999994f` is 1 - 2^-24 only because the decimal rounds that way. The R2 steps are written with 16-17 digits, more than a float holds. `16777216.0f` is 2^24. | Write `0x1.fffffep-1f`, `0x1p-24f` and hex-float R2 steps, or say in a comment that the value rounds. | style |
| Out-parameter locals: coated:22, dielectric:60 `float cos_t;`; conductor:68-69, warp:50-51, sphere_light:88-89 `float3 t, b;`; shapes:44-45 `float3 o, d;`; path.metal.h:173, direct.metal.h:165,270 `LightRecord glowing;`; warp:24-25 `float r, phi;`<br>Structs filled field by field: trace:56 `Hit`; dielectric:52 `Boundary`; coated:56, conductor:110, dielectric:77, rough:42, bsdf:77 `BsdfSample`; sphere_light:50,69,111; shapes:88; resolve:50 `Bsdf`; dielectric:72 `DielectricData glass` (padding indeterminate); path.metal:39, preview.metal:45 `Scene` | ES.20, ES.22 | Always initialize / don't declare before a value exists | Objects are declared with no value. In resolve.metal.h:50, `bsdf.kind` stays indeterminate if `record.kind` is not an enumerator; it is the only switch without a fallback value. | Aggregate-initialize (`T x{...}` or `T x = {}`), or return a struct (next row). | style (resolve kind: low) |
| fresnel.metal.h:21 (coated.metal.h:22 throws the result away); warp.metal.h:38 `basis`; transform.metal.h:49 `transform_ray_to_object`; shapes:41, box:21, sphere:30 `bool` + `thread float& t`; emitter.metal.h:45 `light_at` | F.20, F.21 | Prefer return values; return a struct for several outputs | Outputs travel through `thread T&` parameters. | Return `{F, cos_t}`, `{t, b}`, `Ray{o, d}`, `{bool found; float t}`, and `{bool is_light; LightRecord}`. | style |
| bsdf/coated/conductor/rough/dielectric: every `Bsdf` (48 B) parameter; resolve.metal.h:48 `SurfaceInteraction` (64 B); direct.metal.h:121,206 `Reached` (about 96 B); transform.metal.h:18-49 and sphere_light.metal.h:48 `Transform` (48 B); sphere_light.metal.h:67-137 `SphereLight`/`LightView` (48 B) | F.16 (and COPY.7) | Pass cheaply-copied types by value, others by reference to const | These are passed by value although they exceed the enforcement threshold of 4 x sizeof(void*) = 32 B. | Pass `thread const T&`. Every function is `inline`, so I expect no codegen change; this is unverified. | style |
| direct.metal.h:121 (7 params), :206 (7), :157 (6); trace.metal.h:34 (6), :66 (7); shapes.metal.h:41 (6); box.metal.h:21 (6) | I.23 | Keep the number of function arguments low | These functions take 6 or 7 arguments; the guideline asks for fewer than four. | Bundle them: `Ray{origin, direction, t_min, t_max}` and a shading point `{at, bsdf, wo, medium}`. | low |
| direct.metal.h:157 (`from, origin, direction`, three `float3`); :122 (`count, medium`, two `uint`); path:136 and direct:110 `leave(point, n, direction)`; thin_lens.metal.h:28 `(width, height)` | I.24 | Avoid swappable adjacent parameters | Adjacent parameters of the same type can be passed in either order. At direct.metal.h:231 the caller's local `from` (the offset origin) goes into the parameter named `origin`, and `at.point` goes into the one named `from`. | Use distinct types or a struct, and rename the caller's local. Pass `uint2 size` to `camera_ray`. | low |
| direct.metal.h:292; direct.metal.h:297-302 vs 308-315; dielectric.metal.h:62, :87; transform.metal.h:29, 54-55; bsdf.metal.h:77-82 vs conductor.metal.h:110-114 vs sphere_light.metal.h:114-119,124-128 | ES.3 | Don't repeat yourself | Repeats within files:<br>- direct.metal.h:292 recomputes `entering` from line 268.<br>- The ray-continue block is duplicated in both branches.<br>- dielectric.metal.h:62 recomputes the total-internal-reflection test that `fresnel_reflectance` already made, and :87 recomputes eta as `eta_t`.<br>- transform.metal.h repeats the first-column expression three times instead of using a helper.<br>- The "no sample" result is built three ways; emitter.metal.h:60 uses `LightSample{}`. | Use one variable, a continuation helper, a returned TIR flag, a `transform_column()` helper, and one value-initialized no-sample. | low |
| transform.metal.h:42-43; shapes.metal.h:64+70; coated.metal.h:55,67,68; conductor.metal.h:118-125; swirl.metal.h:20-23; conductor.metal.h:45; direct.metal.h:217; direct.metal.h:166; thin_lens.metal.h:41; tone_map.metal:83,126 | P.9, Per.11 | Don't waste time or space / move computation out of run time | Redundant work:<br>- `transform_to_object` takes a sqrt (`transform_scale`) and then squares it; :54 already does this correctly with `dot`.<br>- The sphere normal is normalized twice.<br>- `coated_sample` evaluates F(cos_o) three times.<br>- `conductor_sample` recomputes h and D in pdf and evaluate.<br>- swirl calls atan2 and then cos/sin of the result; `q.xz / |q.xz|` gives the same.<br>- `pow(x, 5.0f)` is log2+exp2 under fast math.<br>- `sample_2d(...).y` is computed and discarded.<br>- `light_at(...) && see_glow` does its loads before checking the cheap bool.<br>- `normalize(right/up)` runs per ray and `exp2(exposure)` per pixel. | Use `dot(col, col)`, drop the inner normalize, pass f_o down, use x²·x²·x, swap the operands, precompute on the CPU. Unmeasured: ALU is 13% (counters note), and the coated and conductor repeats are probably merged by the compiler after inlining (Per.6, GPU.10). | low |
| noise.metal.h:50-51; tone_map.metal:34-37 | ES.10, NL.21 | One name per declaration | Several names are declared per statement. | Declare one per line. | style |
| tone_map.metal:34-37 (`l`, `m`, and `i` as a float3); conductor.metal.h:39, :51-52 (`l`); sphere.metal.h:43 (`one`); media.metal.h:28 (`media` beside `medium`); direct.metal.h:85-87 (`Pixel` with member `pixel`, so `px.pixel`); direct.metal.h:297 (`t` as a direction) | NL.19, ES.8 | Avoid names easily misread / similar-looking names | `l` reads as 1, and `one` reads as 1.0f. `t` is a hit distance everywhere else. | Rename: `wi`/`light`, `root0`/`root1`, `media_view`, `PixelSamples{xy, position}`, `refracted`. | style |
| bsdf_sample/bsdf_evaluate vs sample_light/light_emitted; wood/swirl `tx`, tone_map `tm` plus `using ToneMap` vs full `serenity::contracts::`; coated:59, dielectric:59 vs conductor:119 `reflect`; thin_lens:36 `== 0` vs path.metal:49 `> 0`; path:194 `any(f > 0)` vs direct:138,189 `all(f == 0)`; sampler:41 `>> 8` vs `>> 8u`; emitter.metal.h:12-16 include order | NL.8, ES.1 | Use a consistent naming style | Inconsistencies:<br>- Function names mix noun_verb and verb_noun.<br>- Some files use namespace aliases, others full `serenity::contracts::` qualification inside `namespace serenity`.<br>- Reflection is hand-written in two files and `metal::reflect` in a third.<br>- The same lens and zero-f conditions are tested differently in different files.<br>- emitter.metal.h:12-16 is the only unsorted include block. | Pick one form for each. | style |
| path.metal.h:173 (`lights::LightRecord`); path.metal.h:146, direct.metal.h:260 (`contracts::no_medium`); path.metal:45 (`PathNumbers`) | SF.10 | Avoid dependencies on implicitly included names | These names arrive through emitter.metal.h, media.metal.h and path.metal.h; direct.metal.h:39 includes light.h directly. | Include core/lights/light.h, core/contracts/medium.h and sampler.metal.h. | style |
| path.metal:10-12, preview.metal:10-12 (core/lights/gradient_sky.h, light.h, sphere_light.h); path.metal.h:103-104; sphere.metal.h:23; conductor.metal.h:19 (only the dead `conductor_reflectance` uses it) | none (include what you use) | n/a | These includes are not used. | Remove them. | style |
| dielectric.metal.h:53-56; path.metal.h:142, direct.metal.h:257 | ES.26 | One variable, one purpose | `cos_i` is signed and then made absolute. The parameters `origin`/`direction` are reused as the loop's current ray. | Use two consts, and name the loop ray state. | style |
| wood.metal.h:29; sampler.metal.h:52; direct.metal.h:237; tone_map.metal:60 | Con.4, Con.5 | Use const / constexpr | `float r` changes once and could be const. The R2 step, `ends[]` and `d` are compile-time values. | Use a const expression, or namespace-scope `constant constexpr`. | style |
| direct.metal.h:237-245 | ES.27, ES.71, ES.77 | std::array / range-for / minimize break | A C array plus an index loop exists only to take the first match and break. | Two explicit tries, or a range-for over a `constexpr metal::array`. | style |
| noise.metal.h:37 `int3(cell)`; wood.metal.h:23 `int(b)` | ES.46 | Avoid narrowing conversions | float to int is undefined at or above 2^31. The range is stated only in core wood.h:69. | Clamp, or state the precondition beside the cast. | low |
| test_pattern.metal:16-17 `width > 1`; direct.metal.h:238 `e < 2`; preview.metal:57; `uint` loop indices throughout | ES.100, ES.102, ES.107 | Don't mix signed and unsigned / signed for arithmetic | Unsigned values are compared with or added to int literals. (Loop indices are `uint`, the GPU idiom.) | Use `1u`, `2u`; keep `uint` indices as the stated idiom. | style |
| path.metal.h:142-232 (90 lines); direct.metal.h:257-319 (62 lines, 3 levels of nesting); direct.metal.h:206-249 | F.3 | Keep functions short and simple | These functions are long and nested. | Extract the step functions (NEE, the delta-surface continuation). | low |
| path.metal.h:154-155, 168-172, 226; sphere_light.metal.h:132 | NL.1, NL.3 | Don't restate code / keep comments crisp | The in-code step comments repeat the header's steps 1 and 3 nearly word for word. "Continue along wi" restates the code. | Keep only the step number and anything new. | style |
| direct.metal.h:29-30; uniform_light.metal.h:23 | NL.2 | State intent (comments must be true) | direct.metal.h:29-30 says shadow rays "cross air alone", but :119-120, :143 and :192 dim them by the surface's medium. uniform_light.metal.h:23 says `count` is "at least 1", but path.metal.h:186 guards `count > 0` because a scene with no lights has 0. That is `select_light`'s precondition, not the struct's. | Correct both comments. | low |
| coated.metal.h:61; dielectric.metal.h:81,90 | I.5, I.7 | State preconditions and postconditions | `f_o / cos_o` and `pdf / cos_o` give infinity at exact grazing (cos 0). The path then multiplies by \|cos wi\| = 0, which gives NaN; it is counted and dropped. | Return pdf 0 when the cosine is 0. | low |
| sampler.metal.h:62-69 | GDSA.3 | Derive random numbers from (seed, stream, counter) | The per-pixel stream key is a 32-bit hash of (pixel, frame). At 3456x2234, about (7.7e6)²/2³³ ≈ 6.9e3 pixel pairs per frame share a key, and so draw identical number sequences. GDSA.3 recommends at least 2^64 streams. | Use a 64-bit key, or Philox-style (key, counter). | low |
| passes/path/path.metal:55-56 | GDSA.6 | Count cost in passes over global memory | The mean is written twice each frame: to `accumulated` and to `radiance`, both RGBA32Float. That is about 123 MB of extra writes per frame at 3456x2234. | Let downstream passes read `accumulated`. This is a graph-interface decision and is unmeasured. | low |
| direct.metal.h:209-232, 279 vs 283 | GPU.4 | Keep SIMD-group control flow coherent | In the preview, glossy lanes run 4 `shade_reflected` traces, each with shadow rays, while diffuse lanes run 2 sky tests. Glass lanes and scattering lanes serialize. | Use an equal sample budget, or split the pass by lobe class. Unmeasured. | low |
| all files | GPU.3, GPU.10 | Occupancy as latency budget | No half-precision arithmetic is used anywhere. Safe candidates are texture color math (checker/wood/swirl, output in [0,1]) and display/tone-map color math; beta and L are HDR and not safe. Little gain is expected: ALU is 13%, and the lever is live state across queries, which is deferred. | Spike only with counters. | low |
| emitter.metal.h:39-50 | CACHE.3, Per.13 | Avoid pointer-chasing / redundant indirection | Next event estimation reaches a light through four dependent loads (shape_lights → records → spheres → transforms/glows). Measured buffer L1 residency is 8%. | Not a priority. Optionally cache the radius and center per light per frame. | low |
| every *.metal.h:1 | SF.8 | Use #include guards | `#pragma once` is used throughout. This is a project-wide convention with a single compiler. | One decision for the whole project. | style |
| scene_block.metal.h:52; preview.metal:28-29 | NL.17 | Layout | The 128-column line is the only one over 120 in scope. There is a double blank line in preview.metal. | Wrap the line and remove the blank. | style |

## Categories walked

**Perf corpus: 67 rules.**

- **gpu (10).** Violated: GPU.3/GPU.10 (half precision), GPU.4.
  - No violation: GPU.2 (texture access is 1:1 per pixel; scene gathers are inherent).
  - No violation: GPU.5 (no threadgroup memory is needed).
  - Divergent constant-space reads (a GPU.4 bullet) judged no violation: they were measured in 2026-10-10-scene-block.md.
  - Host-side or not applicable: GPU.1, 6, 7, 8, 9.
- **gpu-dsa (21).** Violated: GDSA.3, GDSA.6.
  - No violation: GDSA.2 (integer simd_sum and atomic are deterministic), GDSA.5 (simd_sum then one atomic per group).
  - Already covered by the research decision: GDSA.13 (wavefront split).
  - Not applicable: GDSA.1, 4, 7-12, 14-21.
- **cache-layout (8).** Violated: CACHE.3.
  - The layouts behind CACHE.2, 4, 5, 6 are core contracts, outside this scope.
  - Not applicable: CACHE.1, 7, 8.
- **codegen (8).**
  - No violation: GEN.1 (none used), GEN.2 (small branches in box_normal and the wood seam), GEN.6 (none used), GEN.8 (INFINITY under fast math is already known).
  - Not applicable: GEN.3, 4, 5, 7.
- **memory (11).** All not applicable: shaders do not allocate.
- **copy-move (9).** COPY.7 is folded into the F.16 row.
  - No violation: COPY.8.
  - Not applicable: COPY.1-6, 9.

**C++ Core Guidelines: 306 rules.**

- **P (13).** Violated: P.9.
  - No violation: P.1, 3, 4, 5, 6, 7, 10, 11, 12.
  - P.2 is inherent (Metal Shading Language, not ISO C++). Not applicable: P.8, 13.
- **I (20).** Violated: I.1, 5, 7, 23, 24.
  - No violation: I.2, 3, 4, 9 (SceneView's template parameter is documented in its comment; there are no concepts in C++14), I.11-13, I.22.
  - Not applicable: I.6, 8, 10, 25-27, 30.
- **F (40).** Violated: F.3, 10, 16, 20, 21.
  - No violation: F.1, 2, 4, 5, 8, 9 (`light_emitted` and `selection_probability` leave unused parameters unnamed), 11, 15, 17, 43, 49, 50, 52, 56.
  - Not applicable: F.6, 7, 18, 19, 22-27, 42, 44-48, 51, 53-55, 60.
- **C (100).** No violations.
  - Judged: C.1, 2, 4, 5, 7, 8, 131 (all structs are aggregates with no invariant).
  - The other 93 are not applicable: no constructors, destructors, virtuals, operators or unions.
- **Enum (8).** Violated: Enum.3, Enum.7.
  - No violation: Enum.1, 2, 4, 5, 6, 8.
- **ES (66).** Violated: ES.3, 8, 10, 20, 22, 26, 27, 45, 46, 71, 77, 100, 102, 107.
  - No violation: ES.1, 2, 5, 6, 7, 11, 12 (no shadowing found), 21, 25, 28, 40-44 (sampler sequences its draws correctly), 48, 55, 70, 72-76, 78, 79, 84-87, 101, 103-105.
  - ES.23, ES.49 and ES.64 judged idiomatic: Metal's `float3(...)` and `uint(...)` are its conversion syntax, and `as_type` is a named cast.
  - Not applicable: ES.9, 24, 30-34, 47, 50, 56, 60-63, 65.
- **Con (5).** Violated: Con.4, Con.5.
  - No violation: Con.1 (otherwise), Con.2, Con.3.
- **Per (18).** Violated: Per.11, Per.13.
  - Per.1-6: every perf row is marked unmeasured.
  - No violation: Per.7, 10, 12.
  - Per.16-19 concern layouts owned by core. Not applicable: Per.14, 15, 30.
- **SF (16).** Violated: SF.8, SF.10.
  - No violation: SF.1, 2, 3, 4, 7, 9 (no cycle seen), 11 (verified by compiling all 34 headers alone), 12, 13, 20, 21, 22.
  - SF.6 judged no violation: `.metal` files count as local scope per SF.6's note, and headers use `using namespace` only inside functions (trace.metal.h:36,68).
  - Not applicable: SF.5.
- **NL (20).** Violated: NL.1, 2, 3, 7, 8, 11, 17, 19, 21.
  - No violation: NL.4, 5, 9, 10, 15, 18, 20, 25, 26, 27.
  - Not applicable: NL.16.
