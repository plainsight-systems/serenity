# Rule-by-rule review of the host C++ in `src/metal/` (read-only, second sweep)

**Scope:** 46 files: every `.h`/`.cpp` under `src/metal/` except `*.metal` and `*.metal.h`. I read every file in full. `scene_block.h` is a `.h` file, so it is in scope. Nothing from `sweep1-findings.md` is repeated below.

**Corpora:** both were reachable. The C++ Core Guidelines server answered. The perf corpus answered at localhost:7015 through `perf.py`. The `cpp-perf-guidelines` MCP server itself failed to connect (ECONNREFUSED), so `perf.py` was the only route to that corpus.

**Findings at a glance:** no high-severity findings. Two are medium: the missing autorelease pool in `render_to_offscreen` and the scene-block entry check that does not check what its comment says. The rest are low or style.

## Violations

| file:line | rule ID | rule title | violation | fix | severity |
|---|---|---|---|---|---|
| frame/renderer.cpp:277-284 (with renderer.cpp:230) | P.8, R.1 | Don't leak any resources / RAII | `render_to_offscreen` has no autorelease pool, unlike `render_to_window` (renderer.cpp:265). `record()` creates an encoder through `computeCommandEncoder()` every frame. If, as with Metal's other non-`new` returns, that encoder is autoreleased (likely, not verified), it lands in whatever pool the caller has. `headless/main.cpp:59-73` has none, so memory grows with every frame. The GPU tests looping `render_to_offscreen` (path_test.cpp:63) have the same exposure. | Open `NS::AutoreleasePool` in `render_to_offscreen`, or in `Renderer::record`, and state the pool rule in renderer.h:239 | medium |
| scene/scene_buffers.cpp:24-27, 34-60; scene/scene_block.h:92 | P.5 (cited claim false) | Prefer compile-time checking | The comment says the assertion "counts" the entries. It does not. It compares `sizeof(SceneBlock)` with `array_count`. `std::array<Entry,19>` accepts fewer than 19 initializers silently: the missing ones get a null member pointer, so `block_.*entries[i].field` is UB. Duplicate fields also pass. The count `19` is written twice (scene_block.h:92 `19 * 8`, scene_buffers.cpp:34). | Use `constexpr std::array fields{&A::environment, ...}` with `static_assert(fields.size() == sizeof(A)/8)` and a constexpr all-distinct, non-null check. Derive the 19 from one place. | medium |
| frame/renderer.h:77-78 | NL.2 | State intent in comments | Says transforms bind at buffer 12 and glows at buffer 10. path.cpp:21-22, preview.cpp:21-22 and path.metal:26-27 bind glows at 4 and transforms at 5. | Correct the numbers, or point to the binding table instead | low |
| passes/path/path.h:42-44 | NL.2 | State intent in comments | "36 bytes of images" is followed by 16 + 16 + 16 = 48 | Say 48 | low |
| passes/test_pattern/test_pattern.h:35-36 | NL.2 | State intent in comments | "the frame's 16 bytes of constants": `FrameConstants` is 32 bytes (frame_constants.h:48) | Say 32 | low |
| device/submission.h:86-87 | NL.2 | State intent in comments | "resources are added once, at start-up, never per frame". Accumulation (accumulation.cpp:44) and FrameImages (frame_images.cpp:26) add and remove resources on every resize. | Say "at start-up and on a resize" | low |
| device/submission.h:92-94; submission.cpp:140-156 | Per.14, COPY.7, MEM.9 | Minimize allocations / hidden copies in std::function / allocate at init | The comment says "one small allocation a frame". Each commit actually allocates a `CommitOptions`, plus the lambda goes through `CommitFeedbackHandlerFunction` (std::function) → `__block` copy → Block_copy to the heap. That is at least three allocations and two `shared_ptr` refcount bumps per frame (metal-cpp MTL4CommandQueue.hpp:159-163). | Make one `CommitOptions` per slot at construction, with one handler that reads the sequence from `Feedback`, and reuse it. Otherwise state the real count. | low |
| frame/renderer.h:158; acceleration/scene_acceleration.h:83-84 | MEM.9 | Allocate at init, not in steady state | "Nothing is allocated" / "a frame allocates nothing" is asserted but never checked. MEM.9 asks for a checked invariant, and the commit path above already contradicts the frame-level claim. | Add a debug allocation gate or a test around `record()`/`begin()`/`commit()` | low |
| device/submission.cpp:85-92 | CP.40, Per.30, CONC.4 | Minimize context switching / avoid context switches on the critical path / choose the wait by expected time | `settle()`, which `begin()` calls every frame, polls `arrived` with 20 µs sleeps: a context switch per iteration on the frame path, and an unnamed interval | Have the handler signal a second `MTL::SharedEvent` (`setSignaledValue`) and use `waitUntilSignaledValue(seq+1, timeout)`, a blocking wait that still has a timeout | low |
| device/submission.cpp:142-153 | E.12 | Use noexcept when exiting by throw is impossible/unacceptable | The feedback lambda runs on Metal's queue inside an ObjC block and builds `std::string`s (`to_string`, `describe`). A `bad_alloc` there would unwind through Objective-C. | Mark the lambda `noexcept`, which makes such a throw a deterministic terminate | low |
| device/offscreen.cpp:21; frame_array.cpp:36; static_arrays.cpp:48; renderer.cpp:79; tone_map.cpp:46; non_finite.cpp:17; scene_buffers.cpp:86; scene_acceleration.cpp:86-88; presenter.cpp:23 | C.31 | All resources acquired must be released by the destructor | Each `make_resident`, and the layer's `addResidencySet`, is never undone in the owner's destructor. When an owner dies before the `Submission` (e.g. tests making several `Offscreen`s), the set keeps either a stale entry or, if it retains (the SDK header does not say), the memory. | Call `release_resident` in each owner's destructor; give Presenter a `removeResidencySet` | low |
| renderer.cpp:74-75; tone_map.cpp:41; non_finite.cpp:11-12; scene_buffers.cpp:81; plus each FrameArray (shape_transforms, light_glows, boxes) | GPU.9 | Suballocate GPU memory from heaps | Seven-plus separate tiny `MTLBuffer`s (8 B, 16 B, 152 B, 512 B, ...), each its own allocation and residency entry | Suballocate the small static ones from one buffer or heap (e.g. let StaticArrays or one ring own them) | low |
| device/submission.cpp:218-219, 226-227 | GPU.6 | Batch tiny GPU work | Every `make_resident`/`release_resident` commits the residency set. That is about 10 commits at start-up and about 16 per resize (7 textures out, 7 in, plus accumulation). | Add batch add/remove with one `commit()` | low |
| device/frame_array.cpp:40,45; film/non_finite.cpp:36-38 (vs 25-27); acceleration/scene_acceleration.cpp:128,137; frame/renderer.cpp:202 | I.5, I.6, SL.con.3 | State/check preconditions; avoid bounds errors | `slot < frames_in_flight` is checked in `NonFinite::begin_frame` but wraps silently (`% copies`) or is unchecked everywhere else (`NonFinite::address`, `arguments_[frame.slot]`) | Check once, the same way everywhere, or take a slot type that cannot be out of range | low |
| device/static_arrays.h:47; frame/frame_images.h:88 | I.5, E.14 | State preconditions / purpose-designed exception types | `.at()` throws `std::out_of_range`, while each class's documented failure type is `Error` | Check and throw `Error`, or document `out_of_range` | low |
| frame/renderer.cpp:230; acceleration/scene_acceleration.cpp:101 | ES.65, I.12 | Don't dereference an invalid pointer | The result of `computeCommandEncoder()` is used without a null check | Throw `Error` on null | low |
| device/submission.h:173 | C.9 | Minimize exposure of members | `queue()` hands out the raw `MTL4::CommandQueue*`, so any holder can commit or wait outside the begin/commit protocol. Its only use is `addResidencySet`. | Replace with `add_residency_set(MTL::ResidencySet*)` | low |
| device/presenter.cpp:17, 26-29 | C.41 | A constructor should create a fully initialized object | Built with a zero size, `resize()` returns early, leaving `size_` at {0,0} and the drawable unsized. The first frame then throws in `prepare`/`record`. | Reject a zero size in the constructor, or document and handle it in `render_to_window` | low |
| renderer.cpp:199,208; tone_map.cpp:45; scene_buffers.cpp:14-22,85; frame_array.cpp:34; static_arrays.cpp:45; shape_transforms.cpp:26; light_glows.cpp:25; scene_acceleration.cpp:11,25 | SL.con.4, COPY.6, LIFE.4 | memcpy only trivially-copyable types; rely on implicit object creation knowingly | `memcpy` and byte-view reinterpretation of `FrameConstants`, `CameraData`, `ToneMap`, scene records, `Transform` and `Bounds` rely on trivially copyable / implicit-lifetime types, and nothing asserts it. The contracts assert only `sizeof`. scene_acceleration.cpp:11 claims "byte for byte" but checks size only. | `static_assert(std::is_trivially_copyable_v<T>)` in `bytes<T>`, the casts and the contract headers; add `offsetof` checks for `Bounds` | low |
| device/submission.h:70 | CP.3 (miscited) | Minimize explicit sharing of writable data | Cited to justify sharing writable `Feedback` across threads. CP.3 argues for less sharing. The rule for what the code does is CP.32 (share ownership across threads with `shared_ptr`). | Cite R.20 and CP.32 | low |
| frame/frame_images.h:50-51 | Per.6 | Don't make performance claims without measurements | "the barrier's cost is measured at implementation", with no figure or research note | Record the number, or remove the claim | low |
| frame/frame_images.h:75; renderer.cpp:122; accumulation.h:62; shape_transforms.h:43-44; light_glows.h:39; frame_array.h:45-46 | I.4, I.24 | Precisely and strongly typed interfaces; avoid swappable adjacent parameters | `FrameImages(bool radiance, bool pyramid)` takes two adjacent bools (I.4's enforcement case), and `radiance` is always `true` at its one call site, so it is a dead parameter. Bool `moves`/`glows`/`scene_changes`, and `copies` as a `uint32_t` that must be 1 or 2. | Drop `radiance`; use enum classes (e.g. `Copies::one`/`per_frame`) | low |
| passes/*/(path,preview,display,test_pattern).cpp:10-31; tone_map.cpp:62-90 | ES.45, P.5 | Avoid magic constants | Argument-table indices 0..6 are literals, kept in step with the `.metal` signatures only by "Bindings match" comments. The stale renderer.h:77-78 comment shows the drift. | A shared binding-index header used by both C++ and Metal | low |
| passes/path/path.cpp:10-12; preview.cpp:10-12 | P.3 | Express intent | The check covers transforms, glows and the counter, but the message names only scene, camera and images, so it misleads diagnosis | Name every missing resource | low |
| library.cpp:16; submission.cpp:17; renderer.cpp:46; scene_acceleration.cpp:15 / library.cpp:23; submission.cpp:24 + inline pools at offscreen.cpp:11, frame_array.cpp:27, static_arrays.cpp:33, accumulation.cpp:31, frame_images.cpp:71, renderer.cpp:72,265, non_finite.cpp:10, scene_acceleration.cpp:55 / frame_array.cpp:14,20; static_arrays.cpp:13-17; renderer.cpp:22 / accumulation.cpp:32-39 vs frame_images.cpp:14-21 / path.cpp:29-31, preview.cpp:27-29, display.cpp:20-22, test_pattern.cpp:17-19, tone_map.cpp:15-20 / renderer.cpp:22-26,197-209 | ES.3, F.10 | Don't repeat yourself; name reusable operations | Duplicated: `describe()` (4 copies), autorelease-pool creation (2 helpers + 9 inline), the 256 alignment and round-up, the texture-descriptor build, the per-pixel dispatch shape (5 copies), and the constants ring re-implementing FrameArray's slot stride | One device-internal helper header (`describe`, `scoped_pool`, `align_up`, `make_private_texture`, `dispatch_per_pixel`); hold the ring in a FrameArray | style |
| device.h:50; library.h:53; offscreen.h:49; submission.h:173; accumulation.h:65; frame_images.h:87-88; frame_array.h:55; shape_transforms.h:54; light_glows.h:49; non_finite.h:63; scene_acceleration.h:114-115 | Con.2 | By default make member functions const | `const` members hand out mutable handles or spans, or perform writes (`SceneAcceleration::update` writes boxes and records GPU work), so `const` does not mean "no observable change" | Make the writers non-const; keep `const` for true observers | style |
| device.cpp:16-19; submission.cpp:39,91; presenter.cpp:15; offscreen.cpp:25; renderer.h:220 | ES.45 | Avoid magic constants | Family table plus loop bound `4`, capacity `16`, poll `20` µs, drawable count `3`, bytes-per-pixel `4`, sentinel `static_cast<size_t>(-1)` | Named constants; `std::optional<std::size_t>` for `first_cross_frame_` | style |
| device.cpp:16-18; submission.cpp:155; scene_acceleration.cpp:75 | SL.con.1, ES.45 | Prefer std::array over C arrays | C arrays with a literal count (`buffers, 1`, `list, 1`) | `std::array`, passing `.size()` | style |
| every header, line 1 | SF.8 | Use #include guards | `#pragma once`, which SF.8's note calls non-standard (P.2) | Include guards, or record the deviation | style |
| renderer.h:3-7; offscreen.cpp:18,25; static_arrays.cpp:36; non_finite.cpp:26; tone_map.cpp:70; scene_buffers.cpp:15; shape_transforms.cpp:23; scene_acceleration.cpp:22; scene_block.h:87; frame_resources.h:11 | SF.10 | Avoid implicitly #included names | `std::uint*_t`/`std::size_t` without `<cstdint>`/`<cstddef>`; `std::to_string` via error.h; `std::vector` via scene.h; `std::byte` via frame_array.h; unqualified `uint32_t` in the host branch; frame_resources.h includes scene_buffers.h and uses nothing from it | Include what is used; drop the unused include | style |
| accumulation.cpp:27; frame_images.cpp:61; scene_acceleration.cpp:110 / shape_transforms.cpp:26; light_glows.cpp:25; scene_acceleration.cpp:25 | ES.48 | Avoid casts | C-style `(void)` casts (ES.48 enforcement flags them; use `std::ignore =`); three `reinterpret_cast` byte views | `std::ignore =`; one typed-view helper on FrameArray | style |
| frame_images.cpp:55,71 and every `auto drained = pool` (above); submission.h:112,195 + submission.cpp:113,197; renderer.cpp:197; shape_transforms.cpp:9; scene_buffers.cpp:33; scene_acceleration.cpp:56 | ES.8, NL.19, ES.7 | Avoid similar/misleading names | `drained` (a live pool) next to `drained_queue`; `settled` means a sequence+1, an `optional<Completed>` and a vector; `slot` holds a byte offset beside `frame.slot`; `some()`; namespace-scope alias `A`; `mtl` | `pool`, `last_settled`, `slot_offset`, `nonempty()`, `Block` | style |
| renderer.h:189, renderer.cpp:169,271,280; non_finite.cpp:41; light_glows.h:39 | ES.12 | Do not reuse names in nested scopes | Parameter/local `frame` hides namespace `frame` (`frame::reads_radiance` still resolves, but reads ambiguously); local `count` in `count()`; parameter `glows` vs member `glows()` | Rename (`begun`, `total`, `glowing`) | style |
| frame/renderer.h:210-214 | NL.16 | Conventional member order | `struct Prepared` is declared in the middle of the data members | Move the types first | style |
| tone_map.h:41; renderer.h:80-81,96-97,132-134; preview.h:79-80; accumulation.h:66-67; scene_block.h:31; frame_resources.h:46-53 | NL.3, NL.4 | Keep comments crisp; consistent layout | Overlong line; ragged reflow left over from edits; double blank line; garbled sentence ("the block cost frames what bound arrays did not"); the scene-block comment duplicated inline | Reflow and reword | style |
| accumulation.cpp:13; renderer.cpp:201-205 | ES.22, ES.23 | Declare when you have a value / prefer {} init | `frame::Joined joined;` assigned inside a try; `FrameResources` filled field by field | Immediately invoked lambda; designated initializers | style |
| device.cpp:35-40; presenter.cpp:11; submission.cpp:33-58 | C.49 | Prefer initialization to assignment in constructors | Members assigned in the body | Init-list helpers (e.g. `make_queue(device)`) | style |
| tone_map.cpp:79; submission.cpp:198 | ES.40, ES.86 | Avoid complicated expressions | `for (k = size-1; k-- > 0;)`; ternary in the for-initializer | `std::views::reverse`/iota; a named `first` | style |
| device.cpp:10; submission.cpp:85,97; offscreen.cpp:31; passes' `MTL::Size(...)` | ES.64, ES.23 | Use T{e} notation | Functional-style construction | Brace init | style |
| device.h:50-52; offscreen.h:49-50; submission.h:162; frame_array.cpp:44; non_finite.cpp:36,40; scene_buffers.h:49; shape_transforms.h:57; light_glows.h:52; scene_acceleration.cpp:136; renderer.cpp:165 | F.6 | Declare noexcept if must not throw | Non-throwing observers are not `noexcept` | Add `noexcept` | style |
| frame_images.cpp:32; static_arrays.cpp:15 | F.4 | constexpr if may be compile-time | `half()` and `aligned()` are pure arithmetic | `constexpr` | style |
| passes' `record(MTL4::ComputeCommandEncoder*)`; renderer.h:189 (`target`); frame_images.cpp:12; submission.h:165,170 | F.23, I.12 | Use not_null for non-null pointers | Must-not-be-null pointers typed as plain `T*` | References or `not_null` | style |
| path.h:47; preview.h:85; display.h:35; test_pattern.h:41 | I.23 | Keep the number of arguments low | `const Device&` is accepted and never used | Remove the parameter | style |
| device/error.h:23 | C.52 | Use inheriting constructors | Hand-written forwarding constructor | `using std::runtime_error::runtime_error;` | style |
| device/library.cpp:35-39 | R.1, E.19 | RAII / final_action | `dispatch_data_t` released by hand | Scope guard or RAII wrapper | style |
| device/library.h:39 | SL.str.5 | Use std::byte for bytes | `span<const unsigned char>` for metallib bytes; the rest of the backend uses `std::byte` | `span<const std::byte>` (`as_bytes` at the call) | style |
| submission.cpp:15; submission.h:185-186 | I.4 | Strongly typed interfaces (units) | `uint64_t timeout_ms` mixed with chrono; `double gpu_start` where `Completed` uses `frame::Seconds` | `std::chrono::milliseconds`; `frame::Seconds` | style |
| device.cpp:16-18; renderer.h:218-219 | C.1 | Organize related data into structures | Parallel arrays (`families`/`numbers`, `kinds_`/`passes_`) | Array of pairs; a struct `{kind, pass}` | style |
| renderer.cpp:93-97; static_arrays.cpp:43 | ES.1, ES.71, ES.77 | Prefer std algorithms / range-for; minimize continue | Hand-written `find_if` with a compound loop condition; `continue` | `std::ranges::find_if`; invert the `if` | style |
| scene_acceleration.cpp:57-64,93 | P.9 | Don't waste time or space | A residency set is made and committed empty for moving scenes | Make it only when `!moves` | style |
| submission.cpp:100-123,73-98; renderer.cpp:171-187; passes' checks | GEN.7 | Mark error paths cold | String-building throw paths sit inline in per-frame functions | `[[gnu::cold]]` throw helpers (measure first) | style |
| static_arrays.cpp:41; renderer.cpp:243; tone_map.cpp:70 | ES.107 | Don't use unsigned subscripts | `std::size_t` loop indices | `std::ptrdiff_t`/ranges (project-wide convention) | style |

## Guideline IDs cited in code comments

| Citation | Verdict |
|---|---|
| CP.3 at submission.h:70 | Mismatch (row above) |
| P.5 at scene_buffers.cpp:27 | Rule fits, but the claim it backs is false (row above) |
| GPU.7 at frame_array.h:19, submission.h:24/27/53, renderer.h:119/141, scene_acceleration.h:73 | Match. "GPU.7's caveat" exists ("more frames in flight increase latency and memory"). |
| MEM.4 at frame_array.h:19 | Match: GPU.7 cross-references MEM.4 as double buffering |
| MEM.9 (5 sites) | Rule matches; the claims are unverified or overstated (rows above) |
| GPU.1, GPU.2, GPU.6, GPU.8, GPU.9 | Match |
| GDSA.6 (tone_map.h:48), GDSA.16 (frame_images.h:56) | Match |
| CDSA.30 (scene_acceleration.h:77), CDSA.32 (scene_block.h:22) | Match; both are in the perf corpus's cpu-dsa category |
| E.2, E.5, E.14, R.1, R.20, I.4, I.11, F.8, F.25, ES.30 | Match |
| CDSA.29 | Already known, not re-reported |

## Categories walked

**C++ Core Guidelines (14 categories, 428 rules):**

| Category | Rules | Reported | Judged no violation or not applicable |
|---|---|---|---|
| P | 13 | P.3, P.5, P.8, P.9 | P.1, P.2 (only the `#pragma once` note, under SF.8), P.4, P.6, P.7, P.10–P.13 |
| I | 20 | I.4, I.5, I.6, I.12, I.23, I.24 | I.1–I.3, I.7–I.11, I.13, I.22, I.25–I.27, I.30, I.9 |
| F | 40 | F.4, F.6, F.10, F.23 | All others, including F.15–F.21 (parameter passing is clean), F.9 (unused parameters are unnamed in the .cpp files), F.25, F.52–F.54 (captures are correct) |
| C | 100 | C.1, C.9, C.31, C.41, C.49, C.52 | C.2, C.8, C.20–C.22 (deleted copy/move with `= default` dtor, consistent), C.35, C.37, C.46–C.48, C.13 (beyond the known item); hierarchy, operator, union and container rules N/A |
| Enum | 8 | none | All 8; no enum is declared in scope |
| R | 25 | R.1, R.5 | R.2–R.4, R.6, R.10–R.15, R.20–R.24, R.30–R.37 |
| ES | 66 | ES.1, ES.3, ES.7, ES.8, ES.12, ES.22, ES.23, ES.40, ES.45, ES.48, ES.64, ES.65, ES.71, ES.77, ES.86, ES.107 | All others, including ES.20 (beyond the known item), ES.46 (the frame-index narrowing is documented in the contract), ES.47, ES.50, ES.70, ES.78, ES.79 |
| Per | 18 | Per.6, Per.14, Per.30 | All others |
| CP | 33 | CP.3 (citation), CP.40 | All others; CP.100 judged acceptable (one-producer, one-consumer per slot, release/acquire paired; see CONC.5) |
| E | 22 | E.12, E.19 | All others (E.15 catch by reference is correct) |
| Con | 5 | Con.2 | Con.1, Con.3–Con.5 |
| SF | 16 | SF.8, SF.10 | SF.1–SF.7, SF.9, SF.11–SF.13, SF.20–SF.22 |
| SL | 22 | SL.con.1, SL.con.3, SL.con.4, SL.str.5 | All others |
| NL | 20 | NL.2, NL.3, NL.4, NL.16, NL.19 | All others |

**Perf corpus (8 categories, 83 rules):**

| Category | Rules | Reported | Judged no violation or not applicable |
|---|---|---|---|
| memory | 11 | MEM.9 | MEM.1–MEM.8, MEM.10, MEM.11 (no custom allocators; heap use is start-up only) |
| copy-move | 9 | COPY.6, COPY.7 | COPY.1–COPY.5, COPY.8, COPY.9 |
| cache-layout | 8 | none | All 8; CACHE.1 judged negligible (feedback atomics are written once a frame) |
| lifetime | 8 | LIFE.4 (unasserted) | LIFE.1–LIFE.3, LIFE.5–LIFE.8 |
| concurrency | 8 | CONC.4 | CONC.1–CONC.3, CONC.5–CONC.8 (release/acquire pairing on `arrived` and `failed` is correct) |
| codegen | 8 | GEN.7 | GEN.1–GEN.6, GEN.8 |
| gpu | 10 | GPU.6, GPU.9 | GPU.1 (no per-frame readback on the window path), GPU.2, GPU.3–GPU.5, GPU.7 (two frames in flight; per-slot ring, tables, transforms, glows and counters), GPU.8 (queue and encoder barriers each name a real hazard), GPU.10 |
| gpu-dsa | 21 | none | All 21 clean or N/A on the host side |

## Checked and clean (substantive)

- The per-slot write-after-read hazards: ring, argument tables, transforms, glows, non-finite counters and boxes.
- The glow-only animation case: `animate` writes no transforms when there are no movers, so the single shared transforms copy is safe.
- The tone-map dispatch and barrier counts (12 and 11).
- The `max_textures`/`max_buffers` limits against the actual bindings.
- The size and traffic figures in offscreen.h, frame_images.h, preview.h, display.h and renderer.h:151.
