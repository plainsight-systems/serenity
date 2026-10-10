## Review of src/metal host C++ (branch marbles)

Scope: I read all 44 non-shader `.h` and `.cpp` files under `src/metal/`, plus the callers `src/app/main.cpp`, `src/headless/main.cpp` and `tests/gpu/frame*_test.cpp` to check lifetimes, and the Metal SDK headers. Both corpora answered. The `cpp-perf-guidelines` MCP server would not connect, but `perf.py` over localhost:7015 worked, and every perf ID below comes from it.

### Findings

**1. GPU resources are freed before the Submission destructor's wait runs (error path)** — high, with one uncertainty
- Where: `src/metal/device/submission.cpp:62-71` (the destructor and its comment); `src/metal/frame/renderer.h:175` (`~Renderer() = default`); same pattern in `Accumulation`, `FrameImages`, `NonFinite`, `SceneBuffers`, `ShapeTransforms`, `LightGlows`, `SceneAcceleration`, `ToneMapPass`, `Offscreen`.
- Rules:
  - C.31: "All resources acquired by a class must be released by the class's destructor … especially in error cases."
  - R.1: RAII.
  - GPU.9: reuse or free memory "only after the graph proves the old resource has no future users".
- What is wrong:
  - Every resource owner is built after `Submission` and holds a `Submission&` (`app/main.cpp:77-80`, `headless/main.cpp:44-46`). So it is destroyed *before* `~Submission` waits.
  - The destructor comment says its wait protects "everything this queue's work refers to". In fact it protects only its own allocators and command buffers.
  - On the normal path `finish()` runs first, so nothing goes wrong. On any exception (a GPU fault from `settle`, a refused history, `record` throwing), `~Renderer` releases buffers and textures while up to two frames are still running.
  - Those buffers are bound by raw GPU address (`setAddress`), which nothing can retain.
  - None of these classes calls `release_resident` in its destructor either, so residency entries are acquired and never released (C.31).
- Uncertainty: if `MTLResidencySet` keeps a strong reference to each allocation, the result is a leak until `Submission` dies rather than a use-after-free. The SDK header (`MTLResidencySet.h:79-102`) says neither.
- Fix: add a noexcept `Submission::wait_idle()` (a bounded event wait for the last committed sequence). Have `Renderer` (and `Offscreen`) keep the `Submission&`, and in their destructors call `wait_idle()`, then `release_resident` for what they added.
  - Alternative: a deferred-release list in `Submission`, holding `NS::SharedPtr<MTL::Allocation>` until the event passes. That would also remove the drains on resize.

**2. `record()` can throw after `Submission::begin()`, leaving the submission open** — medium
- Where: `src/metal/frame/renderer.cpp:185-187` (camera check), `:237` (`animation::animate` throws `std::invalid_argument`), both after `render_to_window`/`render_to_offscreen` called `begin()` (`renderer.cpp:271`, `:280`).
- Rules:
  - E.4: "every object not destroyed must be in a valid state".
  - E.14: purpose-designed exception types; `animate` throws `std::invalid_argument`, not `metal::Error`.
  - The protocol in `submission.h:44-46`: each `begin()` is followed by exactly one `commit()` or `present()`.
- What is wrong:
  - The missing-camera check depends only on `inputs`, yet it runs after the command buffer is begun and an encoder may be open.
  - The `Submission` then stays `open_` for good, and the drawable is never presented.
  - The exception then reaches finding 1's path.
- Fix:
  - Move the camera check into `Renderer::prepare()`, which runs before `begin()`.
  - Check glower targets against the light count once, at construction (mover targets are already checked in `SceneAcceleration`), so nothing in `record()` after `begin()` can throw except protocol errors.

**3. `renderer.h:141-148` misstates Metal's argument-table contract** — medium (misleading docs)
- Rule: the repo's no-facades rule (the comment justifies a design).
- What is wrong:
  - The comment says "Apple's documentation does not say whether an encoder copies a table's bindings at each dispatch", and that rebinding between dispatches is only "shown on this machine" by a test.
  - The SDK header says otherwise (`MTL4ComputeCommandEncoder.h:564-568`): "Metal takes a snapshot of the resources in the argument table when you make dispatch or execute calls on this encoder." So rebinding within a frame is the documented contract.
  - The per-slot ring of tables (`arguments_`, `renderer.h:217`) rests on the wrong premise. It is harmless, but not needed.
- Fix: cite the header in the comment. Then either keep the ring and say it is optional, or collapse it to one table.

**4. `Accumulation::prepare` updates the history before the image exists** — low
- Where: `src/metal/frame/accumulation.cpp:15` vs `:39-43`.
- Rule: E.4 (objects stay in a valid state through an error).
- What is wrong:
  - `history_.join()` records the new size before `newTexture` succeeds, and the old texture is already reset.
  - If `newTexture` fails, a later `prepare()` at the same size sees `remade == false` and returns `held` with `texture_ == nullptr`.
  - The failure does surface, but later and elsewhere (`PathPass` "no accumulated image").
- Fix: join on a copy (`frame::History next = history_; … make texture …; history_ = next;`).

**5. The window path dispatches at `presenter.size()`, not the drawable's size** — low
- Where: `src/metal/frame/renderer.cpp:270-272`.
- Rule: I.5/I.6 (state and check preconditions).
- What is wrong: `record` dispatches `size.width × size.height` threads that write `drawable->texture()`, and the two sizes are never compared. If Core Animation hands out a drawable of another size (clamping, or a stale drawable), the kernels write outside the texture.
- Fix: take the size from `drawable->texture()->width()/height()`, or throw `Error` if they differ.

**6. `release_resident` does not check its "drain() first" precondition** — low
- Where: `src/metal/device/submission.h:167-170`, `submission.cpp:211-220`.
- Rule: I.5/I.6.
- What is wrong: only `open_` is checked. Removing an allocation from the set while a committed frame still uses it is a GPU fault, and checking costs one read of the event.
- Fix: `if (next_ > 0 && !has_completed(next_ - 1)) throw Error(...)`.

**7. A still scene keeps its build scratch for the whole run** — low
- Where: `src/metal/acceleration/scene_acceleration.cpp:80-81, 90, 110`.
- Rule: GPU.9 (transient memory should not outlive its use).
- What is wrong: for a still scene, `Built::scratch` is used once by the start-up build, which is waited for, and is then held for the program's lifetime.
- Fix: `structures_[0].scratch.reset()` after `wait_until_complete`.

**8. Wrong citation of CDSA.29** — low (misleading docs)
- Where: `src/metal/acceleration/scene_acceleration.h:57-59`, "built … in a submission of its own that is waited for, so it is complete before any frame traces it (CDSA.29)".
- What is wrong: CDSA.29 is "Choose index by queries and mutations per build" (build cost against queries). It says nothing about finishing a build before it is traced.
- Fix: cite GPU.8 or GPU.7 for the ordering, or move CDSA.29 to the sentence about building once for a still scene.

### Checked and fine
- **Submission protocol:**
  - The slot reuse wait settles submission n−2 (event, then feedback) before the allocator is reset.
  - Event values are monotonic, and the arithmetic in `~Submission` (open vs committed) is correct.
  - Wait, commit, `signalDrawable` and `present` come in the right order.
- **Feedback handler state:** a `shared_ptr` outlives `Submission`. The release/acquire on `arrived`/`failed` correctly publishes `failure`, `gpu_start` and `gpu_end`, and only one handler per slot is live at a time. There is no data race. The cited R.20 and CP.3 say what the comment claims.
- **CPU writes vs frames in flight:**
  - The constants/camera ring, `FrameArray` copies (transforms, glows, boxes) and per-slot structures and scratch are all written only after `begin()` waited for the slot.
  - Single-copy arrays are never written per frame, since `animate` writes only movers and glowers.
  - `NonFinite` reads a counter only after its submission has completed.
- **Barriers:**
  - Within a pass (tone map), after the structure build (acceleration → dispatch), and before a pass reading radiance (encoder dispatch → dispatch).
  - Across frames: one queue barrier before the first pass that touches the shared images or the accumulated image.
  - None is missing or redundant for the current pass kinds. GPU.7 and GPU.8 support the comments.
- **Residency:** every buffer and texture a pass binds is made resident. The drawables come from the layer's set, added once. The still-scene build uses its own set for scratch.
- **Resize:** drain first, then `release_resident`, then reset, in `Accumulation` and `FrameImages`. The partial-failure path in `FrameImages` resets `size_`.
- **Metal error handling:**
  - Every `new*` result is checked and wrapped in `TransferPtr`, and every `NS::Error` is described inside a live autorelease pool.
  - Pools wrap the frame (`render_to_window`), each commit, and construction.
  - `RetainPtr` is used for borrowed handles (device, layer).
- **Per-frame allocation:** `record()` allocates nothing. The `CommitOptions` and handler allocation per commit is the API's shape and is documented. Strings are built only on error paths.
- **Other cited IDs:** MEM.4, MEM.9, GPU.1, GPU.2, GPU.6, GDSA.6, GDSA.16, CDSA.30, CDSA.32, E.2, E.5, E.14, I.4, I.11, F.25, ES.30 and P.5 each say what their comment claims.
- **Repo boundary:** Metal's host API appears only under `src/metal/` (AGENTS.md).