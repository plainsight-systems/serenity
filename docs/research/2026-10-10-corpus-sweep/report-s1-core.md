## src/core review (branch `marbles`, read-only)

I read all 64 files under `src/core/`. Corpora: the C++ Core Guidelines MCP and the perf corpus at localhost:7015 both responded. I looked up every rule ID cited below, and every ID the code cites.

### Findings

**1. flight.cpp:141 — undefined behaviour when converting a double to int64 (high)**
- Rule: ES.46, avoid lossy narrowing arithmetic conversions.
- Line 141 casts `std::ceil(v * s.duration / delta)` to `std::int64_t`. The `steps > 1000000` guard only runs after the cast.
- `v` is checked to be finite but can be huge. The reader's `read_positive` accepts any `speed` up to FLT_MAX. Episode 0's drift has `v = speed/3`, so `speed = 1e30` gives about 1.3e32 steps. Converting a value that large to int64 is UB. It happens in the first `clear()` call, from scene-file input the reader accepts. The product `v * s.duration` can also overflow to inf.
- Fix: test the double before casting:
  `const double n = std::ceil(v * s.duration / delta); if (!(n < 1e6)) return false; const auto steps = static_cast<std::int64_t>(n) + 1;`
  Optionally also give `speed` an upper bound in `read_flight_params`.

**2. glow.cpp:30 — the same UB, on every frame (high)**
- Rule: ES.46.
- `static_cast<std::int64_t>(std::floor(t / r.period))`. The reader accepts any `period > 0` (scene.cpp:565, via `read_positive`). With `period = 1e-20`, `t / period` passes 9.2e18 after one second, and the cast is UB on every frame from then on.
- Fix: give `period` a minimum at read time (for example ≥ 1e-3 s) and say so in scene.h. In `rhythm_glow`, also return `dim` (or refuse) when `!(std::abs(t / r.period) < 9.0e18)`.

**3. thin_lens.cpp:48–53 — `valid()` accepts a camera that frames to NaN (medium)**
- Rules: I.6/I.7 (state and check pre/postconditions); repo governance §3.3 (error paths must not look like success).
- Example: `position = [3e38,0,0]`, `look_at = [-3e38,0,0]`. Both pass `Reader::number` and `finite()`.
  - `view.x` overflows to -inf, so `length(view)` is inf and `normalized(view)` is (NaN,0,0).
  - The cross product's length is NaN. `NaN < tiny` is false, so the camera is reported valid, and `shader_form` produces a NaN `forward`.
- Smaller inputs (around 1e20) are refused, but with the wrong reason ("up is parallel"), because the float `a.x*a.x` overflows.
- The scene reader promises "a camera that cannot be framed" is an Error.
- Fix: write each test in the negated form, `!(length(view) >= tiny)` and so on (the file already uses this style for fov and lens). Add `finite(view)`. Either compute the lengths in double or require the camera to lie within `world_extent`, as shapes must.

**4. scene.cpp:933–937 and graph_file.cpp:155–159 — the read-error check can never fire (medium)**
- Rules: I.10 (an error must not be possible to ignore); SL.io.2 (always consider ill-formed input); governance §3.3.
- `text << file.rdbuf()` changes the state of `text`, never of `file`. So `if (!file && !file.eof())` is always false, since `file` was already checked good.
- Effects:
  - A failed read (for example a directory path) gives an empty string. It is parsed and reported as "the scene has no 'camera'" or "no 'passes'".
  - A truncated read could parse as a shorter scene that looks valid.
- Fix: size the buffer from `std::filesystem::file_size(path)`, `file.read(buf.data(), size)`, and throw if `!file` or `file.gcount() != size`.

**5. flight.h:202–204 and flight.cpp:545 — E.17 is cited for the opposite of what it says (medium, misleading docs)**
- E.17 is "Don't try to catch every exception in every function": let exceptions propagate to a function that can handle them.
- The code cites it for catching `(...)` in each worker and storing an `exception_ptr`. That design is correct, but it is CP.60's idea ("Use a future to return a value from a concurrent task… error (exception) return handled simply").
- Fix: cite CP.60, or describe the exception-transport rationale without an ID.

**6. schedule.h:107–108 versus schedule.cpp:10–11 — the "one table" claim is false (low)**
- Rule: ES.45 / "one definition".
- The header says the kind↔name mapping is "One table, so a kind and its name cannot disagree in two places". But `pass_kind()` and `all_pass_kinds()` read a second list, `kinds`.
- `-Wswitch` does not cover that list. A new `PassKind` that has a name but is missing from `kinds` builds fine and cannot be named in a graph file.
- Fix: `static_assert(kinds.size() == static_cast<std::size_t>(PassKind::tone_map) + 1)` beside the enum, or reword the claim.

**7. scene.cpp:503 — reference data member in a copyable class (low)**
- Rule: C.12, don't make data members `const` or references in a copyable/movable type.
- `StillShapes` holds `const shapes::Shapes& shapes_` and is implicitly copyable (and copy-assignment is deleted). It is safe here only because it is a local in `read_scene`.
- Fix: delete the copy and move operations (it is an `Obstacles` interface, so that is C.67-consistent), or hold a `const Shapes*`.

**8. flight.cpp:163–197 — plain `enum Purpose` spills 33 short names into `serenity::animation` (low)**
- Rule: Enum.3, prefer class enums over plain enums.
- Names include `radius`, `tilt`, `lift`, `rise`, `bob`, `direction` and `loops`.
- Locals already shadow some: `bob` at :80, `rise` at :359. Inside `transit()`, any later `d.between(rise, …)` would silently bind to the Segment rather than the enumerator.
- Fix: `enum class Purpose : std::uint64_t`, converted explicitly in `Draws::operator()`.

**9. Implicit includes (low)**
- Rule: SF.10, avoid dependencies on implicitly `#include`d names. Governance "Dependency Checks" also flags reliance on transitive includes.
  - scene.cpp:487 uses `std::min` with no `<algorithm>`.
  - graph_file.cpp:38 uses `std::initializer_list` with no `<initializer_list>`.
  - scene.h:291 uses `std::uint32_t` with no `<cstdint>`; it currently arrives through animate.h.
- Fix: add the three includes.

**10. wander.cpp:45–47 — the error message leaves out a condition it checks (low)**
- Rule: E.2/I.10 (errors should say what failed).
- The check refuses a non-finite or negative `body`, but the message says only "reach and speed must be finite and greater than 0".
- Fix: add "and body finite, 0 or more".

### Checked and fine

- **Shared layouts (`contracts/`, `materials/`, `lights/`, `media/`, `shapes/`, `textures/`, `passes/tone_map.h`):**
  - Every struct is 4-byte fields with no implicit padding, matching its asserted size and the CACHE.5 claim.
  - Metal `constant` guards are consistent.
  - Tone-map step 3's half-float sums check out (10920 → 21840 → 32768 → 43680 → 54592 → 65504).
  - Wood and noise range claims (6.7e8 < 2^31) and the fbm bound check out.
- **make_flights concurrency:** relaxed `fetch_add` only hands out indices. Each `flights[k]` and `failures[k]` slot has a single writer, and the `jthread` joins publish them. There is no data race. The CONC.1, CONC.3, CACHE.1, CP.4, CP.25 and CP.41 citations match their rules. The lowest-k error is deterministic. `Obstacles` is const and only read during the parallel phase.
- **flight.cpp:**
  - Backtracking bounds (rounds, 64 redraws).
  - Step 5 draws use attempt numbers disjoint from episode 63's build, since `round[63]` is always 0.
  - The binary search in `position()` handles negative t and `tau == loop`.
  - The flash spacing (≥ 1 s, wrap included) together with the glow's `flash < 1` means flashes cannot overlap.
- **glow.cpp:** the rhythm neighbour window is correct (jitter ±0.2p, flash ≤ p/2); the schedule wrap-around is correct.
- **wander.cpp:** the float-range guards, outward rounding and the clamp before narrowing are all sound.
- **scene.cpp:**
  - Number parsing checks the double before narrowing (ES.46 cited correctly).
  - Index bookkeeping (`shape_lights`, `lights`, swarm pending, `job_of`), the `FlightsError` message split and the `[[noreturn]]` paths are correct.
  - Movers and glowers are pushed in target order, as animate.h requires.
- **swarm.cpp:** start margin and clearance (`first_drift_reach·√3`) agree with the flight's step-3 test. Seeding matches GDSA.3.
- **shapes.cpp:** the distance and touch tests, the rotation refusal, and the outward rounding in `world_bounds` (hot path, no allocation) are correct.
- **coated.cpp:** Simpson weights are right with an even interval count, and the substitution is right.
- **frame/:** the history rule (held ≤ 2^24−1, LiveHistory reset), schedule validation and switch-without-default are covered by `-Wall -Werror`. The precision claims in frame_constants and frame_inputs are arithmetically correct.
- **png.cpp, frame_times.cpp:** size and int-range checks are fine. The GPU.10 and TLM.11 citations match.
- **Per-frame paths** (`animate`, `position`, `glow`, `world_bounds`): no allocation, matching the MEM.9 claims.
- **Other cited IDs** (F.8, I.1, I.4, I.25, C.48, Enum.2, C.181, ES.45, E.2, E.14, GPU.3, GPU.4, GDSA.3): each says what its comment claims.

Files: `src/core/animation/flight.cpp`, `.../animation/flight.h`, `.../animation/glow.cpp`, `.../animation/wander.cpp`, `.../camera/thin_lens.cpp`, `.../scene/scene.cpp`, `.../scene/scene.h`, `.../frame/graph_file.cpp`, `.../frame/schedule.cpp`, `.../frame/schedule.h`.