# The dark opening's core: the guideline pass

*2026-10-10. A rule-by-rule pass over the implementation of the dark
opening (branch `dark-opening-core`): `src/core/animation/glow.{h,cpp}`,
`flight.{h,cpp}`, `src/core/scene/swarm.{h,cpp}`, `scene.cpp`, and the tests
added to `tests/flight_test.cpp`, `swarm_test.cpp` and `scene_test.cpp`. The
design it implements is in those headers and in
[../2026-10-10-dark-opening.md](../2026-10-10-dark-opening.md).*

Both corpora answered for the whole pass:

- the C++ Core Guidelines, over the `cpp-guidelines` MCP server
  (`list_category` for every category below, `get_guideline` for the rules
  cited);
- the C++ performance guidelines, over HTTP at `localhost:7015/mcp` (its MCP
  server did not connect to the session; the HTTP endpoint answered every
  call).

Every category of both was walked. For each: the rules that bear on the
diff, each finding with its rule and verdict, and the rules checked and
found clean. A verdict is **fixed** (in this change), **convention** (a
project-wide departure recorded in AGENTS.md, "Guideline deviations"), or
**rejected** with the rule-based reason.

## Summary

| Verdict | Count |
|---|---|
| Fixed | 15 |
| Convention (AGENTS.md) | 2 |
| Rejected, with reason | 6 |

The review of 55f968a found five more; they are fixed in the commit after
it, and listed under [The review of 55f968a](#the-review-of-55f968a) below.

## C++ Core Guidelines

### P (Philosophy)

- Clean. P.1 (ideas in code: `Prelude` and `SwarmStart` are variants, not a
  kind and loose fields); P.4/P.5 (a prelude kind without a case in a
  `std::visit` fails the build, shown by deleting the `Hold` case of
  `check_prelude`'s visit: the build failed); P.6/P.7 (every precondition is
  checked at run time and refused early: `check_prelude`, `check_numbers`
  in swarm.cpp, `wake_factor`, `schedule_glow`'s begin); P.9 (nothing
  allocated per frame, below under Per).

### I (Interfaces)

- **Fixed, I.24.** glow.cpp's `loop_pulse(starts, loop, flash, t)` and
  `opening_pulse(schedule, flash, t)` took three doubles side by side; both
  now take the `ScheduleGlow` and t.
- **Fixed, I.24.** flight.cpp's `outward(double x, bool up)`: a two-valued
  bool, now `enum class Toward { down, up }` (the project's named enums for
  two-valued arguments, d1a5702).
- **Fixed, I.24.** swarm.cpp's `trace_down(box, x, z, height, obstacles)`:
  x, z and height adjacent doubles; now a `Vertical { x, z }` and the height
  computed inside.
- **Fixed, I.23.** swarm.cpp's `perch_from` and `perch_for` took six
  arguments and `start_in` five; each now takes a `Drawing` (the swarm,
  firefly, seed, range, distance and obstacles) and what is its own.
- **Fixed, I.5.** `make_firefly` took a swarm's start and wake numbers as
  given; it now refuses out-of-range ones with `std::invalid_argument`
  before drawing, and swarm.h says so (a header change).
- Clean: I.1 (`make_flight` takes its job, `rounding_allowance` its
  argument; no ambient state), I.2/I.22 (no new globals; the tests' text
  constants are `constexpr const char*`), I.4 (variants for kinds), I.5/I.6/
  I.7 (each new function states its preconditions and what it throws in
  its header or comment), I.10 (failures are exceptions), I.11-I.13, I.25-
  I.27, I.30 (not touched).

### F (Functions)

- **Fixed, F.6 and F.4.** The small pure helpers of flight.cpp and
  swarm.cpp are `noexcept`, and `constexpr` where they can be:
  `to_float`, `controls_of_transit`, `still_at`, `lay_out_opening`,
  `outward`, `largest_coordinate`.
- **Fixed, F.16.** `perch_opening` took `End` (six doubles, 48 bytes) by
  value; it takes `const End&`. (`hermite`, which also takes `End` by value,
  is not part of this change.)
- Clean: F.1/F.2/F.3 (one step a function, carried by number: P1, P2, P3,
  E0, 2p, 3), F.8 (pure functions of their inputs; the draws keyed), F.9
  (the visits' unused kinds unnamed), F.20/F.21 (results returned: `Perched`,
  `Rates`, `Firefly`), F.51 (a default argument for the tests' start),
  F.52 (lambdas capture by reference locally), F.48/F.49.

### C (Classes)

- **Fixed, C.21.** `Drawing` (new) and `Context` (given `rho`) deleted
  their copies only; both now delete their moves and default their
  destructor, as contract 11's `Obstacles` does.
- Clean: C.2 (`Wake`, `Hold`, `Perch`, `SwarmStart` kinds are structs of
  independent values, checked where read, as the headers design), C.12
  (`Drawing` and `Context` hold references and are neither copied nor
  moved), C.181/C.182 (tagged unions as `std::variant`), C.40/C.41 (Drawing's
  constructor computes every member), C.67 (`AtDistance` and `Ground`, the
  tests' fakes, derive from an uncopyable interface), C.128 (`override` on
  every fake's functions).

### Enum (Enumerations)

- Clean: Enum.2/Enum.3 (`Toward` and `Behaviour::still` are in class
  enums), Enum.5/Enum.6, Enum.7/Enum.8 (no underlying type or values given).

### R (Resource management)

- Clean: no owning pointer, `new` or `delete` in the diff; R.1 (vectors);
  R.3/R.4 (references and pointers non-owning: `Pending::motion`, the
  reader's `const toml::node&`); R.5 (scoped objects).

### ES (Expressions and statements)

- **Fixed, ES.45.** The refusals say "an hour" for `most_wait`; a
  `static_assert(most_wait == 3600.0)` ties each message to the constant
  (flight.cpp and scene.cpp), as the reader already ties "under a second" to
  `flash_spacing`.
- **Fixed, ES.77.** `opening_flashes` skipped the rise with `continue`; it
  is now an `if` over the waits.
- **Fixed, ES.73.** `check_rise` walked with `for (;;)`; it is `while
  (true)`, with the reason in a comment: no loop variable steps evenly, each
  sample's slack sets the next, and every step is at least
  rise_least_slack / V, so it ends.
- **Fixed, ES.3.** scene.cpp's flight refusal (`report`) and the new reach
  check each chose the swarm's table or the motion's line; one
  `flight_node()` now chooses for both.
- **Fixed, ES.46 (stated).** Every float narrowing new in the diff is
  either checked before (`rounding_allowance` and `reach_of` compare as
  doubles with float's max first; `opening_flashes` compares the count as a
  double with `most_opening_flashes` before `static_cast<int>`) or carries a
  comment naming why its value is in float's range (`to_float` in both
  files, `faces_up`).
- **Rejected, ES.23.** `=` initialization of arithmetic
  locals, the files' existing style; ES.23 allows `=` where no narrowing
  can occur, and `-Wconversion -Wsign-conversion -Werror` refuses any that
  would.
- Clean: ES.5/ES.6/ES.21/ES.22 (names declared where first given a value),
  ES.12 (`-Wshadow` on), ES.20/ES.25 (const where not mutated: `Range top`
  is mutated on purpose), ES.27 (`std::array`), ES.40/ES.41, ES.43/ES.44,
  ES.48/ES.49 (named casts only), ES.55 (bounds checked before every
  `*(after - 1)`, `.front()`, `.back()`, `(*linger)[i]`), ES.70/ES.78/ES.79
  (`reach_of`'s switch names every `Behaviour`, no default), ES.71 (the one
  index loop over the opening keeps its index for the draws' keys, and says
  so), ES.100-ES.107 (unsigned only for counters and keys).

### Per (Performance)

- **Fixed, Per.6 (stated cost).** glow.h counted a wake's per-frame
  operations without its three checks of `at` and `ramp`, nor the `min`
  that holds the ramp's x to 1; it counts both now (a header change).
- Clean: Per.11 (constants `constexpr`), Per.14/Per.15 (nothing allocated in
  `position()` or `glow()`; the opening's vectors are built once, at load),
  Per.16-Per.19 (the opening is at most two segments; its search touches at
  most two), Per.7.

### CP (Concurrency)

- Clean. `make_flights` is unchanged but for passing the whole job;
  `make_flight` reads its job and the obstacles through `const&` and writes
  only its result (CP.2, CP.3, CP.31); the swarm's draws are made before the
  threads, on the reader's thread. CP.4/CP.25/CP.41 as before.

### E (Error handling)

- Clean: E.2/E.3 (exceptions only for refusals), E.14 (`MotionError` for a
  perch, rise or far perch the flight refuses; `std::invalid_argument` for
  numbers out of range, as flight.h and swarm.h now state; `SceneError` with
  file and line from the reader), E.7 (preconditions stated), E.12
  (`noexcept` only where no throw can happen), E.15/E.18 (the reader's one
  `try` per refusal it translates, unchanged in shape).

### Con (Constants)

- Clean: Con.1/Con.4 (locals `const` where unchanged), Con.3 (references
  to const throughout), Con.5 (`perch_keeps`, `perch_most_rounding`,
  `rise_least_slack`, the step-7 keys, `perch_tolerance` and `facing_step`
  are `constexpr`).

### T (Templates)

- **Fixed, T.10 and T.47.** `Visit<Cases...>`, in a public header, was an
  unconstrained template with a common name; it now requires every case to
  be a class type (`std::is_class_v`), all an overload set can inherit.
- Clean: T.100 (variadic template for a heterogeneous set of lambdas),
  T.44 (aggregate deduction), T.141 (unnamed lambdas for one-place cases).

### SF (Source files)

- **Convention, SF.8.** `#pragma once`, per AGENTS.md.
- **Convention, SF.12.** Quoted `"core/..."` includes, per AGENTS.md.
- Clean: SF.10 (each file includes what it names: `<variant>`, `<optional>`,
  `<array>`, `<type_traits>`, `"core/animation/draw.h"` in swarm_test),
  SF.11 (flight.h self-contained with `<type_traits>` for `Visit`), SF.21/
  SF.22 (helpers in unnamed namespaces in the .cpp files only).

### SL (Standard library)

- Clean: SL.1/SL.2 (`std::variant`, `std::visit`, `std::optional`,
  `std::nextafter`, `std::upper_bound`), SL.con.1/SL.con.2/SL.con.3 (vector
  and array, bounds guarded), SL.str.1/SL.str.2 (`std::string` messages,
  `std::string_view` keys).

### NL (Naming and layout)

- Clean: NL.1/NL.2/NL.3 (comments state intent and cite the header's
  steps), NL.4/NL.17 (the files' layout, 4 spaces, lines at most 120),
  NL.8/NL.10 (underscore_style, kinds' arrays plural), NL.16 (Drawing's
  constructor, special members, then members), NL.19 (no 0/O or 1/l names).

## C++ performance guidelines

Every category `list_category` returned: cache-layout, codegen, concurrency,
copy-move, cpu-dsa, embedded, gpu, gpu-dsa, lifetime, memory, simd,
telemetry, wasm.

### MEM (memory)

- Clean: MEM.9 (the project's frame-path reading of it: no allocation in
  steady state; `position()` and `glow()` allocate nothing, the opening is
  built at load). MEM.1-MEM.8, MEM.10, MEM.11: no allocator in the diff.

### COPY (copy-move)

- Clean: COPY.1/COPY.8 (results returned as prvalues), COPY.3 (the reader
  moves `FlightParams` into each job), COPY.7 (no hidden copy: the range-for
  over `controls_of_transit(...)` binds a temporary whose life is extended;
  `Pending` copies a `Prelude` of 32 bytes, once per firefly at load:
  measured with the pinned clang on arm64, `sizeof(Perch)` 24, its
  `Float3` padded to the `double`'s alignment, and `sizeof(Prelude)` 32 with
  the variant's index; corrected after the review, below), COPY.4 (rule of zero for
  every new data struct; `Drawing`, which holds references, deletes all).

### CACHE (cache-layout)

- Clean: CACHE.2/CACHE.6 (the per-frame path reads `opening.empty()`,
  `begin`, then the loop's segments; `Flight` grows by a box, a vector and a
  double, cold beside 140 segments); CACHE.1 (no new shared writes; each
  `flights[k]` written once by one thread, as before).

### GEN (codegen)

- **Rejected, GEN.7.** `[[gnu::cold]]` on the refusal paths: they run at
  load or never; nothing measured asks for it (Per.1, Per.2, GEN.6), and the
  codebase marks none.
- Clean: GEN.1 (no `[[likely]]`), GEN.2 (the wake's branches are as
  predictable as a light is awake), GEN.8.

### CONC (concurrency)

- Clean: CONC.1 (the relaxed counter of `make_flights` unchanged and
  justified there); no new atomic.

### CDSA (cpu-dsa)

- **Rejected, CDSA.9.** A linear scan instead of `upper_bound` for the
  opening's at most two segments: at n ≤ 2 the binary search is at most two
  comparisons, the scan's count, and flight.h states the search; no
  measurement could separate them. The glow's search over at most some 300
  opening flashes is within L1, where CDSA.9 keeps the binary search.
- Clean: CDSA.21 (every threshold named and injectable or constant:
  `most_steps` bounds the rise as it does a segment, `perch_steps`,
  `perch_tolerance`, `perch_most_rounding`), CDSA.32 (the opening laid out
  once, at load, in the order `position()` reads it).

### GDSA (gpu-dsa, the draws and determinism)

- Clean: GDSA.3 (every new draw keyed by (seed, stream, counter): the
  opening's at episode numbers 67 and 68, past every loop key; the swarm's
  perch, wake and linger draws at 2^32 and up, past every start attempt;
  both shown: the opening pinned for one seed, and a wake drawn with step
  2's key caught by the test that compares the two); GDSA.2 (the rounding
  allowance moves a threshold by some 4 x 10^-7 m; the pinned loop of
  tests/flight_test.cpp is unmoved, so the determinism level flight.h
  states holds).

### GPU, LIFE, EMB, SIMD, TLM, WASM

- Not applicable: no GPU code, raw storage, lifetime tricks, embedded target,
  vector intrinsics, telemetry or WebAssembly in the diff. Checked by
  listing each category's rules against the diff; none applies.

## Rejected, gathered

| Rule | What | Why rejected |
|---|---|---|
| ES.23 | `=` initialization of arithmetic locals | Allowed where no narrowing; `-Wconversion -Werror` refuses any |
| GEN.7 | `[[gnu::cold]]` on refusal paths | Load-time or never-run paths; no measurement (Per.1, Per.2) |
| CDSA.9 | Linear scan for the opening's segments | n ≤ 2: the same comparisons; the header's choice |
| F.16 | `hermite()` takes `End` by value | Existing code outside this change; the new `perch_opening` takes `const End&` |
| C.21 | Other existing classes | Only the two this change touches (`Context`, `Drawing`) were brought in line |
| I.4 | Seconds as `double` in `Wake`, `Hold`, `Perch` | The headers' types, as every scene time (`Rhythm::period`) is; `frame::Seconds` is a frame's time |

## The review of 55f968a

Codex reviewed 55f968a for performance and architecture
(`.cache/reviews/55f968a-*.json`). Its findings in `src/core` and in this
note, each checked against both corpora again and fixed in the next commit.
The GPU finding (shadow rays toward lights of zero radiance,
`metal/integrator/`) is outside this change and is being handled
separately.

| Finding | Rule | Verdict and what changed |
|---|---|---|
| A perched firefly's rise is not vertical: the perch's x and z were clamped into the flight's range, and the loop's first point is episode 0's drift at time 0, up to 10 cm off its center (P1) | I.7 (the postcondition the header states: the first point straight above) | **Fixed.** `first_offset(seed)` (flight.h) gives the first point less the start, a function of the seed alone; step 2p places the start at the drawn line less that offset, never clamped, an attempt whose start leaves step 2's range redrawn, the line traced the first point's, its height required above the perch. position() at the loop's begin has the perch's x and z, bit for bit; tested from the made flights, with a box partly and one wholly outside the range, every perch on a line some attempt drew (never moved), and a shelf with clear air below it, where no first point is below its perch |
| `make_firefly` accepts waits past most_wait and a perch box not finite (P2) | I.5, P.6 | **Fixed.** A wake's to at most most_wait, and a linger's most at most most_wait less the wake's to (0 with no wake), compared without adding (ES.103: no sum to overflow), which holds the linger's most to most_wait as well; every box coordinate finite. Each tested for make_firefly's own refusal, not the MotionError a box it cannot draw in would give. The reader compares the sum the same way, so the two cannot disagree at a rounding. The perch's until is held to most_wait against the sum's rounding |
| At at = 2^100 and a ramp of 1, at + ramp rounds to at, and t = at answered 1 (P2) | I.7 (the header's w(t)) | **Fixed.** The wake's tests are t < at, then t - at >= ramp, then the smoothstep; t - at < ramp there, so x is at most 1 and the clamp is gone. A ramp of 0 is answered by the same comparison (t - at >= 0), with no division: a test of its own for it could never change the answer, and a mutation removing one survived every test, so there is none (P.9). Tested at 2^100 |
| Sleeping glows evaluated their schedule or rhythm before the wake discarded it (P2) | COPY.9 (arguments are evaluated before the call; the frame's path, per glowing light) | **Fixed.** The kind's checks and the wake's run on every call, awake or not (I.5, E.2: a bad record never hides behind a wake); the glow itself only when the factor is above 0. A rhythm's check of t is its beat, which its glow then reuses. Results bit for bit as before, tested at every millisecond across sleep, ramp and waking |
| This note said `Pending` copies a `Prelude` of at most 20 bytes (P1, doc) | COPY.7 | **Fixed.** Measured with the pinned clang on arm64: `Perch` 24 bytes, `Prelude` 32; the text says so |

Rules checked for the new code and found clean: I.23/I.24 (`perch_from`
takes its drawing, its start kind, the offset and the attempt; the offset
an array, not three doubles), F.4/F.8 (`first_offset` a pure function of
its seed), ES.3 (`drift_shape` drawn once for the drift and the offset, so
they cannot disagree), ES.46 (each narrowing to float is of a value inside
the world), GDSA.3 (the draws keep their keys: an air swarm's starts and
every loop are unchanged, the pinned loop included), CDSA.21 (no new
threshold), MEM.9 (nothing allocated per frame; a sleeping glow now does
less).
