# Workflow

This repo adapts the Lite Factory workflow from
`governance/product_memory_workflow.md`. The design of each change lives in
the code it changes, not in a separate packet.

## Default Chain

```text
Coordinator
  -> file headers: the contract, the design, and the guidelines behind it
  -> review of the headers
  -> implementation and tests
  -> review
  -> MEMORY.md and QUEUE.md updates
```

## Local Notes

This repo is C++-dominant and performance-sensitive.

- **The headers are the plan.** Each new or changed file's header states its
  contract and its design, and cites by ID the guidelines from the
  `cpp-guidelines` and `cpp-perf-guidelines` corpora that shape it (for
  example R.1, E.5, GDSA.2). The headers are written and reviewed before the
  implementation.
- **Every C++ step is checked against `cpp-guidelines` before its code is
  written, and performance-sensitive work against `cpp-perf-guidelines`.**
  The citations in the headers are the record. If either server is
  unreachable, say so and get explicit agreement before writing C++.
- **Optimizations are designed in, not deferred.** Each carries
  `// Optimization: …` with its concrete reason and the guideline or system
  it follows, so a reader sees a choice rather than an accident. Where it can
  be measured, it is, and the figures go in the note.
- **A data path's cost is derived before it is built.** Its header walks
  every stage and counts, as functions of the resolution, the light count and
  the scene size: the bytes it moves and the copies it makes, the operations
  in its inner loop and whether each can be inlined, and the crossings
  between CPU and GPU in a frame (command buffers, encoders, dispatches,
  synchronizations, readbacks). A count that scales with pixels or lights
  where it could scale with tiles or passes is designed out before code is
  written. Once the path runs it is measured against that floor.
- **Math the CPU and GPU share is written to `src/core/portable_math.h`'s
  contract,** and a GPU test compares the two bit for bit where the contract
  covers the inputs. Anything outside the contract is compared with a stated
  tolerance, never with equality that happens to hold.
- **Independent review is on request,** not per commit:
  `scripts/codex-review.sh <commit>` checks a commit for correctness and
  speed and writes its findings to `.cache/reviews/`.
- A frame's path, acceleration-structure builds, memory footprint and GPU
  dispatch are performance-sensitive by default.
- GPU behavior is environment-sensitive. Verification names the machine it
  ran on (the device test prints it); a green build is not proof. There is
  one target, the development machine (MEMORY.md).

Do not weaken inherited governance without a decision; MEMORY.md records the
ones taken.
