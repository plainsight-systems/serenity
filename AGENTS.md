# Agent Instructions

This repository inherits Plainsight Systems governance from `docs/process/`.

## Session Entry

1. Read `docs/process/MEMORY.md`.
2. Read `docs/process/QUEUE.md`.
3. Read `docs/process/workflow.md`.
4. Read the inherited governance in `docs/process/governance/` (submodule;
   run `git submodule update --init` if empty). `docs/process/inherited.md`
   explains the pin.
5. Inspect product-specific source and tests before editing.

## Operating Rules

- Follow Lite Factory workflow from `docs/process/governance/product_memory_workflow.md`,
  as adapted in `docs/process/workflow.md`: new work is designed in its file
  headers, not in packets.
- Keep product decisions in `docs/process/` and research in `docs/research/`.
- Update `MEMORY.md` and `QUEUE.md` when work state or durable knowledge changes.
- Do not weaken inherited governance without an explicit decision.

## Repo-Specific Gates

This repo is C++-dominant and performance-sensitive. Both C++ gates bind:

- Non-trivial C++ changes require a C++ architecture note in the file headers
  before implementation, and a review against `docs/process/governance/cpp_architecture_review.md`
  before acceptance.
- Performance-sensitive C++ changes require a C++ performance note in the file
  headers before implementation, and a review against
  `docs/process/governance/cpp_performance_review.md` before acceptance.
- Treat a frame's path, acceleration-structure builds, memory footprint and
  GPU dispatch as performance-sensitive by default.
- `src/core/` uses no GPU or windowing API; only `src/metal/` uses Metal's.
  `make check` enforces it.

## Scope Posture

`engineering_philosophies.md` forbids demo-ware. In this repo, "tech demo" means
the real renderer on a deliberately narrow slice at full intended quality. It
does not mean a lower-fidelity preview to be hardened later.

- Narrow the slice rather than lowering the quality inside it.
- A contract relaxation must be defensible without using the word "demo."
