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

## Guideline deviations

Project-wide departures from the C++ Core Guidelines, each decided once, here,
so a review does not raise them file by file:

- **SF.8, `#pragma once` instead of include guards.** Every header is
  compiled by Apple clang and many by the Metal compiler as well; both support
  it, and one form is used everywhere.
- **SF.12, quoted includes resolved through `-I src`.** One include root,
  `"core/..."`, `"metal/..."`, for C++ and Metal alike; angle brackets are kept
  for third-party and system headers.
- **ES.23, ES.49 and ES.64 in the Metal Shading Language.** `float3(x)` and
  `uint(x)` are MSL's conversion syntax, and `as_type` is its named cast.
- **ES.27 and SL.con.1 for arrays inside shared GPU layouts.** `padding[]`,
  `Transform::m` and the noise tables stay C arrays: MSL has no `std::array`
  in `constant` layouts shared with the host.
- **Enum.7 and Enum.8 on enums shared with Metal.** Their fixed underlying
  type and values are the layout the shaders read.
- **ES.100, ES.102 and ES.107: `uint` loop indices in shaders,** the GPU
  idiom.
