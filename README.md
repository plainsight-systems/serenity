# Serenity

Serenity is a real-time path tracer for a scene of simple procedural
geometry, lit by hundreds to thousands of moving lights, with the lighting
correct as everything moves. It is meant to sample those lights with ReSTIR
DI (Bitterli et al. 2020), denoise the result with a denoiser of its own, and
compare both against a reference rendered at thousands of samples per pixel.

**Status: skeleton.** Nothing renders yet. The repository has its build, its
process, and a Metal backend that acquires a device and runs compiled
shaders. The design
comes next ([`docs/architecture/`](docs/architecture/README.md)), then the
first rendering code.

Serenity is R&D under Plainsight Systems LLC.

## Requirements

- An Apple silicon Mac. The development machine is an M3 Max; Apple9 GPUs
  (M3 and later) trace rays in hardware, earlier ones in software.
- Xcode 26.2 exactly, with its Metal toolchain. The build checks the
  toolchain against [`cmake/toolchain.json`](cmake/toolchain.json) when it
  is configured and stops on a mismatch.
- CMake 3.25 or later, and Python 3.

## Build and test

```sh
git clone --recurse-submodules <this repository>
cd serenity
make test     # the tests, the GPU tests on this machine's GPU among them
make check    # structural rules, and tests proving each check fires
```

The first build fetches metal-cpp and doctest, each pinned by commit.

## Find your way around

| Path | What it holds |
|---|---|
| `src/core/` | platform-neutral C++: no GPU or windowing API. Empty until the first rendering change |
| `src/metal/` | the Metal backend, the only code that uses Metal's host API |
| `cmake/` | the toolchain pin, and how shaders are compiled and compiled in |
| `tests/` | the core's tests; `tests/gpu/` runs on the GPU |
| `tools/` | the toolchain and boundary checks |
| `scripts/` | an independent review of a commit, on request |
| `docs/architecture/` | the design (next) |
| `docs/research/` | measurements and investigations, each dated, with its machine and toolchain |
| `docs/process/` | decisions and the work queue |

## License

Code is licensed under the Apache License 2.0; documentation, prose and
original visual assets under CC BY 4.0. See [`LICENSE`](LICENSE) for the
split, and [`NOTICE`](NOTICE) for third-party components and their licenses.
