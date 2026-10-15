# The denoiser's data set: Phase 2

2026-10-14. Phase 2 of Milestone 3 (2026-10-14-denoiser-training-design.md,
"The data", "Splits"): denoiser-v1, rendered by
`scripts/make_denoiser_dataset.py --out DIR` at commit ab6dc9c, release
build (Apple clang -O3, Metal -O2, fast math; the toolchain in
`cmake/toolchain.json`), Apple M3 Max. Kept outside the repo, at
`../serenity-data/denoiser-v1/` beside the checkout; never committed.

## What is in it

| Split | Scenes | Sequences | Frames |
|---|---|---|---|
| Training | generated seeds 101-140 less the 4 below; brass_sphere, brass_sphere_flight, brass_sphere_wander | 39 | 312 |
| Validation | generated seeds 105, 115, 125, 135 | 4 | 32 |
| Test | the real marbles from t = 2, 4, 6, 8, 30, 60, 120 s | 7 | 56 |

- **Scenes.** `make_marbles.py --seed N` for N = 101 to 140, never seed 7
  (the real marbles' layout); each scene file is copied into the data set
  and hashed. The strata, by seed: 101-110 start in the opening (1 to 8 s),
  111-120 in the flood (8 to 24 s, the most fireflies lit, the perched rising
  past the descending), 121-140 awake (24 to 120 s). Validation is one
  opening, one flood and two awake. The start of each is drawn by its name
  alone (`random.Random("denoiser-v1-time-<name>")`); the drawn starts run
  from 2.97 s to 115.8 s. The test's t = 8 was asked for twice and is one
  sequence.
- **A sequence** is 8 consecutive frames at 60 frames a second, f to f + 7.
- **Input, per frame:** `graphs/restir_pt_features.toml` (ReSTIR PT + MNEE
  and the features) at 960 x 540, one sample: linear radiance and albedo,
  normal, depth and motion, five PFMs. One run a sequence. The headless
  renderer replays a reusing graph's frames 0 to f - 1 first
  (`core/frame/history.h`), so each frame is exactly the frame a run from
  the start shows, its history at least 120 frames, not the design's 30 of
  warm-up: the same frame the runtime makes, at the price of the replay
  (about 12 ms a frame; most of the input time).
- **Target, per frame:** `graphs/path.toml` at 1920 x 1080, 4,096 samples,
  time frozen at the frame's instant (`--time`, the same double as the
  input's step x i), linear PFM; and the 1920 x 1080 albedo from
  `graphs/features.toml`, run over the input's frame indices, so its rays
  are those the runtime's full-size albedo would take (one a pixel; QUEUE.md
  already holds averaging it).
- **Random numbers.** The input's frame indices are the timeline's, 0 to
  7,207. The target of frame i is headless frame 100,000 + i, its samples
  indices (100,000 + i) x 4,096 onward, a block of its own: none meets an
  input's, another target's, or a reference's 0 to 65,535.
- **`manifest.json`:** the renderer's commit, that the tree was clean, the
  binary's SHA-256 and build flags; the plan; the splits; and per frame its
  sequence, scene, seed, split, stratum, time, frame indices, sizes, sample
  counts and sample index range, and every file's name and SHA-256, with
  the count of non-finite floats in it.

## Running it

Work is done in units (a scene file, a sequence's input run, its albedo
run, each target), each rendered into a staging directory, scanned for
non-finite floats, moved into place, hashed and recorded, the manifest
written whole and renamed. A rerun re-verifies every recorded checksum and
renders only what is missing or changed; a manifest of another plan or
another renderer is refused. Shown on a trial (2 scenes, 2 frames, 64 and
1,024 target samples): killed mid-sequence, one finished target corrupted
by one bit; the rerun found the bit, rendered that target and the
unfinished units, and a third run rendered nothing. `--verify` checks every
planned unit against the manifest. `tests/test_make_denoiser_dataset.py`
(ctest `make_denoiser_dataset`) pins the plan by its SHA-256 and checks the
splits, seed 7's absence, the times and the sample indices; a moved stratum
and a moved target offset each fail it.

## Cost

Estimated from the first sequence's input run (3.0 s for 244 timeline
frames, start-up included) and its first target (27.9 s): 3.58 h, under the
5 h past which the targets would have dropped to 2,048 samples, so they
kept 4,096. Rendered 2026-10-14, 17:06 to 21:09, with no other serenity
process on the GPU at any sequence (utilization 4% before the start):

| Work | Runs | Hours | Each |
|---|---|---|---|
| Inputs (136,858 timeline frames, replays included) | 50 | 0.48 | 1.6 to 110 s |
| Full-size albedo | 50 | 0.01 | 0.5 to 0.7 s |
| Targets, 4,096 samples at 1920 x 1080 | 400 | 3.48 | 13.5 to 43.7 s, mean 31.3 |
| In all | | 3.98 | |

The estimate was 10% short: the first target, an opening scene with few
fireflies lit, was quicker than the mean. 32.3 GB on disk (30.1 GiB), 2,844
files: 31.1 MB of input and 49.8 MB of target a frame. `--verify` reads
and checks it all in 14 s: every one of the 544 units present, every
checksum matching, no non-finite float in any file, and no run refused for
non-finite samples (serenity-headless fails a frame that has any).

A sequence's input run and a target rendered again from the manifest's
commit and arguments (gen-0108, frames 178 to 185 and frame 181's target)
matched their recorded checksums byte for byte.

## Anything odd

- **No training sequence starts before 3 s.** The opening's ten draws fell
  between 2.97 and 6.97 s; the test's t = 2 is earlier than any training
  frame (queued).
- **The targets still sparkle inside glass.** At 4,096 samples the path
  tracer (no MNEE, the reference's estimator) leaves speckle in the cat's-eye
  cores and tinted glass lit through the glass, plainest late in the
  animation (gen-0123 at 115.8 s); the network is trained toward it there
  (queued).
- **Input and target agree in the mean.** On three frames picked at random
  the input's mean radiance is within 1.2% of the target's (0.1122 against
  0.1129, 0.1944 against 0.1969, 0.1700 against 0.1700).
- **The test inputs use frame indices 0 to 7,207**, the timeline's, which a
  65,536-sample reference made by `make reference` (batches from index 0)
  would share; Phase 4's references for scoring start past them (queued).
- The scenes are copied into the data set beside the frames, so a frame
  can be rendered again from the data set and the commit alone.
