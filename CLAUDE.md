# Claude/Codex Entry

## How this project works (read first)

Serenity is a tech demo, not a research project. It shows that its author
understands GPU performance and the parallel between attention and path
tracing, with real frame times on real hardware. This section is the
project's own constitution: where it conflicts with a global operator
profile or the inherited governance in `docs/process/governance/`, this
section wins.

- **Production practice by default.** Choose what shipped engines and the
  platform vendor do (RTXDI, Falcor, Unreal, Apple's samples), checked
  before designing, not after.
- **The design order stays:** architecture documents first
  (`docs/architecture/`: logical overview, change axes, file mapping),
  then the design in the headers, with both C++ guideline corpora (the
  C++ Core Guidelines and the C++ performance guidelines) checked as each
  header is written and the deciding rules cited by ID, then the code.
- **One measurement:** the demo graph's frame time (and frames a second)
  at the sizes in play, on AC power with the GPU idle. Per-pass GPU
  timestamps when a change targets a pass. Nothing else unless asked: no
  reference renders, no convergence runs, no timing of graphs the change
  isn't for, no statistical machinery.
- **Images:** if a change alters the picture, render the frame before and
  after, show them side by side, and let the user decide. That is the
  whole image check.
- **Byte identity** of existing graphs' output is the guard for changes
  meant to keep the image; it is cheap and stays.
- **Tests guard invariants and contracts** (layouts, determinism,
  refusals that protect a contract, no per-frame allocation, matches
  between implementations), not coverage. A test that can't fail is
  removed. Tests never assert the shipped graphs' or scenes' current
  settings, which change with the demo; they build their own inputs.
- **Unit tests test units of code.** Tests that run the program are
  integration tests, a small handful. Logic checked by rendering (a file
  name, a setting) moves into a function tested directly.
- **A decision removes the alternative.** Superseded code, graphs,
  settings and weights are deleted, not kept "just because". Git and the
  milestone tags (`milestone-N`, tagged at every close) are the archive.
- **Past milestones' films, images and numbers are kept as made**, never
  re-rendered to compare against.
- **Gates run once, at the end** of a piece of work, not after each step.
- **Fix valid findings together.** No deferring for "one intent", no
  review-fix-review chains: one pass, fix what matters, move on.
- **Quality trades are the user's call.** Any change that lowers image
  quality to gain speed, or any parameter chosen to fit a time budget, is
  measured where the scene is hardest and brought to the user before it is
  kept.
- **Reports are short:** what was built, the frame time before and after,
  any image change, anything the user must decide.

### The default brief for an agent

1. The goal, in a sentence or two.
2. Design order: architecture docs, headers with both corpora, code.
3. Build it, following production practice.
4. Measure the demo graph's frame time, before and after.
5. If the image changes: before/after frames side by side for the user.
6. Gates: existing graphs byte-identical where the image should not
   change; all three presets build with 0 warnings; ctest and `make
   check` green.
7. Report briefly. Don't push or merge.

Add anything beyond this only when the user asks for it.

## Where things are

Start with:

1. `docs/process/MEMORY.md`
2. `docs/process/QUEUE.md`
3. `docs/process/workflow.md`
4. `docs/process/governance/` for inherited governance (submodule),
   subordinate to the section above

Do not rely on chat history as project memory. Durable decisions belong in
`docs/process/`; investigations and lessons belong in `docs/research/`.

See `AGENTS.md` for the repo-specific C++ architecture and performance gates.
