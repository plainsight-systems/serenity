# Architecture

The design, read in order:

1. [`logical-overview.md`](logical-overview.md): phases, responsibilities and
   principles: what the renderer does in a frame, and what it keeps between
   frames. Figure: [`logical-overview.svg`](logical-overview.svg).
2. [`change-axes.md`](change-axes.md): the reasons a file changes, the
   contracts between them, and the one-reason rule that places code.
3. `file-mapping.md`: the modules, the axis each changes on, and the contracts
   between them. Next.

Then the code: each file's header states its contract and its design.
Measurements are in [`../research/`](../research/README.md).
