# Architecture

The design, to be written in this order before any rendering code:

1. `logical-overview.md`: what the renderer does, as phases, responsibilities
   and principles. Names no files or types.
2. `change-axes.md`: the reasons a file changes, and the rule that places
   code: one translation unit, one reason to change.
3. `file-mapping.md`: the modules, the axis each changes on, and the
   contracts between them.

None is written yet (`../process/QUEUE.md`). Until then, the code holds two
boundaries, stated in `CMakeLists.txt` and enforced by
`tools/check_boundaries.sh`: `src/core/` uses no GPU or windowing API, and
only `src/metal/` uses Metal's.
