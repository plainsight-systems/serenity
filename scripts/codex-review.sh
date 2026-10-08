#!/usr/bin/env bash
# codex-review.sh — independent review of a commit.
#
# Usage:  ./scripts/codex-review.sh [--post] [--focus performance|architecture]
#                                   <commit> [output-file]
#
# Hand it a commit. It looks for two things only: where the code is wrong, and,
# by default, where it is slower than it needs to be; with --focus
# architecture, where it breaks the design in docs/architecture/ instead of
# where it is slow — the latter by walking the changed
# path and counting what it costs, grounded in the cpp-guidelines and
# cpp-performance MCP servers. Process, style and governance are out of scope.
#
# The review is JSON in the shape of the schema below, written to the output
# file (default .cache/reviews/<sha>.json). With --post it is also posted as
# comments on the commit on GitHub (scripts/post_commit_review.py): one holding
# the whole review, and one on each finding's line in the diff.
#
# Reviewer independence is the point: this session's author should not be the
# only one judging whether the commit's claims hold.

set -euo pipefail

cd "$(dirname "$0")/.."

# Pinned here rather than in ~/.codex/config.toml so this script's behavior does
# not drift with the user's interactive default. Override with CODEX_MODEL.
MODEL="${CODEX_MODEL:-gpt-5.6-sol}"
EFFORT="${CODEX_EFFORT:-high}"

# The MCP servers this review is required to consult. Names must match the
# server keys in ~/.codex/config.toml (cpp-guidelines, cpp-performance) or the
# reviewer will cite tools it never called.
GUIDELINES_URL="${CPP_GUIDELINES_URL:-http://127.0.0.1:7011}"
PERF_URL="${CPP_PERF_URL:-http://127.0.0.1:7015}"

# ----------------------------------------------------------------------
# Preflight. Every check below fails loudly: a review that silently skips
# its guideline grounding is worse than no review, because it produces an
# artifact that looks like one.
# ----------------------------------------------------------------------

die() { echo "codex-review.sh: $*" >&2; exit 1; }

command -v codex >/dev/null 2>&1 || die "'codex' CLI not found on PATH"

POST=0
FOCUS=performance
while [ $# -gt 0 ]; do
  case "$1" in
    --post) POST=1; shift ;;
    --focus)
      [ $# -ge 2 ] || die "--focus needs performance or architecture"
      case "$2" in performance|architecture) FOCUS="$2" ;; *) die "--focus is performance or architecture, not '$2'" ;; esac
      shift 2 ;;
    *) break ;;
  esac
done
COMMIT_ARG="${1:-}"
OUTPUT_ARG="${2:-}"
[ -n "${COMMIT_ARG}" ] || die "usage: $0 [--post] <commit> [output-file]"
COMMIT="$(git rev-parse --verify --quiet "${COMMIT_ARG}^{commit}")" \
  || die "not a commit: ${COMMIT_ARG}"

# Comments anchor to the commit on GitHub, so it must be there.
if [ "${POST}" = 1 ]; then
  command -v gh >/dev/null 2>&1 || die "'gh' CLI not found on PATH; it posts the comments"
  [ -n "$(git branch -r --contains "${COMMIT}" 2>/dev/null)" ] \
    || die "commit ${COMMIT:0:7} is not on any remote branch — push it before posting a review of it"
fi

# The MCP servers are a hard requirement of this review, not a nice-to-have.
# Checking here converts a silent mid-review skip into an upfront failure.
check_mcp() {
  name="$1"; url="$2"
  # Any HTTP status means the server is listening. A bare GET returns 406
  # because the MCP streamable transport wants its own Accept headers, so
  # `curl -f` must NOT be used here — it would reject a healthy server.
  #
  # Branch on curl's exit status, not on the body. An earlier version used
  # `... || echo 000` as a fallback, but on connection failure curl BOTH
  # prints 000 and exits non-zero, producing "000000" and silently passing
  # the check this function exists to enforce. Covered by
  # tests/test_codex_review_preflight.sh.
  if ! curl -s -o /dev/null --max-time 5 "${url}" 2>/dev/null; then
    die "MCP server '${name}' is not reachable at ${url}
  This review is required to be grounded in it. Start the server and retry,
  rather than running a review that cannot check what it claims to check."
  fi
}
check_mcp "cpp-guidelines" "${GUIDELINES_URL}"
check_mcp "cpp-performance" "${PERF_URL}"

OUTPUT="${OUTPUT_ARG:-.cache/reviews/${COMMIT}.json}"
[ -e "${OUTPUT}" ] && die "review already exists: ${OUTPUT}
  delete it or pass a different output path"

PROMPT_FILE="$(mktemp -t serenity-codex-prompt.XXXXXX)"
SCHEMA_FILE="$(mktemp -t serenity-codex-schema.XXXXXX)"
trap 'rm -f "${PROMPT_FILE}" "${SCHEMA_FILE}"' EXIT

# ----------------------------------------------------------------------
# Output schema. Every finding names a file and, where it has one, a line of
# the commit's new version, so it can be posted on that line.
# ----------------------------------------------------------------------

cat > "${SCHEMA_FILE}" <<'SCHEMAEOF'
{
  "type": "object",
  "additionalProperties": false,
  "required": ["outcome", "summary", "findings", "cost_walk", "mcp_grounding"],
  "properties": {
    "outcome": {"type": "string", "enum": ["approved", "approved_with_notes", "changes_requested"]},
    "summary": {"type": "string"},
    "findings": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["severity", "kind", "path", "line", "title", "body", "guidelines"],
        "properties": {
          "severity": {"type": "string", "enum": ["P0", "P1", "P2", "P3"]},
          "kind": {"type": "string", "enum": ["correctness", "performance", "architecture"]},
          "path": {"type": "string"},
          "line": {"type": ["integer", "null"]},
          "title": {"type": "string"},
          "body": {"type": "string"},
          "guidelines": {"type": "array", "items": {"type": "string"}}
        }
      }
    },
    "cost_walk": {"type": "string"},
    "mcp_grounding": {"type": "string"}
  }
}
SCHEMAEOF

# ----------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------

{
  cat <<'PROMPTEOF'
You are reviewing one commit in the Serenity repository. You are the second
reader: the author believes the change is correct and well designed. Find
where it is not. Two questions only:

PROMPTEOF
  if [ "${FOCUS}" = architecture ]; then
    cat <<'PROMPTEOF'
  1. Is it correct?
  2. Does it hold to the architecture in docs/architecture/?

PROMPTEOF
  else
    cat <<'PROMPTEOF'
  1. Is it correct?
  2. Could it be faster, and is every performance claim it makes true?

PROMPTEOF
  fi
  cat <<'PROMPTEOF'
PROJECT CONTEXT
---------------
Serenity is a real-time path tracer: a scene lit by many moving lights,
sampled with resampled importance sampling (ReSTIR), then denoised. C++20,
native, on macOS through Metal (metal-cpp and Metal shaders compiled at build
time); a Vulkan backend for an AMD machine comes later. src/core/ is
platform-neutral and uses no GPU API; only src/metal/ uses Metal's host API.
There is one target: an Apple M3 Max,
on the toolchain pinned in cmake/toolchain.json.

Design lives in file headers; each optimization is labelled "Optimization:"
with its reason and, where measured, its figures.

CORRECTNESS
-----------
Bugs, undefined behavior, overflow, lifetimes and dangling spans, off-by-one,
alignment, unchecked error paths that let a failure pass as success, state
left inconsistent after a failure, Metal rules the code breaks (object
ownership and autorelease, resource hazards, synchronization between CPU and
GPU), and claims in comments or the commit message that the code does not honor. Read
the code the change depends on, not just the diff. A test that cannot fail for
the bug it names is a finding; a request for more tests in general is not.

PROMPTEOF
  if [ "${FOCUS}" = architecture ]; then
    cat <<'PROMPTEOF'
ARCHITECTURE — AGAINST THE DESIGN AS WRITTEN
--------------------------------------------
Read docs/architecture/logical-overview.md, change-axes.md and
file-mapping.md first; they are the standard, not your taste. Then, for every
file the commit adds or changes:

  - Name the one axis of change-axes.md it changes on. A file that changes
    for two of them is a finding: say which two, and name a change that
    would touch one and not the other.
  - Check the dependency direction: the app and the headless renderer on the
    backend and the core, the backend on the core, the core on nothing
    platform-specific. Any include or call against that direction is a
    finding.
  - Check that the core decides what a frame computes and the backend only
    how (principle 10): a backend choosing what to render, or the core
    reaching into a backend, is a finding.
  - Check the contracts: each has one owner; a layout shared with shaders is
    defined once; a family depends on a contract, not on another family.
  - Check the principles the commit touches, for example: nothing that
    renders reads a clock or other ambient state (1); values are data, not
    code (change-axes.md); failure is visible, never a silent fallback or a
    success that did not happen.
  - Check that every claim a file header makes about its design is true of
    the code under it.

Put the axis you named for each file in "cost_walk", one line each.

PROMPTEOF
  else
    cat <<'PROMPTEOF'
PERFORMANCE — WALK THE PATH AND COUNT
-------------------------------------
For each hot path the commit touches (a frame: ray generation, traversal,
shading, light sampling, reservoir reuse, denoising, presentation), walk it
stage by stage and write down, as functions of the resolution, the light
count and the scene size: the bytes each stage moves and the copies it makes;
the operations in each inner loop and whether each is a call the compiler
cannot inline; and the crossings between CPU and GPU — command buffers,
encoders, dispatches, synchronizations and readbacks per frame. A count that
scales with pixels, lights or primitives where it could scale with tiles or
passes is a finding, with the count and the cheaper design. Compare each
stage with the ceiling the same bytes would reach at the GPU's memory
bandwidth. If the commit states measured figures, check they are consistent
with your counts. Put the walk in "cost_walk".

Setup code that runs once at start-up is not hot; say so and move on.

PROMPTEOF
  fi
  cat <<'PROMPTEOF'
OUT OF SCOPE — DO NOT RAISE
---------------------------
Process, governance, change classification, commit hygiene, documentation
style, naming, guideline citation bookkeeping, architecture taste that the
documents in docs/architecture/ do not state, requests for device matrices,
budgets, baselines, benchmarks before landing, or more tests in general. A
finding must change what the program computes, how fast it does it, or (with
the architecture focus) where the design says code belongs. If the commit is
sound, say so briefly; do not pad.

GUIDELINES
----------
Two MCP servers are available: cpp-guidelines and cpp-performance
(search_guidelines / get_guideline). When a finding rests on a rule, look it
up and cite its ID; do not cite from memory. If a server is unavailable or a
call is cancelled, say so in mcp_grounding.

SEVERITY
--------
  P0  wrong results, memory corruption, undefined behavior, or a crash on a
      path the renderer runs
  P1  wrong in a reachable case; or a hot path whose cost scales with the
      wrong unit; or a stated measurement or performance claim that is false;
      or a dependency against the design's direction, or a backend deciding
      what is computed
  P2  a latent bug that needs an unlikely input; or a missed optimization
      with a stated, material gain; or a file on two axes, or a header whose
      design claim the code does not honor
  P3  minor

OUTPUT
------
Answer in the JSON schema you were given:

  outcome        approved | approved_with_notes | changes_requested
  summary        what the commit does, and whether it is correct and fast
  findings       each with severity, kind (correctness | performance |
                 architecture), the
                 file path from the repository root, the line in the commit's
                 version of that file (null for the file as a whole), a short
                 title, a body that stands on its own (what is wrong, the
                 input or count that shows it, the fix), and guideline IDs
  cost_walk      with the performance focus, the per-stage counts for the
                 hot paths touched, or one line saying none is touched; with
                 the architecture focus, each file and its axis
  mcp_grounding  which servers and tools you called, and any failure

Write the text fields in Markdown.
PROMPTEOF

  printf '\n=== COMMIT UNDER REVIEW: %s ===\n' "${COMMIT}"
  git show --stat --format='%H%n%an <%ae>%n%ad%n%n%B' "${COMMIT}"

  printf '\n=== END COMMIT SUMMARY ===\n\n'
  printf 'Repository root is the current working directory. Run git show %s for\n' "${COMMIT}"
  printf 'the full diff, and read whatever source, tests and build files you need.\n'
} > "${PROMPT_FILE}"

# ----------------------------------------------------------------------
# Invoke codex
# ----------------------------------------------------------------------

# '-a on-request' is REQUIRED: the cpp-guidelines / cpp-perf-guidelines MCP
# tools are approval-gated in ~/.codex/config.toml. Under a bare 'codex exec'
# those calls are cancelled and the review proceeds having read no guideline
# at all. Do not drop this flag.
echo "codex-review.sh: reviewing ${COMMIT:0:7} (correctness and ${FOCUS})" >&2
echo "  model:  ${MODEL} (effort ${EFFORT})" >&2
echo "  output: ${OUTPUT}" >&2

mkdir -p "$(dirname "${OUTPUT}")"

codex -a on-request exec \
  -m "${MODEL}" \
  -c model_reasoning_effort="\"${EFFORT}\"" \
  --output-schema "${SCHEMA_FILE}" \
  -o "${OUTPUT}" \
  - < "${PROMPT_FILE}" || {
  RC=$?
  echo "codex-review.sh: codex exited non-zero (${RC})" >&2
  exit "${RC}"
}

[ -s "${OUTPUT}" ] || die "codex produced an empty review — not keeping it"
python3 -c 'import json, sys; json.load(open(sys.argv[1]))' "${OUTPUT}" \
  || die "codex's review is not JSON — kept at ${OUTPUT} for inspection, not posted"

echo "codex-review.sh: review written to ${OUTPUT}" >&2

if [ "${POST}" = 1 ]; then
  python3 scripts/post_commit_review.py "${OUTPUT}" "${COMMIT}"
fi
