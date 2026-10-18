#!/bin/sh
# serenity-headless, run as its user runs it (src/headless/main.cpp,
# headless/options.h): an integration test, kept small. A good run writes
# the frames --write selects, under their names, and prints their paths; a
# command line or a graph it cannot run is refused before anything renders
# or is made; a frame that left samples out ends the run, written nowhere.
# Each failure: status 1, its message on stderr, nothing on stdout. What the
# names, the written frames and the refusals are is unit-tested in
# tests/options_test.cpp; what a PFM holds, in tests/gpu/headless_pfm_test.cpp.
#
# Usage: test_headless.sh SERENITY_HEADLESS SCRATCH_DIR
# SCRATCH_DIR is removed and made anew; the test writes nothing outside it,
# and runs graphs it writes there, not graphs/'s.
set -eu

[ "$#" -eq 2 ] || { echo "usage: $0 SERENITY_HEADLESS SCRATCH_DIR" >&2; exit 2; }
BIN=$1
SCRATCH=$2

fail() { echo "FAIL: $*" >&2; exit 1; }

rm -rf "${SCRATCH}"
mkdir -p "${SCRATCH}"

# The two graphs it runs: the test pattern, and the path tracer shown
# through the tone map.
GRAPHS="${SCRATCH}/graphs"
mkdir -p "${GRAPHS}"
printf 'passes = ["test_pattern"]\n' >"${GRAPHS}/test_pattern.toml"
printf 'passes = ["path", "tone_map"]\n[tone_map]\nexposure = 0.0\nbloom = 0.0\n' >"${GRAPHS}/path.toml"

# The names in directory $1, sorted, on one line.
names() { (cd "$1" && ls | tr '\n' ' ' | sed 's/ $//'); }

# A PNG's width and height, from its header (bytes 16 to 23, big-endian).
png_size() {
    od -An -tu1 -j16 -N8 "$1" |
        awk 'NF { printf "%dx%d", (($1 * 256 + $2) * 256 + $3) * 256 + $4, (($5 * 256 + $6) * 256 + $7) * 256 + $8 }'
}

# Five frames from frame 7, written doubling: the 1st, 2nd, 4th and the
# last (frames 7, 8, 10 and 11), each 16 x 8, their paths printed in order.
OUT="${SCRATCH}/doubling"
"${BIN}" --graph "${GRAPHS}/test_pattern.toml" --out "${OUT}" --size 16x8 --first 7 --frames 5 \
    --write doubling >"${SCRATCH}/stdout" 2>"${SCRATCH}/stderr" || fail "a good run exited with $?"
# Nothing on stderr but Metal's own notice that its validation is on.
if grep -v 'Metal [A-Z]* *Validation Enabled$' "${SCRATCH}/stderr" | grep -q .; then
    fail "a good run wrote to stderr: $(cat "${SCRATCH}/stderr")"
fi
EXPECTED="frame-000007.png frame-000008.png frame-000010.png frame-000011.png"
[ "$(names "${OUT}")" = "${EXPECTED}" ] || fail "doubling wrote '$(names "${OUT}")', not '${EXPECTED}'"
for n in 7 8 10 11; do printf '%s/frame-%06d.png\n' "${OUT}" "${n}"; done >"${SCRATCH}/expected"
cmp -s "${SCRATCH}/stdout" "${SCRATCH}/expected" || fail "doubling printed '$(cat "${SCRATCH}/stdout")'"
for f in "${OUT}"/*.png; do
    [ "$(png_size "${f}")" = "16x8" ] || fail "${f} is $(png_size "${f}"), not 16x8"
done

# Refused: status 1, a message naming the program on stderr, nothing on
# stdout, and nothing written.
expect_refusal() {
    description=$1
    shift
    status=0
    "${BIN}" "$@" >"${SCRATCH}/stdout" 2>"${SCRATCH}/stderr" || status=$?
    [ "${status}" -eq 1 ] || fail "${description}: status ${status}, not 1"
    grep -q '^serenity-headless: ' "${SCRATCH}/stderr" || fail "${description}: no message on stderr"
    [ ! -s "${SCRATCH}/stdout" ] || fail "${description}: printed '$(cat "${SCRATCH}/stdout")'"
}

expect_refusal "an unknown option" --graph "${GRAPHS}/test_pattern.toml" --out "${SCRATCH}/unknown" --fast
[ ! -e "${SCRATCH}/unknown" ] || fail "an unknown option made its output directory"

printf 'passes = ["no_such_pass"]\n' >"${SCRATCH}/broken.toml"
expect_refusal "a graph that does not read" --graph "${SCRATCH}/broken.toml" --out "${SCRATCH}/broken"
grep -q 'no_such_pass' "${SCRATCH}/stderr" || fail "a broken graph's message does not name the pass"
[ ! -e "${SCRATCH}/broken" ] || fail "a broken graph made its output directory"

# A frame that left samples out for not being finite fails the run once it
# has completed, naming the count, before it is written: a light at the
# most radiance a float holds inside a closed white sphere, whose light
# overflows (as tests/support/scene_text.h's blinding_lights()).
cat >"${SCRATCH}/blinding.toml" <<'SCENE'
[camera]
position = [0, 0, 0.8]
look_at = [0, 0, 1]
vertical_fov_degrees = 60
[environment]
kind = "gradient"
zenith = [0, 0, 0]
horizon = [0, 0, 0]
[materials.wall]
kind = "rough"
color = [1, 1, 1]
[materials.blinding]
kind = "emissive"
radiance = [3e38, 3e38, 3e38]
[[shapes]]
kind = "sphere"
center = [0, 0, 0]
radius = 1
material = "wall"
[[shapes]]
kind = "sphere"
center = [0, 0, 0]
radius = 0.5
material = "blinding"
SCENE
expect_refusal "samples not finite" --graph "${GRAPHS}/path.toml" --scene "${SCRATCH}/blinding.toml" \
    --out "${SCRATCH}/blinding" --size 16x16 --frames 2
grep -q 'frame 0: [1-9][0-9]* samples were not finite' "${SCRATCH}/stderr" ||
    fail "samples not finite: '$(cat "${SCRATCH}/stderr")'"
[ -z "$(names "${SCRATCH}/blinding")" ] || fail "a frame with samples not finite was written"

rm -rf "${SCRATCH}"
echo "serenity-headless: every check passed"
