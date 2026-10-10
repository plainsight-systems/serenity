#!/bin/sh
# serenity-headless, run as its user runs it (src/headless/main.cpp,
# headless/options.h): which frames it writes, under which names, what it
# prints, the size of what it writes; and that a failure ends the run with
# its message on stderr, status 1 and nothing written.
#
# Usage: test_headless.sh SERENITY_HEADLESS GRAPHS_DIR SCRATCH_DIR
# SCRATCH_DIR is removed and made anew; the test writes nothing outside it.
set -eu

[ "$#" -eq 3 ] || { echo "usage: $0 SERENITY_HEADLESS GRAPHS_DIR SCRATCH_DIR" >&2; exit 2; }
BIN=$1
GRAPHS=$2
SCRATCH=$3

fail() { echo "FAIL: $*" >&2; exit 1; }

rm -rf "${SCRATCH}"
mkdir -p "${SCRATCH}"

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

# The same frames again: the same bytes (principle 1).
AGAIN="${SCRATCH}/again"
"${BIN}" --graph "${GRAPHS}/test_pattern.toml" --out "${AGAIN}" --size 16x8 --first 7 --frames 5 \
    --write doubling >/dev/null || fail "the second run exited with $?"
for n in 7 8 10 11; do
    name=$(printf 'frame-%06d.png' "${n}")
    cmp -s "${OUT}/${name}" "${AGAIN}/${name}" || fail "${name} differs between two runs of one command"
done

# The last only, of three.
LAST="${SCRATCH}/last"
"${BIN}" --graph "${GRAPHS}/test_pattern.toml" --out "${LAST}" --size 4x4 --frames 3 --write last >/dev/null ||
    fail "--write last exited with $?"
[ "$(names "${LAST}")" = "frame-000002.png" ] || fail "--write last wrote '$(names "${LAST}")'"

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
# has completed, naming the count, before it is written: two lights at the
# most radiance a float holds, close over a floor, whose light overflows
# (as tests/support/scene_text.h's blinding_lights()).
cat >"${SCRATCH}/blinding.toml" <<'SCENE'
[camera]
position = [0, 1, 2]
look_at = [0, 0, 0]
vertical_fov_degrees = 60
[environment]
kind = "gradient"
zenith = [0, 0, 0]
horizon = [0, 0, 0]
[materials.floor]
kind = "rough"
color = [0.9, 0.9, 0.9]
[materials.blinding]
kind = "emissive"
radiance = [3e38, 3e38, 3e38]
[[shapes]]
kind = "box"
min = [-5, -1, -5]
max = [5, 0, 5]
material = "floor"
[[shapes]]
kind = "sphere"
center = [0, 0.3, 0]
radius = 0.25
material = "blinding"
[[shapes]]
kind = "sphere"
center = [0.6, 0.3, 0]
radius = 0.25
material = "blinding"
SCENE
expect_refusal "samples not finite" --graph "${GRAPHS}/path.toml" --scene "${SCRATCH}/blinding.toml" \
    --out "${SCRATCH}/blinding" --size 16x16 --frames 2
grep -q 'frame 0: [1-9][0-9]* samples were not finite' "${SCRATCH}/stderr" ||
    fail "samples not finite: '$(cat "${SCRATCH}/stderr")'"
[ -z "$(names "${SCRATCH}/blinding")" ] || fail "a frame with samples not finite was written"

# An output directory holding an earlier run's frame: refused, the frame
# left as it was.
expect_refusal "a directory holding a frame" --graph "${GRAPHS}/test_pattern.toml" --out "${OUT}" --size 16x8
[ "$(names "${OUT}")" = "${EXPECTED}" ] || fail "a refused run changed what its directory held"

rm -rf "${SCRATCH}"
echo "serenity-headless: every check passed"
