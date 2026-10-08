#!/bin/sh
# Proves codex-review.sh refuses to run when a required MCP server is down.
#
# This guard shipped broken once: the liveness check produced "000000" on a
# dead port and passed. An untested guard is not a guard.
#
# Each invocation runs under a watchdog. If the guard regresses, the script
# sails past preflight and starts a real multi-minute model call — so a
# regression must fail fast here rather than burn one.
set -eu

cd "$(dirname "$0")/.."
SCRIPT=./scripts/codex-review.sh
COMMIT=HEAD
DEAD=http://127.0.0.1:59997
WATCHDOG_SECS=20

OUT="$(mktemp -t serenity-preflight-out.XXXXXX)"; rm -f "${OUT}"
ERR="$(mktemp -t serenity-preflight-err.XXXXXX)"
trap 'rm -f "${OUT}" "${ERR}"' EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

# Runs the script with one env override, under a watchdog. Echoes its exit
# status, or 'TIMEOUT' if the watchdog had to kill it.
run_guarded() {
    var="$1"
    env "${var}=${DEAD}" "${SCRIPT}" "${COMMIT}" "${OUT}" >/dev/null 2>"${ERR}" &
    pid=$!
    waited=0
    while kill -0 "${pid}" 2>/dev/null; do
        if [ "${waited}" -ge "${WATCHDOG_SECS}" ]; then
            kill -9 "${pid}" 2>/dev/null || true
            wait "${pid}" 2>/dev/null || true
            echo TIMEOUT
            return
        fi
        sleep 1
        waited=$((waited + 1))
    done
    wait "${pid}" 2>/dev/null && echo 0 || echo $?
}

for var in CPP_GUIDELINES_URL CPP_PERF_URL; do
    status="$(run_guarded "${var}")"
    case "${status}" in
        TIMEOUT) fail "${var} unreachable: script did not exit within ${WATCHDOG_SECS}s.
  The preflight guard is broken — it let a dead server through and began a
  real model call." ;;
        0) fail "${var} unreachable: script exited 0; the guard did not fire" ;;
    esac
    [ -e "${OUT}" ] && fail "${var}: a review file was written despite preflight failure"
    # The error must name the server, so the operator knows what to start.
    grep -q 'not reachable' "${ERR}" || fail "${var}: error did not report unreachability: $(cat "${ERR}")"
done

echo "codex-review preflight: OK (refuses to run without its MCP servers)"
