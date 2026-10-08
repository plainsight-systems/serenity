# Thin task wrapper. Every target is a one-liner you could run by hand.
#
# Two configurations: native-debug (make test) and native-release
# (make test-release). Both run the same tests, including the GPU tests on
# this machine's GPU. There is no CI yet (docs/process/QUEUE.md).

.PHONY: test test-release check toolchain clean

## Every test, debug build: the core's tests and the GPU tests.
test:
	cmake --preset native-debug
	cmake --build --preset native-debug
	ctest --preset native-debug

## The same tests, release build.
test-release:
	cmake --preset native-release
	cmake --build --preset native-release
	ctest --preset native-release

## Whether this machine's toolchain is the pinned one (cmake/toolchain.json).
## Configuring runs the same check and stops on a mismatch.
toolchain:
	./tools/check_toolchain.py

## Structural invariants, plus the tests proving each guard actually fires.
check:
	./tools/check_boundaries.sh
	./tools/check_toolchain.py
	./tests/test_check_boundaries.sh
	./tests/test_check_toolchain.sh
	./tests/test_codex_review_preflight.sh

clean:
	rm -rf build
