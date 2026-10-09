# Thin task wrapper. Every target is a one-liner you could run by hand.
#
# Two configurations: native-debug (make test) and native-release
# (make test-release). Both run the same tests, including the GPU tests on
# this machine's GPU. There is no CI yet (docs/process/QUEUE.md).

.PHONY: test test-release run headless check toolchain clean

# Configure quietly: dependencies' status summaries (SDL prints its whole
# option list) are hidden; warnings and errors still show. VERBOSE=1 shows all.
CONFIGURE_QUIET := $(if $(VERBOSE),,--log-level=NOTICE)

## Every test, debug build: the core's tests and the GPU tests.
test:
	cmake --preset native-debug $(CONFIGURE_QUIET)
	cmake --build --preset native-debug
	ctest --preset native-debug

## The same tests, release build.
test-release:
	cmake --preset native-release $(CONFIGURE_QUIET)
	cmake --build --preset native-release
	ctest --preset native-release

## The window, release build, running the frame graph GRAPH (default: the
## test pattern) over the scene SCENE, if given (a graph that reads a scene
## needs one). Escape or closing the window ends it.
GRAPH ?= graphs/test_pattern.toml
SCENE ?=
SCENE_ARG := $(if $(SCENE),--scene $(SCENE),)
run:
	cmake --preset native-release $(CONFIGURE_QUIET)
	cmake --build --preset native-release --target serenity
	./build/native-release/serenity --graph $(GRAPH) $(SCENE_ARG)

## Headless frames of GRAPH, over SCENE if given, release build, as PNGs in
## OUT. FRAMES frames from frame 0, a sixtieth of a second apart.
OUT ?= frames
FRAMES ?= 60
headless:
	cmake --preset native-release $(CONFIGURE_QUIET)
	cmake --build --preset native-release --target serenity-headless
	./build/native-release/serenity-headless --graph $(GRAPH) $(SCENE_ARG) --out $(OUT) --frames $(FRAMES)

## Whether this machine's toolchain is the pinned one (cmake/toolchain.json).
## Configuring runs the same check and stops on a mismatch.
toolchain:
	./tools/check_toolchain.py

## Structural invariants, plus the tests proving each guard actually fires.
check:
	for f in scripts/*.sh tools/*.sh tests/*.sh; do bash -n "$$f" || exit 1; done
	./tools/check_boundaries.sh
	./tools/check_toolchain.py
	./tests/test_check_boundaries.sh
	./tests/test_check_toolchain.sh
	./tests/test_codex_review_preflight.sh

clean:
	rm -rf build
