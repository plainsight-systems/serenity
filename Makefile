# Thin task wrapper. Every target is a one-liner you could run by hand.
#
# Three configurations, each running the same tests, including the GPU
# tests on this machine's GPU: native-debug (make test), native-release
# (make test-release), and native-sanitize (make test-sanitize), a debug
# build under UndefinedBehaviorSanitizer. AddressSanitizer is not among
# them: its runtime hangs before main on this machine's macOS
# (docs/research/2026-10-10-sanitizers.md). There is no CI yet
# (docs/process/QUEUE.md).

.PHONY: test test-release test-sanitize run headless movie reference convergence check toolchain clean

# Configure quietly: dependencies' status summaries (SDL prints its whole
# option list) are hidden; warnings and errors still show. VERBOSE=1 shows all.
CONFIGURE_QUIET := $(if $(VERBOSE),,--log-level=NOTICE)

## Every test, debug build: the core's tests, the GPU tests, and the
## boundary and toolchain checks with the proofs that they fire.
test:
	cmake --preset native-debug $(CONFIGURE_QUIET)
	cmake --build --preset native-debug
	ctest --preset native-debug

## The same tests, release build.
test-release:
	cmake --preset native-release $(CONFIGURE_QUIET)
	cmake --build --preset native-release
	ctest --preset native-release

## The same tests, debug build under UndefinedBehaviorSanitizer: an overflow
## of a signed integer, a shift past the width, a misaligned or null access,
## an index past an array's bound or a value outside its enum or bool
## stops the test that made it (CMakePresets.json, native-sanitize;
## cmake/ubsan_ignorelist.txt names the one third-party idiom let pass).
test-sanitize:
	cmake --preset native-sanitize $(CONFIGURE_QUIET)
	cmake --build --preset native-sanitize
	ctest --preset native-sanitize

## The window, release build, running the frame graph GRAPH (default: the
## test pattern) over the scene SCENE, if given (a graph that reads a scene
## needs one), rendered at SCALE of the window's pixels if given (0.5 is a
## quarter of the pixels). Escape or closing the window ends it.
GRAPH ?= graphs/test_pattern.toml
SCENE ?=
SCENE_ARG := $(if $(SCENE),--scene $(SCENE),)
SCALE ?=
SCALE_ARG := $(if $(SCALE),--scale $(SCALE),)
run:
	cmake --preset native-release $(CONFIGURE_QUIET)
	cmake --build --preset native-release --target serenity
	./build/native-release/serenity --graph $(GRAPH) $(SCENE_ARG) $(SCALE_ARG)

## Headless frames of GRAPH, over SCENE if given, release build, as PNGs in
## OUT. FRAMES frames from frame 0, a sixtieth of a second apart. The frames
## of an earlier run in OUT are removed first; the renderer refuses an OUT
## that still holds anything else (headless/options.h).
OUT ?= frames
FRAMES ?= 60
headless:
	cmake --preset native-release $(CONFIGURE_QUIET)
	cmake --build --preset native-release --target serenity-headless
	rm -f $(OUT)/frame-*.png
	./build/native-release/serenity-headless --graph $(GRAPH) $(SCENE_ARG) --out $(OUT) --frames $(FRAMES)

## A movie of GRAPH over SCENE, for sharing: SECONDS seconds at FPS frames a
## second, SIZE pixels, each frame SAMPLES renders of its instant
## (headless/options.h), rendered headless as PNGs (the lossless frames, which
## are what is measured) and encoded with ffmpeg as H.264 into media/, named
## by date, scene and graph. ffmpeg is a tool for sharing, outside the build
## and not pinned: brew install ffmpeg.
SECONDS ?= 10
FPS ?= 60
SIZE ?= 1920x1080
SAMPLES ?= 64
MOVIE_FRAMES := build/movie-frames
MOVIE ?= media/$(shell date +%Y-%m-%d)-$(basename $(notdir $(or $(SCENE),none)))-$(basename $(notdir $(GRAPH))).mp4
movie:
	@command -v ffmpeg >/dev/null || { echo "make movie needs ffmpeg: brew install ffmpeg" >&2; exit 1; }
	cmake --preset native-release $(CONFIGURE_QUIET)
	cmake --build --preset native-release --target serenity-headless
	rm -rf $(MOVIE_FRAMES) && mkdir -p media
	./build/native-release/serenity-headless --graph $(GRAPH) $(SCENE_ARG) --out $(MOVIE_FRAMES) \
		--frames $$(( $(SECONDS) * $(FPS) )) --step $$(awk 'BEGIN { printf "%.17g", 1 / $(FPS) }') --size $(SIZE) \
		--samples $(SAMPLES) >/dev/null
	ffmpeg -hide_banner -loglevel error -y -framerate $(FPS) -i $(MOVIE_FRAMES)/frame-%06d.png \
		-vf "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p" \
		-colorspace bt709 -color_primaries bt709 -color_trc bt709 \
		-c:v libx264 -preset slow -crf 16 -movflags +faststart $(MOVIE)
	@echo $(MOVIE)

## A reference image of SCENE, time frozen at TIME, SIZE pixels, release
## build (docs/research/2026-10-10-reference.md, "How it is run"): BATCHES
## batches of BATCH_SAMPLES samples of graphs/path.toml, batch k its own
## headless run, --first k --frames 1, so its samples are frame indices
## k x N .. k x N + N - 1 and no two batches share one. Each is written as a
## PFM into a scratch directory beside the reference, folded in double by
## serenity-measure into REFERENCE, and the scratch directory removed; the
## reference's own error, its floors (serenity-measure's stdout), goes beside
## it as .txt. A reference is never replaced: remove it to make it again.
## The defaults, 64 x 1024 samples at 1920 x 1080, take some ten minutes of
## GPU, scaling with pixels and samples; see
## docs/research/2026-10-10-reference.md. For example:
## make reference SCENE=scenes/marbles.toml TIME=120
TIME ?=
BATCHES ?= 64
BATCH_SAMPLES ?= 1024
REFERENCES := .cache/references
REFERENCE_NAME = $(basename $(notdir $(SCENE)))-t$(TIME)-$(SIZE)-$(BATCHES)x$(BATCH_SAMPLES)
REFERENCE = $(REFERENCES)/$(REFERENCE_NAME).pfm
REFERENCE_FLOORS = $(REFERENCES)/$(REFERENCE_NAME).txt
REFERENCE_BATCHES = $(REFERENCES)/$(REFERENCE_NAME).batches
reference:
	@[ -n "$(SCENE)" ] && [ -n "$(TIME)" ] || { echo "make reference needs SCENE and TIME" >&2; exit 1; }
	@[ ! -e $(REFERENCE) ] || { echo "$(REFERENCE) exists, and a reference is never replaced: remove it to make it again" >&2; exit 1; }
	cmake --preset native-release $(CONFIGURE_QUIET)
	cmake --build --preset native-release --target serenity-headless serenity-measure
	rm -rf $(REFERENCE_BATCHES) $(REFERENCE_FLOORS)
	k=0; while [ $$k -lt $(BATCHES) ]; do \
		./build/native-release/serenity-headless --graph graphs/path.toml --scene $(SCENE) \
			--out $(REFERENCE_BATCHES)/$$k --size $(SIZE) --first $$k --frames 1 --samples $(BATCH_SAMPLES) \
			--time $(TIME) --format pfm || exit 1; \
		k=$$((k + 1)); \
	done
	./build/native-release/serenity-measure reference --out $(REFERENCE) \
		$$(k=0; while [ $$k -lt $(BATCHES) ]; do \
			printf '%s/%d/frame-%06d.pfm\n' $(REFERENCE_BATCHES) $$k $$k; k=$$((k + 1)); done) \
		>$(REFERENCE_FLOORS) || { rm -f $(REFERENCE_FLOORS); exit 1; }
	rm -rf $(REFERENCE_BATCHES)
	@cat $(REFERENCE_FLOORS)
	@echo $(REFERENCE)

## The naive path tracer's convergence against the reference above (made
## first, by make reference with the same SCENE, TIME, SIZE, BATCHES and
## BATCH_SAMPLES): CONVERGENCE_FRAMES frames of CONVERGENCE_SAMPLES samples
## of graphs/path.toml at TIME, from the first frame index whose samples
## follow every one of the reference's, so none is any of them and their
## noise is independent (core/measurement/error.h). Written at doubling
## frames as PFMs (--write doubling), each measured against the reference by
## serenity-measure, into a CSV beside it: samples, image, MSE and relative
## MSE, a row a written frame. The frames are removed once measured.
CONVERGENCE_FRAMES ?= 16384
CONVERGENCE_SAMPLES ?= 1
CONVERGENCE_NAME = $(REFERENCE_NAME)-convergence-$(CONVERGENCE_FRAMES)x$(CONVERGENCE_SAMPLES)
CONVERGENCE_FRAMES_DIR = $(REFERENCES)/$(CONVERGENCE_NAME).frames
CONVERGENCE = $(REFERENCES)/$(CONVERGENCE_NAME).csv
convergence:
	@[ -n "$(SCENE)" ] && [ -n "$(TIME)" ] || { echo "make convergence needs SCENE and TIME" >&2; exit 1; }
	@[ -f $(REFERENCE) ] || { echo "$(REFERENCE) is not there: make reference first, with the same SCENE, TIME, SIZE, BATCHES and BATCH_SAMPLES" >&2; exit 1; }
	cmake --preset native-release $(CONFIGURE_QUIET)
	cmake --build --preset native-release --target serenity-headless serenity-measure
	rm -rf $(CONVERGENCE_FRAMES_DIR) $(CONVERGENCE)
	first=$$(( ($(BATCHES) * $(BATCH_SAMPLES) + $(CONVERGENCE_SAMPLES) - 1) / $(CONVERGENCE_SAMPLES) )) && \
	frames=$$(./build/native-release/serenity-headless --graph graphs/path.toml --scene $(SCENE) \
		--out $(CONVERGENCE_FRAMES_DIR) --size $(SIZE) --first $$first --frames $(CONVERGENCE_FRAMES) \
		--samples $(CONVERGENCE_SAMPLES) --time $(TIME) --write doubling --format pfm) && \
	errors=$$(./build/native-release/serenity-measure error --reference $(REFERENCE) $$frames) && \
	printf '%s\n' "$$errors" | awk -F, -v first=$$first -v m=$(CONVERGENCE_SAMPLES) \
		'NR == 1 { print "samples," $$0; next } \
		 match($$0, /frame-[0-9]+\.pfm/) { frame = substr($$0, RSTART + 6, RLENGTH - 10) + 0; \
			printf "%d,%s\n", (frame - first + 1) * m, $$0; next } \
		 { print "unmatched row: " $$0 > "/dev/stderr"; exit 1 }' >$(CONVERGENCE) || { rm -f $(CONVERGENCE); exit 1; }
	rm -rf $(CONVERGENCE_FRAMES_DIR)
	@cat $(CONVERGENCE)
	@echo $(CONVERGENCE)

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
