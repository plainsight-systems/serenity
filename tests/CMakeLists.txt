# Pinned by commit: dependency resolution must be deterministic, and a
# floating ref would make the test build a moving target.
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        2d0a9359a60c51affe2a9bebb1be1dca47868151  # v2.5.3
)
FetchContent_MakeAvailable(doctest)
get_target_property(doctest_includes doctest::doctest INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(doctest PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${doctest_includes}")

# What every test executable shares: the tests' own helpers (tests/support,
# included as "support/..."), where the repository's scenes and graphs are
# (generated, support/test_paths.h.in), doctest, and the warnings. The tests
# are held to more than the project's warnings: conversions that change a
# value or its sign, names that hide others, and C-style casts fail their
# build (P.12, ES.46, ES.49, ES.12).
configure_file(support/test_paths.h.in "${CMAKE_CURRENT_BINARY_DIR}/generated/test_paths.h" @ONLY)
# And where the programs the tests run are (support/programs.h.in), from the
# targets themselves; a test executable that runs one depends on it, below.
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/programs.h" INPUT support/programs.h.in)
add_library(serenity_test_support INTERFACE)
target_include_directories(serenity_test_support INTERFACE
    "${PROJECT_SOURCE_DIR}/tests" "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(serenity_test_support INTERFACE doctest::doctest)
target_compile_options(serenity_test_support INTERFACE
    ${SERENITY_WARNINGS} -Wconversion -Wsign-conversion -Wshadow -Wold-style-cast)

# The tests that need no GPU: unit tests of the core, and of the three
# programs' command lines and of what they do with them; measure_test also
# runs serenity-measure once, as its user does, an integration test.
add_executable(serenity_tests
    schedule_test.cpp
    frame_refusals_test.cpp
    unet_test.cpp
    structure_plan_test.cpp
    denoise_core_test.cpp
    graph_file_test.cpp
    restir_pt_core_test.cpp
    checkerboard_test.cpp
    scene_test.cpp
    packed_range_test.cpp
    thin_lens_test.cpp
    depth_of_field_test.cpp
    jitter_test.cpp
    shapes_test.cpp
    animation_test.cpp
    history_test.cpp
    flight_test.cpp
    swarm_test.cpp
    coated_test.cpp
    frame_times_test.cpp
    png_test.cpp
    image_format_test.cpp
    linear_image_test.cpp
    pfm_test.cpp
    reference_test.cpp
    error_test.cpp
    measure_test.cpp
    options_test.cpp
    programs_test.cpp
    main_test.cpp
)
target_link_libraries(serenity_tests PRIVATE
    serenity_app_options serenity_headless_options serenity_measure_options serenity_test_support)
# measure_test runs serenity-measure as its user does (support/programs.h).
add_dependencies(serenity_tests serenity-measure)
# stb_image, to decode what png_test writes; the same pinned stb as the
# writer's (CMakeLists.txt), test-only.
target_include_directories(serenity_tests SYSTEM PRIVATE "${stb_SOURCE_DIR}")
add_test(NAME serenity_tests COMMAND serenity_tests)

# The structural checks, so the default verification path (ctest, make test)
# runs them, not make check alone: the boundary the core keeps
# (tools/check_boundaries.sh), and the proofs that each of its rules and the
# toolchain check's fire. The probe tests write files into src/ for a moment,
# so none runs beside another that reads the tree. The review script's
# preflight test (tests/test_codex_review_preflight.sh) is not here, nor in
# make check: it is our review workflow's, run by make review-check, since
# it needs the codex CLI on PATH, which nothing here builds or pins, and
# should its guard regress it starts a model call before its watchdog stops
# it, which a test run must never do.
add_test(NAME boundaries COMMAND sh "${PROJECT_SOURCE_DIR}/tools/check_boundaries.sh")
add_test(NAME boundaries_fire COMMAND sh "${PROJECT_SOURCE_DIR}/tests/test_check_boundaries.sh")
add_test(NAME toolchain_fires COMMAND sh "${PROJECT_SOURCE_DIR}/tests/test_check_toolchain.sh")
set_tests_properties(boundaries boundaries_fire PROPERTIES RESOURCE_LOCK source_tree)

# The Python tests run under the interpreter the configure found and gave a
# floor (CMakeLists.txt, find_package(Python3)), not whichever python3 is
# first on PATH, so the one the build checked is the one whose draws the
# byte pins below hold: random.Random guarantees its sequence across
# versions for random() alone, so another Python fails them.

# The marbles' scene generator as its user runs it: the real scene byte for
# byte by default, and generated scenes reproducible by seed.
add_test(NAME make_marbles COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_make_marbles.py")

# The denoiser data set's plan: the same seeds, times and splits every run,
# pinned; the real marbles only the test.
add_test(NAME make_denoiser_dataset
         COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_make_denoiser_dataset.py")

# The headless renderer as its user runs it, an integration test: a good
# run's frames and what it prints, and its failures' status and messages.
# On the GPU, under Metal's validation as the GPU tests are (tests/gpu).
# On the platforms whose backend builds it (CMakeLists.txt).
if(TARGET serenity-headless)
    add_test(NAME headless
        COMMAND sh "${PROJECT_SOURCE_DIR}/tests/test_headless.sh" $<TARGET_FILE:serenity-headless>
                "${CMAKE_CURRENT_BINARY_DIR}/headless_test")
    set_tests_properties(headless PROPERTIES
        ENVIRONMENT "MTL_DEBUG_LAYER=1;MTL_SHADER_VALIDATION=1;MTL_DEBUG_LAYER_ERROR_MODE=assert")
endif()
