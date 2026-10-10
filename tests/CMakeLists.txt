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

# The tests that need no GPU: the core, and both command lines.
add_executable(serenity_tests
    schedule_test.cpp
    graph_file_test.cpp
    scene_test.cpp
    thin_lens_test.cpp
    shapes_test.cpp
    animation_test.cpp
    history_test.cpp
    flight_test.cpp
    swarm_test.cpp
    coated_test.cpp
    frame_times_test.cpp
    png_test.cpp
    options_test.cpp
    programs_test.cpp
    main_test.cpp
)
target_link_libraries(serenity_tests PRIVATE
    serenity_app_options serenity_headless_options doctest::doctest)
target_compile_options(serenity_tests PRIVATE ${SERENITY_WARNINGS})
target_compile_definitions(serenity_tests PRIVATE
    SERENITY_GRAPHS_DIR="${PROJECT_SOURCE_DIR}/graphs"
    SERENITY_SCENES_DIR="${PROJECT_SOURCE_DIR}/scenes")
add_test(NAME serenity_tests COMMAND serenity_tests)

# The structural checks, so the default verification path (ctest, make test)
# runs them, not make check alone: the boundary the core keeps
# (tools/check_boundaries.sh), and the proofs that each of its rules and the
# toolchain check's fire. The probe tests write files into src/ for a moment,
# so none runs beside another that reads the tree.
add_test(NAME boundaries COMMAND sh "${PROJECT_SOURCE_DIR}/tools/check_boundaries.sh")
add_test(NAME boundaries_fire COMMAND sh "${PROJECT_SOURCE_DIR}/tests/test_check_boundaries.sh")
add_test(NAME toolchain_fires COMMAND sh "${PROJECT_SOURCE_DIR}/tests/test_check_toolchain.sh")
set_tests_properties(boundaries boundaries_fire PROPERTIES RESOURCE_LOCK source_tree)
