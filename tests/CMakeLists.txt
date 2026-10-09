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
    pinhole_test.cpp
    shapes_test.cpp
    animation_test.cpp
    frame_times_test.cpp
    png_test.cpp
    options_test.cpp
    main_test.cpp
)
target_link_libraries(serenity_tests PRIVATE
    serenity_app_options serenity_headless_options doctest::doctest)
target_compile_options(serenity_tests PRIVATE ${SERENITY_WARNINGS})
target_compile_definitions(serenity_tests PRIVATE
    SERENITY_GRAPHS_DIR="${PROJECT_SOURCE_DIR}/graphs"
    SERENITY_SCENES_DIR="${PROJECT_SOURCE_DIR}/scenes")
add_test(NAME serenity_tests COMMAND serenity_tests)
