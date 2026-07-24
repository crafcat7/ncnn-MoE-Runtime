cmake_minimum_required(VERSION 3.21)

project(ncnn_moe_runtime VERSION 0.1.0 LANGUAGES CXX)

option(NCNN_BUILD_MOE_RUNTIME "Build the ncnn MoE transformer runtime" ON)
option(NCNN_MOE_BUILD_TESTS "Build ncnn_moe tests" ON)
option(NCNN_MOE_BUILD_EXAMPLES "Build ncnn_moe examples" ON)
set(NCNN_MOE_USE_NCNN_DEFAULT ON)
option(NCNN_MOE_USE_NCNN "Use ncnn optimized CPU operators when available" ${NCNN_MOE_USE_NCNN_DEFAULT})
set(NCNN_MOE_USE_VULKAN_DEFAULT ON)
option(NCNN_MOE_USE_VULKAN "Build ncnn with Vulkan support" ${NCNN_MOE_USE_VULKAN_DEFAULT})
option(NCNN_MOE_ENABLE_EXPERIMENTAL_MACOS_VULKAN "Enable the experimental MoltenVK hybrid backend on macOS" ON)

if(NOT NCNN_BUILD_MOE_RUNTIME)
    return()
endif()

add_library(ncnn_moe STATIC
    src/cpu_attention.cpp
    src/cpu_executor.cpp
    src/cpu_mxfp4.cpp
    src/cpu_ops.cpp
    src/cpu_topology.cpp
    src/model_compiler.cpp
    src/ncnn_linear.cpp
    src/runtime.cpp
    src/safetensors.cpp
    src/scheduler.cpp
    src/session.cpp
    src/moe_adapter.cpp
)

add_library(ncnn::moe ALIAS ncnn_moe)
set_target_properties(ncnn_moe PROPERTIES EXPORT_NAME moe)

target_compile_features(ncnn_moe PUBLIC cxx_std_20)
target_include_directories(ncnn_moe
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

include(cmake/ncnn_moe_openmp.cmake)
set(NCNN_MOE_WITH_OPENMP OFF)
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(ncnn_moe PUBLIC OpenMP::OpenMP_CXX)
    set(NCNN_MOE_WITH_OPENMP ON)
    message(STATUS "ncnn_moe OpenMP expert parallelism enabled")
else()
    message(STATUS "ncnn_moe OpenMP unavailable; expert groups use one CPU thread")
endif()

set(NCNN_MOE_WITH_NCNN OFF)
set(NCNN_MOE_WITH_VULKAN OFF)
if(APPLE)
    set(NCNN_MOE_MOLTENVK_LIBRARY "" CACHE FILEPATH
        "Path to libMoltenVK.dylib used by ncnn_moe on macOS")
endif()
if(NCNN_MOE_USE_NCNN)
    if(APPLE AND NCNN_MOE_USE_VULKAN AND NCNN_MOE_ENABLE_EXPERIMENTAL_MACOS_VULKAN)
        message(STATUS
            "macOS Vulkan support uses MoltenVK. Build the bundled ncnn submodule "
            "with a local MoltenVK installation.")
        if(NOT NCNN_MOE_MOLTENVK_LIBRARY)
            set(_ncnn_moe_moltenvk_prefixes)
            if(DEFINED ENV{HOMEBREW_PREFIX})
                list(APPEND _ncnn_moe_moltenvk_prefixes "$ENV{HOMEBREW_PREFIX}/opt/molten-vk")
            endif()
            list(APPEND _ncnn_moe_moltenvk_prefixes
                "/opt/homebrew/opt/molten-vk"
                "/usr/local/opt/molten-vk"
            )
            find_library(_ncnn_moe_moltenvk_library
                NAMES MoltenVK
                HINTS ${_ncnn_moe_moltenvk_prefixes}
                PATH_SUFFIXES lib
                NO_DEFAULT_PATH
            )
            if(_ncnn_moe_moltenvk_library)
                set(NCNN_MOE_MOLTENVK_LIBRARY
                    "${_ncnn_moe_moltenvk_library}"
                    CACHE FILEPATH
                    "Path to libMoltenVK.dylib used by ncnn_moe on macOS"
                    FORCE
                )
            endif()
        endif()
        if(NOT NCNN_MOE_MOLTENVK_LIBRARY)
            message(FATAL_ERROR
                "macOS Vulkan requires libMoltenVK.dylib. Install MoltenVK or set "
                "-DNCNN_MOE_MOLTENVK_LIBRARY=/absolute/path/to/libMoltenVK.dylib.")
        endif()

        # ncnn simplevk uses this path for its static-link fallback on Apple platforms.
        set(Vulkan_LIBRARY "${NCNN_MOE_MOLTENVK_LIBRARY}" CACHE FILEPATH
            "Vulkan loader library used by bundled ncnn" FORCE)
    endif()
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/CMakeLists.txt")
        message(FATAL_ERROR
            "Bundled ncnn is unavailable. Initialize the submodule with "
            "'git submodule update --init --recursive'.")
    endif()

    set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "Build ncnn benchmark")
    set(NCNN_BUILD_TESTS OFF CACHE BOOL "Build ncnn tests")
    set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "Build ncnn examples")
    set(NCNN_BUILD_TOOLS OFF CACHE BOOL "Build ncnn tools")
    set(NCNN_VULKAN ${NCNN_MOE_USE_VULKAN} CACHE BOOL "Build ncnn with Vulkan support" FORCE)
    set(NCNN_BF16 ON CACHE BOOL "Build ncnn with bfloat16 support")
    add_subdirectory(third_party/ncnn EXCLUDE_FROM_ALL)
    if(TARGET ncnn)
        target_link_libraries(ncnn_moe PUBLIC ncnn)
        target_compile_definitions(ncnn_moe PRIVATE NCNN_MOE_USE_NCNN=1)
        set(NCNN_MOE_WITH_NCNN ON)
        if(NCNN_MOE_USE_VULKAN AND NOT NCNN_VULKAN)
            message(FATAL_ERROR
                "NCNN_MOE_USE_VULKAN is ON, but bundled ncnn was configured with "
                "NCNN_VULKAN=OFF. Reconfigure with -DNCNN_MOE_USE_VULKAN=ON."
            )
        endif()
        if(NCNN_MOE_USE_VULKAN AND NCNN_VULKAN
            AND (NOT APPLE OR NCNN_MOE_ENABLE_EXPERIMENTAL_MACOS_VULKAN))
            target_compile_definitions(ncnn_moe PRIVATE NCNN_MOE_WITH_VULKAN=1)
            set(NCNN_MOE_WITH_VULKAN ON)
            if(APPLE)
                target_compile_definitions(ncnn_moe PRIVATE
                    NCNN_MOE_MOLTENVK_LIBRARY_PATH="${NCNN_MOE_MOLTENVK_LIBRARY}"
                )
            endif()
        endif()
        if(APPLE AND NCNN_MOE_USE_VULKAN AND NCNN_VULKAN
            AND NOT NCNN_MOE_ENABLE_EXPERIMENTAL_MACOS_VULKAN)
            message(STATUS
                "The MoltenVK hybrid backend is disabled by configuration on macOS; "
                "set NCNN_MOE_ENABLE_EXPERIMENTAL_MACOS_VULKAN=ON to enable it.")
        endif()
    endif()
endif()

if(MSVC)
    target_compile_options(ncnn_moe PRIVATE /W4 /permissive-)
else()
    target_compile_options(ncnn_moe PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(NCNN_MOE_BUILD_TESTS)
    enable_testing()
    add_executable(ncnn_moe_tests tests/test_runtime.cpp)
    target_link_libraries(ncnn_moe_tests PRIVATE ncnn::moe)
    target_compile_features(ncnn_moe_tests PRIVATE cxx_std_20)
    target_include_directories(ncnn_moe_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(ncnn_moe_tests PRIVATE NCNN_MOE_WITH_NCNN=$<BOOL:${NCNN_MOE_WITH_NCNN}>)
    add_test(NAME ncnn_moe_tests COMMAND ncnn_moe_tests)
    add_test(
        NAME ncnn_moe_style
        COMMAND ${CMAKE_COMMAND}
            -DNCNN_MOE_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_style.cmake
    )
endif()

if(NCNN_MOE_BUILD_EXAMPLES)
    add_executable(ncnn_moe_tiny examples/run_tiny.cpp)
    target_link_libraries(ncnn_moe_tiny PRIVATE ncnn::moe)
    target_compile_features(ncnn_moe_tiny PRIVATE cxx_std_20)

    add_executable(ncnn_moe_gpt_oss examples/run_gpt_oss.cpp)
    target_link_libraries(ncnn_moe_gpt_oss PRIVATE ncnn::moe)
    target_compile_features(ncnn_moe_gpt_oss PRIVATE cxx_std_20)
endif()

if(NCNN_MOE_BUILD_TESTS AND NCNN_MOE_BUILD_EXAMPLES)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME ncnn_moe_gpt_oss_fixture
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/create_gpt_oss_fixture.py
                ${CMAKE_CURRENT_BINARY_DIR}/gpt_oss_fixture
                --verify-runner $<TARGET_FILE:ncnn_moe_gpt_oss>
        )
    endif()
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(NCNN_MOE_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/ncnn_moe")

configure_package_config_file(
    cmake/ncnn_moeConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/ncnn_moeConfig.cmake"
    INSTALL_DESTINATION "${NCNN_MOE_INSTALL_CMAKEDIR}"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ncnn_moeConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(TARGETS ncnn_moe
    EXPORT ncnn_moeTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT ncnn_moeTargets
    NAMESPACE ncnn::
    DESTINATION "${NCNN_MOE_INSTALL_CMAKEDIR}"
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ncnn_moeConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ncnn_moeConfigVersion.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ncnn_moe_openmp.cmake"
    DESTINATION "${NCNN_MOE_INSTALL_CMAKEDIR}"
)
