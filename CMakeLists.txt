cmake_minimum_required(VERSION 3.21)

project(ncnn_moe_runtime VERSION 0.1.0 LANGUAGES CXX)

option(NCNN_BUILD_MOE_RUNTIME "Build the ncnn MoE transformer runtime" ON)
option(NCNN_MOE_BUILD_TESTS "Build ncnn_moe tests" ON)
option(NCNN_MOE_BUILD_EXAMPLES "Build ncnn_moe examples" ON)
option(
    NCNN_MOE_BUILD_REFERENCE_RUNNERS
    "Build model-specific token-ID reference runners used by benchmarks and CTest"
    OFF
)
option(
    NCNN_MOE_BUILD_MXFP4_BENCHMARK
    "Build the standalone MXFP4 microbenchmark"
    OFF
)
option(
    NCNN_MOE_ENABLE_MSVC_MP
    "Enable MSVC multiprocessor compilation"
    ON
)
set(NCNN_MOE_USE_NCNN_DEFAULT ON)
option(NCNN_MOE_USE_NCNN "Use ncnn optimized CPU operators when available" ${NCNN_MOE_USE_NCNN_DEFAULT})
set(NCNN_MOE_USE_VULKAN_DEFAULT ON)
option(NCNN_MOE_USE_VULKAN "Build ncnn with Vulkan support" ${NCNN_MOE_USE_VULKAN_DEFAULT})
option(NCNN_MOE_ENABLE_EXPERIMENTAL_MACOS_VULKAN "Enable the experimental MoltenVK hybrid backend on macOS" ON)
include(GNUInstallDirs)

if(MSVC AND NCNN_MOE_ENABLE_MSVC_MP)
    add_compile_options(/MP)
endif()

if(NOT NCNN_BUILD_MOE_RUNTIME)
    return()
endif()

set(NCNN_MOE_SOURCES
    src/backends/ncnn/ncnn_linear.cpp
    src/compiler/graph_builder.cpp
    src/compiler/moe_graph.cpp
    src/engine/cpu_executor.cpp
    src/engine/cpu_features.cpp
    src/engine/cpu_task_worker.cpp
    src/engine/cpu_thread_budget.cpp
    src/engine/cpu_topology.cpp
    src/engine/expert_backend.cpp
    src/engine/memory_manager.cpp
    src/engine/runtime.cpp
    src/engine/scheduler.cpp
    src/engine/session.cpp
    src/graph/execution_graph.cpp
    src/graph/expert_dispatcher.cpp
    src/graph/memory_planner.cpp
    src/graph/model_compiler.cpp
    src/kernels/cpu_attention.cpp
    src/kernels/cpu_bfloat16.cpp
    src/kernels/cpu_float8.cpp
    src/kernels/cpu_gated_delta_net.cpp
    src/kernels/cpu_hyper_connection.cpp
    src/kernels/cpu_latent_attention.cpp
    src/kernels/cpu_mxfp4.cpp
    src/kernels/cpu_qnk.cpp
    src/kernels/cpu_ops.cpp
    src/kernels/cpu_state_cache.cpp
    src/kernels/cpu_vector.cpp
    src/models/builtin_model_adapter.cpp
    src/models/deepseek_v4_model_adapter.cpp
    src/models/qwen3_5_moe_model_adapter.cpp
    src/models/safetensors.cpp
    src/storage/expert_cache.cpp
    src/storage/expert_store.cpp
    src/storage/expert_victim_cache.cpp
    src/storage/mapped_file.cpp
    src/storage/system_memory.cpp
    src/storage/weight_store.cpp
)
add_library(ncnn_moe STATIC ${NCNN_MOE_SOURCES})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" FILES ${NCNN_MOE_SOURCES})

if(MSVC AND CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "x64")
    set(NCNN_MOE_MSVC_SIMD_SOURCES
        src/kernels/cpu_float8_msvc_avx2.cpp
        src/kernels/cpu_float8_msvc_avx512.cpp
        src/kernels/cpu_mxfp4_msvc_avx2.cpp
        src/kernels/cpu_mxfp4_msvc_avx512.cpp
        src/kernels/cpu_qnk_msvc_avx2.cpp
        src/kernels/cpu_qnk_msvc_avx512.cpp
        src/kernels/cpu_vector_msvc_avx2.cpp
        src/kernels/cpu_vector_msvc_avx512.cpp
    )
    target_sources(ncnn_moe PRIVATE ${NCNN_MOE_MSVC_SIMD_SOURCES})
    source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" FILES ${NCNN_MOE_MSVC_SIMD_SOURCES})
    set_source_files_properties(
        src/kernels/cpu_float8_msvc_avx2.cpp
        src/kernels/cpu_mxfp4_msvc_avx2.cpp
        src/kernels/cpu_qnk_msvc_avx2.cpp
        src/kernels/cpu_vector_msvc_avx2.cpp
        PROPERTIES COMPILE_OPTIONS "/arch:AVX2"
    )
    set_source_files_properties(
        src/kernels/cpu_float8_msvc_avx512.cpp
        src/kernels/cpu_mxfp4_msvc_avx512.cpp
        src/kernels/cpu_qnk_msvc_avx512.cpp
        src/kernels/cpu_vector_msvc_avx512.cpp
        PROPERTIES COMPILE_OPTIONS "/arch:AVX512"
    )
    target_compile_definitions(ncnn_moe PRIVATE NCNN_MOE_MSVC_X86_SIMD=1)
endif()

if(
    CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$"
    AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"
)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(
        "-march=armv8.2-a+sve2"
        NCNN_MOE_COMPILER_SUPPORTS_SVE2
    )
    if(NCNN_MOE_COMPILER_SUPPORTS_SVE2)
        target_sources(
            ncnn_moe
            PRIVATE src/kernels/cpu_mxfp4_sve2.cpp
        )
        set_source_files_properties(
            src/kernels/cpu_mxfp4_sve2.cpp
            PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+sve2"
        )
        target_compile_definitions(
            ncnn_moe
            PRIVATE NCNN_MOE_ARM_SVE2_KERNEL=1
        )
    endif()
endif()

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

if(NCNN_MOE_WITH_VULKAN)
    set(NCNN_MOE_VULKAN_SHADER_SOURCES
        src/kernels/vulkan/attention_decode_sdpa.comp
        src/kernels/vulkan/attention_output_gate.comp
        src/kernels/vulkan/attention_qkv_norm_rope.comp
        src/kernels/vulkan/attention_qkv_rope.comp
        src/kernels/vulkan/attention_ring_append.comp
        src/kernels/vulkan/attention_ring_zero.comp
        src/kernels/vulkan/bfloat16_cooperative_projection.comp
        src/kernels/vulkan/bfloat16_projection.comp
        src/kernels/vulkan/bfloat16_rms_norm_projection.comp
        src/kernels/vulkan/bfloat16_swiglu_down.comp
        src/kernels/vulkan/float8_projection.comp
        src/kernels/vulkan/float8_projection_tiled4.comp
        src/kernels/vulkan/float8_quantize.comp
        src/kernels/vulkan/float8_rms_norm_quantize.comp
        src/kernels/vulkan/float8_swiglu_quantize.comp
        src/kernels/vulkan/gated_delta_net.comp
        src/kernels/vulkan/mxfp4_gate_up.comp
        src/kernels/vulkan/mxfp4_indexed.comp
        src/kernels/vulkan/mxfp4_projection.comp
        src/kernels/vulkan/mxfp4_route_aggregation.comp
    )
    set(NCNN_MOE_VULKAN_SHADER_HEADERS)
    foreach(NCNN_MOE_SHADER_SOURCE IN LISTS NCNN_MOE_VULKAN_SHADER_SOURCES)
        get_filename_component(
            NCNN_MOE_SHADER_NAME
            "${NCNN_MOE_SHADER_SOURCE}"
            NAME_WE)
        set(NCNN_MOE_SHADER_HEADER
            "${CMAKE_CURRENT_BINARY_DIR}/kernels/vulkan/${NCNN_MOE_SHADER_NAME}.comp.hex.h")
        add_custom_command(
            OUTPUT "${NCNN_MOE_SHADER_HEADER}"
            COMMAND ${CMAKE_COMMAND}
                -DSHADER_SRC=${CMAKE_CURRENT_SOURCE_DIR}/${NCNN_MOE_SHADER_SOURCE}
                -DSHADER_COMP_HEADER=${NCNN_MOE_SHADER_HEADER}
                -DSHADER_VARIABLE=${NCNN_MOE_SHADER_NAME}_shader
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ncnn_moe_generate_shader_comp_header.cmake"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/${NCNN_MOE_SHADER_SOURCE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ncnn_moe_generate_shader_comp_header.cmake"
            COMMENT "Preprocessing MoE Vulkan shader ${NCNN_MOE_SHADER_NAME}.comp"
            VERBATIM
        )
        set_source_files_properties("${NCNN_MOE_SHADER_HEADER}" PROPERTIES GENERATED TRUE)
        list(APPEND NCNN_MOE_VULKAN_SHADER_HEADERS "${NCNN_MOE_SHADER_HEADER}")
    endforeach()
    set_source_files_properties(
        ${NCNN_MOE_VULKAN_SHADER_SOURCES}
        PROPERTIES HEADER_FILE_ONLY TRUE)
    target_sources(
        ncnn_moe
        PRIVATE
            ${NCNN_MOE_VULKAN_SHADER_SOURCES}
            ${NCNN_MOE_VULKAN_SHADER_HEADERS})
    target_include_directories(ncnn_moe PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    add_custom_target(
        ncnn-moe-generate-vulkan-shaders
        DEPENDS ${NCNN_MOE_VULKAN_SHADER_HEADERS})
    add_dependencies(ncnn_moe ncnn-moe-generate-vulkan-shaders)
    source_group(
        TREE "${CMAKE_CURRENT_SOURCE_DIR}/src"
        FILES ${NCNN_MOE_VULKAN_SHADER_SOURCES})
endif()

if(MSVC)
    target_compile_options(ncnn_moe PRIVATE /W4 /permissive-)
else()
    target_compile_options(ncnn_moe PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(NCNN_MOE_BUILD_TESTS)
    enable_testing()
    add_executable(ncnn_moe_tests
        tests/fixture_model_adapter.cpp
        tests/test_runtime.cpp
    )
    target_link_libraries(ncnn_moe_tests PRIVATE ncnn::moe)
    target_compile_features(ncnn_moe_tests PRIVATE cxx_std_20)
    target_include_directories(ncnn_moe_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(ncnn_moe_tests PRIVATE NCNN_MOE_WITH_NCNN=$<BOOL:${NCNN_MOE_WITH_NCNN}>)
    add_test(NAME ncnn_moe_tests COMMAND ncnn_moe_tests)
    if(NCNN_MOE_WITH_VULKAN)
        add_test(
            NAME ncnn_moe_pipeline_bind_elision_disabled
            COMMAND ncnn_moe_tests --disable-vulkan-pipeline-bind-elision
        )
    endif()
    add_test(
        NAME ncnn_moe_style
        COMMAND ${CMAKE_COMMAND}
            -DNCNN_MOE_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_style.cmake
    )
endif()

if(NCNN_MOE_BUILD_EXAMPLES)
    if(NCNN_MOE_BUILD_REFERENCE_RUNNERS)
        add_library(
            ncnn_moe_reference_runner
            STATIC
            examples/internal/run_model.cpp)
        target_link_libraries(
            ncnn_moe_reference_runner
            PUBLIC ncnn::moe)
        target_compile_features(
            ncnn_moe_reference_runner
            PRIVATE cxx_std_20)

        add_executable(
            ncnn_moe_gpt_oss
            examples/run_gpt_oss.cpp)
        target_link_libraries(
            ncnn_moe_gpt_oss
            PRIVATE ncnn_moe_reference_runner)
        target_compile_features(
            ncnn_moe_gpt_oss
            PRIVATE cxx_std_20)

        add_executable(
            ncnn_moe_deepseek_v4
            examples/run_deepseek_v4.cpp)
        target_link_libraries(
            ncnn_moe_deepseek_v4
            PRIVATE ncnn_moe_reference_runner)
        target_compile_features(
            ncnn_moe_deepseek_v4
            PRIVATE cxx_std_20)

        add_executable(
            ncnn_moe_qwen3_6
            examples/run_qwen3_6.cpp)
        target_link_libraries(
            ncnn_moe_qwen3_6
            PRIVATE ncnn_moe_reference_runner)
        target_compile_features(
            ncnn_moe_qwen3_6
            PRIVATE cxx_std_20)
    endif()

    add_executable(
        ncnn_moe_worker
        examples/ncnn_moe_worker.cpp
        examples/internal/gpu_telemetry.cpp)
    target_link_libraries(
        ncnn_moe_worker
        PRIVATE ncnn::moe)
    target_compile_features(
        ncnn_moe_worker
        PRIVATE cxx_std_20)
    if(WIN32)
        target_link_libraries(ncnn_moe_worker PRIVATE psapi)
    endif()
    if(CMAKE_DL_LIBS)
        target_link_libraries(ncnn_moe_worker PRIVATE ${CMAKE_DL_LIBS})
    endif()
    install(TARGETS ncnn_moe_worker
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    if(NCNN_MOE_BUILD_MXFP4_BENCHMARK)
        add_executable(
            ncnn_moe_mxfp4_bench
            examples/benchmark_mxfp4.cpp)
        target_link_libraries(
            ncnn_moe_mxfp4_bench
            PRIVATE ncnn::moe)
        target_compile_features(
            ncnn_moe_mxfp4_bench
            PRIVATE cxx_std_20)
        target_include_directories(
            ncnn_moe_mxfp4_bench
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    endif()
endif()

if(NCNN_MOE_BUILD_TESTS AND NCNN_MOE_BUILD_EXAMPLES)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        if(NCNN_MOE_BUILD_REFERENCE_RUNNERS)
            add_test(
                NAME ncnn_moe_gpt_oss_fixture
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/create_gpt_oss_fixture.py
                    ${CMAKE_CURRENT_BINARY_DIR}/gpt_oss_fixture
                    --verify-runner $<TARGET_FILE:ncnn_moe_gpt_oss>
            )
        else()
            add_test(
                NAME ncnn_moe_gpt_oss_fixture
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/create_gpt_oss_fixture.py
                    ${CMAKE_CURRENT_BINARY_DIR}/gpt_oss_fixture
            )
        endif()
        add_test(
            NAME ncnn_moe_python_fixture
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/test_ncnn_moe.py
                --worker $<TARGET_FILE:ncnn_moe_worker>
                --model ${CMAKE_CURRENT_BINARY_DIR}/gpt_oss_fixture
        )
        set_tests_properties(
            ncnn_moe_python_fixture
            PROPERTIES DEPENDS ncnn_moe_gpt_oss_fixture
        )
        add_test(
            NAME ncnn_moe_python_auto_fixture
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/test_ncnn_moe.py
                --auto
                --worker $<TARGET_FILE:ncnn_moe_worker>
                --model ${CMAKE_CURRENT_BINARY_DIR}/gpt_oss_fixture
        )
        set_tests_properties(
            ncnn_moe_python_auto_fixture
            PROPERTIES DEPENDS ncnn_moe_gpt_oss_fixture
        )
    endif()
endif()

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
