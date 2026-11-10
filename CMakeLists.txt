cmake_minimum_required(VERSION 3.21)

project(ncnn_moe_runtime VERSION 0.1.0 LANGUAGES CXX)

# ==============================================================================
# Build options
# ==============================================================================
# NCNN_MOE_USE_* requests backends; NCNN_MOE_WITH_* records capabilities.
option(NCNN_BUILD_MOE_RUNTIME "Build the ncnn MoE transformer runtime" ON)
option(NCNN_MOE_BUILD_TESTS "Build ncnn_moe tests" ON)
option(NCNN_MOE_BUILD_EXAMPLES "Build ncnn_moe examples" ON)
option(NCNN_MOE_ENABLE_MSVC_MP "Enable MSVC multiprocessor compilation" ON)
option(NCNN_MOE_ENABLE_LTO "Enable Release LTO (smaller worker, much larger static .lib)" OFF)
option(NCNN_MOE_SIZE_OPTIMIZE_COLD_PATHS "Compile load/import/tokenizer paths for size in Release" ON)
set(NCNN_MOE_OPT_LEVEL "O3" CACHE STRING "Release optimization level: O0, O1, O2, O3, or Os")
# Optional CPU ISA extensions. Disable these for portable/minimal builds; the
# runtime selector falls back to scalar kernels when they are off.
option(NCNN_MOE_ENABLE_X86_SIMD "Build optional AVX2/AVX512 x86 kernels" ON)
option(NCNN_MOE_ENABLE_ARM_SVE2 "Build optional ARM SVE2 MXFP4 kernels" ON)
option(NCNN_MOE_ENABLE_OPENMP "Enable OpenMP CPU parallelism when the toolchain provides it" ON)
set(NCNN_MOE_USE_NCNN_DEFAULT ON)
option(NCNN_MOE_USE_NCNN "Use ncnn optimized CPU operators when available" ${NCNN_MOE_USE_NCNN_DEFAULT})
set(NCNN_MOE_USE_VULKAN_DEFAULT ON)
option(NCNN_MOE_USE_VULKAN "Build ncnn with Vulkan support" ${NCNN_MOE_USE_VULKAN_DEFAULT})

string(TOUPPER "${NCNN_MOE_OPT_LEVEL}" _NCNN_MOE_OPT_LEVEL)
if(_NCNN_MOE_OPT_LEVEL MATCHES "^[0123]$")
    set(NCNN_MOE_OPT_LEVEL "O${_NCNN_MOE_OPT_LEVEL}")
elseif(_NCNN_MOE_OPT_LEVEL STREQUAL "S")
    set(NCNN_MOE_OPT_LEVEL "Os")
elseif(_NCNN_MOE_OPT_LEVEL MATCHES "^O[0123]$")
    set(NCNN_MOE_OPT_LEVEL "${_NCNN_MOE_OPT_LEVEL}")
elseif(_NCNN_MOE_OPT_LEVEL STREQUAL "OS")
    set(NCNN_MOE_OPT_LEVEL "Os")
else()
    message(FATAL_ERROR "NCNN_MOE_OPT_LEVEL must be O0, O1, O2, O3, or Os (got: ${NCNN_MOE_OPT_LEVEL})")
endif()
set(NCNN_MOE_OPT_LEVEL "${NCNN_MOE_OPT_LEVEL}" CACHE STRING "Release optimization level: O0, O1, O2, O3, or Os" FORCE)
set_property(CACHE NCNN_MOE_OPT_LEVEL PROPERTY STRINGS O0 O1 O2 O3 Os)

include(GNUInstallDirs)

# Keep Release artifacts small without changing -O level: one section per
# function/data plus LTO lets the linker drop unreferenced operator paths.
function(ncnn_moe_optimize_target target)
    if(NOT TARGET ${target})
        return()
    endif()
    if(MSVC)
        if(NCNN_MOE_OPT_LEVEL STREQUAL "O0")
            target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:/Od>)
        elseif(NCNN_MOE_OPT_LEVEL STREQUAL "O1")
            target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:/O1>)
        elseif(NCNN_MOE_OPT_LEVEL STREQUAL "O2")
            target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:/O2>)
        elseif(NCNN_MOE_OPT_LEVEL STREQUAL "O3")
            # /Ob3 and /Ot are the closest MSVC equivalent of -O3.
            target_compile_options(${target} PRIVATE
                $<$<CONFIG:Release>:/O2>
                $<$<CONFIG:Release>:/Ob3>
                $<$<CONFIG:Release>:/Ot>)
        else()
            target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:/O1>)
        endif()
        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Release>:/Gy>
            $<$<CONFIG:Release>:/Gw>)
        target_link_options(${target} PRIVATE
            $<$<CONFIG:Release>:/OPT:REF>
            $<$<CONFIG:Release>:/OPT:ICF>)
    else()
        if(NCNN_MOE_OPT_LEVEL STREQUAL "O0")
            target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:-O0>)
        elseif(NCNN_MOE_OPT_LEVEL STREQUAL "O1")
            target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:-O1>)
        elseif(NCNN_MOE_OPT_LEVEL STREQUAL "O2")
            target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:-O2>)
        elseif(NCNN_MOE_OPT_LEVEL STREQUAL "O3")
            target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:-O3>)
        else()
            target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:-Os>)
        endif()
        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Release>:-ffunction-sections>
            $<$<CONFIG:Release>:-fdata-sections>)
        if(APPLE)
            target_link_options(${target} PRIVATE $<$<CONFIG:Release>:-Wl,-dead_strip>)
        elseif(UNIX)
            target_link_options(${target} PRIVATE $<$<CONFIG:Release>:-Wl,--gc-sections>)
        endif()
    endif()
    if(NCNN_MOE_ENABLE_LTO)
        if(NOT DEFINED NCNN_MOE_IPO_SUPPORTED)
            include(CheckIPOSupported)
            check_ipo_supported(RESULT NCNN_MOE_IPO_SUPPORTED
                OUTPUT NCNN_MOE_IPO_ERROR LANGUAGES CXX)
            set(NCNN_MOE_IPO_SUPPORTED ${NCNN_MOE_IPO_SUPPORTED} CACHE INTERNAL
                "ncnn_moe toolchain LTO support")
        endif()
        if(NCNN_MOE_IPO_SUPPORTED)
            set_property(TARGET ${target} PROPERTY
                INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
        endif()
    endif()
endfunction()

if(MSVC)
    # Set CRT and link policy before creating runtime/ncnn targets.
    include(third_party/Mile.Windows.UniCrt/Mile.Windows.UniCrt.cmake)
endif()

if(MSVC AND NCNN_MOE_ENABLE_MSVC_MP)
    add_compile_options(/MP)
endif()

if(NOT NCNN_BUILD_MOE_RUNTIME)
    return()
endif()

# ==============================================================================
# Runtime library
# ==============================================================================
# Backend files also provide CPU-only stubs.
set(NCNN_MOE_SOURCES
    # ncnn backends
    src/backends/ncnn/attention_vulkan.cpp
    src/backends/ncnn/attentiondecode_vulkan.cpp
    src/backends/ncnn/expertbackend_vulkan.cpp
    src/backends/ncnn/experttransfer_vulkan.cpp
    src/backends/ncnn/graphallocator_vulkan.cpp
    src/backends/ncnn/gateddeltanet_vulkan.cpp
    src/backends/ncnn/gatedresidual_vulkan.cpp
    src/backends/ncnn/gatedresiduallayer_vulkan.cpp
    src/backends/ncnn/gpusegment_vulkan.cpp
    src/backends/ncnn/expertbank_vulkan.cpp
    src/backends/ncnn/sharedffn_vulkan.cpp
    src/backends/ncnn/qsaattention_vulkan.cpp
    src/backends/ncnn/hyperconnection_vulkan.cpp
    src/backends/ncnn/rmsnorm_vulkan.cpp
    src/backends/ncnn/layernorm_vulkan.cpp
    src/backends/ncnn/router_vulkan.cpp
    src/backends/ncnn/layerhead_vulkan.cpp
    src/backends/ncnn/latentlayer_vulkan.cpp
    src/backends/ncnn/moecombine_vulkan.cpp
    src/backends/ncnn/residentffn_vulkan.cpp
    src/backends/ncnn/latentattention_vulkan.cpp
    src/backends/ncnn/linear.cpp
    src/backends/ncnn/linear_bf16.cpp
    src/backends/ncnn/linear_expert.cpp
    src/backends/ncnn/linear_float8.cpp
    src/backends/ncnn/linear_mxfp4.cpp
    src/backends/ncnn/linear_qnk.cpp
    src/backends/ncnn/modelpipeline.cpp
    src/backends/ncnn/vulkancontext.cpp
    # Execution
    src/engine/cpu.cpp
    src/engine/executor.cpp
    src/engine/executor_speculative.cpp
    src/engine/expert.cpp
    src/engine/expertbackend.cpp
    src/engine/modelloader.cpp
    src/engine/runtime.cpp
    src/engine/scheduler.cpp
    src/engine/session.cpp
    # Graph
    src/graph/compiler.cpp
    src/graph/graph.cpp
    src/graph/graphmemory.cpp
    src/graph/memoryplan.cpp
    src/graph/resourceplan.cpp
    src/engine/graphresources.cpp
    src/graph/router.cpp
    # Kernels
    src/kernels/attention.cpp
    src/kernels/bfloat16.cpp
    src/kernels/float8.cpp
    src/kernels/gateddeltanet.cpp
    src/kernels/gatedresidual.cpp
    src/kernels/hyperconnection.cpp
    src/kernels/latentattention.cpp
    src/kernels/mxfp4.cpp
    src/kernels/ops.cpp
    src/kernels/ple.cpp
    src/kernels/qnk.cpp
    src/kernels/int4.cpp
    src/kernels/statecache.cpp
    src/kernels/vector.cpp
    # Models and package I/O
    src/models/modeladapter.cpp
    src/models/modeladapter_gptoss.cpp
    src/models/modeladapter_deepseekv4.cpp
    src/models/modeladapter_qwen3.cpp
    src/models/modeladapter_qwen3_5.cpp
    src/models/modeladapter_qwen4exp.cpp
    src/models/modeladapter_phimoe.cpp
    src/models/modeladapter_ling.cpp
    src/models/safetensors.cpp
    src/models/gguf.cpp
    src/models/ggufmodel.cpp
    src/models/ggufmodel_gptoss.cpp
    src/models/ggufmodel_qwen.cpp
    src/models/ggufmodel_qwen3.cpp
    src/models/tokenizer.cpp
    # Storage
    src/storage/expertcache.cpp
    src/storage/expertresidency.cpp
    src/storage/expertcache_victim.cpp
    src/storage/mappedfile.cpp
    src/storage/weightstore.cpp
)
add_library(ncnn_moe STATIC ${NCNN_MOE_SOURCES})
ncnn_moe_optimize_target(ncnn_moe)

# Loading, importing and tokenization are cold paths. Optimize them for size
# while keeping kernels and execution at the target's Release optimization.
if(NCNN_MOE_SIZE_OPTIMIZE_COLD_PATHS AND NOT NCNN_MOE_OPT_LEVEL STREQUAL "O0")
    set(NCNN_MOE_COLD_SOURCES
        src/engine/modelloader.cpp
        src/engine/runtime.cpp
        src/graph/compiler.cpp
        src/graph/graphmemory.cpp
        src/graph/memoryplan.cpp
        src/graph/resourceplan.cpp
        src/models/gguf.cpp
        src/models/ggufmodel.cpp
        src/models/ggufmodel_gptoss.cpp
        src/models/ggufmodel_qwen.cpp
        src/models/ggufmodel_qwen3.cpp
        src/models/modeladapter.cpp
        src/models/modeladapter_deepseekv4.cpp
        src/models/modeladapter_gptoss.cpp
        src/models/modeladapter_ling.cpp
        src/models/modeladapter_phimoe.cpp
        src/models/modeladapter_qwen3.cpp
        src/models/modeladapter_qwen3_5.cpp
        src/models/modeladapter_qwen4exp.cpp
        src/models/safetensors.cpp
        src/models/tokenizer.cpp
        src/storage/mappedfile.cpp
        src/storage/weightstore.cpp
    )
    if(MSVC)
        set(_ncnn_moe_cold_size_option "$<$<CONFIG:Release>:/Os>")
    else()
        set(_ncnn_moe_cold_size_option "$<$<CONFIG:Release>:-Os>")
    endif()
    set_property(SOURCE ${NCNN_MOE_COLD_SOURCES} APPEND PROPERTY
        COMPILE_OPTIONS ${_ncnn_moe_cold_size_option})
endif()
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" FILES ${NCNN_MOE_SOURCES})

if(MSVC)
    # Preserve UTF-8 tokenizer literals.
    set_property(SOURCE src/models/tokenizer.cpp tests/test_tokenizer.cpp APPEND PROPERTY COMPILE_OPTIONS /utf-8)
endif()
install(FILES src/models/unicode.LICENSE
    DESTINATION ${CMAKE_INSTALL_DATADIR}/ncnn-moe-runtime/licenses)

if(APPLE)
    # Static SDK consumers also need the Accelerate link dependency.
    target_link_libraries(ncnn_moe PRIVATE "-framework Accelerate")
endif()

# ==============================================================================
# Architecture-specific compilation
# ==============================================================================
# Keep ISA flags local so baseline code remains portable.
if(NCNN_MOE_ENABLE_X86_SIMD AND MSVC AND CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "x64")
    set(NCNN_MOE_MSVC_SIMD_SOURCES
        src/kernels/float8_msvc_avx2.cpp
        src/kernels/float8_msvc_avx512.cpp
        src/kernels/mxfp4_msvc_avx2.cpp
        src/kernels/mxfp4_msvc_avx512.cpp
        src/kernels/qnk_msvc_avx2.cpp
        src/kernels/qnk_msvc_avx512.cpp
        src/kernels/vector_x86_avx2.cpp
        src/kernels/vector_x86_avx512.cpp
    )
    target_sources(ncnn_moe PRIVATE ${NCNN_MOE_MSVC_SIMD_SOURCES})
    source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" FILES ${NCNN_MOE_MSVC_SIMD_SOURCES})
    set_source_files_properties(
        src/kernels/float8_msvc_avx2.cpp
        src/kernels/mxfp4_msvc_avx2.cpp
        src/kernels/qnk_msvc_avx2.cpp
        src/kernels/vector_x86_avx2.cpp
        PROPERTIES COMPILE_OPTIONS "/arch:AVX2"
    )
    set_source_files_properties(
        src/kernels/float8_msvc_avx512.cpp
        src/kernels/mxfp4_msvc_avx512.cpp
        src/kernels/qnk_msvc_avx512.cpp
        src/kernels/vector_x86_avx512.cpp
        PROPERTIES COMPILE_OPTIONS "/arch:AVX512"
    )
    target_compile_definitions(ncnn_moe PRIVATE NCNN_MOE_MSVC_X86_SIMD=1 NCNN_MOE_X86_SIMD=1)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(NCNN_MOE_TARGET_HAS_X86 OFF)
    if(APPLE AND CMAKE_OSX_ARCHITECTURES)
        # Universal builds may include x86 on ARM hosts.
        foreach(NCNN_MOE_ARCH IN LISTS CMAKE_OSX_ARCHITECTURES)
            if(NCNN_MOE_ARCH MATCHES "^(x86_64|AMD64|amd64|X86|x86|x64|i[3-6]86)$")
                set(NCNN_MOE_TARGET_HAS_X86 ON)
            endif()
        endforeach()
    elseif(
        CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|X86|x86|x64|i[3-6]86)$"
        OR CMAKE_CXX_COMPILER_ARCHITECTURE_ID MATCHES "^(x86_64|AMD64|amd64|X86|x86|x64|i[3-6]86)$"
    )
        set(NCNN_MOE_TARGET_HAS_X86 ON)
    endif()

    if(NCNN_MOE_TARGET_HAS_X86 AND NCNN_MOE_ENABLE_X86_SIMD)
        set(NCNN_MOE_X86_SIMD_SOURCES
            src/kernels/vector_x86_avx2.cpp
            src/kernels/vector_x86_avx512.cpp
        )
        target_sources(ncnn_moe PRIVATE ${NCNN_MOE_X86_SIMD_SOURCES})
        source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" FILES ${NCNN_MOE_X86_SIMD_SOURCES})
        target_compile_definitions(ncnn_moe PRIVATE NCNN_MOE_X86_SIMD=1)
    endif()
endif()
# Compiler support does not imply runtime CPU support.
if(
    NCNN_MOE_ENABLE_ARM_SVE2
    AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$"
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
            PRIVATE src/kernels/mxfp4_sve2.cpp
        )
        set_source_files_properties(
            src/kernels/mxfp4_sve2.cpp
            PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+sve2"
        )
        target_compile_definitions(
            ncnn_moe
            PRIVATE NCNN_MOE_ARM_SVE2_KERNEL=1
        )
    endif()
endif()

# ==============================================================================
# Public target interface
# ==============================================================================
# Expose only public headers through ncnn::moe.
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

# --- OpenMP dependency --------------------------------------------------------
# Discover OpenMP before ncnn so both share the imported target.
set(NCNN_MOE_WITH_OPENMP OFF)
if(NCNN_MOE_ENABLE_OPENMP)
    include(cmake/ncnn_moe_openmp.cmake)
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(ncnn_moe PUBLIC OpenMP::OpenMP_CXX)
        set(NCNN_MOE_WITH_OPENMP ON)
        message(STATUS "ncnn_moe OpenMP expert parallelism enabled")
    else()
        message(STATUS "ncnn_moe OpenMP unavailable; expert groups use one CPU thread")
    endif()
else()
    message(STATUS "ncnn_moe OpenMP disabled by NCNN_MOE_ENABLE_OPENMP=OFF")
endif()

# ==============================================================================
# Bundled ncnn and Vulkan configuration
# ==============================================================================

set(NCNN_MOE_WITH_NCNN OFF)
set(NCNN_MOE_WITH_VULKAN OFF)
if(APPLE)
    set(NCNN_MOE_MOLTENVK_LIBRARY "" CACHE FILEPATH
        "Path to libMoltenVK.dylib used by ncnn_moe on macOS")
endif()
if(NCNN_MOE_USE_NCNN)
    if(APPLE AND NCNN_MOE_USE_VULKAN)
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

        # ncnn simplevk uses this path for its Apple loader fallback.
        set(Vulkan_LIBRARY "${NCNN_MOE_MOLTENVK_LIBRARY}" CACHE FILEPATH
            "Vulkan loader library used by bundled ncnn" FORCE)
    endif()
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/CMakeLists.txt")
        message(FATAL_ERROR
            "Bundled ncnn is unavailable. Initialize the submodule with "
            "'git submodule update --init --recursive'.")
    endif()

    include(cmake/ncnn_moe_prepare_ncnn.cmake)
    ncnn_moe_prepare_ncnn("${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn"
        "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/patches/ncnn"
        NCNN_MOE_NCNN_SOURCE_DIR)

    # Force runtime-owned options when reconfiguring an existing cache.
    set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "Build ncnn benchmark")
    set(NCNN_BUILD_TESTS OFF CACHE BOOL "Build ncnn tests")
    set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "Build ncnn examples")
    set(NCNN_BUILD_TOOLS OFF CACHE BOOL "Build ncnn tools")
    set(NCNN_VULKAN ${NCNN_MOE_USE_VULKAN} CACHE BOOL "Build ncnn with Vulkan support" FORCE)
    set(NCNN_BF16 ON CACHE BOOL "Build ncnn with bfloat16 support")
    # Quantized runtime kernels do not use ncnn's INT8 operators.
    set(NCNN_INT8 OFF CACHE BOOL "Build ncnn with int8 inference" FORCE)
    # Runtime layers use numeric IDs and ParamDict entries, not text graphs.
    set(NCNN_STRING OFF CACHE BOOL "Build ncnn with string support" FORCE)

    # Disable unneeded upstream layers to keep the dependency small.
    set(NCNN_MOE_REQUIRED_LAYERS
        binaryop cast flatten gemm innerproduct packing permute rmsnorm
        reshape rotaryembed sdpa slice softmax)
    if(NCNN_MOE_USE_VULKAN)
        # RMSNorm reuses the LayerNorm Vulkan shaders.
        list(APPEND NCNN_MOE_REQUIRED_LAYERS layernorm)
    endif()
    file(STRINGS "${NCNN_MOE_NCNN_SOURCE_DIR}/src/CMakeLists.txt"
        NCNN_MOE_LAYER_DECLARATIONS REGEX [=[^ncnn_add_layer\(]=])
    foreach(NCNN_MOE_LAYER_DECLARATION IN LISTS NCNN_MOE_LAYER_DECLARATIONS)
        string(REGEX MATCH [=[^ncnn_add_layer\(([A-Za-z0-9_]+)]=]
            NCNN_MOE_LAYER_MATCH "${NCNN_MOE_LAYER_DECLARATION}")
        string(TOLOWER "${CMAKE_MATCH_1}" NCNN_MOE_LAYER_NAME)
        if(NCNN_MOE_LAYER_NAME IN_LIST NCNN_MOE_REQUIRED_LAYERS)
            set(WITH_LAYER_${NCNN_MOE_LAYER_NAME} ON CACHE BOOL
                "build with layer ${NCNN_MOE_LAYER_NAME}" FORCE)
        else()
            set(WITH_LAYER_${NCNN_MOE_LAYER_NAME} OFF CACHE BOOL
                "build with layer ${NCNN_MOE_LAYER_NAME}" FORCE)
        endif()
    endforeach()
    add_subdirectory("${NCNN_MOE_NCNN_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/third_party/ncnn")
    if(TARGET ncnn)
        if(NCNN_OPENMP AND NOT NCNN_SIMPLEOMP AND TARGET OpenMP::OpenMP_CXX
            AND NOT OpenMP_CXX_FOUND AND NOT OPENMP_FOUND)
            # Apple fallback targets may exist even when FindOpenMP reports failure.
            target_link_libraries(ncnn PUBLIC OpenMP::OpenMP_CXX)
        endif()
        target_link_libraries(ncnn_moe PUBLIC ncnn)
        target_compile_definitions(ncnn_moe PRIVATE NCNN_MOE_USE_NCNN=1)
        set(NCNN_MOE_WITH_NCNN ON)
        if(NCNN_MOE_USE_VULKAN AND NOT NCNN_VULKAN)
            message(FATAL_ERROR
                "NCNN_MOE_USE_VULKAN is ON, but bundled ncnn was configured with "
                "NCNN_VULKAN=OFF. Reconfigure with -DNCNN_MOE_USE_VULKAN=ON."
            )
        endif()
        if(NCNN_MOE_USE_VULKAN AND NCNN_VULKAN)
            target_compile_definitions(ncnn_moe PRIVATE NCNN_MOE_WITH_VULKAN=1)
            target_include_directories(ncnn_moe PRIVATE
                ${NCNN_MOE_NCNN_SOURCE_DIR}/src/layer)
            set(NCNN_MOE_WITH_VULKAN ON)
            if(APPLE)
                target_compile_definitions(ncnn_moe PRIVATE
                    NCNN_MOE_MOLTENVK_LIBRARY_PATH="${NCNN_MOE_MOLTENVK_LIBRARY}"
                )
            endif()
        endif()
    endif()
endif()

# ==============================================================================
# Runtime Vulkan shaders
# ==============================================================================
# Embed GLSL for runtime compilation.
if(NCNN_MOE_WITH_VULKAN)
    set(NCNN_MOE_VULKAN_SHADER_SOURCES
        src/kernels/vulkan/attention_decode_merge.comp
        src/kernels/vulkan/attention_decode_sdpa.comp
        src/kernels/vulkan/attention_decode_split.comp
        src/kernels/vulkan/attention_gqa_matrix.comp
        src/kernels/vulkan/attention_gqa_reduce.comp
        src/kernels/vulkan/attention_output_gate.comp
        src/kernels/vulkan/attention_qkv_norm_rope.comp
        src/kernels/vulkan/attention_qkv_rope.comp
        src/kernels/vulkan/attention_ring_append.comp
        src/kernels/vulkan/attention_ring_zero.comp
        src/kernels/vulkan/bfloat16_cooperative_projection.comp
        src/kernels/vulkan/bfloat16_projection.comp
        src/kernels/vulkan/bfloat16_rms_norm_projection.comp
        src/kernels/vulkan/bfloat16_row_rms.comp
        src/kernels/vulkan/bfloat16_swiglu_down.comp
        src/kernels/vulkan/bfloat16_tile_64x64.comp
        src/kernels/vulkan/bfloat16_tile_32x64.comp
        src/kernels/vulkan/float8_projection.comp
        src/kernels/vulkan/float8_quantize.comp
        src/kernels/vulkan/float8_rms_norm_quantize.comp
        src/kernels/vulkan/float8_swiglu_quantize.comp
        src/kernels/vulkan/projection_matvec.comp
        src/kernels/vulkan/gated_delta_conv.comp
        src/kernels/vulkan/gated_delta_conv_history.comp
        src/kernels/vulkan/gated_delta_conv_parallel.comp
        src/kernels/vulkan/gated_delta_rule.comp
        src/kernels/vulkan/gated_delta_rule_reg.comp
        src/kernels/vulkan/gated_delta_qk.comp
        src/kernels/vulkan/gated_delta_norm.comp
        src/kernels/vulkan/residual_ple.comp
        src/kernels/vulkan/qsa_norm_rope.comp
        src/kernels/vulkan/qsa_cache_write.comp
        src/kernels/vulkan/qsa_pool.comp
        src/kernels/vulkan/qsa_scores.comp
        src/kernels/vulkan/qsa_select.comp
        src/kernels/vulkan/qsa_attention.comp
        src/kernels/vulkan/qsa_copy.comp
        src/kernels/vulkan/activation_rms_norm.comp
        src/kernels/vulkan/hyper_connection_project.comp
        src/kernels/vulkan/hyper_connection_mix.comp
        src/kernels/vulkan/hyper_connection_reduce.comp
        src/kernels/vulkan/hyper_connection_post.comp
        src/kernels/vulkan/router_select.comp
        src/kernels/vulkan/router_select_raw.comp
        src/kernels/vulkan/expertbank_gate_up.comp
        src/kernels/vulkan/expertbank_down.comp
        src/kernels/vulkan/expertbank_combine.comp
        src/kernels/vulkan/latent_attention_transform.comp
        src/kernels/vulkan/latent_attention_compress.comp
        src/kernels/vulkan/latent_attention_index_score.comp
        src/kernels/vulkan/latent_attention_index_topk.comp
        src/kernels/vulkan/latent_attention_index_sort.comp
        src/kernels/vulkan/latent_attention_index_merge.comp
        src/kernels/vulkan/latent_attention_cache_append.comp
        src/kernels/vulkan/latent_attention_core.comp
        src/kernels/vulkan/latent_attention_score.comp
        src/kernels/vulkan/latent_attention_scored_core.comp
        src/kernels/vulkan/latent_attention_query.comp
        src/kernels/vulkan/expert_gather.comp
        src/kernels/vulkan/moe_combine.comp
        src/kernels/vulkan/moe_combine_single_row.comp
        src/kernels/vulkan/mxfp4_gate_up.comp
        src/kernels/vulkan/mxfp4_row_tile.comp
        src/kernels/vulkan/mxfp4_indexed.comp
        src/kernels/vulkan/mxfp4_projection.comp
        src/kernels/vulkan/mxfp4_route_aggregation.comp
        src/kernels/vulkan/qnk_projection.comp
        src/kernels/vulkan/qnk_swiglu.comp
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

# ==============================================================================
# Tests: framework, quantization and model-level precision checks only
# ==============================================================================
# Per-kernel Vulkan and micro-benchmark tests are deliberately not registered;
# they duplicate the model-level coverage below and dominate build time.
if(NCNN_MOE_BUILD_TESTS)
    enable_testing()

    add_executable(ncnn_moe_tests
        tests/modeladapter_fixture.cpp
        tests/test_batch_combine.cpp
        tests/test_dense.cpp
        tests/test_int4.cpp
        tests/test_quant_archive.cpp
        tests/test_native_awq.cpp
        tests/test_runtime_main.cpp
        tests/test_runtime_continuation.cpp
        tests/test_runtime_cpu_kernels.cpp
        tests/test_runtime_deepseek.cpp
        tests/test_runtime_generation.cpp
        tests/test_runtime_graph_adapter.cpp
        tests/test_runtime_linear_vulkan.cpp
        tests/test_runtime_prefill_session.cpp
        tests/test_runtime_quant_cpu.cpp
        tests/test_runtime_qwen.cpp
        tests/test_runtime_storage_cache.cpp
    )
    target_link_libraries(ncnn_moe_tests PRIVATE ncnn::moe)
    target_compile_features(ncnn_moe_tests PRIVATE cxx_std_20)
    target_include_directories(ncnn_moe_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(ncnn_moe_tests PRIVATE
        NCNN_MOE_WITH_NCNN=$<BOOL:${NCNN_MOE_WITH_NCNN}>
        NCNN_MOE_WITH_VULKAN=$<BOOL:${NCNN_MOE_WITH_VULKAN}>
    )
    add_test(NAME ncnn_moe_tests COMMAND ncnn_moe_tests)
    if(UNIX)
        # Isolate the low-FD regression in its own process.
        add_test(NAME ncnn_moe_mapped_file_low_fd COMMAND ncnn_moe_tests --mapped-file)
    endif()

    foreach(NCNN_MOE_MODEL_TEST IN ITEMS qwen3 qwen_dense gguf q8_0 gguf_blocks gguf_mxfp4_cache gguf_tokenizer)
        add_executable(ncnn_moe_${NCNN_MOE_MODEL_TEST}_test tests/test_${NCNN_MOE_MODEL_TEST}.cpp)
        target_link_libraries(ncnn_moe_${NCNN_MOE_MODEL_TEST}_test PRIVATE ncnn::moe)
        target_compile_features(ncnn_moe_${NCNN_MOE_MODEL_TEST}_test PRIVATE cxx_std_20)
        target_include_directories(ncnn_moe_${NCNN_MOE_MODEL_TEST}_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        if(MSVC AND NCNN_MOE_MODEL_TEST STREQUAL "gguf_tokenizer")
            target_compile_options(ncnn_moe_${NCNN_MOE_MODEL_TEST}_test PRIVATE /utf-8)
        endif()
        add_test(NAME ncnn_moe_${NCNN_MOE_MODEL_TEST} COMMAND ncnn_moe_${NCNN_MOE_MODEL_TEST}_test)
    endforeach()
    if(NCNN_MOE_WITH_VULKAN)
        add_test(NAME ncnn_moe_qwen3_vulkan COMMAND ncnn_moe_qwen3_test --vulkan)
        set_tests_properties(ncnn_moe_qwen3_vulkan PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    add_executable(ncnn_moe_public_api_test tests/public_api/main.cpp)
    target_link_libraries(ncnn_moe_public_api_test PRIVATE ncnn::moe)
    add_test(NAME ncnn_moe_public_api_test COMMAND ncnn_moe_public_api_test)
    set_tests_properties(ncnn_moe_public_api_test PROPERTIES TIMEOUT 60)

    # Model-driven JSONL harness; run manually against a tokenizer package.
    add_executable(ncnn_moe_tokenizer_test tests/test_tokenizer.cpp)
    target_link_libraries(ncnn_moe_tokenizer_test PRIVATE ncnn::moe)
    target_include_directories(ncnn_moe_tokenizer_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

    add_test(
        NAME ncnn_moe_style
        COMMAND ${CMAKE_COMMAND}
            -DNCNN_MOE_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_style.cmake
    )

    # Keep temporary fixtures in the build tree.
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test-tmp")
    get_property(NCNN_MOE_TESTS DIRECTORY PROPERTY TESTS)
    set_property(TEST ${NCNN_MOE_TESTS} APPEND PROPERTY ENVIRONMENT
        "TEMP=${CMAKE_CURRENT_BINARY_DIR}/test-tmp"
        "TMP=${CMAKE_CURRENT_BINARY_DIR}/test-tmp"
        "TMPDIR=${CMAKE_CURRENT_BINARY_DIR}/test-tmp")
endif()

# ==============================================================================
# Tools and optional reference runners
# ==============================================================================

if(NCNN_MOE_BUILD_EXAMPLES)
    add_executable(
        ncnn_moe_worker
        examples/ncnn_moe_worker.cpp
        examples/internal/gputelemetry.cpp)
    target_link_libraries(
        ncnn_moe_worker
        PRIVATE ncnn::moe)
    target_compile_features(
        ncnn_moe_worker
        PRIVATE cxx_std_20)
    target_include_directories(
        ncnn_moe_worker
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

    if(WIN32)
        # Worker telemetry uses Windows process APIs.
        target_link_libraries(ncnn_moe_worker PRIVATE psapi)
    endif()
    if(CMAKE_DL_LIBS)
        target_link_libraries(ncnn_moe_worker PRIVATE ${CMAKE_DL_LIBS})
    endif()
    ncnn_moe_optimize_target(ncnn_moe_worker)
    install(TARGETS ncnn_moe_worker
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(APPLE AND TARGET ncnn_moe_worker)
    target_link_options(ncnn_moe_worker PRIVATE
        "$<$<CONFIG:Release>:LINKER:-dead_strip>")
endif()

# ==============================================================================
# Installed SDK
# ==============================================================================
# Export public SDK headers and transitive link dependencies.
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
install(FILES
    include/ncnn/moe/modeladapter.h
    include/ncnn/moe/modeldescriptor.h
    include/ncnn/moe/model.h
    include/ncnn/moe/option.h
    include/ncnn/moe/result.h
    include/ncnn/moe/runtime.h
    include/ncnn/moe/scheduler.h
    include/ncnn/moe/session.h
    include/ncnn/moe/types.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ncnn/moe
)
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
