include_guard(GLOBAL)

# AppleClang does not discover Homebrew's libomp without explicit flags. Define
# the imported target before ncnn's package configuration requests it.
find_package(OpenMP QUIET COMPONENTS CXX)

if(APPLE AND NOT TARGET OpenMP::OpenMP_CXX)
    set(_ncnn_moe_libomp_prefixes)
    if(DEFINED ENV{HOMEBREW_PREFIX})
        list(APPEND _ncnn_moe_libomp_prefixes "$ENV{HOMEBREW_PREFIX}/opt/libomp")
    endif()
    list(APPEND _ncnn_moe_libomp_prefixes
        "/opt/homebrew/opt/libomp"
        "/usr/local/opt/libomp"
    )

    find_path(_ncnn_moe_libomp_include_dir
        NAMES omp.h
        HINTS ${_ncnn_moe_libomp_prefixes}
        PATH_SUFFIXES include
        NO_DEFAULT_PATH
        NO_CACHE
    )
    find_library(_ncnn_moe_libomp_library
        NAMES omp
        HINTS ${_ncnn_moe_libomp_prefixes}
        PATH_SUFFIXES lib
        NO_DEFAULT_PATH
        NO_CACHE
    )

    if(_ncnn_moe_libomp_include_dir AND _ncnn_moe_libomp_library)
        add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED GLOBAL)
        set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
            INTERFACE_COMPILE_OPTIONS "-Xpreprocessor;-fopenmp"
            INTERFACE_INCLUDE_DIRECTORIES "${_ncnn_moe_libomp_include_dir}"
            INTERFACE_LINK_LIBRARIES "${_ncnn_moe_libomp_library}"
        )
    endif()
endif()
