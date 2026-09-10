if(NOT DEFINED NCNN_MOE_SOURCE_DIR)
    message(FATAL_ERROR "NCNN_MOE_SOURCE_DIR is required")
endif()

get_filename_component(NCNN_MOE_SOURCE_DIR "${NCNN_MOE_SOURCE_DIR}" ABSOLUTE)

file(GLOB_RECURSE NCNN_MOE_STYLE_FILES
    LIST_DIRECTORIES false
    "${NCNN_MOE_SOURCE_DIR}/include/*.h"
    "${NCNN_MOE_SOURCE_DIR}/include/*.hpp"
    "${NCNN_MOE_SOURCE_DIR}/src/*.h"
    "${NCNN_MOE_SOURCE_DIR}/src/*.hpp"
    "${NCNN_MOE_SOURCE_DIR}/src/*.cpp"
    "${NCNN_MOE_SOURCE_DIR}/tests/*.h"
    "${NCNN_MOE_SOURCE_DIR}/tests/*.hpp"
    "${NCNN_MOE_SOURCE_DIR}/tests/*.cpp"
    "${NCNN_MOE_SOURCE_DIR}/examples/*.h"
    "${NCNN_MOE_SOURCE_DIR}/examples/*.hpp"
    "${NCNN_MOE_SOURCE_DIR}/examples/*.cpp"
)

if(NOT NCNN_MOE_STYLE_FILES)
    message(FATAL_ERROR "No C++ source files found in ${NCNN_MOE_SOURCE_DIR}")
endif()

set(NCNN_MOE_STYLE_ERRORS "")

foreach(SOURCE_FILE IN LISTS NCNN_MOE_STYLE_FILES)
    file(READ "${SOURCE_FILE}" SOURCE_TEXT)
    string(REPLACE "\r\n" "\n" SOURCE_TEXT "${SOURCE_TEXT}")
    file(RELATIVE_PATH RELATIVE_FILE "${NCNN_MOE_SOURCE_DIR}" "${SOURCE_FILE}")

    # Check direct includes at the public API and layer boundaries.
    string(REGEX MATCHALL "(^|\n)[ \t]*#[ \t]*include[ \t]*[<\"][^>\"]+[>\"]" INCLUDE_DIRECTIVES "${SOURCE_TEXT}")
    foreach(INCLUDE_DIRECTIVE IN LISTS INCLUDE_DIRECTIVES)
        if(RELATIVE_FILE MATCHES "^include/" AND INCLUDE_DIRECTIVE MATCHES "[<\"]((src/)?(engine|graph|kernels|models|storage|backends)/|\\.\\./)")
            string(APPEND NCNN_MOE_STYLE_ERRORS
                "${RELATIVE_FILE}: public headers must not include runtime internals\n")
        elseif(RELATIVE_FILE MATCHES "^src/models/" AND INCLUDE_DIRECTIVE MATCHES "[<\"](src/|\\.\\./)?(engine|graph|backends)/")
            string(APPEND NCNN_MOE_STYLE_ERRORS
                "${RELATIVE_FILE}: model adapters must not include execution or backend internals\n")
        elseif(RELATIVE_FILE MATCHES "^src/backends/"
               AND INCLUDE_DIRECTIVE MATCHES "[<\"]((\\.\\./)*(src/)?models/|ncnn/moe/modeladapter\\.h)")
            string(APPEND NCNN_MOE_STYLE_ERRORS
                "${RELATIVE_FILE}: backend code must not include model adapters\n")
        elseif(RELATIVE_FILE MATCHES "^src/graph/(graph|layerplan)\\.h$"
               AND INCLUDE_DIRECTIVE MATCHES "[<\"]((\\.\\./)*(src/)?(backends|engine|kernels|models)/[^>\"]+|((\\.\\./)*(src/)?graph/)?compiledoperator\\.h)[>\"]")
            string(APPEND NCNN_MOE_STYLE_ERRORS
                "${RELATIVE_FILE}: graph foundation headers must not include backend or execution implementations\n")
        endif()
    endforeach()

    if(SOURCE_TEXT MATCHES "namespace[ \t]*\\{")
        string(APPEND NCNN_MOE_STYLE_ERRORS "${RELATIVE_FILE}: anonymous namespace is not allowed\n")
    endif()
    if(SOURCE_TEXT MATCHES "namespace[ \t]+(detail|internal)([ \t]*\\{|[ \t]*:)")
        string(APPEND NCNN_MOE_STYLE_ERRORS "${RELATIVE_FILE}: auxiliary namespace is not allowed\n")
    endif()
    if(SOURCE_TEXT MATCHES "namespace[ \t]+ncnn::moe")
        string(APPEND NCNN_MOE_STYLE_ERRORS "${RELATIVE_FILE}: use explicit nested ncnn/moe namespaces\n")
    endif()
    string(REGEX MATCHALL "namespace[ \t]+[A-Za-z_][A-Za-z0-9_]*" NAMESPACE_DECLARATIONS "${SOURCE_TEXT}")
    foreach(NAMESPACE_DECLARATION IN LISTS NAMESPACE_DECLARATIONS)
        string(REGEX REPLACE "namespace[ \t]+" "" NAMESPACE_NAME "${NAMESPACE_DECLARATION}")
        if(NOT NAMESPACE_NAME STREQUAL "ncnn" AND NOT NAMESPACE_NAME STREQUAL "moe")
            string(APPEND NCNN_MOE_STYLE_ERRORS
                "${RELATIVE_FILE}: project symbols must use the explicit ncnn/moe namespace nesting\n")
        endif()
    endforeach()
    if(SOURCE_TEXT MATCHES "#pragma[ \t]+once")
        string(APPEND NCNN_MOE_STYLE_ERRORS "${RELATIVE_FILE}: use an include guard instead of pragma once\n")
    endif()
    if(SOURCE_TEXT MATCHES "R[ \t]+\"[A-Za-z0-9_]*\\(")
        string(APPEND NCNN_MOE_STYLE_ERRORS "${RELATIVE_FILE}: malformed raw string literal\n")
    endif()
    if(SOURCE_TEXT MATCHES "(1[uUlL]*|UINT(32|64)_C\\(1\\))[ \t]*<<[ \t]*[0-9]+")
        string(APPEND NCNN_MOE_STYLE_ERRORS
            "${RELATIVE_FILE}: name bitmap positions with NCNN_MOE_*_BIT macros\n")
    endif()
    if(SOURCE_TEXT MATCHES "\n[ \t]+(=|\\+=|-=|\\*=|/=|%=|&=|\\|=|\\^=)[ \t]+")
        string(APPEND NCNN_MOE_STYLE_ERRORS
            "${RELATIVE_FILE}: keep an assignment operator with its left-hand side\n")
    endif()
    if(SOURCE_TEXT MATCHES "\n[ \t]+(\\.|->)[A-Za-z_]")
        string(APPEND NCNN_MOE_STYLE_ERRORS
            "${RELATIVE_FILE}: keep chained member access with the preceding expression\n")
    endif()
    if(SOURCE_TEXT MATCHES "\\}[ \t]+(else|catch)([ \t]|\\()")
        string(APPEND NCNN_MOE_STYLE_ERRORS "${RELATIVE_FILE}: text after a closing brace must start on a new line\n")
    endif()
    if(NOT RELATIVE_FILE STREQUAL "include/ncnn/moe/result.h" AND SOURCE_TEXT MATCHES "template[ \t]*<")
        string(APPEND NCNN_MOE_STYLE_ERRORS "${RELATIVE_FILE}: avoid templates when a concrete implementation is sufficient\n")
    endif()
endforeach()

if(NCNN_MOE_STYLE_ERRORS)
    message(FATAL_ERROR "ncnn_moe style violations:\n${NCNN_MOE_STYLE_ERRORS}")
endif()

# ColumnLimit: 0 preserves existing newlines, so clang-format alone does not
# reject a first argument placed below the opening parenthesis.
find_package(Python3 3.10 COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${NCNN_MOE_SOURCE_DIR}/cmake/check_format.py"
        --source-dir "${NCNN_MOE_SOURCE_DIR}"
    RESULT_VARIABLE NCNN_MOE_FORMAT_RESULT
)
if(NOT NCNN_MOE_FORMAT_RESULT EQUAL 0)
    message(FATAL_ERROR "ncnn_moe argument placement or clang-format check failed")
endif()
