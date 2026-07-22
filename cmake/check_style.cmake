if(NOT DEFINED NCNN_MOE_SOURCE_DIR)
    message(FATAL_ERROR "NCNN_MOE_SOURCE_DIR is required")
endif()

file(GLOB_RECURSE NCNN_MOE_STYLE_FILES
    LIST_DIRECTORIES false
    "${NCNN_MOE_SOURCE_DIR}/include/*.h"
    "${NCNN_MOE_SOURCE_DIR}/src/*.h"
    "${NCNN_MOE_SOURCE_DIR}/src/*.cpp"
    "${NCNN_MOE_SOURCE_DIR}/tests/*.h"
    "${NCNN_MOE_SOURCE_DIR}/tests/*.cpp"
    "${NCNN_MOE_SOURCE_DIR}/examples/*.h"
    "${NCNN_MOE_SOURCE_DIR}/examples/*.cpp"
)

set(NCNN_MOE_STYLE_ERRORS "")

foreach(SOURCE_FILE IN LISTS NCNN_MOE_STYLE_FILES)
    file(READ "${SOURCE_FILE}" SOURCE_TEXT)
    string(REPLACE "\r\n" "\n" SOURCE_TEXT "${SOURCE_TEXT}")
    file(RELATIVE_PATH RELATIVE_FILE "${NCNN_MOE_SOURCE_DIR}" "${SOURCE_FILE}")

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
    if(SOURCE_TEXT MATCHES "(^|\n)[ \t]*(if|for|while|switch|catch)[^\n]*\n[ \t]*\\{")
        string(APPEND NCNN_MOE_STYLE_ERRORS "${RELATIVE_FILE}: control-flow opening brace must stay on the condition line\n")
    endif()
    if(SOURCE_TEXT MATCHES "\\}[ \t]+(else|catch|while)([ \t]|\\()")
        string(APPEND NCNN_MOE_STYLE_ERRORS "${RELATIVE_FILE}: text after a closing brace must start on a new line\n")
    endif()
    if(NOT RELATIVE_FILE STREQUAL "include/ncnn/moe/result.h" AND SOURCE_TEXT MATCHES "template[ \t]*<")
        string(APPEND NCNN_MOE_STYLE_ERRORS "${RELATIVE_FILE}: avoid templates when a concrete implementation is sufficient\n")
    endif()
endforeach()

if(NCNN_MOE_STYLE_ERRORS)
    message(FATAL_ERROR "ncnn_moe style violations:\n${NCNN_MOE_STYLE_ERRORS}")
endif()
