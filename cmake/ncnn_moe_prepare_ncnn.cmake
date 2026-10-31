# Prepare a patched build-tree copy without modifying the upstream checkout.
function(ncnn_moe_prepare_ncnn source_dir build_dir patch_dir output_var)
    file(REAL_PATH "${source_dir}" source_dir)
    file(MAKE_DIRECTORY "${build_dir}")
    file(REAL_PATH "${build_dir}" build_dir)
    set(prepared_dir "${build_dir}/_deps/ncnn-src")
    set(stage_dir "${build_dir}/_deps/ncnn-src-stage")
    cmake_path(IS_PREFIX source_dir "${prepared_dir}" NORMALIZE inside_source)
    cmake_path(IS_PREFIX build_dir "${prepared_dir}" NORMALIZE inside_build)
    if(inside_source OR NOT inside_build)
        message(FATAL_ERROR "The patched ncnn copy must be inside the build directory and outside the upstream source directory.")
    endif()
    if(NOT EXISTS "${source_dir}/CMakeLists.txt")
        message(FATAL_ERROR "ncnn source is missing: ${source_dir}")
    endif()

    # Track contents as well as file additions/removals, including submodules.
    if(NOT CMAKE_SCRIPT_MODE_FILE)
        set(glob_depends CONFIGURE_DEPENDS)
    endif()
    file(GLOB_RECURSE source_files ${glob_depends} LIST_DIRECTORIES false
        RELATIVE "${source_dir}" "${source_dir}/*")
    list(FILTER source_files EXCLUDE REGEX "(^|/)\\.git(/|$)")
    file(GLOB patches ${glob_depends} LIST_DIRECTORIES false "${patch_dir}/*.patch")
    if(NOT patches)
        message(FATAL_ERROR "No ncnn patches found in ${patch_dir}")
    endif()
    list(SORT source_files)
    list(SORT patches)
    set(fingerprint "${source_dir}\n")
    set(dependencies "${CMAKE_CURRENT_FUNCTION_LIST_FILE}")
    foreach(relative_file IN LISTS source_files)
        set(source_file "${source_dir}/${relative_file}")
        file(SHA256 "${source_file}" source_hash)
        string(APPEND fingerprint "${relative_file}:${source_hash}\n")
        list(APPEND dependencies "${source_file}")
    endforeach()
    foreach(patch IN LISTS patches)
        file(SHA256 "${patch}" patch_hash)
        string(APPEND fingerprint "${patch}:${patch_hash}\n")
        list(APPEND dependencies "${patch}")
    endforeach()
    file(SHA256 "${CMAKE_CURRENT_FUNCTION_LIST_FILE}" helper_hash)
    string(SHA256 fingerprint "${fingerprint}${helper_hash}")
    if(NOT CMAKE_SCRIPT_MODE_FILE)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${dependencies})
    endif()
    set(stamp "${prepared_dir}/.ncnn-moe-patches.sha256")
    if(EXISTS "${stamp}" AND EXISTS "${prepared_dir}/CMakeLists.txt")
        file(READ "${stamp}" previous_fingerprint)
        if(previous_fingerprint STREQUAL fingerprint)
            set(${output_var} "${prepared_dir}" PARENT_SCOPE)
            return()
        endif()
    endif()

    find_package(Git REQUIRED QUIET)
    message(STATUS "Preparing patched ncnn sources in ${prepared_dir}")
    # Both paths are fixed children of the resolved build directory above.
    file(REMOVE_RECURSE "${stage_dir}")
    file(MAKE_DIRECTORY "${stage_dir}")
    file(COPY "${source_dir}/" DESTINATION "${stage_dir}" PATTERN ".git" EXCLUDE)
    foreach(patch IN LISTS patches)
        foreach(mode IN ITEMS check apply)
            set(check_arg)
            if(mode STREQUAL "check")
                set(check_arg --check)
            endif()
            # Do not discover the enclosing Runtime repository or its index.
            execute_process(
                COMMAND "${CMAKE_COMMAND}" -E env
                    --unset=GIT_DIR --unset=GIT_WORK_TREE --unset=GIT_INDEX_FILE
                    "GIT_CEILING_DIRECTORIES=${build_dir}/_deps"
                    "${GIT_EXECUTABLE}" apply ${check_arg} -- "${patch}"
                WORKING_DIRECTORY "${stage_dir}"
                RESULT_VARIABLE patch_result
                OUTPUT_VARIABLE patch_output
                ERROR_VARIABLE patch_error)
            if(NOT patch_result EQUAL 0)
                message(FATAL_ERROR
                    "Cannot ${mode} ncnn patch ${patch}:\n${patch_output}${patch_error}\n"
                    "Update the patches for the selected upstream ncnn revision. "
                    "The upstream checkout and previous prepared sources were preserved.")
            endif()
        endforeach()
    endforeach()
    file(WRITE "${stage_dir}/.ncnn-moe-patches.sha256" "${fingerprint}")
    # Publish only after the entire patch series has applied successfully.
    file(REMOVE_RECURSE "${prepared_dir}")
    file(RENAME "${stage_dir}" "${prepared_dir}")
    set(${output_var} "${prepared_dir}" PARENT_SCOPE)
endfunction()
