if(NOT DEFINED SHADER_SRC
    OR NOT DEFINED SHADER_COMP_HEADER
    OR NOT DEFINED SHADER_VARIABLE)
    message(FATAL_ERROR
        "SHADER_SRC, SHADER_COMP_HEADER, and SHADER_VARIABLE are required")
endif()

file(READ "${SHADER_SRC}" comp_data)

string(FIND "${comp_data}" "#version" version_start)
if(NOT version_start EQUAL -1)
    string(SUBSTRING "${comp_data}" ${version_start} -1 comp_data)
endif()

string(REGEX REPLACE "\n +" "\n" comp_data "${comp_data}")
string(REGEX REPLACE "//[^\n]*" "" comp_data "${comp_data}")
string(REGEX REPLACE "[ \t]+" " " comp_data "${comp_data}")
string(REGEX REPLACE "\n[\n]+" "\n" comp_data "${comp_data}")

get_filename_component(shader_header_directory "${SHADER_COMP_HEADER}" DIRECTORY)
file(MAKE_DIRECTORY "${shader_header_directory}")
set(shader_text_file "${SHADER_COMP_HEADER}.text2hex.txt")
file(WRITE "${shader_text_file}" "${comp_data}")
file(READ "${shader_text_file}" comp_data_hex HEX)
file(REMOVE "${shader_text_file}")

string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," comp_data_hex "${comp_data_hex}")
string(FIND "${comp_data_hex}" "," tail_comma REVERSE)
if(tail_comma GREATER -1)
    string(SUBSTRING "${comp_data_hex}" 0 ${tail_comma} comp_data_hex)
endif()

file(WRITE "${SHADER_COMP_HEADER}"
    "#if defined(_MSC_VER)\n"
    "#pragma warning(push)\n"
    "#pragma warning(disable : 4309)\n"
    "#endif\n"
    "static const char ${SHADER_VARIABLE}[] = {${comp_data_hex},0x00};\n"
    "#if defined(_MSC_VER)\n"
    "#pragma warning(pop)\n"
    "#endif\n")
