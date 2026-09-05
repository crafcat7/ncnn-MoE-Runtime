cmake_minimum_required(VERSION 3.21)

project(ncnn_moe_public_api_test LANGUAGES CXX)

find_package(ncnn_moe CONFIG REQUIRED)

add_executable(ncnn_moe_public_api_test main.cpp)
target_link_libraries(ncnn_moe_public_api_test PRIVATE ncnn::moe)

enable_testing()
add_test(NAME ncnn_moe_public_api_test COMMAND ncnn_moe_public_api_test)
set_tests_properties(ncnn_moe_public_api_test PROPERTIES TIMEOUT 60)
