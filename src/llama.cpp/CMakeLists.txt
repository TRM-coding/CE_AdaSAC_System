cmake_minimum_required(VERSION 3.10)

project(OPS_TEST)


add_subdirectory(${CMAKE_SOURCE_DIR}/3dparty/llamacpp)
add_subdirectory(${CMAKE_SOURCE_DIR}/3dparty/pybind11)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development Development.Module)

include_directories(${CMAKE_SOURCE_DIR}/ops)
add_subdirectory(ops)
add_executable(ops_test exp1/main.cpp)
target_link_libraries(ops_test PRIVATE Python3::Python llama opslib)

