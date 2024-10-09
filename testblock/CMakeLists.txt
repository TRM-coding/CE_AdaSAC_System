project(test_block)
cmake_minimum_required(VERSION 3.16)

add_executable(test_block test.cpp)
target_link_libraries(test_block armadillo)
