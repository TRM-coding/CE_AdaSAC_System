project(Class-Tensor)
cmake_minimum_required(VERSION 3.16)

message(STATUS  "CMAKE_CURRENT_SOURCE_DIR============${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(main ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(main gtest)