project(MINI_MLsys_Layer)
cmake_minimum_required(VERSION 3.16)

aux_source_directory(./src DIR_SRCS_FILE)
aux_source_directory(./test DIR_TEST_FILE)
add_library(MINI_MLsys_Layer STATIC ${DIR_SRCS_FILE})

add_executable(MINI_MLsys_Layer_test main.cpp ${DIR_TEST_FILE})
target_link_libraries(MINI_MLsys_Layer_test Operand MINI_MLsys_Layer armadillo gtest Operators Tensor)
