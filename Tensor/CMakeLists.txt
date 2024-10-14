project(Class-Tensor)
cmake_minimum_required(VERSION 3.16)

aux_source_directory(./test DIR_TEST_Tensor)
add_library(Tensor STATIC ${DIR_TEST_Tensor})
add_executable(MINI_MLsys_Tensor main.cpp ${DIR_TEST_Tensor})

target_link_libraries(MINI_MLsys_Tensor gtest armadillo)