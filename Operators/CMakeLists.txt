project(Operators)
cmake_minimum_required(VERSION 3.16)

aux_source_directory(./test DIR_TEST_Operators)
aux_source_directory(./src DIR_SRC_FILE)
add_library(Operators STATIC ${DIR_SRC_FILE})
add_executable(MINI_MLsys_Operators main.cpp ${DIR_TEST_Operators} ${DIR_SRC_FILE})

target_link_libraries(MINI_MLsys_Operators gtest armadillo Operand pnnx_mini_mlSys Attribute)
