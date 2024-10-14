project(Class-Operand)
cmake_minimum_required(VERSION 3.16)

aux_source_directory(./test DIR_TEST_Operand)
aux_source_directory(./src DIR_SOURCE_FILE)
add_library(Operand STATIC ${DIR_SOURCE_FILE})
add_executable(MINI_MLsys_Operand main.cpp ${DIR_TEST_Operand} ${DIR_SOURCE_FILE} )

target_link_libraries(MINI_MLsys_Operand gtest armadillo pnnx_mini_mlSys)