project(Graph)
cmake_minimum_required(VERSION 3.16)

aux_source_directory(./test DIR_TEST_Graph)
aux_source_directory(./src DIR_SRC_FILE)
add_executable(MINI_MLsys_Graph main.cpp ${DIR_SRC_FILE})

target_link_libraries(MINI_MLsys_Graph MINI_MLsys_Layer gtest armadillo Operators Operand pnnx_mini_mlSys Attribute )