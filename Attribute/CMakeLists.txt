project(Attribute)
cmake_minimum_required(VERSION 3.16)

aux_source_directory(./src DIR_SRC_FILE)
aux_source_directory(./test DIR_TEST_FILE)
add_library(Attribute ${DIR_SRC_FILE})

add_executable(ATTRIBUTE_TEST ${DIR_TEST_FILE})
target_link_libraries(ATTRIBUTE_TEST Attribute gtest armadillo)