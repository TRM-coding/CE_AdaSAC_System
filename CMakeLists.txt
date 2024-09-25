cmake_minimum_required(VERSION 3.16)
project(Mini-mlSys)


add_subdirectory(${CMAKE_SOURCE_DIR}/3dparty/armadillo)
add_subdirectory(${CMAKE_SOURCE_DIR}/3dparty/googleTest)


include_directories(${CMAKE_SOURCE_DIR}/include)
add_subdirectory(${CMAKE_SOURCE_DIR}/Tensor)


message(
    STATUS "CMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}"
)

# aux_source_directory()
