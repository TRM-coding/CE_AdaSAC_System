cmake_minimum_required(VERSION 3.16)
project(Mini-mlSys)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



add_subdirectory(${CMAKE_SOURCE_DIR}/3dparty/armadillo)
add_subdirectory(${CMAKE_SOURCE_DIR}/3dparty/googleTest)
add_subdirectory(${CMAKE_SOURCE_DIR}/3dparty/pnnx)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/3dparty/armadillo/include)
include_directories(${CMAKE_SOURCE_DIR}/3dparty/googleTest/googletest/include)
include_directories(${CMAKE_SOURCE_DIR}/3dparty/pnnx/include)


add_subdirectory(${CMAKE_SOURCE_DIR}/Tensor)
add_subdirectory(${CMAKE_SOURCE_DIR}/Attribute)

add_subdirectory(${CMAKE_SOURCE_DIR}/Operand)
add_subdirectory(${CMAKE_SOURCE_DIR}/Operators)
add_subdirectory(${CMAKE_SOURCE_DIR}/Layer)
add_subdirectory(${CMAKE_SOURCE_DIR}/Graph)
add_subdirectory(${CMAKE_SOURCE_DIR}/testblock)



message(
    STATUS "CMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}"
)

# aux_source_directory()
