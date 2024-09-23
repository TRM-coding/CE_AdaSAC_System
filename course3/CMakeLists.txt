cmake_minimum_required(VERSION 3.16)
project(kuiper_datawhale_course3)
set(CMAKE_CXX_STANDARD 17)
find_package(benchmark REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(GTest REQUIRED)

set(link_lib glog::glog GTest::gtest)
if (!WIN32)
    set(link_lib "${link_lib} pthread")
endif ()
set(link_math_lib ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
aux_source_directory(./test DIR_TEST_ARMA)
aux_source_directory(./source DIR_SOURCE_ARMA)

add_executable(kuiper_datawhale_course3 main.cpp ${DIR_TEST_ARMA} ${DIR_SOURCE_ARMA})
target_link_libraries(kuiper_datawhale_course3 ${link_lib} ${link_math_lib} OpenMP::OpenMP_CXX)

target_include_directories(kuiper_datawhale_course3 PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(kuiper_datawhale_course3 PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(kuiper_datawhale_course3 PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(kuiper_datawhale_course3 PUBLIC ./include)

enable_testing()


# set(SOURCE_FILE "${CMAKE_SOURCE_DIR}/data/file.txt")
# set(TARGET_DIR "${CMAKE_BINARY_DIR}/data")

# # 创建目标目录
# file(MAKE_DIRECTORY ${TARGET_DIR})

# # 复制文件到目标目录
# file(COPY ${SOURCE_FILE} DESTINATION ${TARGET_DIR})

#这哥cmake写的不行啊，编译后要用的源文件都没去build里，相对路径直接挂了
