cmake_minimum_required(VERSION 3.10)

project(OPS_TEST)

# 检测是否为安卓平台
if(ANDROID)
    message("Android Version Compiling")
    set(IS_ANDROID ON)
    # 强制 llamacpp 编译为静态库
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
    set(LLAMA_BUILD_SHARED OFF CACHE BOOL "Build shared llama library" FORCE)
    set(GGML_BUILD_SHARED OFF CACHE BOOL "Build shared ggml library" FORCE)
else()
    set(IS_ANDROID OFF)
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/3dparty/llamacpp)
include_directories(${CMAKE_SOURCE_DIR}/3dparty/llamacpp/src)
include_directories(${CMAKE_SOURCE_DIR}/3dparty/llamacpp/common)
include_directories(${CMAKE_SOURCE_DIR}/3dparty/llamacpp/include)
include_directories(${CMAKE_SOURCE_DIR}/3dparty/llamacpp/ggml)
include_directories(${CMAKE_SOURCE_DIR}/3dparty/llamacpp/ggml/include)


# 桌面版需要 Python
if(NOT IS_ANDROID)
    add_subdirectory(${CMAKE_SOURCE_DIR}/3dparty/pybind11)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development Development.Module)
endif()

include_directories(${CMAKE_SOURCE_DIR}/ops)
add_subdirectory(ops)

if(IS_ANDROID)
    add_executable(decode_svd_test exp6_decode_svd_model/decode_svd_model.cpp)
    target_link_libraries(decode_svd_test PRIVATE llama)
endif()

# 桌面版才编译 ops_test
if(NOT IS_ANDROID)
    add_executable(load_test exp3_load_svd_to_llamacpp/load_llamacpp.cpp)
    add_executable(run_svd_test exp5_run_svd_model/run_svd_model.cpp)
    add_executable(decode_svd_test exp6_decode_svd_model/decode_svd_model.cpp)
    add_executable(simple ${CMAKE_SOURCE_DIR}/3dparty/llamacpp/examples/simple/simple.cpp)
    add_executable(llama-bench
        ${CMAKE_SOURCE_DIR}/3dparty/llamacpp/examples/llama-bench/llama-bench.cpp
        ${CMAKE_SOURCE_DIR}/3dparty/llamacpp/common/common.cpp
        ${CMAKE_SOURCE_DIR}/3dparty/llamacpp/common/log.cpp
        ${CMAKE_SOURCE_DIR}/3dparty/llamacpp/common/build-info.cpp  # 定义 LLAMA_* 符号
    )
    # target_link_libraries(ops_test PRIVATE Python3::Python llama opslib)
    target_link_libraries(load_test PRIVATE llama)
    target_link_libraries(run_svd_test PRIVATE llama)
    target_link_libraries(decode_svd_test PRIVATE llama)
    target_link_libraries(simple PRIVATE opslib llama)
    target_link_libraries(llama-bench PRIVATE opslib llama ggml)
endif()
