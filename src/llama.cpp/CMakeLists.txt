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


# 桌面版需要 Python
if(NOT IS_ANDROID)
    add_subdirectory(${CMAKE_SOURCE_DIR}/3dparty/pybind11)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development Development.Module)
endif()

include_directories(${CMAKE_SOURCE_DIR}/ops)
add_subdirectory(ops)

if(IS_ANDROID)
    add_executable(ops_test exp1/main.cpp)
    target_link_libraries(ops_test PRIVATE 
        opslib
        llama
    )
    # 强制静态链接所有库和 C++ 运行时
    target_link_options(ops_test PRIVATE 
        -static-libgcc 
        -static-libstdc++
    )
endif()

# 桌面版才编译 ops_test
if(NOT IS_ANDROID)
    add_executable(ops_test exp1/main.cpp)
    target_link_libraries(ops_test PRIVATE Python3::Python llama opslib)
endif()

