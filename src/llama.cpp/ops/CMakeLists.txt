file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# 复制一份公共源码
set(SRC_COMMON ${SRC_FILES})

# 去掉绑定文件，防止重复
list(FILTER SRC_COMMON EXCLUDE REGEX "ops_py\\.cpp$")
list(FILTER SRC_COMMON EXCLUDE REGEX "ops_jni\\.cpp$")

# 根据平台选择编译目标
if(ANDROID)
    # 安卓版：编译为静态库，包含 ops_jni.cpp
    add_library(opslib STATIC ${SRC_COMMON} ops_jni.cpp)
    
    target_include_directories(opslib PUBLIC 
        ${CMAKE_SOURCE_DIR}/3dparty/llamacpp/include 
        ${CMAKE_SOURCE_DIR}/ops
        ${CMAKE_SOURCE_DIR}/3dparty/llamacpp/ggml/include
        $ENV{JAVA_HOME}/include
        $ENV{JAVA_HOME}/include/linux
    )
    
    find_library(log-lib log)
    target_link_libraries(opslib PRIVATE ggml ${log-lib})
    
else()
    # 桌面版：编译为动态库，包含 ops_py.cpp
    add_library(opslib SHARED ${SRC_COMMON} ops_py.cpp)
    
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development Development.Module)
    
    target_include_directories(opslib PUBLIC 
        ${CMAKE_SOURCE_DIR}/3dparty/llamacpp/include 
        ${CMAKE_SOURCE_DIR}/3dparty/pybind11/include 
        ${CMAKE_SOURCE_DIR}/ops
        ${CMAKE_SOURCE_DIR}/3dparty/llamacpp/ggml/include
    )
    
    target_link_libraries(opslib PRIVATE 
        Python3::Python
        ggml
    )
endif()

# 设置输出名称
set_target_properties(opslib PROPERTIES 
    OUTPUT_NAME "opslib"
)

# 仅桌面版去掉 lib 前缀
if(NOT ANDROID)
    set_target_properties(opslib PROPERTIES PREFIX "")
endif()