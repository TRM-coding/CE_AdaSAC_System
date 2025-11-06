file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(opslib SHARED ${SRC_FILES} ops_py.cpp)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development Development.Module)

target_include_directories(
    opslib PUBLIC ${CMAKE_SOURCE_DIR}/3dparty/llamacpp/include 
    ${CMAKE_SOURCE_DIR}/3dparty/pybind11/include 
    ${CMAKE_SOURCE_DIR}/ops
    ${CMAKE_SOURCE_DIR}/3dparty/llamacpp/ggml/include)

# 链接 Python 和 GGML 库（关键修改！）
target_link_libraries(opslib PRIVATE 
    Python3::Python
    ggml  # 添加这一行
)

# 设置输出名称，去掉 lib 前缀
set_target_properties(opslib PROPERTIES 
    PREFIX ""
    OUTPUT_NAME "opslib"
)