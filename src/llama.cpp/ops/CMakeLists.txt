file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(opslib SHARED ${SRC_FILES})
find_package(Python3 REQUIRED COMPONENTS Interpreter Development Development.Module)

target_include_directories(
    opslib PUBLIC ${CMAKE_SOURCE_DIR}/3dparty/llamacpp/include 
    ${CMAKE_SOURCE_DIR}/3dparty/pybind11/include 
    ${CMAKE_SOURCE_DIR}/ops
    ${CMAKE_SOURCE_DIR}/3dparty/llamacpp/ggml/include)

# 关键：把 Python 头与库提供给该目标
target_link_libraries(opslib PRIVATE Python3::Python)
# 如需显式加入头：
# target_include_directories(opslib PRIVATE ${Python3_INCLUDE_DIRS})