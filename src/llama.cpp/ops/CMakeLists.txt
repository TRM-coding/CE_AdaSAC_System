# 递归查找所有 .cpp 文件
file(GLOB_RECURSE SRC_FILES 
    "*.cpp"
    "*/*.cpp"
)

# 调试：显示找到的文件
message(STATUS "=== Found CPP Files ===")
foreach(file ${SRC_FILES})
    message(STATUS "  ${file}")
endforeach()
message(STATUS "=== Total: ${SRC_FILES} files ===")

add_library(opslib SHARED ${SRC_FILES})

find_package(Python3 REQUIRED COMPONENTS Interpreter Development Development.Module)

target_include_directories(
    opslib PUBLIC ${CMAKE_SOURCE_DIR}/3dparty/llamacpp/include
    ${CMAKE_SOURCE_DIR}/3dparty/pybind11/include
    ${CMAKE_SOURCE_DIR}/ops
    ${CMAKE_SOURCE_DIR}/3dparty/llamacpp/ggml/include)

target_link_libraries(opslib PRIVATE
    Python3::Python
    ggml
)

set_target_properties(opslib PROPERTIES
    PREFIX ""
    OUTPUT_NAME "opslib"
)
