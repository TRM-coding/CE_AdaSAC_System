file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(opslib SHARED ${SRC_FILES})

target_include_directories(
    opslib PUBLIC ${CMAKE_SOURCE_DIR}/llamacpp/include 
    ${CMAKE_SOURCE_DIR}/ops
    ${CMAKE_SOURCE_DIR}/llamacpp/ggml/include)