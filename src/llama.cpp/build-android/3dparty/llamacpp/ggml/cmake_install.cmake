# Install script for directory: /home/tianruiming/InfraPowerTest/src/llama.cpp/3dparty/llamacpp/ggml

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "TRUE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/home/tianruiming/Android/Sdk/ndk/26.3.11579264/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/tianruiming/InfraPowerTest/src/llama.cpp/build-android/3dparty/llamacpp/ggml/src/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/tianruiming/InfraPowerTest/src/llama.cpp/build-android/bin/libggml.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/home/tianruiming/Android/Sdk/ndk/26.3.11579264/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/home/tianruiming/InfraPowerTest/src/llama.cpp/3dparty/llamacpp/ggml/include/ggml.h"
    "/home/tianruiming/InfraPowerTest/src/llama.cpp/3dparty/llamacpp/ggml/include/ggml-cpu.h"
    "/home/tianruiming/InfraPowerTest/src/llama.cpp/3dparty/llamacpp/ggml/include/ggml-alloc.h"
    "/home/tianruiming/InfraPowerTest/src/llama.cpp/3dparty/llamacpp/ggml/include/ggml-backend.h"
    "/home/tianruiming/InfraPowerTest/src/llama.cpp/3dparty/llamacpp/ggml/include/ggml-blas.h"
    "/home/tianruiming/InfraPowerTest/src/llama.cpp/3dparty/llamacpp/ggml/include/ggml-cann.h"
    "/home/tianruiming/InfraPowerTest/src/llama.cpp/3dparty/llamacpp/ggml/include/ggml-cpp.h"
    "/home/tianruiming/InfraPowerTest/src/llama.cpp/3dparty/llamacpp/ggml/include/ggml-cuda.h"
    "/home/tianruiming/InfraPowerTest/src/llama.cpp/3dparty/llamacpp/ggml/include/ggml-opt.h"
    "/home/tianruiming/InfraPowerTest/src/llama.cpp/3dparty/llamacpp/ggml/include/ggml-metal.h"
    "/home/tianruiming/InfraPowerTest/src/llama.cpp/3dparty/llamacpp/ggml/include/ggml-rpc.h"
    "/home/tianruiming/InfraPowerTest/src/llama.cpp/3dparty/llamacpp/ggml/include/ggml-sycl.h"
    "/home/tianruiming/InfraPowerTest/src/llama.cpp/3dparty/llamacpp/ggml/include/ggml-vulkan.h"
    "/home/tianruiming/InfraPowerTest/src/llama.cpp/3dparty/llamacpp/ggml/include/ggml-webgpu.h"
    "/home/tianruiming/InfraPowerTest/src/llama.cpp/3dparty/llamacpp/ggml/include/gguf.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml-base.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml-base.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml-base.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/tianruiming/InfraPowerTest/src/llama.cpp/build-android/bin/libggml-base.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml-base.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml-base.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/home/tianruiming/Android/Sdk/ndk/26.3.11579264/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml-base.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ggml" TYPE FILE FILES
    "/home/tianruiming/InfraPowerTest/src/llama.cpp/build-android/3dparty/llamacpp/ggml/ggml-config.cmake"
    "/home/tianruiming/InfraPowerTest/src/llama.cpp/build-android/3dparty/llamacpp/ggml/ggml-version.cmake"
    )
endif()

