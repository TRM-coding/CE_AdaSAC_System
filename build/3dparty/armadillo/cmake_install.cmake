# Install script for directory: /home/Science_research/Eckart-Young-based-ML-Inference-framework/3dparty/armadillo

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/Science_research/Eckart-Young-based-ML-Inference-framework/out/install/gcc 11.3.0")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
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
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/Science_research/Eckart-Young-based-ML-Inference-framework/build/3dparty/armadillo/tmp/include/" REGEX "/\\.git$" EXCLUDE REGEX "/[^/]*\\.cmake$" EXCLUDE REGEX "/[^/]*\\~$" EXCLUDE REGEX "/[^/]*orig$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libarmadillo.so.14.0.2"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libarmadillo.so.14"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/Science_research/Eckart-Young-based-ML-Inference-framework/build/3dparty/armadillo/libarmadillo.so.14.0.2"
    "/home/Science_research/Eckart-Young-based-ML-Inference-framework/build/3dparty/armadillo/libarmadillo.so.14"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libarmadillo.so.14.0.2"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libarmadillo.so.14"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/Science_research/Eckart-Young-based-ML-Inference-framework/build/3dparty/armadillo/libarmadillo.so")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/Science_research/Eckart-Young-based-ML-Inference-framework/build/3dparty/armadillo/tmp/misc/armadillo.pc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/Armadillo/CMake/ArmadilloLibraryDepends.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/Armadillo/CMake/ArmadilloLibraryDepends.cmake"
         "/home/Science_research/Eckart-Young-based-ML-Inference-framework/build/3dparty/armadillo/CMakeFiles/Export/d5f1c8e7a348faccefbb4ca918240c99/ArmadilloLibraryDepends.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/Armadillo/CMake/ArmadilloLibraryDepends-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/Armadillo/CMake/ArmadilloLibraryDepends.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/Armadillo/CMake" TYPE FILE FILES "/home/Science_research/Eckart-Young-based-ML-Inference-framework/build/3dparty/armadillo/CMakeFiles/Export/d5f1c8e7a348faccefbb4ca918240c99/ArmadilloLibraryDepends.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/Armadillo/CMake" TYPE FILE FILES "/home/Science_research/Eckart-Young-based-ML-Inference-framework/build/3dparty/armadillo/CMakeFiles/Export/d5f1c8e7a348faccefbb4ca918240c99/ArmadilloLibraryDepends-debug.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/Science_research/Eckart-Young-based-ML-Inference-framework/out/install/gcc 11.3.0/share/Armadillo/CMake/ArmadilloConfig.cmake;/home/Science_research/Eckart-Young-based-ML-Inference-framework/out/install/gcc 11.3.0/share/Armadillo/CMake/ArmadilloConfigVersion.cmake")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/Science_research/Eckart-Young-based-ML-Inference-framework/out/install/gcc 11.3.0/share/Armadillo/CMake" TYPE FILE FILES
    "/home/Science_research/Eckart-Young-based-ML-Inference-framework/build/3dparty/armadillo/InstallFiles/ArmadilloConfig.cmake"
    "/home/Science_research/Eckart-Young-based-ML-Inference-framework/build/3dparty/armadillo/InstallFiles/ArmadilloConfigVersion.cmake"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/Science_research/Eckart-Young-based-ML-Inference-framework/build/3dparty/armadillo/tests1/cmake_install.cmake")

endif()

