project(hello)
cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_CXX_FLAGS "-I/usr/include/eigen3 -L${CMAKE_BINARY_DIR}/lib/")

find_package(Libint2 REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(hello helloworld.cpp)
