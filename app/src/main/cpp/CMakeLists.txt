cmake_minimum_required(VERSION 3.22)
project(gsplat_jni LANGUAGES CXX)

# The engine lives at the repository root so it can also be built and tested on Linux.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../../engine" "${CMAKE_CURRENT_BINARY_DIR}/engine")

add_library(gsplat_jni SHARED jni_bridge.cpp)
target_link_libraries(gsplat_jni PRIVATE gsengine android log)
