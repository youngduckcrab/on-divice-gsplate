cmake_minimum_required(VERSION 3.22)
project(gsengine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(GS_BUILD_TESTS "Build desktop tests and CLI tools" OFF)

# --- Vulkan -----------------------------------------------------------------
if(ANDROID)
  # The NDK sysroot provides vulkan headers and the libvulkan.so stub.
  set(GS_VULKAN_LIB vulkan)
else()
  find_package(Vulkan REQUIRED)
  set(GS_VULKAN_LIB Vulkan::Vulkan)
endif()

# --- Shader compiler ----------------------------------------------------------
if(ANDROID)
  file(GLOB _gs_ndk_glslc
    "${ANDROID_NDK}/shader-tools/*/glslc"
    "${CMAKE_ANDROID_NDK}/shader-tools/*/glslc")
  if(_gs_ndk_glslc)
    list(GET _gs_ndk_glslc 0 GS_GLSLC)
  endif()
endif()
if(NOT GS_GLSLC)
  find_program(GS_GLSLC glslc NO_CMAKE_FIND_ROOT_PATH)
endif()
if(NOT GS_GLSLC)
  message(FATAL_ERROR "glslc not found (install shaderc / glslc, or use an NDK with shader-tools)")
endif()
message(STATUS "gsengine: using glslc at ${GS_GLSLC}")

# Each shader is compiled to SPIR-V and embedded as a uint32_t array header
# (gs::shaders::<name>), so the engine needs no asset loading on any platform.
set(GS_SHADERS
  bandwidth_copy
  fma_fp32
  fma_fp16
  atomic_u32
  atomic_f32_native
  atomic_f32_cas
  radix_count
  radix_scan
  radix_scatter
)
set(GS_SHADER_OUT "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GS_SHADER_OUT}/shaders")
set(GS_SHADER_HEADERS "")
foreach(name IN LISTS GS_SHADERS)
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${name}.comp")
  set(spv "${GS_SHADER_OUT}/shaders/${name}.spv")
  set(hdr "${GS_SHADER_OUT}/shaders/${name}.h")
  add_custom_command(
    OUTPUT "${hdr}"
    COMMAND "${GS_GLSLC}" --target-env=vulkan1.1 -O -o "${spv}" "${src}"
    COMMAND "${CMAKE_COMMAND}" -DINPUT=${spv} -DOUTPUT=${hdr} -DNAME=${name}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake"
    DEPENDS "${src}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake"
    COMMENT "Compiling shader ${name}"
    VERBATIM)
  list(APPEND GS_SHADER_HEADERS "${hdr}")
endforeach()
add_custom_target(gs_shaders DEPENDS ${GS_SHADER_HEADERS})

# --- Engine library -----------------------------------------------------------
add_library(gsengine STATIC
  src/benchmarks.cpp
  src/device_probe.cpp
  src/log.cpp
  src/radix_sort.cpp
  src/vk_context.cpp
)
add_dependencies(gsengine gs_shaders)
target_include_directories(gsengine
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${GS_SHADER_OUT}")
target_link_libraries(gsengine PUBLIC ${GS_VULKAN_LIB})
if(ANDROID)
  target_link_libraries(gsengine PUBLIC log)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(gsengine PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
endif()

# --- Desktop tests and tools --------------------------------------------------
if(GS_BUILD_TESTS AND NOT ANDROID)
  enable_testing()
  add_executable(gs_tests tests/test_engine.cpp)
  target_link_libraries(gs_tests PRIVATE gsengine)
  add_test(NAME gs_tests COMMAND gs_tests)

  add_executable(gs_probe cli/gs_probe.cpp)
  target_link_libraries(gs_probe PRIVATE gsengine)
endif()
