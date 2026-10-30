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
# (gs::shaders::<name>: radix_count.comp -> radix_count, splat.vert -> splat_vert),
# so the engine needs no asset loading on any platform.
set(GS_SHADERS
  bandwidth_copy.comp
  fma_fp32.comp
  fma_fp16.comp
  atomic_u32.comp
  atomic_f32_native.comp
  atomic_f32_cas.comp
  radix_count.comp
  radix_scan.comp
  radix_scatter.comp
  splat_preprocess.comp
  splat.vert
  splat.frag
  scan_local.comp
  scan_add.comp
  train_project.comp
  train_emit.comp
  train_ranges.comp
  train_forward.comp
  train_backward.comp
  train_backward_shared.comp
  train_step.comp
  train_densify_mark.comp
  train_densify_spawn.comp
  train_reset_opacity.comp
)
file(GLOB GS_SHADER_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.glsl")
set(GS_SHADER_OUT "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GS_SHADER_OUT}/shaders")
set(GS_SHADER_HEADERS "")
foreach(file IN LISTS GS_SHADERS)
  string(REGEX REPLACE "\\.comp$" "" name "${file}")
  string(REPLACE "." "_" name "${name}")
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${file}")
  set(spv "${GS_SHADER_OUT}/shaders/${name}.spv")
  set(hdr "${GS_SHADER_OUT}/shaders/${name}.h")
  add_custom_command(
    OUTPUT "${hdr}"
    COMMAND "${GS_GLSLC}" --target-env=vulkan1.1 -O -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders" -o "${spv}" "${src}"
    COMMAND "${CMAKE_COMMAND}" -DINPUT=${spv} -DOUTPUT=${hdr} -DNAME=${name}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake"
    DEPENDS "${src}" ${GS_SHADER_INCLUDES} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake"
    COMMENT "Compiling shader ${file}"
    VERBATIM)
  list(APPEND GS_SHADER_HEADERS "${hdr}")
endforeach()
add_custom_target(gs_shaders DEPENDS ${GS_SHADER_HEADERS})

# --- Engine library -----------------------------------------------------------
add_library(gsengine STATIC
  src/benchmarks.cpp
  src/camera.cpp
  src/dataset.cpp
  src/device_probe.cpp
  src/image_io.cpp
  src/json_parse.cpp
  src/log.cpp
  src/offscreen.cpp
  src/radix_sort.cpp
  src/splat_renderer.cpp
  src/splat_scene.cpp
  src/train_selftest.cpp
  src/trainer.cpp
  src/vk_context.cpp
)
add_dependencies(gsengine gs_shaders)
# Public headers use C++20 (concepts, std::span), so consumers need it too.
target_compile_features(gsengine PUBLIC cxx_std_20)
target_include_directories(gsengine
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${GS_SHADER_OUT}" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb")
target_link_libraries(gsengine PUBLIC ${GS_VULKAN_LIB})
if(ANDROID)
  target_link_libraries(gsengine PUBLIC log z)
else()
  find_package(ZLIB REQUIRED)
  target_link_libraries(gsengine PUBLIC ZLIB::ZLIB)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(gsengine PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
  # Third-party single-header code is compiled in image_io.cpp only.
  set_source_files_properties(src/image_io.cpp PROPERTIES COMPILE_OPTIONS "-w")
endif()

# --- Desktop tests and tools --------------------------------------------------
if(GS_BUILD_TESTS AND NOT ANDROID)
  enable_testing()
  add_executable(gs_tests tests/test_engine.cpp)
  target_link_libraries(gs_tests PRIVATE gsengine)
  add_test(NAME gs_tests COMMAND gs_tests)

  add_executable(gs_splat_tests tests/test_splats.cpp)
  target_link_libraries(gs_splat_tests PRIVATE gsengine)
  add_test(NAME gs_splat_tests COMMAND gs_splat_tests)

  add_executable(gs_trainer_tests tests/test_trainer.cpp)
  target_link_libraries(gs_trainer_tests PRIVATE gsengine)
  add_test(NAME gs_trainer_tests COMMAND gs_trainer_tests)

  add_executable(gs_train cli/gs_train.cpp)
  target_link_libraries(gs_train PRIVATE gsengine)

  add_executable(gs_probe cli/gs_probe.cpp)
  target_link_libraries(gs_probe PRIVATE gsengine)

  add_executable(gs_render cli/gs_render.cpp)
  target_link_libraries(gs_render PRIVATE gsengine)
endif()
