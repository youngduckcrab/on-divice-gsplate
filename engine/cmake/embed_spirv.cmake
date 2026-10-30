# Converts a SPIR-V binary into a C++ header exposing it as a uint32_t array.
# Usage: cmake -DINPUT=<file.spv> -DOUTPUT=<file.h> -DNAME=<identifier> -P embed_spirv.cmake
file(READ "${INPUT}" hex HEX)
string(LENGTH "${hex}" hex_len)
math(EXPR remainder "${hex_len} % 8")
if(NOT remainder EQUAL 0)
  message(FATAL_ERROR "${INPUT}: size is not a multiple of 4 bytes")
endif()
# SPIR-V words are little-endian: bytes aa bb cc dd -> 0xddccbbaa.
string(REGEX REPLACE "(..)(..)(..)(..)" "0x\\4\\3\\2\\1u," words "${hex}")
file(WRITE "${OUTPUT}"
  "// Generated from ${NAME}.comp - do not edit.\n"
  "#pragma once\n"
  "#include <cstdint>\n"
  "namespace gs::shaders {\n"
  "inline constexpr uint32_t ${NAME}[] = {${words}};\n"
  "}  // namespace gs::shaders\n")
