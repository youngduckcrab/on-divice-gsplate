#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gs {

struct Image8 {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t channels = 0;
  std::vector<uint8_t> data;
};

struct Image16 {
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<uint16_t> data;  // single channel
};

std::vector<uint8_t> read_file(const std::string& path);  // throws std::runtime_error

// JPEG or PNG, converted to `channels` (1..4). Throws std::runtime_error.
Image8 decode_image(const std::vector<uint8_t>& bytes, int channels);
// 16-bit (or 8-bit, widened) single-channel PNG. Throws std::runtime_error.
Image16 decode_image16(const std::vector<uint8_t>& bytes);

// Area-average downscale (also handles equal size). `channels` interleaved.
std::vector<uint8_t> resize_area(const uint8_t* src, uint32_t src_w, uint32_t src_h, uint32_t channels,
                                 uint32_t dst_w, uint32_t dst_h);

}  // namespace gs
