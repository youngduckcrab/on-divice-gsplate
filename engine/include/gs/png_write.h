#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gs {

// Single-channel PNG files: 8-bit (masks, depth confidence) and 16-bit
// (depth in millimetres), the formats the dataset loader reads.
std::vector<uint8_t> encode_png_gray8(const uint8_t* pixels, uint32_t width, uint32_t height);
std::vector<uint8_t> encode_png_gray16(const uint16_t* pixels, uint32_t width, uint32_t height);
// Baseline JPEG of an RGB image; quality 90 or less keeps chroma at half
// resolution, as camera images already have it.
std::vector<uint8_t> encode_jpeg_rgb(const uint8_t* rgb, uint32_t width, uint32_t height, int quality = 90);
// Writes `bytes` to `path`; throws std::runtime_error.
void write_file(const std::string& path, const std::vector<uint8_t>& bytes);

}  // namespace gs
