#include "gs/png_write.h"

#include <zlib.h>

#include <cstdio>
#include <stdexcept>

namespace gs {
namespace {

void put_u32(std::vector<uint8_t>& out, uint32_t v) {
  for (int s = 24; s >= 0; s -= 8) out.push_back(uint8_t(v >> s));
}

void chunk(std::vector<uint8_t>& out, const char type[4], const std::vector<uint8_t>& data) {
  put_u32(out, uint32_t(data.size()));
  const size_t start = out.size();
  out.insert(out.end(), type, type + 4);
  out.insert(out.end(), data.begin(), data.end());
  const uLong crc = crc32(0L, out.data() + start, uInt(out.size() - start));
  put_u32(out, uint32_t(crc));
}

// `rows` holds each scanline (big-endian samples) prefixed by filter type 0.
std::vector<uint8_t> encode(const std::vector<uint8_t>& rows, uint32_t width, uint32_t height, uint8_t depth) {
  std::vector<uint8_t> out = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
  std::vector<uint8_t> header;
  put_u32(header, width);
  put_u32(header, height);
  header.insert(header.end(), {depth, 0 /* greyscale */, 0, 0, 0});
  chunk(out, "IHDR", header);
  uLongf size = compressBound(uLong(rows.size()));
  std::vector<uint8_t> packed(size);
  if (compress2(packed.data(), &size, rows.data(), uLong(rows.size()), 6) != Z_OK) {
    throw std::runtime_error("PNG 압축 실패");
  }
  packed.resize(size);
  chunk(out, "IDAT", packed);
  chunk(out, "IEND", {});
  return out;
}

}  // namespace

std::vector<uint8_t> encode_png_gray8(const uint8_t* pixels, uint32_t width, uint32_t height) {
  std::vector<uint8_t> rows;
  rows.reserve(size_t(width + 1) * height);
  for (uint32_t y = 0; y < height; ++y) {
    rows.push_back(0);
    rows.insert(rows.end(), pixels + size_t(y) * width, pixels + size_t(y + 1) * width);
  }
  return encode(rows, width, height, 8);
}

std::vector<uint8_t> encode_png_gray16(const uint16_t* pixels, uint32_t width, uint32_t height) {
  std::vector<uint8_t> rows;
  rows.reserve(size_t(2 * width + 1) * height);
  for (uint32_t y = 0; y < height; ++y) {
    rows.push_back(0);
    for (uint32_t x = 0; x < width; ++x) {
      const uint16_t v = pixels[size_t(y) * width + x];
      rows.push_back(uint8_t(v >> 8));
      rows.push_back(uint8_t(v));
    }
  }
  return encode(rows, width, height, 16);
}

void write_file(const std::string& path, const std::vector<uint8_t>& bytes) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) throw std::runtime_error("파일을 쓸 수 없음: " + path);
  const size_t written = std::fwrite(bytes.data(), 1, bytes.size(), f);
  const bool ok = std::fclose(f) == 0 && written == bytes.size();
  if (!ok) throw std::runtime_error("파일을 쓸 수 없음: " + path);
}

}  // namespace gs
