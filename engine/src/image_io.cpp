#include "image_io.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#include "stb_image.h"

namespace gs {

std::vector<uint8_t> read_file(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) throw std::runtime_error("파일을 열 수 없음: " + path);
  std::vector<uint8_t> data;
  uint8_t buffer[1 << 16];
  size_t n = 0;
  while ((n = std::fread(buffer, 1, sizeof(buffer), f)) > 0) data.insert(data.end(), buffer, buffer + n);
  std::fclose(f);
  return data;
}

Image8 decode_image(const std::vector<uint8_t>& bytes, int channels) {
  int w = 0, h = 0, n = 0;
  stbi_uc* pixels = stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()), &w, &h, &n, channels);
  if (pixels == nullptr) throw std::runtime_error(std::string("이미지를 해석할 수 없음: ") + stbi_failure_reason());
  Image8 image;
  image.width = static_cast<uint32_t>(w);
  image.height = static_cast<uint32_t>(h);
  image.channels = static_cast<uint32_t>(channels);
  image.data.assign(pixels, pixels + size_t(w) * h * channels);
  stbi_image_free(pixels);
  return image;
}

Image16 decode_image16(const std::vector<uint8_t>& bytes) {
  int w = 0, h = 0, n = 0;
  stbi_us* pixels = stbi_load_16_from_memory(bytes.data(), static_cast<int>(bytes.size()), &w, &h, &n, 1);
  if (pixels == nullptr) throw std::runtime_error(std::string("16-bit 이미지를 해석할 수 없음: ") + stbi_failure_reason());
  Image16 image;
  image.width = static_cast<uint32_t>(w);
  image.height = static_cast<uint32_t>(h);
  image.data.assign(pixels, pixels + size_t(w) * h);
  stbi_image_free(pixels);
  return image;
}

std::vector<uint8_t> resize_area(const uint8_t* src, uint32_t src_w, uint32_t src_h, uint32_t channels,
                                 uint32_t dst_w, uint32_t dst_h) {
  // Separable box filter with fractional coverage: horizontal pass, then vertical.
  auto pass = [](const float* in, uint32_t in_len, uint32_t out_len, uint32_t lines, size_t in_line_stride,
                 size_t in_step, size_t out_line_stride, size_t out_step, uint32_t channels, float* out) {
    const double ratio = double(in_len) / out_len;
    for (uint32_t o = 0; o < out_len; ++o) {
      const double x0 = o * ratio, x1 = (o + 1) * ratio;
      const uint32_t i0 = static_cast<uint32_t>(std::floor(x0));
      const uint32_t i1 = std::min(in_len, static_cast<uint32_t>(std::ceil(x1)));
      for (uint32_t line = 0; line < lines; ++line) {
        for (uint32_t c = 0; c < channels; ++c) {
          double sum = 0.0;
          for (uint32_t i = i0; i < i1; ++i) {
            const double w = std::min<double>(i + 1, x1) - std::max<double>(i, x0);
            sum += w * in[line * in_line_stride + i * in_step + c];
          }
          out[line * out_line_stride + o * out_step + c] = static_cast<float>(sum / ratio);
        }
      }
    }
  };
  std::vector<float> in(size_t(src_w) * src_h * channels);
  for (size_t i = 0; i < in.size(); ++i) in[i] = src[i];
  std::vector<float> horizontal(size_t(dst_w) * src_h * channels);
  pass(in.data(), src_w, dst_w, src_h, size_t(src_w) * channels, channels, size_t(dst_w) * channels, channels,
       channels, horizontal.data());
  std::vector<float> vertical(size_t(dst_w) * dst_h * channels);
  // Vertical pass: "lines" are columns.
  pass(horizontal.data(), src_h, dst_h, dst_w, channels, size_t(dst_w) * channels, channels,
       size_t(dst_w) * channels, channels, vertical.data());
  std::vector<uint8_t> out(vertical.size());
  for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<uint8_t>(std::clamp(vertical[i] + 0.5f, 0.0f, 255.0f));
  return out;
}

}  // namespace gs
