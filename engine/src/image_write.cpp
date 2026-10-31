#include "gs/png_write.h"

#include <stdexcept>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_WRITE_NO_STDIO
#include "stb_image_write.h"

namespace gs {

std::vector<uint8_t> encode_jpeg_rgb(const uint8_t* rgb, uint32_t width, uint32_t height, int quality) {
  std::vector<uint8_t> out;
  auto append = [](void* context, void* data, int size) {
    auto* bytes = static_cast<std::vector<uint8_t>*>(context);
    bytes->insert(bytes->end(), static_cast<uint8_t*>(data), static_cast<uint8_t*>(data) + size);
  };
  if (stbi_write_jpg_to_func(append, &out, int(width), int(height), 3, rgb, quality) == 0) {
    throw std::runtime_error("JPEG 인코딩 실패");
  }
  return out;
}

}  // namespace gs
