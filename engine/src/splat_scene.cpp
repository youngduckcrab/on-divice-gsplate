#include "gs/splat_scene.h"

#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <array>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace gs {
namespace {

constexpr float kShC0 = 0.28209479177387814f;
constexpr float kPi = 3.14159265358979f;

float logit(float p) {
  p = std::clamp(p, 0.5f / 255.0f, 1.0f - 0.5f / 255.0f);
  return std::log(p / (1.0f - p));
}

int degree_for_rest_coeffs(int coeffs) {
  switch (coeffs) {
    case 0: return 0;
    case 3: return 1;
    case 8: return 2;
    case 15: return 3;
    case 24: return 4;
    default: return -1;
  }
}

// --- PLY ------------------------------------------------------------------------

enum class PlyType { I8, U8, I16, U16, I32, U32, F32, F64 };

bool parse_ply_type(std::string_view name, PlyType* type, size_t* size) {
  struct Entry {
    std::string_view name;
    PlyType type;
    size_t size;
  };
  static const Entry kTypes[] = {
      {"char", PlyType::I8, 1},     {"int8", PlyType::I8, 1},       {"uchar", PlyType::U8, 1},
      {"uint8", PlyType::U8, 1},    {"short", PlyType::I16, 2},     {"int16", PlyType::I16, 2},
      {"ushort", PlyType::U16, 2},  {"uint16", PlyType::U16, 2},    {"int", PlyType::I32, 4},
      {"int32", PlyType::I32, 4},   {"uint", PlyType::U32, 4},      {"uint32", PlyType::U32, 4},
      {"float", PlyType::F32, 4},   {"float32", PlyType::F32, 4},   {"double", PlyType::F64, 8},
      {"float64", PlyType::F64, 8},
  };
  for (const Entry& e : kTypes) {
    if (e.name == name) {
      *type = e.type;
      *size = e.size;
      return true;
    }
  }
  return false;
}

template <typename T>
T read_le(const uint8_t* p) {
  T v;
  std::memcpy(&v, p, sizeof(T));
  return v;
}

float read_ply_value(const uint8_t* p, PlyType type) {
  switch (type) {
    case PlyType::I8: return static_cast<float>(read_le<int8_t>(p));
    case PlyType::U8: return static_cast<float>(read_le<uint8_t>(p));
    case PlyType::I16: return static_cast<float>(read_le<int16_t>(p));
    case PlyType::U16: return static_cast<float>(read_le<uint16_t>(p));
    case PlyType::I32: return static_cast<float>(read_le<int32_t>(p));
    case PlyType::U32: return static_cast<float>(read_le<uint32_t>(p));
    case PlyType::F32: return read_le<float>(p);
    case PlyType::F64: return static_cast<float>(read_le<double>(p));
  }
  return 0.0f;
}

struct PlyProperty {
  std::string name;
  PlyType type;
  size_t offset;
};

struct PlyElement {
  std::string name;
  uint64_t count = 0;
  size_t stride = 0;
  std::vector<PlyProperty> properties;
};

std::vector<std::string_view> split_words(std::string_view line) {
  std::vector<std::string_view> words;
  size_t i = 0;
  while (i < line.size()) {
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t' || line[i] == '\r')) ++i;
    size_t start = i;
    while (i < line.size() && line[i] != ' ' && line[i] != '\t' && line[i] != '\r') ++i;
    if (i > start) words.push_back(line.substr(start, i - start));
  }
  return words;
}

// --- SPZ ------------------------------------------------------------------------

constexpr uint32_t kSpzMagic = 0x5053474e;  // "NGSP"
constexpr float kSpzColorScale = 0.15f;

std::vector<uint8_t> gunzip(const uint8_t* data, size_t size) {
  z_stream stream{};
  stream.next_in = const_cast<Bytef*>(data);
  stream.avail_in = static_cast<uInt>(size);
  if (inflateInit2(&stream, 16 + MAX_WBITS) != Z_OK) throw std::runtime_error("gzip 초기화 실패");
  std::vector<uint8_t> out;
  out.reserve(size * 4);
  uint8_t buffer[1 << 16];
  int result = Z_OK;
  while (result == Z_OK) {
    stream.next_out = buffer;
    stream.avail_out = sizeof(buffer);
    result = inflate(&stream, Z_NO_FLUSH);
    if (result != Z_OK && result != Z_STREAM_END) break;
    out.insert(out.end(), buffer, buffer + (sizeof(buffer) - stream.avail_out));
  }
  inflateEnd(&stream);
  if (result != Z_STREAM_END) throw std::runtime_error("SPZ 압축 해제 실패 (파일이 손상됐을 수 있음)");
  return out;
}

// Ported from nianticlabs/spz (MIT): "smallest three" quaternion packing used by SPZ v3.
void unpack_quaternion_smallest_three(const uint8_t r[4], float xyzw[4]) {
  uint32_t comp = uint32_t(r[0]) | (uint32_t(r[1]) << 8) | (uint32_t(r[2]) << 16) | (uint32_t(r[3]) << 24);
  constexpr uint32_t kMask = (1u << 9) - 1;
  const int largest = static_cast<int>(comp >> 30);
  float sum_squares = 0.0f;
  for (int i = 3; i >= 0; --i) {
    if (i == largest) continue;
    uint32_t magnitude = comp & kMask;
    uint32_t negative = (comp >> 9) & 1u;
    comp >>= 10;
    float v = 0.70710678f * static_cast<float>(magnitude) / static_cast<float>(kMask);
    xyzw[i] = negative ? -v : v;
    sum_squares += xyzw[i] * xyzw[i];
  }
  xyzw[largest] = std::sqrt(std::max(0.0f, 1.0f - sum_squares));
}

}  // namespace

int sh_rest_coeffs(int degree) {
  return degree <= 0 ? 0 : (degree + 1) * (degree + 1) - 1;
}

void SplatScene::resize(uint32_t n, int degree) {
  count = n;
  sh_degree = degree;
  positions.assign(size_t(n) * 3, 0.0f);
  log_scales.assign(size_t(n) * 3, 0.0f);
  rotations.assign(size_t(n) * 4, 0.0f);
  opacity_logits.assign(n, 0.0f);
  sh_dc.assign(size_t(n) * 3, 0.0f);
  sh_rest.assign(size_t(n) * sh_rest_coeffs(degree) * 3, 0.0f);
}

SceneBounds compute_bounds(const SplatScene& scene) {
  SceneBounds bounds{{0.0f, 0.0f, 0.0f}, 1.0f};
  const size_t n = scene.count;
  if (n == 0) return bounds;
  std::vector<float> values(n);
  for (int axis = 0; axis < 3; ++axis) {
    for (size_t i = 0; i < n; ++i) values[i] = scene.positions[i * 3 + axis];
    std::nth_element(values.begin(), values.begin() + n / 2, values.end());
    bounds.center[axis] = values[n / 2];
  }
  for (size_t i = 0; i < n; ++i) {
    float dx = scene.positions[i * 3 + 0] - bounds.center[0];
    float dy = scene.positions[i * 3 + 1] - bounds.center[1];
    float dz = scene.positions[i * 3 + 2] - bounds.center[2];
    values[i] = std::sqrt(dx * dx + dy * dy + dz * dz);
  }
  size_t k = std::min(n - 1, n * 9 / 10);
  std::nth_element(values.begin(), values.begin() + k, values.end());
  bounds.radius = std::max(values[k], 1e-3f);
  return bounds;
}

SplatScene load_ply(const uint8_t* data, size_t size) {
  const std::string_view text(reinterpret_cast<const char*>(data), std::min<size_t>(size, 1 << 16));
  const size_t header_end = text.find("end_header\n");
  if (text.substr(0, 3) != "ply" || header_end == std::string_view::npos) {
    throw std::runtime_error("PLY 헤더를 찾을 수 없음");
  }
  const size_t body_offset = header_end + std::strlen("end_header\n");

  std::vector<PlyElement> elements;
  size_t pos = 0;
  while (pos < header_end) {
    size_t eol = text.find('\n', pos);
    std::vector<std::string_view> w = split_words(text.substr(pos, eol - pos));
    pos = eol + 1;
    if (w.empty()) continue;
    if (w[0] == "format") {
      if (w.size() < 2 || w[1] != "binary_little_endian") {
        throw std::runtime_error("binary_little_endian PLY만 지원 (이 파일: " +
                                 std::string(w.size() > 1 ? w[1] : "?") + ")");
      }
    } else if (w[0] == "element" && w.size() >= 3) {
      PlyElement e;
      e.name = std::string(w[1]);
      e.count = std::stoull(std::string(w[2]));
      elements.push_back(std::move(e));
    } else if (w[0] == "property" && !elements.empty()) {
      if (w.size() >= 2 && w[1] == "list") throw std::runtime_error("list 속성이 있는 PLY는 지원하지 않음");
      PlyType type;
      size_t type_size;
      if (w.size() < 3 || !parse_ply_type(w[1], &type, &type_size)) {
        throw std::runtime_error("알 수 없는 PLY 속성 형식");
      }
      PlyElement& e = elements.back();
      e.properties.push_back({std::string(w[2]), type, e.stride});
      e.stride += type_size;
    }
  }

  size_t offset = body_offset;
  const PlyElement* vertex = nullptr;
  for (const PlyElement& e : elements) {
    if (e.name == "vertex") {
      vertex = &e;
      break;
    }
    offset += e.count * e.stride;
  }
  if (vertex == nullptr || vertex->count == 0) throw std::runtime_error("PLY에 vertex가 없음");
  if (vertex->count > 0xffffffffull || offset + vertex->count * vertex->stride > size) {
    throw std::runtime_error("PLY 파일이 잘려 있음 (크기가 헤더와 맞지 않음)");
  }

  std::unordered_map<std::string, const PlyProperty*> props;
  for (const PlyProperty& p : vertex->properties) props[p.name] = &p;
  auto require = [&](const std::string& name) {
    auto it = props.find(name);
    if (it == props.end()) {
      if (name == "f_dc_0" && props.count("packed_position")) {
        throw std::runtime_error("압축 PLY(SuperSplat 등)는 아직 지원하지 않음");
      }
      throw std::runtime_error("가우시안 스플랫 PLY가 아님 (" + name + " 속성 없음)");
    }
    return it->second;
  };

  int rest_count = 0;
  while (props.count("f_rest_" + std::to_string(rest_count))) ++rest_count;
  const int file_coeffs = rest_count / 3;
  const int file_degree = degree_for_rest_coeffs(file_coeffs);
  if (file_degree < 0) throw std::runtime_error("f_rest 개수가 SH 차수와 맞지 않음");
  const int degree = std::min(file_degree, 3);
  const int coeffs = sh_rest_coeffs(degree);

  const PlyProperty* xyz[3] = {require("x"), require("y"), require("z")};
  const PlyProperty* dc[3] = {require("f_dc_0"), require("f_dc_1"), require("f_dc_2")};
  const PlyProperty* opacity = require("opacity");
  const PlyProperty* scale[3] = {require("scale_0"), require("scale_1"), require("scale_2")};
  const PlyProperty* rot[4] = {require("rot_0"), require("rot_1"), require("rot_2"), require("rot_3")};
  std::vector<const PlyProperty*> rest(rest_count);
  for (int i = 0; i < rest_count; ++i) rest[i] = props["f_rest_" + std::to_string(i)];

  SplatScene scene;
  scene.convention = AxisConvention::RDF;
  scene.resize(static_cast<uint32_t>(vertex->count), degree);
  for (uint32_t i = 0; i < scene.count; ++i) {
    const uint8_t* v = data + offset + size_t(i) * vertex->stride;
    for (int k = 0; k < 3; ++k) {
      scene.positions[i * 3 + k] = read_ply_value(v + xyz[k]->offset, xyz[k]->type);
      scene.sh_dc[i * 3 + k] = read_ply_value(v + dc[k]->offset, dc[k]->type);
      scene.log_scales[i * 3 + k] = read_ply_value(v + scale[k]->offset, scale[k]->type);
    }
    for (int k = 0; k < 4; ++k) scene.rotations[i * 4 + k] = read_ply_value(v + rot[k]->offset, rot[k]->type);
    scene.opacity_logits[i] = read_ply_value(v + opacity->offset, opacity->type);
    // PLY stores f_rest channel-major: f_rest_[channel * file_coeffs + coeff].
    for (int j = 0; j < coeffs; ++j) {
      for (int c = 0; c < 3; ++c) {
        const PlyProperty* p = rest[c * file_coeffs + j];
        scene.sh_rest[(size_t(i) * coeffs + j) * 3 + c] = read_ply_value(v + p->offset, p->type);
      }
    }
  }
  return scene;
}

std::vector<uint8_t> save_ply(const SplatScene& scene) {
  const int coeffs = sh_rest_coeffs(scene.sh_degree);
  std::string header = "ply\nformat binary_little_endian 1.0\nelement vertex " + std::to_string(scene.count) +
                       "\nproperty float x\nproperty float y\nproperty float z\n"
                       "property float nx\nproperty float ny\nproperty float nz\n"
                       "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n";
  for (int i = 0; i < coeffs * 3; ++i) header += "property float f_rest_" + std::to_string(i) + "\n";
  header += "property float opacity\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\n"
            "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\nend_header\n";

  const size_t floats_per_vertex = 17 + size_t(coeffs) * 3;
  std::vector<uint8_t> out(header.begin(), header.end());
  std::vector<float> row(floats_per_vertex);
  out.reserve(out.size() + scene.count * floats_per_vertex * 4);
  for (uint32_t i = 0; i < scene.count; ++i) {
    size_t k = 0;
    for (int a = 0; a < 3; ++a) row[k++] = scene.positions[i * 3 + a];
    for (int a = 0; a < 3; ++a) row[k++] = 0.0f;
    for (int a = 0; a < 3; ++a) row[k++] = scene.sh_dc[i * 3 + a];
    for (int c = 0; c < 3; ++c) {
      for (int j = 0; j < coeffs; ++j) row[k++] = scene.sh_rest[(size_t(i) * coeffs + j) * 3 + c];
    }
    row[k++] = scene.opacity_logits[i];
    for (int a = 0; a < 3; ++a) row[k++] = scene.log_scales[i * 3 + a];
    for (int a = 0; a < 4; ++a) row[k++] = scene.rotations[i * 4 + a];
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(row.data());
    out.insert(out.end(), bytes, bytes + floats_per_vertex * 4);
  }
  return out;
}

SplatScene load_spz(const uint8_t* data, size_t size) {
  if (size >= 4 && read_le<uint32_t>(data) == kSpzMagic) {
    throw std::runtime_error("SPZ v4(zstd) 형식은 아직 지원하지 않음 (v2/v3만 가능)");
  }
  if (size < 2 || data[0] != 0x1f || data[1] != 0x8b) throw std::runtime_error("SPZ 파일이 아님");
  const std::vector<uint8_t> raw = gunzip(data, size);
  if (raw.size() < 16 || read_le<uint32_t>(raw.data()) != kSpzMagic) {
    throw std::runtime_error("SPZ 헤더가 올바르지 않음");
  }
  const uint32_t version = read_le<uint32_t>(raw.data() + 4);
  const uint32_t n = read_le<uint32_t>(raw.data() + 8);
  const int file_degree = raw[12];
  const int fractional_bits = raw[13];
  const uint8_t flags = raw[14];
  if (version < 2 || version > 3) {
    throw std::runtime_error("SPZ 버전 " + std::to_string(version) + "은 지원하지 않음 (v2/v3만 가능)");
  }
  if (file_degree > 4) throw std::runtime_error("SPZ SH 차수가 올바르지 않음");
  const int file_coeffs = sh_rest_coeffs(file_degree);
  const int degree = std::min(file_degree, 3);
  const int coeffs = sh_rest_coeffs(degree);
  const size_t rot_bytes = version >= 3 ? 4 : 3;

  const size_t positions = 16;
  const size_t alphas = positions + size_t(n) * 9;
  const size_t colors = alphas + n;
  const size_t scales = colors + size_t(n) * 3;
  const size_t rotations = scales + size_t(n) * 3;
  const size_t sh = rotations + size_t(n) * rot_bytes;
  const size_t end = sh + size_t(n) * file_coeffs * 3;
  if (n == 0 || raw.size() < end) throw std::runtime_error("SPZ 데이터가 잘려 있음");

  SplatScene scene;
  scene.convention = AxisConvention::RUB;
  scene.antialiased = (flags & 0x1) != 0;
  scene.resize(n, degree);
  const float position_scale = 1.0f / float(1 << fractional_bits);
  for (uint32_t i = 0; i < n; ++i) {
    for (int k = 0; k < 3; ++k) {
      const uint8_t* p = &raw[positions + (size_t(i) * 3 + k) * 3];
      int32_t fixed = int32_t(p[0]) | (int32_t(p[1]) << 8) | (int32_t(p[2]) << 16);
      if (fixed & 0x800000) fixed |= int32_t(0xff000000u);
      scene.positions[i * 3 + k] = float(fixed) * position_scale;
      scene.sh_dc[i * 3 + k] = (raw[colors + i * 3 + k] / 255.0f - 0.5f) / kSpzColorScale;
      scene.log_scales[i * 3 + k] = raw[scales + i * 3 + k] / 16.0f - 10.0f;
    }
    scene.opacity_logits[i] = logit(raw[alphas + i] / 255.0f);

    float xyzw[4];
    const uint8_t* r = &raw[rotations + size_t(i) * rot_bytes];
    if (version >= 3) {
      unpack_quaternion_smallest_three(r, xyzw);
    } else {
      for (int k = 0; k < 3; ++k) xyzw[k] = r[k] / 127.5f - 1.0f;
      xyzw[3] = std::sqrt(std::max(0.0f, 1.0f - (xyzw[0] * xyzw[0] + xyzw[1] * xyzw[1] + xyzw[2] * xyzw[2])));
    }
    scene.rotations[i * 4 + 0] = xyzw[3];
    scene.rotations[i * 4 + 1] = xyzw[0];
    scene.rotations[i * 4 + 2] = xyzw[1];
    scene.rotations[i * 4 + 3] = xyzw[2];

    // SPZ stores SH as [coeff][rgb] per splat, quantized around 128.
    for (int j = 0; j < coeffs; ++j) {
      for (int c = 0; c < 3; ++c) {
        uint8_t q = raw[sh + (size_t(i) * file_coeffs + j) * 3 + c];
        scene.sh_rest[(size_t(i) * coeffs + j) * 3 + c] = (float(q) - 128.0f) / 128.0f;
      }
    }
  }
  return scene;
}

SplatScene load_splat_file(const uint8_t* data, size_t size) {
  if (size >= 3 && std::memcmp(data, "ply", 3) == 0) return load_ply(data, size);
  if (size >= 4 && ((data[0] == 0x1f && data[1] == 0x8b) || read_le<uint32_t>(data) == kSpzMagic)) {
    return load_spz(data, size);
  }
  throw std::runtime_error("PLY 또는 SPZ 파일이 아님");
}

SplatScene make_synthetic_scene(uint32_t count, uint32_t seed) {
  SplatScene scene;
  scene.convention = AxisConvention::RUB;
  scene.resize(count, 1);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  std::normal_distribution<float> normal(0.0f, 1.0f);

  // Splats shrink as the count grows so the knot keeps a similar look.
  const float size_factor = std::cbrt(100000.0f / float(std::max<uint32_t>(count, 1)));
  for (uint32_t i = 0; i < count; ++i) {
    const float t = uniform(rng) * 2.0f * kPi;
    // (2,3) torus knot.
    auto knot = [](float u) {
      float r = 2.0f + std::cos(3.0f * u);
      return std::array<float, 3>{0.5f * r * std::cos(2.0f * u), 0.5f * std::sin(3.0f * u),
                                  0.5f * r * std::sin(2.0f * u)};
    };
    const auto p = knot(t);
    const auto q = knot(t + 1e-3f);
    float tangent[3] = {q[0] - p[0], q[1] - p[1], q[2] - p[2]};
    float len = std::sqrt(tangent[0] * tangent[0] + tangent[1] * tangent[1] + tangent[2] * tangent[2]);
    for (float& v : tangent) v /= len;

    const float spread = 0.07f;
    for (int k = 0; k < 3; ++k) scene.positions[i * 3 + k] = p[k] + spread * normal(rng);

    // Rotate the local x axis onto the tangent: q = (1 + x·t, x × t), normalised later.
    float w = 1.0f + tangent[0];
    float qx = 0.0f, qy = -tangent[2], qz = tangent[1];
    if (w < 1e-4f) {  // tangent is -x: rotate 180 degrees about y
      w = 0.0f;
      qx = 0.0f;
      qy = 1.0f;
      qz = 0.0f;
    }
    scene.rotations[i * 4 + 0] = w;
    scene.rotations[i * 4 + 1] = qx;
    scene.rotations[i * 4 + 2] = qy;
    scene.rotations[i * 4 + 3] = qz;

    const float jitter = 0.5f + uniform(rng);
    scene.log_scales[i * 3 + 0] = std::log(0.045f * size_factor * jitter);
    scene.log_scales[i * 3 + 1] = std::log(0.012f * size_factor * jitter);
    scene.log_scales[i * 3 + 2] = std::log(0.012f * size_factor * jitter);
    scene.opacity_logits[i] = logit(0.6f + 0.35f * uniform(rng));

    // Rainbow along the knot; SH degree 1 adds a view-dependent tint.
    const float hue = t / (2.0f * kPi);
    float rgb[3];
    for (int c = 0; c < 3; ++c) {
      float phase = hue + float(c) / 3.0f;
      rgb[c] = 0.5f + 0.45f * std::cos(2.0f * kPi * phase);
    }
    for (int c = 0; c < 3; ++c) scene.sh_dc[i * 3 + c] = (rgb[c] - 0.5f) / kShC0;
    for (int j = 0; j < 3; ++j) {
      for (int c = 0; c < 3; ++c) {
        scene.sh_rest[(size_t(i) * 3 + j) * 3 + c] = 0.25f * tangent[j] * (c == 2 ? 1.0f : -0.5f);
      }
    }
  }
  return scene;
}

}  // namespace gs
