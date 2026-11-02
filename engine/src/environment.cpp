#include "gs/environment.h"

#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#include "gs/json.h"
#include "gs/json_parse.h"

namespace gs {
namespace {

constexpr float kShC0 = 0.28209479177387814f;
constexpr float kPi = 3.14159265358979f;
constexpr float kLuma[3] = {0.2126f, 0.7152f, 0.0722f};

std::vector<uint8_t> gunzip(const uint8_t* data, size_t size) {
  z_stream stream{};
  if (inflateInit2(&stream, 16 + MAX_WBITS) != Z_OK) throw std::runtime_error("gzip 초기화 실패");
  stream.next_in = const_cast<Bytef*>(data);
  stream.avail_in = static_cast<uInt>(size);
  std::vector<uint8_t> out;
  std::vector<uint8_t> chunk(1 << 20);
  int result = Z_OK;
  while (result == Z_OK) {
    stream.next_out = chunk.data();
    stream.avail_out = static_cast<uInt>(chunk.size());
    result = inflate(&stream, Z_NO_FLUSH);
    out.insert(out.end(), chunk.data(), chunk.data() + (chunk.size() - stream.avail_out));
  }
  inflateEnd(&stream);
  if (result != Z_STREAM_END) throw std::runtime_error("장소 파일 압축 해제 실패");
  return out;
}

std::vector<uint8_t> gzip(const std::vector<uint8_t>& data) {
  z_stream stream{};
  if (deflateInit2(&stream, 6, Z_DEFLATED, 16 + MAX_WBITS, 8, Z_DEFAULT_STRATEGY) != Z_OK) {
    throw std::runtime_error("gzip init failed");
  }
  std::vector<uint8_t> out(deflateBound(&stream, static_cast<uLong>(data.size())) + 64);
  stream.next_in = const_cast<Bytef*>(data.data());
  stream.avail_in = static_cast<uInt>(data.size());
  stream.next_out = out.data();
  stream.avail_out = static_cast<uInt>(out.size());
  const int result = deflate(&stream, Z_FINISH);
  deflateEnd(&stream);
  if (result != Z_STREAM_END) throw std::runtime_error("gzip failed");
  out.resize(stream.total_out);
  return out;
}

uint32_t read_u32(const uint8_t* p) { return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24; }

void put_u32(std::vector<uint8_t>& out, uint32_t v) {
  for (int k = 0; k < 4; ++k) out.push_back(uint8_t(v >> (8 * k)));
}

std::vector<float> numbers(const JsonValue* v) {
  std::vector<float> out;
  if (v == nullptr || v->type != JsonValue::Type::Array) return out;
  for (const JsonValue& e : v->array) out.push_back(float(e.number));
  return out;
}

// 256 values spread over the sorted data, and the nearest one for each value.
std::vector<float> codebook(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  std::vector<float> book(256, 0.0f);
  for (int k = 0; k < 256; ++k) {
    if (values.empty()) break;
    book[size_t(k)] = values[std::min(values.size() - 1, size_t((double(k) + 0.5) / 256.0 * values.size()))];
  }
  return book;
}

uint8_t nearest(const std::vector<float>& book, float v) {
  const auto it = std::lower_bound(book.begin(), book.end(), v);
  size_t i = size_t(it - book.begin());
  if (i == book.size()) return 255;
  if (i > 0 && std::abs(book[i - 1] - v) <= std::abs(book[i] - v)) --i;
  return uint8_t(i);
}

float srgb_to_linear(float c) { return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f); }

void quat_mul(const float a[4], const float b[4], float out[4]) {
  out[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
  out[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
  out[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
  out[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

}  // namespace

SplatScene read_gsenv(const uint8_t* data, size_t size) {
  const std::vector<uint8_t> raw = gunzip(data, size);
  if (raw.size() < 12 || std::memcmp(raw.data(), "GSEV", 4) != 0) throw std::runtime_error("장소 파일이 아님");
  if (read_u32(&raw[4]) != 1) throw std::runtime_error("모르는 장소 파일 버전");
  const uint32_t meta_len = read_u32(&raw[8]);
  if (12 + size_t(meta_len) > raw.size()) throw std::runtime_error("장소 파일이 잘려 있음");
  const JsonValue meta = parse_json(std::string_view(reinterpret_cast<const char*>(&raw[12]), meta_len));
  const uint32_t n = uint32_t(meta.number_or("count", 0));
  const JsonValue* means = meta.find("means");
  const std::vector<float> mins = numbers(means ? means->find("mins") : nullptr);
  const std::vector<float> maxs = numbers(means ? means->find("maxs") : nullptr);
  const JsonValue* scales = meta.find("scales");
  const JsonValue* sh0 = meta.find("sh0");
  const std::vector<float> scale_book = numbers(scales ? scales->find("codebook") : nullptr);
  const std::vector<float> colour_book = numbers(sh0 ? sh0->find("codebook") : nullptr);
  if (mins.size() != 3 || maxs.size() != 3 || scale_book.size() != 256 || colour_book.size() != 256) {
    throw std::runtime_error("장소 파일 정보가 이상함");
  }
  const size_t body = 12 + meta_len;
  if (raw.size() < body + size_t(n) * 17) throw std::runtime_error("장소 파일이 잘려 있음");
  const uint8_t* means_l = &raw[body];
  const uint8_t* means_u = means_l + size_t(n) * 3;
  const uint8_t* quats = means_u + size_t(n) * 3;
  const uint8_t* scale_idx = quats + size_t(n) * 4;
  const uint8_t* colour = scale_idx + size_t(n) * 3;

  SplatScene scene;
  scene.convention = AxisConvention::RDF;
  scene.antialiased = meta.find("antialias") != nullptr && meta.find("antialias")->boolean;
  scene.resize(n, 0);
  const float root2 = std::sqrt(2.0f);
  for (uint32_t i = 0; i < n; ++i) {
    for (int k = 0; k < 3; ++k) {
      const float t = float((uint32_t(means_u[i * 3 + k]) << 8) | means_l[i * 3 + k]) / 65535.0f;
      const float v = mins[k] + (maxs[k] - mins[k]) * t;
      scene.positions[size_t(i) * 3 + k] = (v < 0 ? -1.0f : 1.0f) * (std::exp(std::abs(v)) - 1.0f);
      scene.log_scales[size_t(i) * 3 + k] = scale_book[scale_idx[i * 3 + k]];
      scene.sh_dc[size_t(i) * 3 + k] = colour_book[colour[i * 4 + k]];
    }
    const uint8_t* q = &quats[size_t(i) * 4];
    float abc[3], sum = 0.0f;
    for (int k = 0; k < 3; ++k) {
      abc[k] = (float(q[k]) / 255.0f - 0.5f) * root2;
      sum += abc[k] * abc[k];
    }
    const int largest = std::clamp(int(q[3]) - 252, 0, 3);
    float* r = &scene.rotations[size_t(i) * 4];
    for (int k = 0, j = 0; k < 4; ++k) r[k] = k == largest ? std::sqrt(std::max(0.0f, 1.0f - sum)) : abc[j++];
    const float alpha = std::clamp(float(colour[i * 4 + 3]) / 255.0f, 1e-4f, 1.0f - 1e-4f);
    scene.opacity_logits[i] = std::log(alpha / (1.0f - alpha));
  }
  return scene;
}

std::vector<uint8_t> write_gsenv(const SplatScene& scene) {
  const uint32_t n = scene.count;
  float mins[3] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                   std::numeric_limits<float>::max()};
  float maxs[3] = {-mins[0], -mins[0], -mins[0]};
  std::vector<float> logged(size_t(n) * 3);
  for (size_t i = 0; i < logged.size(); ++i) {
    const float p = scene.positions[i];
    logged[i] = (p < 0 ? -1.0f : 1.0f) * std::log(std::abs(p) + 1.0f);
    mins[i % 3] = std::min(mins[i % 3], logged[i]);
    maxs[i % 3] = std::max(maxs[i % 3], logged[i]);
  }
  const std::vector<float> scale_book = codebook(scene.log_scales);
  const std::vector<float> colour_book = codebook(scene.sh_dc);
  JsonWriter w;
  w.begin_object();
  w.field("count", n);
  w.key("means").begin_object();
  w.key("mins").begin_array();
  for (float v : mins) w.value(double(v));
  w.end_array();
  w.key("maxs").begin_array();
  for (float v : maxs) w.value(double(v));
  w.end_array();
  w.end_object();
  w.key("scales").begin_object().key("codebook").begin_array();
  for (float v : scale_book) w.value(double(v));
  w.end_array().end_object();
  w.key("sh0").begin_object().key("codebook").begin_array();
  for (float v : colour_book) w.value(double(v));
  w.end_array().end_object();
  w.field("antialias", scene.antialiased);
  w.end_object();
  const std::string meta = w.str();

  std::vector<uint8_t> raw = {'G', 'S', 'E', 'V'};
  put_u32(raw, 1);
  put_u32(raw, uint32_t(meta.size()));
  raw.insert(raw.end(), meta.begin(), meta.end());
  std::vector<uint8_t> lo(size_t(n) * 3), hi(size_t(n) * 3), quats(size_t(n) * 4), scales(size_t(n) * 3),
      colour(size_t(n) * 4);
  for (uint32_t i = 0; i < n; ++i) {
    for (int k = 0; k < 3; ++k) {
      const float span = std::max(maxs[k] - mins[k], 1e-9f);
      const uint32_t v = uint32_t(std::lround(std::clamp((logged[size_t(i) * 3 + k] - mins[k]) / span, 0.0f, 1.0f) * 65535.0f));
      lo[size_t(i) * 3 + k] = uint8_t(v & 0xff);
      hi[size_t(i) * 3 + k] = uint8_t(v >> 8);
      scales[size_t(i) * 3 + k] = nearest(scale_book, scene.log_scales[size_t(i) * 3 + k]);
      colour[size_t(i) * 4 + k] = nearest(colour_book, scene.sh_dc[size_t(i) * 3 + k]);
    }
    colour[size_t(i) * 4 + 3] = uint8_t(std::lround(sigmoid(scene.opacity_logits[i]) * 255.0f));
    float q[4];
    float len = 0.0f;
    for (int k = 0; k < 4; ++k) {
      q[k] = scene.rotations[size_t(i) * 4 + k];
      len += q[k] * q[k];
    }
    len = std::sqrt(len);
    int largest = 0;
    for (int k = 0; k < 4; ++k) {
      q[k] = len > 0 ? q[k] / len : (k == 0 ? 1.0f : 0.0f);
      if (std::abs(q[k]) > std::abs(q[largest])) largest = k;
    }
    const float sign = q[largest] < 0 ? -1.0f : 1.0f;
    for (int k = 0, j = 0; k < 4; ++k) {
      if (k == largest) continue;
      quats[size_t(i) * 4 + j++] = uint8_t(std::lround(std::clamp(sign * q[k] / std::sqrt(2.0f) + 0.5f, 0.0f, 1.0f) * 255.0f));
    }
    quats[size_t(i) * 4 + 3] = uint8_t(252 + largest);
  }
  for (const auto* plane : {&lo, &hi, &quats, &scales, &colour}) raw.insert(raw.end(), plane->begin(), plane->end());
  return gzip(raw);
}

EnvLight light_at(const SplatScene& scene, const float eye[3], bool flip_y, const float sky[2][3], uint32_t first) {
  constexpr int kW = 128, kH = 64;
  const float s = flip_y ? -1.0f : 1.0f;
  std::vector<float> depth(size_t(kW) * kH, std::numeric_limits<float>::max());
  std::vector<float> rgb(size_t(kW) * kH * 3);
  // The sky where nothing is.
  for (int j = 0; j < kH; ++j) {
    const float y = std::cos((float(j) + 0.5f) / kH * kPi);
    for (int i = 0; i < kW; ++i) {
      for (int c = 0; c < 3; ++c) {
        const float up = sky[0][c] * std::pow(std::max(y, 0.0f), 0.6f) + sky[1][c] * (1.0f - std::pow(std::max(y, 0.0f), 0.6f));
        rgb[(size_t(j) * kW + i) * 3 + c] = srgb_to_linear(y >= 0 ? up : sky[1][c] * 0.7f);
      }
    }
  }
  const uint32_t total = scene.count > first ? scene.count - first : 0;
  const uint32_t stride = std::max(1u, total / 400000);
  const float pixel = kPi / kH;
  for (uint32_t i = first; i < scene.count; i += stride) {
    const float opacity = sigmoid(scene.opacity_logits[i]);
    if (opacity < 0.3f) continue;
    const float* p = &scene.positions[size_t(i) * 3];
    const float rx = p[0] - eye[0], ry = (p[1] - eye[1]) * s, rz = (p[2] - eye[2]) * s;
    const float dist = std::sqrt(rx * rx + ry * ry + rz * rz);
    if (dist < 0.3f) continue;
    const float size = std::exp(std::max({scene.log_scales[size_t(i) * 3], scene.log_scales[size_t(i) * 3 + 1],
                                          scene.log_scales[size_t(i) * 3 + 2]}));
    const int reach = int(std::clamp(std::atan(2.0f * size / dist) / pixel, 1.0f, 6.0f));
    const float u = std::atan2(rx, -rz) / (2 * kPi) + 0.5f;
    const float v = std::acos(std::clamp(ry / dist, -1.0f, 1.0f)) / kPi;
    const int ci = int(u * kW), cj = int(v * kH);
    float colour[3];
    for (int c = 0; c < 3; ++c) colour[c] = srgb_to_linear(std::clamp(scene.sh_dc[size_t(i) * 3 + c] * kShC0 + 0.5f, 0.0f, 1.0f));
    for (int dj = -reach; dj <= reach; ++dj) {
      const int j = cj + dj;
      if (j < 0 || j >= kH) continue;
      for (int di = -reach; di <= reach; ++di) {
        const int ii = ((ci + di) % kW + kW) % kW;
        const size_t q = size_t(j) * kW + ii;
        if (dist >= depth[q]) continue;
        depth[q] = dist;
        for (int c = 0; c < 3; ++c) rgb[q * 3 + c] = colour[c];
      }
    }
  }
  double mean = 0, weight = 0;
  for (int j = 0; j < kH; ++j) {
    const float solid = std::sin((float(j) + 0.5f) / kH * kPi);
    for (int i = 0; i < kW; ++i) {
      const float* c = &rgb[(size_t(j) * kW + i) * 3];
      mean += double(kLuma[0] * c[0] + kLuma[1] * c[1] + kLuma[2] * c[2]) * solid;
      weight += solid;
    }
  }
  mean /= std::max(weight, 1e-9);
  const float brightness = std::clamp(float(std::pow(mean / 0.18, 0.6)), 0.45f, 1.15f);
  return light_from_directions(
      [&](float x, float y, float z, float out[3]) {
        float u = std::atan2(x, -z) / (2 * kPi) + 0.5f;
        u -= std::floor(u);
        const int i = std::min(kW - 1, int(u * kW));
        const int j = std::min(kH - 1, int(std::acos(std::clamp(y, -1.0f, 1.0f)) / kPi * kH));
        for (int c = 0; c < 3; ++c) out[c] = rgb[(size_t(j) * kW + i) * 3 + c];
      },
      brightness);
}

Staging stage(const SplatScene& model, const SplatScene& place, const Spot& spot) {
  Staging out;
  const bool rdf = model.convention == AxisConvention::RDF;
  const float scale = std::max(spot.scale, 1e-4f);

  // The place's floor at the spot (RDF: y down, so the floor is the largest
  // y of what is there): the densest 2 cm layer in the lower half of a
  // column around it.
  {
    std::vector<float> ys;
    const float reach = 0.5f / scale;
    for (uint32_t i = 0; i < place.count; ++i) {
      const float* p = &place.positions[size_t(i) * 3];
      if (std::abs(p[0] - spot.x) > reach || std::abs(p[2] - spot.z) > reach) continue;
      if (sigmoid(place.opacity_logits[i]) < 0.3f) continue;
      ys.push_back(p[1]);
    }
    if (ys.size() < 20) throw std::runtime_error("그 자리에 바닥이 없음");
    std::sort(ys.begin(), ys.end());
    const float lo = ys[ys.size() / 2], hi = ys[ys.size() - 1 - ys.size() / 200];
    const float bin = 0.02f / scale;
    const int bins = std::max(1, int((hi - lo) / bin) + 1);
    std::vector<int> counts(size_t(bins), 0);
    for (float y : ys) {
      if (y >= lo && y <= hi) counts[size_t(std::min(bins - 1, int((y - lo) / bin)))]++;
    }
    const int best = int(std::max_element(counts.begin(), counts.end()) - counts.begin());
    // The middle of that layer.
    std::vector<float> layer;
    for (float y : ys) {
      if (y >= lo + float(best) * bin && y <= lo + float(best + 1) * bin) layer.push_back(y);
    }
    out.floor = layer.empty() ? lo + (float(best) + 0.5f) * bin : layer[layer.size() / 2];
  }

  // The model's feet and middle.
  const Ground ground = find_ground(model);
  const float anchor[3] = {ground.centre[0], rdf ? ground.high : ground.low, ground.centre[1]};
  const float up[3] = {0.0f, rdf ? -1.0f : 1.0f, 0.0f};

  SplatScene& scene = out.scene;
  scene = model;
  out.model_count = model.count;
  const int coeffs = sh_rest_coeffs(model.sh_degree);
  const uint32_t n = model.count + place.count;
  scene.count = n;
  scene.positions.resize(size_t(n) * 3);
  scene.log_scales.resize(size_t(n) * 3);
  scene.rotations.resize(size_t(n) * 4);
  scene.opacity_logits.resize(n);
  scene.sh_dc.resize(size_t(n) * 3);
  scene.sh_rest.resize(size_t(n) * coeffs * 3, 0.0f);

  const float yaw = spot.yaw_deg * kPi / 180.0f;
  const float cy = std::cos(yaw), sy = std::sin(yaw);
  // Half a turn about x takes RDF to RUB; then the turn about the vertical.
  const float to_model[4] = {rdf ? 1.0f : 0.0f, rdf ? 0.0f : 1.0f, 0.0f, 0.0f};
  const float turn[4] = {std::cos(0.5f * yaw), 0.0f, std::sin(0.5f * yaw), 0.0f};
  float frame[4];
  quat_mul(turn, to_model, frame);
  const float log_scale = std::log(scale);
  for (uint32_t i = 0; i < place.count; ++i) {
    const uint32_t o = model.count + i;
    const float* p = &place.positions[size_t(i) * 3];
    float q[3] = {(p[0] - spot.x) * scale, (p[1] - out.floor) * scale, (p[2] - spot.z) * scale};
    if (!rdf) {
      q[1] = -q[1];
      q[2] = -q[2];
    }
    scene.positions[size_t(o) * 3 + 0] = anchor[0] + cy * q[0] + sy * q[2];
    scene.positions[size_t(o) * 3 + 1] = anchor[1] + q[1];
    scene.positions[size_t(o) * 3 + 2] = anchor[2] - sy * q[0] + cy * q[2];
    quat_mul(frame, &place.rotations[size_t(i) * 4], &scene.rotations[size_t(o) * 4]);
    for (int k = 0; k < 3; ++k) {
      scene.log_scales[size_t(o) * 3 + k] = place.log_scales[size_t(i) * 3 + k] + log_scale;
      scene.sh_dc[size_t(o) * 3 + k] = place.sh_dc[size_t(i) * 3 + k];
    }
    scene.opacity_logits[o] = place.opacity_logits[i];
  }

  // The light where the model stands, at chest height.
  const float eye[3] = {anchor[0] + up[0] * 1.1f, anchor[1] + up[1] * 1.1f, anchor[2] + up[2] * 1.1f};
  out.light = light_at(scene, eye, rdf, spot.sky, model.count);

  // A soft shadow on the floor around the feet, longer away from the main light.
  const float s = rdf ? -1.0f : 1.0f;
  const float key[3] = {out.light.key[0], out.light.key[1] * s, out.light.key[2] * s};  // model axes
  const float key_up = out.light.key[1];
  float away[2] = {-key[0], -key[2]};
  const float horizontal = std::sqrt(away[0] * away[0] + away[1] * away[1]);
  const bool cast = horizontal > 1e-3f && key_up > 0.05f;
  if (cast) {
    away[0] /= horizontal;
    away[1] /= horizontal;
  }
  const float length = std::clamp(1.7f * horizontal / std::max(key_up, 0.05f), 0.3f, 3.0f);
  const float strength = std::clamp(1.1f * out.light.key_share - 0.1f, 0.0f, 0.6f);
  for (uint32_t o = model.count; o < n; ++o) {
    const float* p = &scene.positions[size_t(o) * 3];
    const float dx = p[0] - anchor[0], dz = p[2] - anchor[2];
    const float height = (p[1] - anchor[1]) * up[1];
    if (height > 0.25f || height < -0.1f || dx * dx + dz * dz > 16.0f) continue;
    float lit = 1.0f - 0.5f * std::exp(-(dx * dx + dz * dz) / 0.09f);
    if (cast && strength > 0) {
      const float along = dx * away[0] + dz * away[1], across = -dx * away[1] + dz * away[0];
      const float a = (along - 0.5f * length) / (0.5f * length + 0.1f);
      const float width = 0.25f + 0.06f * length;
      lit *= 1.0f - strength * std::exp(-a * a * 1.6f - across * across / (width * width));
    }
    for (int c = 0; c < 3; ++c) {
      float& dc = scene.sh_dc[size_t(o) * 3 + c];
      dc = ((dc * kShC0 + 0.5f) * lit - 0.5f) / kShC0;
    }
  }
  return out;
}

}  // namespace gs
