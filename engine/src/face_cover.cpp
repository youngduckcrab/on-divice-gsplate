#include "gs/face_cover.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <thread>

#include "gs/json_parse.h"

namespace gs {
namespace {

constexpr float kShC0 = 0.28209479177387814f;
constexpr float kPi = 3.14159265358979f;
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kDilation = 0.3f;  // the viewer's low-pass filter, px^2

struct Vec3 {
  float x = 0, y = 0, z = 0;
  Vec3() = default;
  Vec3(float a, float b, float c) : x(a), y(b), z(c) {}
  explicit Vec3(const float* p) : x(p[0]), y(p[1]), z(p[2]) {}
  Vec3 operator+(Vec3 o) const { return {x + o.x, y + o.y, z + o.z}; }
  Vec3 operator-(Vec3 o) const { return {x - o.x, y - o.y, z - o.z}; }
  Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
  Vec3 operator-() const { return {-x, -y, -z}; }
  void store(float* p) const {
    p[0] = x;
    p[1] = y;
    p[2] = z;
  }
};

float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(Vec3 a, Vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
float length(Vec3 a) { return std::sqrt(dot(a, a)); }
Vec3 normalize(Vec3 a) {
  const float l = length(a);
  return l > 0.0f ? a * (1.0f / l) : a;
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float logit(float p) { return std::log(p / (1.0f - p)); }
float smoothstep(float e0, float e1, float x) {
  const float t = std::clamp((x - e0) / (e1 - e0), 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}
float mix(float a, float b, float t) { return a + (b - a) * t; }

// Rotation matrix (row-major) of a (w, x, y, z) quaternion.
void quat_matrix(const float* q, float m[9]) {
  const float len = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  const float w = len > 0 ? q[0] / len : 1.0f, x = len > 0 ? q[1] / len : 0.0f;
  const float y = len > 0 ? q[2] / len : 0.0f, z = len > 0 ? q[3] / len : 0.0f;
  m[0] = 1 - 2 * (y * y + z * z);
  m[1] = 2 * (x * y - w * z);
  m[2] = 2 * (x * z + w * y);
  m[3] = 2 * (x * y + w * z);
  m[4] = 1 - 2 * (x * x + z * z);
  m[5] = 2 * (y * z - w * x);
  m[6] = 2 * (x * z - w * y);
  m[7] = 2 * (y * z + w * x);
  m[8] = 1 - 2 * (x * x + y * y);
}

// (w, x, y, z) of the rotation whose columns are x, y, z (orthonormal, right-handed).
void matrix_quat(Vec3 cx, Vec3 cy, Vec3 cz, float q[4]) {
  const float m00 = cx.x, m10 = cx.y, m20 = cx.z;
  const float m01 = cy.x, m11 = cy.y, m21 = cy.z;
  const float m02 = cz.x, m12 = cz.y, m22 = cz.z;
  const float trace = m00 + m11 + m22;
  float w, x, y, z;
  if (trace > 0) {
    const float s = 0.5f / std::sqrt(trace + 1.0f);
    w = 0.25f / s;
    x = (m21 - m12) * s;
    y = (m02 - m20) * s;
    z = (m10 - m01) * s;
  } else if (m00 > m11 && m00 > m22) {
    const float s = 2.0f * std::sqrt(1.0f + m00 - m11 - m22);
    w = (m21 - m12) / s;
    x = 0.25f * s;
    y = (m01 + m10) / s;
    z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const float s = 2.0f * std::sqrt(1.0f + m11 - m00 - m22);
    w = (m02 - m20) / s;
    x = (m01 + m10) / s;
    y = 0.25f * s;
    z = (m12 + m21) / s;
  } else {
    const float s = 2.0f * std::sqrt(1.0f + m22 - m00 - m11);
    w = (m10 - m01) / s;
    x = (m02 + m20) / s;
    y = (m12 + m21) / s;
    z = 0.25f * s;
  }
  const float len = std::sqrt(w * w + x * x + y * y + z * z);
  q[0] = w / len;
  q[1] = x / len;
  q[2] = y / len;
  q[3] = z / len;
}

// Splat i's colour seen along direction d (3DGS real SH, as the renderer evaluates it).
void sh_color(const SplatScene& s, uint32_t i, Vec3 d, float out[3]) {
  const int coeffs = sh_rest_coeffs(s.sh_degree);
  float b[15] = {};
  if (coeffs > 0) {
    const float x = d.x, y = d.y, z = d.z;
    const float C1 = 0.4886025119029199f;
    b[0] = -C1 * y;
    b[1] = C1 * z;
    b[2] = -C1 * x;
    if (coeffs > 3) {
      const float xx = x * x, yy = y * y, zz = z * z, xy = x * y, yz = y * z, xz = x * z;
      b[3] = 1.0925484305920792f * xy;
      b[4] = -1.0925484305920792f * yz;
      b[5] = 0.31539156525252005f * (2 * zz - xx - yy);
      b[6] = -1.0925484305920792f * xz;
      b[7] = 0.5462742152960396f * (xx - yy);
      if (coeffs > 8) {
        b[8] = -0.5900435899266435f * y * (3 * xx - yy);
        b[9] = 2.890611442640554f * xy * z;
        b[10] = -0.4570457994644658f * y * (4 * zz - xx - yy);
        b[11] = 0.3731763325901154f * z * (2 * zz - 3 * xx - 3 * yy);
        b[12] = -0.4570457994644658f * x * (4 * zz - xx - yy);
        b[13] = 1.445305721320277f * z * (xx - yy);
        b[14] = -0.5900435899266435f * x * (xx - 3 * yy);
      }
    }
  }
  const float* rest = coeffs > 0 ? &s.sh_rest[size_t(i) * coeffs * 3] : nullptr;
  for (int c = 0; c < 3; ++c) {
    float v = kShC0 * s.sh_dc[size_t(i) * 3 + c];
    for (int j = 0; j < coeffs; ++j) v += b[j] * rest[j * 3 + c];
    out[c] = std::max(v + 0.5f, 0.0f);
  }
}

// The face's frame: P(a, b, c) = origin + e * (a * right + b * up + c * forward).
struct Frame {
  Vec3 o, r, u, f;
  float e = 1;
  Vec3 at(float a, float b, float c) const { return o + (r * a + u * b + f * c) * e; }
  void local(const float* p, float out[3]) const {
    const Vec3 d = Vec3(p) - o;
    out[0] = dot(d, r) / e;
    out[1] = dot(d, u) / e;
    out[2] = dot(d, f) / e;
  }
};

// A regular grid over the face's (a, b) plane; cell (i, j) is centred at
// a = a0 + (i + 0.5) * step, b = b1 - (j + 0.5) * step (rows go down, as images do).
struct Grid {
  float a0 = 0, b1 = 0, step = 1;
  int w = 0, h = 0;
  float a(int i) const { return a0 + (float(i) + 0.5f) * step; }
  float b(int j) const { return b1 - (float(j) + 0.5f) * step; }
  float fi(float a) const { return (a - a0) / step - 0.5f; }
  float fj(float b) const { return (b1 - b) / step - 0.5f; }
  size_t size() const { return size_t(w) * h; }
};

// The face seen from the front: per cell the surface's c (NaN where nothing
// is opaque), coverage and colour (not premultiplied).
struct FaceField {
  Grid g;
  std::vector<float> height;
  std::vector<float> alpha;
  std::vector<float> rgb;

  // Value of `values` at (a, b) from the nearest cells that have one; NaN if none nearby.
  float sample(const std::vector<float>& values, float a, float b) const {
    const float x = g.fi(a), y = g.fj(b);
    const int x0 = int(std::floor(x)), y0 = int(std::floor(y));
    float sum = 0, weight = 0;
    for (int dy = 0; dy <= 1; ++dy) {
      for (int dx = 0; dx <= 1; ++dx) {
        const int xi = x0 + dx, yi = y0 + dy;
        if (xi < 0 || yi < 0 || xi >= g.w || yi >= g.h) continue;
        const float v = values[size_t(yi) * g.w + xi];
        if (std::isnan(v)) continue;
        const float wgt = (dx ? x - x0 : 1 - (x - x0)) * (dy ? y - y0 : 1 - (y - y0)) + 1e-4f;
        sum += wgt * v;
        weight += wgt;
      }
    }
    return weight > 0 ? sum / weight : kNaN;
  }
};

FaceField face_field(const SplatScene& scene, const Frame& fr) {
  FaceField field;
  Grid& g = field.g;
  g.step = 1.0f / 40.0f;
  g.a0 = -2.4f;
  g.b1 = 2.6f;
  g.w = int(4.8f / g.step + 0.5f);
  g.h = int(5.8f / g.step + 0.5f);
  constexpr float kFront = 3.5f;  // the picture's plane, in front of the face

  OrthoView v;
  (fr.at(0.0f, g.b1 - 0.5f * float(g.h) * g.step, kFront)).store(v.origin);
  fr.r.store(v.right);
  (-fr.u).store(v.down);
  (-fr.f).store(v.forward);
  v.pixel = g.step * fr.e;
  v.width = uint32_t(g.w);
  v.height = uint32_t(g.h);
  std::vector<uint32_t> subset;
  float l[3];
  for (uint32_t i = 0; i < scene.count; ++i) {
    fr.local(&scene.positions[size_t(i) * 3], l);
    if (l[0] > -2.8f && l[0] < 2.8f && l[1] > -3.4f && l[1] < 3.0f && l[2] > -3.6f && l[2] < kFront) {
      subset.push_back(i);
    }
  }
  const float black[3] = {0, 0, 0};
  render_ortho(scene, subset, v, black);

  field.height.assign(g.size(), kNaN);
  field.alpha = v.alpha;
  field.rgb.assign(g.size() * 3, 0.0f);
  for (size_t p = 0; p < g.size(); ++p) {
    if (!std::isnan(v.depth[p])) field.height[p] = kFront - v.depth[p] / fr.e;
    const float a = v.alpha[p];
    for (int c = 0; c < 3; ++c) field.rgb[p * 3 + c] = a > 1e-3f ? std::min(1.0f, v.rgb[p * 3 + c] / a) : 0.0f;
  }
  return field;
}

// Separable filters over a w x h grid, skipping NaN.
std::vector<float> max_filter(const std::vector<float>& in, int w, int h, int radius) {
  std::vector<float> tmp(in.size(), kNaN), out(in.size(), kNaN);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float m = kNaN;
      for (int k = std::max(0, x - radius); k <= std::min(w - 1, x + radius); ++k) {
        const float v = in[size_t(y) * w + k];
        if (!std::isnan(v) && (std::isnan(m) || v > m)) m = v;
      }
      tmp[size_t(y) * w + x] = m;
    }
  }
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float m = kNaN;
      for (int k = std::max(0, y - radius); k <= std::min(h - 1, y + radius); ++k) {
        const float v = tmp[size_t(k) * w + x];
        if (!std::isnan(v) && (std::isnan(m) || v > m)) m = v;
      }
      out[size_t(y) * w + x] = m;
    }
  }
  return out;
}

// Gaussian blur of `channels` interleaved values weighted by `weight` (NaN values weigh nothing).
std::vector<float> blur(const std::vector<float>& in, const std::vector<float>& weight, int channels, int w, int h,
                        float sigma_px, float min_weight = 1e-3f) {
  const int radius = std::max(1, int(std::ceil(3.0f * sigma_px)));
  std::vector<float> kernel(size_t(2 * radius + 1));
  for (int k = -radius; k <= radius; ++k) kernel[size_t(k + radius)] = std::exp(-0.5f * float(k * k) / (sigma_px * sigma_px));
  const size_t n = size_t(w) * h;
  std::vector<float> num(n * channels, 0.0f), den(n, 0.0f);
  for (size_t p = 0; p < n; ++p) {
    bool valid = weight[p] > 0;
    for (int c = 0; c < channels && valid; ++c) valid = !std::isnan(in[p * channels + c]);
    if (!valid) continue;
    den[p] = weight[p];
    for (int c = 0; c < channels; ++c) num[p * channels + c] = in[p * channels + c] * weight[p];
  }
  auto pass = [&](std::vector<float>& values, int stride_c, bool horizontal) {
    std::vector<float> out(values.size(), 0.0f);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const size_t p = size_t(y) * w + x;
        for (int k = -radius; k <= radius; ++k) {
          const int xx = horizontal ? x + k : x, yy = horizontal ? y : y + k;
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
          const size_t q = size_t(yy) * w + xx;
          const float kw = kernel[size_t(k + radius)];
          for (int c = 0; c < stride_c; ++c) out[p * stride_c + c] += kw * values[q * stride_c + c];
        }
      }
    }
    values.swap(out);
  };
  pass(num, channels, true);
  pass(num, channels, false);
  pass(den, 1, true);
  pass(den, 1, false);
  float peak = 0;
  for (float k : kernel) peak += k;
  peak *= peak;
  std::vector<float> out(n * channels, kNaN);
  for (size_t p = 0; p < n; ++p) {
    if (den[p] < min_weight * peak) continue;
    for (int c = 0; c < channels; ++c) out[p * channels + c] = num[p * channels + c] / den[p];
  }
  return out;
}

// Median of each cell's neighbourhood (NaN skipped): drops lone spikes.
std::vector<float> median_filter(const std::vector<float>& in, int w, int h, int radius) {
  std::vector<float> out(in.size(), kNaN), values;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (std::isnan(in[size_t(y) * w + x])) continue;
      values.clear();
      for (int yy = std::max(0, y - radius); yy <= std::min(h - 1, y + radius); ++yy) {
        for (int xx = std::max(0, x - radius); xx <= std::min(w - 1, x + radius); ++xx) {
          const float v = in[size_t(yy) * w + xx];
          if (!std::isnan(v)) values.push_back(v);
        }
      }
      std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
      out[size_t(y) * w + x] = values[values.size() / 2];
    }
  }
  return out;
}

// A surface just in front of the face: its heights (lone spikes dropped)
// spread by `spread` and smoothed by `sigma` (eye distances), never behind the
// face itself, plus `lift`.
std::vector<float> shell_over(const FaceField& field, float spread, float sigma, float lift) {
  const Grid& g = field.g;
  std::vector<float> h = field.height;
  for (size_t p = 0; p < h.size(); ++p) {
    if (field.alpha[p] < 0.5f) h[p] = kNaN;
  }
  h = median_filter(h, g.w, g.h, 2);
  const std::vector<float> spreaded = max_filter(h, g.w, g.h, std::max(1, int(spread / g.step + 0.5f)));
  const std::vector<float> ones(h.size(), 1.0f);
  std::vector<float> smooth = blur(spreaded, ones, 1, g.w, g.h, sigma / g.step, 0.05f);
  for (size_t p = 0; p < h.size(); ++p) {
    float s = smooth[p];
    if (!std::isnan(h[p])) s = std::isnan(s) ? h[p] : std::max(s, h[p]);
    smooth[p] = std::isnan(s) ? kNaN : s + lift;
  }
  return smooth;
}

// What a style draws on the face, per cell of a finer grid: the layer's
// height (NaN: nothing), its colour as drawn (light baked in) and before
// lighting, how much of the cell it covers, and the colour the face's own
// splats under it take.
struct Layer {
  Grid g;
  std::vector<float> height;
  std::vector<float> rgb;
  std::vector<float> base;
  std::vector<float> under;
  std::vector<float> alpha;
  float thickness = 0.02f;  // splat depth across the layer, eye distances

  void init(float a0, float b1, float width, float height_extent, float step) {
    g.a0 = a0;
    g.b1 = b1;
    g.step = step;
    g.w = int(width / step + 0.5f);
    g.h = int(height_extent / step + 0.5f);
    height.assign(g.size(), kNaN);
    rgb.assign(g.size() * 3, 0.0f);
    base.assign(g.size() * 3, 0.0f);
    under.assign(g.size() * 3, 0.0f);
    alpha.assign(g.size(), 0.0f);
  }
  size_t cell(float a, float b) const {
    const int i = int(std::floor(g.fi(a) + 0.5f)), j = int(std::floor(g.fj(b) + 0.5f));
    if (i < 0 || j < 0 || i >= g.w || j >= g.h) return SIZE_MAX;
    return size_t(j) * g.w + i;
  }
  // Normal of the layer at cell p, in face coordinates (and the slopes).
  Vec3 normal(size_t p, float* sa = nullptr, float* sb = nullptr) const {
    const int i = int(p % size_t(g.w)), j = int(p / size_t(g.w));
    auto at = [&](int x, int y) {
      if (x < 0 || y < 0 || x >= g.w || y >= g.h) return kNaN;
      return height[size_t(y) * g.w + x];
    };
    const float c = height[p];
    auto slope = [&](float lo, float hi) {
      if (!std::isnan(lo) && !std::isnan(hi)) return (hi - lo) / (2 * g.step);
      if (!std::isnan(hi)) return (hi - c) / g.step;
      if (!std::isnan(lo)) return (c - lo) / g.step;
      return 0.0f;
    };
    const float da = slope(at(i - 1, j), at(i + 1, j));
    const float db = slope(at(i, j + 1), at(i, j - 1));  // rows go down
    if (sa) *sa = da;
    if (sb) *sb = db;
    return normalize(Vec3(-da, -db, 1.0f));
  }
};

const Vec3 kLight = normalize(Vec3(-0.35f, 0.55f, 0.75f));  // face coordinates: from the upper left, in front

void emit(SplatScene& out, Vec3 pos, Vec3 x_axis, Vec3 y_axis, Vec3 z_axis, Vec3 scales, const float rgb[3],
          float alpha) {
  const uint32_t i = out.count;
  const int coeffs = sh_rest_coeffs(out.sh_degree);
  out.count += 1;
  out.positions.resize(size_t(out.count) * 3);
  pos.store(&out.positions[size_t(i) * 3]);
  out.log_scales.push_back(std::log(std::max(scales.x, 1e-7f)));
  out.log_scales.push_back(std::log(std::max(scales.y, 1e-7f)));
  out.log_scales.push_back(std::log(std::max(scales.z, 1e-7f)));
  float q[4];
  matrix_quat(x_axis, y_axis, z_axis, q);
  out.rotations.insert(out.rotations.end(), q, q + 4);
  out.opacity_logits.push_back(logit(std::clamp(alpha, 0.004f, 0.995f)));
  for (int c = 0; c < 3; ++c) out.sh_dc.push_back((std::clamp(rgb[c], 0.0f, 1.0f) - 0.5f) / kShC0);
  out.sh_rest.resize(out.sh_rest.size() + size_t(coeffs) * 3, 0.0f);
}

// One flat splat per drawn cell, lying in the layer, overlapping its neighbours.
uint32_t emit_layer(const Layer& layer, const Frame& fr, SplatScene& out) {
  uint32_t added = 0;
  const float sigma = 0.62f * layer.g.step;
  for (size_t p = 0; p < layer.g.size(); ++p) {
    const float c = layer.height[p];
    if (std::isnan(c) || layer.alpha[p] < 0.02f) continue;
    const int i = int(p % size_t(layer.g.w)), j = int(p / size_t(layer.g.w));
    float sa = 0, sb = 0;
    layer.normal(p, &sa, &sb);
    const Vec3 ta = fr.r + fr.f * sa;
    const Vec3 tb = fr.u + fr.f * sb;
    const Vec3 n = normalize(cross(ta, tb));
    const Vec3 x = normalize(ta);
    const Vec3 y = cross(n, x);
    const Vec3 scales(sigma * length(ta) * fr.e, sigma * length(tb) * fr.e, layer.thickness * fr.e);
    emit(out, fr.at(layer.g.a(i), layer.g.b(j), c), x, y, n, scales, &layer.rgb[p * 3], 0.97f * layer.alpha[p]);
    ++added;
  }
  return added;
}

// The colour a style paints at (a, b) and how much of that point it covers (0..1).
using Paint = std::function<float(float a, float b, float rgb[3])>;

// Fills the layer from `paint` with 3 x 3 samples per cell, so edges come out
// smooth, at heights from `height` (NaN leaves the cell empty).
void fill(Layer& layer, const Paint& paint, const std::function<float(float, float)>& height) {
  const float s = layer.g.step;
  for (int j = 0; j < layer.g.h; ++j) {
    for (int i = 0; i < layer.g.w; ++i) {
      const float a = layer.g.a(i), b = layer.g.b(j);
      float sum[3] = {0, 0, 0}, cover = 0;
      for (int sy = 0; sy < 3; ++sy) {
        for (int sx = 0; sx < 3; ++sx) {
          float rgb[3] = {0, 0, 0};
          const float wgt = paint(a + float(sx - 1) * s / 3.0f, b + float(sy - 1) * s / 3.0f, rgb);
          if (!(wgt > 0.0f)) continue;
          for (int c = 0; c < 3; ++c) sum[c] += wgt * rgb[c];
          cover += wgt;
        }
      }
      if (cover <= 0.0f) continue;
      const float c = height(a, b);
      if (std::isnan(c)) continue;
      const size_t p = size_t(j) * layer.g.w + i;
      layer.height[p] = c;
      layer.alpha[p] = cover / 9.0f;
      for (int k = 0; k < 3; ++k) layer.rgb[p * 3 + k] = layer.base[p * 3 + k] = layer.under[p * 3 + k] = sum[k] / cover;
    }
  }
}

// Bakes a light from the upper left into the layer's colours.
void light(Layer& layer, float ambient, float diffuse) {
  for (size_t p = 0; p < layer.g.size(); ++p) {
    if (layer.alpha[p] <= 0.0f || std::isnan(layer.height[p])) continue;
    const float shade = ambient + diffuse * std::max(0.0f, dot(layer.normal(p), kLight));
    for (int k = 0; k < 3; ++k) layer.rgb[p * 3 + k] = std::min(1.0f, layer.base[p * 3 + k] * shade);
  }
}

// The face's own colour without its features: the median over the middle of
// the face (dark eyes, brows and hair are outvoted by skin).
void skin_tone(const FaceField& field, float rgb[3]);

// Under styles that hide the whole face, the face's splats become plain skin.
void set_under(Layer& layer, const float rgb[3]) {
  for (size_t p = 0; p < layer.g.size(); ++p) {
    for (int k = 0; k < 3; ++k) layer.under[p * 3 + k] = rgb[k];
  }
}

// Gives the face's splats under the layer (down to `depth` behind the face
// surface) the layer's under-colour without view-dependent colour, and keeps
// their centres behind the layer so it is always drawn over them.
uint32_t recolour_under(SplatScene& scene, const Layer& layer, const FaceField& field, const Frame& fr, float depth) {
  uint32_t count = 0;
  const int coeffs = sh_rest_coeffs(scene.sh_degree);
  float l[3];
  for (uint32_t i = 0; i < scene.count; ++i) {
    float* pos = &scene.positions[size_t(i) * 3];
    fr.local(pos, l);
    const size_t p = layer.cell(l[0], l[1]);
    if (p == SIZE_MAX || layer.alpha[p] < 0.5f) continue;
    const float surface = field.sample(field.height, l[0], l[1]);
    if (std::isnan(surface) || l[2] < surface - depth) continue;
    for (int c = 0; c < 3; ++c) scene.sh_dc[size_t(i) * 3 + c] = (layer.under[p * 3 + c] - 0.5f) / kShC0;
    std::fill_n(scene.sh_rest.begin() + ptrdiff_t(size_t(i) * coeffs * 3), coeffs * 3, 0.0f);
    const float limit = layer.height[p] - 0.02f;
    if (!std::isnan(limit) && l[2] > limit) (Vec3(pos) + fr.f * ((limit - l[2]) * fr.e)).store(pos);
    ++count;
  }
  return count;
}

// --- Styles ------------------------------------------------------------------------
// Face coordinates in eye distances: eyes at (+-0.5, 0), mouth near (0, -1.1).

constexpr float kOvalB = -0.3f;  // centre of the face oval
constexpr float kOvalA = 1.3f;   // half width
constexpr float kOvalH = 1.72f;  // half height

float oval(float a, float b) { return std::sqrt((a / kOvalA) * (a / kOvalA) + ((b - kOvalB) / kOvalH) * ((b - kOvalB) / kOvalH)); }

// Whether the face (or hair over it) is at (a, b) seen from the front, rather
// than nothing, the ears and sides of the head, or the neck under the chin,
// which lie well behind the face: covers stay on the face.
bool on_face(const FaceField& field, float a, float b) {
  if (field.sample(field.alpha, a, b) < 0.5f) return false;
  const float c = field.sample(field.height, a, b);
  return !std::isnan(c) && c > -0.9f;
}

void skin_tone(const FaceField& field, float rgb[3]) {
  std::vector<float> ch[3];
  const Grid& g = field.g;
  for (int j = 0; j < g.h; ++j) {
    for (int i = 0; i < g.w; ++i) {
      const size_t p = size_t(j) * g.w + i;
      if (field.alpha[p] < 0.5f || oval(g.a(i), g.b(j)) > 0.8f) continue;
      for (int c = 0; c < 3; ++c) ch[c].push_back(field.rgb[p * 3 + c]);
    }
  }
  for (int c = 0; c < 3; ++c) {
    if (ch[c].empty()) {
      rgb[c] = 0.6f;
      continue;
    }
    std::nth_element(ch[c].begin(), ch[c].begin() + ch[c].size() / 2, ch[c].end());
    rgb[c] = ch[c][ch[c].size() / 2];
  }
}

void mosaic(const FaceField& field, Layer& layer) {
  const std::vector<float> shell = shell_over(field, 0.2f, 0.15f, 0.05f);
  constexpr float kBlock = 0.42f;
  constexpr int kBlocks = 16;
  // Blocks aligned so one column is centred between the eyes.
  auto block_index = [](float a, float b) -> int {
    const int bi = int(std::floor(a / kBlock + 0.5f)) + kBlocks / 2;
    const int bj = int(std::floor((b - kOvalB) / kBlock + 0.5f)) + kBlocks / 2;
    if (bi < 0 || bj < 0 || bi >= kBlocks || bj >= kBlocks) return -1;
    return bj * kBlocks + bi;
  };
  struct Block {
    double rgb[3] = {0, 0, 0};
    double weight = 0;
    int cells = 0, covered = 0;
    bool used = false;
  };
  std::vector<Block> blocks(size_t(kBlocks * kBlocks));
  const Grid& g = field.g;
  for (int j = 0; j < g.h; ++j) {
    for (int i = 0; i < g.w; ++i) {
      const int k = block_index(g.a(i), g.b(j));
      if (k < 0) continue;
      const size_t p = size_t(j) * g.w + i;
      Block& blk = blocks[size_t(k)];
      blk.cells += 1;
      const float a = field.alpha[p];
      if (a > 0.5f) blk.covered += 1;
      for (int c = 0; c < 3; ++c) blk.rgb[c] += double(field.rgb[p * 3 + c]) * a;
      blk.weight += a;
    }
  }
  for (int k = 0; k < kBlocks * kBlocks; ++k) {
    Block& blk = blocks[size_t(k)];
    // The block's centre decides, so the edge is a staircase of whole blocks.
    const float ca = float(k % kBlocks - kBlocks / 2) * kBlock, cb = kOvalB + float(k / kBlocks - kBlocks / 2) * kBlock;
    blk.used = oval(ca, cb) <= 1.0f && blk.covered * 2 >= blk.cells && blk.weight > 0;
    if (blk.used) {
      for (double& c : blk.rgb) c /= blk.weight;
    }
  }
  fill(
      layer,
      [&](float a, float b, float rgb[3]) {
        const int k = block_index(a, b);
        if (k < 0 || !blocks[size_t(k)].used || !on_face(field, a, b)) return 0.0f;
        for (int c = 0; c < 3; ++c) rgb[c] = float(blocks[size_t(k)].rgb[c]);
        return 1.0f;
      },
      [&](float a, float b) { return field.sample(shell, a, b); });
}

void blur_face(const FaceField& field, Layer& layer) {
  const std::vector<float> shell = shell_over(field, 0.2f, 0.15f, 0.05f);
  const Grid& g = field.g;
  const std::vector<float> soft = blur(field.rgb, field.alpha, 3, g.w, g.h, 0.36f / g.step);
  fill(
      layer,
      [&](float a, float b, float rgb[3]) {
        const float edge = 1.0f - smoothstep(0.85f, 1.0f, oval(a, b));
        if (edge <= 0.0f) return 0.0f;
        const int fi = std::clamp(int(g.fi(a) + 0.5f), 0, g.w - 1), fj = std::clamp(int(g.fj(b) + 0.5f), 0, g.h - 1);
        const size_t p = size_t(fj) * g.w + fi;
        if (!on_face(field, a, b) || std::isnan(soft[p * 3])) return 0.0f;
        for (int c = 0; c < 3; ++c) rgb[c] = soft[p * 3 + c];
        return edge;
      },
      [&](float a, float b) { return field.sample(shell, a, b); });
}

// Half width of the mask at height b: an oval narrowing towards the chin.
float mask_half_width(float b) {
  constexpr float kCentre = -0.27f, kA = 1.22f, kB = 1.68f;
  const float t = (b - kCentre) / kB;
  if (std::abs(t) >= 1.0f) return 0.0f;
  float w = kA * std::sqrt(1.0f - t * t);
  if (t < 0) w *= 1.0f - 0.22f * (-t);
  return w;
}

bool in_almond(float a, float b, float ca, float cb, float ha, float hb) {
  const float x = (a - ca) / ha;
  return std::abs(x) < 1.0f && std::abs(b - cb) < hb * (1.0f - x * x);
}

void mask(const FaceField& field, Layer& layer) {
  // Spread and smoothed: a mask is a smooth shell over the face, not a cast of
  // it. Less spread brings back the face's own bumps; more turns the nose
  // into a beak.
  const std::vector<float> shell = shell_over(field, 0.2f, 0.2f, 0.06f);
  const float ivory[3] = {0.95f, 0.93f, 0.88f};
  const float rim[3] = {0.80f, 0.78f, 0.74f};
  const float hole[3] = {0.03f, 0.03f, 0.035f};
  const float gold[3] = {0.80f, 0.63f, 0.28f};
  const float lip[3] = {0.72f, 0.20f, 0.22f};
  fill(
      layer,
      [&](float a, float b, float rgb[3]) {
        const float half = mask_half_width(b);
        if (std::abs(a) >= half || !on_face(field, a, b)) return 0.0f;
        const float* col = ivory;
        bool eye = false, outline = false;
        for (float side : {-0.5f, 0.5f}) {
          eye = eye || in_almond(a, b, side, 0.03f, 0.3f, 0.14f);
          outline = outline || in_almond(a, b, side, 0.03f, 0.36f, 0.2f);
        }
        const float x = a / 0.34f;
        const bool mouth = std::abs(x) < 1.0f && std::abs(b - (-1.13f + 0.1f * x * x)) < 0.055f * (1.0f - x * x) + 0.012f;
        const float edge = std::min(half - std::abs(a), std::min(1.41f - b, b + 1.95f));
        if (eye) {
          col = hole;
        } else if (outline) {
          col = gold;
        } else if (mouth) {
          col = lip;
        } else if (edge < 0.06f) {
          col = rim;  // a slightly darker rim, so the outline reads
        }
        for (int k = 0; k < 3; ++k) rgb[k] = col[k];
        return 1.0f;
      },
      [&](float a, float b) { return field.sample(shell, a, b); });
  light(layer, 0.62f, 0.42f);
  float skin[3];
  skin_tone(field, skin);
  set_under(layer, skin);
  layer.thickness = 0.03f;
}

void smile(const FaceField& field, Layer& layer) {
  constexpr float kCb = -0.3f, kRadius = 1.55f, kCurve = 0.22f;
  // A dome in front of everything it covers.
  float top = -1e9f;
  const Grid& g = field.g;
  for (int j = 0; j < g.h; ++j) {
    for (int i = 0; i < g.w; ++i) {
      const size_t p = size_t(j) * g.w + i;
      const float a = g.a(i), b = g.b(j);
      const float rho2 = a * a + (b - kCb) * (b - kCb);
      if (rho2 > kRadius * kRadius || field.alpha[p] < 0.5f || std::isnan(field.height[p])) continue;
      top = std::max(top, field.height[p] + kCurve * rho2);
    }
  }
  if (top < -1e8f) return;
  top += 0.06f;
  const float yellow[3] = {1.0f, 0.82f, 0.18f};
  const float ring[3] = {0.86f, 0.55f, 0.06f};
  const float ink[3] = {0.24f, 0.14f, 0.05f};
  fill(
      layer,
      [&](float a, float b, float rgb[3]) {
        const float y = b - kCb;
        const float rho = std::sqrt(a * a + y * y);
        if (rho > kRadius) return 0.0f;
        const float* col = rho > kRadius - 0.07f ? ring : yellow;
        // Eyes: upright ovals.
        for (float side : {-0.45f, 0.45f}) {
          const float ex = (a - side) / 0.13f, ey = (y - 0.42f) / 0.22f;
          if (ex * ex + ey * ey < 1.0f) col = ink;
        }
        // Mouth: an arc below the centre with round ends.
        constexpr float kMouthY = 0.15f, kMouthR = 0.85f, kHalf = 0.075f, kSpan = 55.0f * kPi / 180.0f;
        const float my = y - kMouthY;
        const float r = std::sqrt(a * a + my * my);
        if (std::abs(std::atan2(a, -my)) < kSpan && std::abs(r - kMouthR) < kHalf) col = ink;
        for (float s : {-1.0f, 1.0f}) {
          const float ea = s * kMouthR * std::sin(kSpan), eb = -kMouthR * std::cos(kSpan);
          if ((a - ea) * (a - ea) + (my - eb) * (my - eb) < kHalf * kHalf) col = ink;
        }
        for (int k = 0; k < 3; ++k) rgb[k] = col[k];
        return 1.0f;
      },
      [&](float a, float b) { return top - kCurve * (a * a + (b - kCb) * (b - kCb)); });
  light(layer, 0.74f, 0.30f);
  float skin[3];
  skin_tone(field, skin);
  set_under(layer, skin);
  layer.thickness = 0.03f;
}


// Sunglasses: two lenses in a black frame on a gently wrapped surface in
// front of the eyes, a bridge, and arms back to the ears.
uint32_t sunglasses(const SplatScene& scene, const FaceField& field, const Frame& fr, SplatScene& out) {
  constexpr float kLensA = 0.56f, kLensB = -0.08f, kHalfW = 0.43f, kHalfH = 0.34f;
  constexpr float kSide = 0.06f, kTop = 0.10f, kBottom = 0.05f, kWrap = 0.12f, kGap = 0.10f;
  auto superellipse = [](float x, float y) { return std::pow(std::abs(x), 3.0f) + std::pow(std::abs(y), 3.0f); };
  // 0 lens, 1 frame, -1 nothing; (lx, ly) lens-local coordinates in -1..1.
  auto part = [&](float a, float b, float* lx, float* ly) {
    const float side = a < 0 ? -1.0f : 1.0f;
    const float dx = a - side * kLensA, dy = b - kLensB;
    *lx = dx / kHalfW * side;  // +1 towards the temple
    *ly = dy / kHalfH;
    if (superellipse(*lx, *ly) <= 1.0f) return 0;
    const float fh = kHalfH + (dy > 0 ? kTop : kBottom);
    if (superellipse(dx / (kHalfW + kSide), dy / fh) <= 1.0f) return 1;
    if (std::abs(a) < 0.16f && b > 0.06f && b < 0.17f) return 1;  // bridge
    return -1;
  };
  // The front: far enough out that the lenses clear the brows, nose and lashes.
  const Grid& g = field.g;
  std::vector<float> clear;
  for (int j = 0; j < g.h; ++j) {
    for (int i = 0; i < g.w; ++i) {
      const size_t p = size_t(j) * g.w + i;
      float lx, ly;
      const float a = g.a(i), b = g.b(j);
      if (field.alpha[p] < 0.5f || std::isnan(field.height[p])) continue;
      if (part(a, b, &lx, &ly) < 0 && !(std::abs(a) < kLensA + kHalfW + 0.1f && std::abs(b - kLensB) < kHalfH + 0.15f)) continue;
      clear.push_back(field.height[p] + kWrap * a * a);
    }
  }
  if (clear.empty()) return 0;
  std::sort(clear.begin(), clear.end());
  const float front = clear[std::min(clear.size() - 1, clear.size() * 995 / 1000)] + kGap;
  auto surface = [&](float a) { return front - kWrap * a * a; };

  const float step = 1.0f / 34.0f;
  const float sigma = 0.62f * step;
  uint32_t added = 0;
  const float frame_rgb[3] = {0.035f, 0.035f, 0.04f};
  for (float b = kLensB - kHalfH - kBottom - step; b <= kLensB + kHalfH + kTop + step; b += step) {
    for (float a = -(kLensA + kHalfW + kSide + step); a <= kLensA + kHalfW + kSide + step; a += step) {
      float lx, ly;
      const int what = part(a, b, &lx, &ly);
      if (what < 0) continue;
      const float c = surface(a);
      const Vec3 ta = fr.r + fr.f * (-2.0f * kWrap * a);
      const Vec3 n = normalize(cross(ta, fr.u));
      const Vec3 x = normalize(ta);
      const Vec3 y = cross(n, x);
      float rgb[3];
      float alpha = 0.97f;
      float thick = 0.02f;
      if (what == 0) {
        // Smoky glass, darker at the top, with two faint reflections.
        const float t = 0.5f * (ly + 1.0f);
        for (int k = 0; k < 3; ++k) rgb[k] = mix(0.24f + 0.02f * float(k), 0.05f + 0.01f * float(k), t);
        const float s = 0.7f * (-lx) + 0.7f * ly;
        const float glint = 0.32f * (1.0f - smoothstep(0.0f, 0.07f, std::abs(s - 0.35f))) +
                            0.2f * (1.0f - smoothstep(0.0f, 0.035f, std::abs(s - 0.6f)));
        for (int k = 0; k < 3; ++k) rgb[k] = mix(rgb[k], 0.75f, glint);
      } else {
        const Vec3 nl(dot(n, fr.r), dot(n, fr.u), dot(n, fr.f));
        const float shade = 0.8f + 1.6f * std::max(0.0f, dot(nl, kLight) - 0.6f);
        for (int k = 0; k < 3; ++k) rgb[k] = frame_rgb[k] * shade + (b > kLensB + kHalfH ? 0.03f : 0.0f);
        thick = 0.05f;
      }
      emit(out, fr.at(a, b, c), x, y, n, Vec3(sigma * length(ta) * fr.e, sigma * fr.e, thick * fr.e), rgb, alpha);
      ++added;
    }
  }

  // Arms: from the frame's outer top corners back along the head's sides.
  float l[3];
  std::vector<float> left, right;
  for (uint32_t i = 0; i < scene.count; ++i) {
    fr.local(&scene.positions[size_t(i) * 3], l);
    if (std::abs(l[1] - 0.1f) > 0.25f || l[2] < -2.0f || l[2] > -0.2f || std::abs(l[0]) > 2.2f) continue;
    if (sigmoid(scene.opacity_logits[i]) < 0.3f) continue;
    (l[0] < 0 ? left : right).push_back(std::abs(l[0]));
  }
  auto side_width = [](std::vector<float>& v) {
    if (v.size() < 20) return 1.2f;
    std::sort(v.begin(), v.end());
    return std::clamp(v[v.size() * 97 / 100], 0.9f, 1.6f);
  };
  const float widths[2] = {side_width(left), side_width(right)};
  for (int s = 0; s < 2; ++s) {
    const float side = s == 0 ? -1.0f : 1.0f;
    const float hinge_a = kLensA + kHalfW + kSide - 0.02f;
    const float w = std::max(widths[s] + 0.04f, hinge_a + 0.05f);
    const Vec3 p0(side * hinge_a, 0.16f, surface(hinge_a) - 0.03f);
    const Vec3 p1(side * w, 0.14f, std::min(p0.z - 0.35f, -0.3f));
    const Vec3 p2(side * w, 0.06f, -1.55f);
    for (const auto& seg : {std::pair<Vec3, Vec3>{p0, p1}, std::pair<Vec3, Vec3>{p1, p2}}) {
      const Vec3 d = seg.second - seg.first;
      const int steps = std::max(2, int(length(d) / 0.035f));
      const Vec3 dw = normalize(fr.r * d.x + fr.u * d.y + fr.f * d.z);
      const Vec3 zw = normalize(cross(dw, fr.u));
      const Vec3 yw = cross(zw, dw);
      for (int k = 0; k <= steps; ++k) {
        const Vec3 q = seg.first + d * (float(k) / float(steps));
        emit(out, fr.at(q.x, q.y, q.z), dw, yw, zw, Vec3(0.03f * fr.e, 0.04f * fr.e, 0.02f * fr.e), frame_rgb, 0.97f);
        ++added;
      }
    }
  }
  return added;
}

// Only the eyes, blurred: no new splats. The face's own splats in a band over
// the eyes take the colour of a heavily blurred picture of the face, fading
// into their own colour at the band's edge, so the rest of the face (and its
// shape) stays as it was. Inside the band they lose their view-dependent
// colour too.
uint32_t blur_eyes(SplatScene& scene, const FaceField& field, const Frame& fr) {
  constexpr float kHalfA = 1.02f, kHalfB = 0.3f, kCentreB = 0.02f;
  const Grid& g = field.g;
  const std::vector<float> soft = blur(field.rgb, field.alpha, 3, g.w, g.h, 0.25f / g.step);
  std::vector<float> channel[3];
  for (int c = 0; c < 3; ++c) {
    channel[c].resize(g.size());
    for (size_t p = 0; p < g.size(); ++p) channel[c][p] = soft[p * 3 + c];
  }
  const int coeffs = sh_rest_coeffs(scene.sh_degree);
  uint32_t count = 0;
  float l[3];
  for (uint32_t i = 0; i < scene.count; ++i) {
    fr.local(&scene.positions[size_t(i) * 3], l);
    // A rounded bar over both eyes, soft over its outer quarter.
    const float x = l[0] / kHalfA, y = (l[1] - kCentreB) / kHalfB;
    const float r = std::pow(std::pow(std::abs(x), 4.0f) + std::pow(std::abs(y), 4.0f), 0.25f);
    const float w = 1.0f - smoothstep(0.75f, 1.0f, r);
    if (w <= 0.0f) continue;
    const float surface = field.sample(field.height, l[0], l[1]);
    if (std::isnan(surface) || l[2] < surface - 0.5f) continue;
    float rgb[3];
    bool ok = true;
    for (int c = 0; c < 3 && ok; ++c) {
      rgb[c] = field.sample(channel[c], l[0], l[1]);
      ok = !std::isnan(rgb[c]);
    }
    if (!ok) continue;
    for (int c = 0; c < 3; ++c) {
      float& dc = scene.sh_dc[size_t(i) * 3 + c];
      dc = mix(dc, (rgb[c] - 0.5f) / kShC0, w);
    }
    for (int k = 0; k < coeffs * 3; ++k) scene.sh_rest[size_t(i) * coeffs * 3 + k] *= w >= 0.999f ? 0.0f : 1.0f - w;
    ++count;
  }
  return count;
}

}  // namespace

// --- Pictures ------------------------------------------------------------------------

void OrthoView::point(float x, float y, float d, float out[3]) const {
  const float ox = (x - 0.5f * float(width)) * pixel, oy = (y - 0.5f * float(height)) * pixel;
  for (int k = 0; k < 3; ++k) out[k] = origin[k] + ox * right[k] + oy * down[k] + d * forward[k];
}

float OrthoView::depth_near(float x, float y, int reach) const {
  const int cx = int(std::floor(x)), cy = int(std::floor(y));
  auto valid = [&](int px, int py, float* d) {
    if (px < 0 || py < 0 || px >= int(width) || py >= int(height)) return false;
    const size_t p = size_t(py) * width + px;
    if (alpha[p] < 0.5f || std::isnan(depth[p])) return false;
    *d = depth[p];
    return true;
  };
  std::vector<float> near;
  float d;
  for (int dy = -2; dy <= 2; ++dy) {
    for (int dx = -2; dx <= 2; ++dx) {
      if (valid(cx + dx, cy + dy, &d)) near.push_back(d);
    }
  }
  if (!near.empty()) {
    std::nth_element(near.begin(), near.begin() + near.size() / 2, near.end());
    return near[near.size() / 2];
  }
  for (int r = 3; r <= reach; ++r) {
    for (int dy = -r; dy <= r; ++dy) {
      for (int dx = -r; dx <= r; ++dx) {
        if (std::max(std::abs(dx), std::abs(dy)) != r) continue;
        if (valid(cx + dx, cy + dy, &d)) return d;
      }
    }
  }
  return kNaN;
}

std::vector<uint8_t> OrthoView::rgb8() const {
  std::vector<uint8_t> out(rgb.size());
  for (size_t i = 0; i < rgb.size(); ++i) out[i] = uint8_t(std::clamp(rgb[i], 0.0f, 1.0f) * 255.0f + 0.5f);
  return out;
}

void render_ortho(const SplatScene& scene, const std::vector<uint32_t>& subset, OrthoView& v,
                  const float background[3]) {
  const size_t pixels = size_t(v.width) * v.height;
  v.rgb.assign(pixels * 3, 0.0f);
  v.alpha.assign(pixels, 0.0f);
  v.depth.assign(pixels, kNaN);
  std::vector<float> transmit(pixels, 1.0f);
  const Vec3 o(v.origin), R(v.right), D(v.down), F(v.forward);
  struct Item {
    float depth;
    uint32_t index;
  };
  std::vector<Item> items;
  const size_t n = subset.empty() ? scene.count : subset.size();
  items.reserve(n);
  for (size_t k = 0; k < n; ++k) {
    const uint32_t i = subset.empty() ? uint32_t(k) : subset[k];
    items.push_back({dot(Vec3(&scene.positions[size_t(i) * 3]) - o, F), i});
  }
  std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.depth < b.depth; });
  const float inv = 1.0f / v.pixel;
  const int w = int(v.width), h = int(v.height);
  for (const Item& it : items) {
    const uint32_t i = it.index;
    float opacity = sigmoid(scene.opacity_logits[i]);
    if (opacity < 1.0f / 255.0f) continue;
    const Vec3 rel = Vec3(&scene.positions[size_t(i) * 3]) - o;
    const float gx = dot(rel, R) * inv + 0.5f * float(w) - 0.5f;
    const float gy = dot(rel, D) * inv + 0.5f * float(h) - 0.5f;
    float m[9];
    quat_matrix(&scene.rotations[size_t(i) * 4], m);
    float a0[3], a1[3];
    for (int c = 0; c < 3; ++c) {
      const float s = std::exp(scene.log_scales[size_t(i) * 3 + c]);
      a0[c] = (R.x * m[0 + c] + R.y * m[3 + c] + R.z * m[6 + c]) * s * inv;
      a1[c] = (D.x * m[0 + c] + D.y * m[3 + c] + D.z * m[6 + c]) * s * inv;
    }
    float sxx = a0[0] * a0[0] + a0[1] * a0[1] + a0[2] * a0[2];
    const float sxy = a0[0] * a1[0] + a0[1] * a1[1] + a0[2] * a1[2];
    float syy = a1[0] * a1[0] + a1[1] * a1[1] + a1[2] * a1[2];
    const float det_raw = sxx * syy - sxy * sxy;
    sxx += kDilation;
    syy += kDilation;
    const float det = sxx * syy - sxy * sxy;
    if (!(det > 0.0f)) continue;
    if (scene.antialiased) opacity *= std::sqrt(std::max(det_raw, 0.0f) / det);
    const float ia = syy / det, ib = -sxy / det, ic = sxx / det;
    const float mid = 0.5f * (sxx + syy);
    const float l1 = mid + std::sqrt(std::max(0.0f, mid * mid - det));
    const float r = 3.0f * std::sqrt(l1);
    const int x0 = std::max(0, int(std::floor(gx - r))), x1 = std::min(w - 1, int(std::ceil(gx + r)));
    const int y0 = std::max(0, int(std::floor(gy - r))), y1 = std::min(h - 1, int(std::ceil(gy + r)));
    if (x0 > x1 || y0 > y1) continue;
    float color[3];
    sh_color(scene, i, F, color);
    for (int y = y0; y <= y1; ++y) {
      for (int x = x0; x <= x1; ++x) {
        const size_t p = size_t(y) * w + x;
        const float t = transmit[p];
        if (t < 1e-4f) continue;
        const float dx = float(x) - gx, dy = float(y) - gy;
        const float power = -0.5f * (ia * dx * dx + 2.0f * ib * dx * dy + ic * dy * dy);
        if (power > 0.0f) continue;
        const float alpha = std::min(0.99f, opacity * std::exp(power));
        if (alpha < 1.0f / 255.0f) continue;
        for (int c = 0; c < 3; ++c) v.rgb[p * 3 + c] += t * alpha * color[c];
        const float next = t * (1.0f - alpha);
        if (t >= 0.5f && next < 0.5f) v.depth[p] = it.depth;
        transmit[p] = next;
      }
    }
  }
  for (size_t p = 0; p < pixels; ++p) {
    for (int c = 0; c < 3; ++c) v.rgb[p * 3 + c] += transmit[p] * background[c];
    v.alpha[p] = 1.0f - transmit[p];
  }
}

// --- Finding the face ----------------------------------------------------------------

HeadRegion find_head(const SplatScene& scene, const float up_in[3]) {
  const Vec3 up = normalize(Vec3(up_in));
  float u[3];
  up.store(u);
  HeadRegion head = estimate_head(scene, u);
  if (head.found || scene.count < 200) return head;
  // Shorter than a standing person (head and shoulders): the head is at the top.
  std::vector<float> heights(scene.count);
  for (uint32_t i = 0; i < scene.count; ++i) heights[i] = dot(Vec3(&scene.positions[size_t(i) * 3]), up);
  std::vector<float> sorted = heights;
  std::sort(sorted.begin(), sorted.end());
  const float lo = sorted[sorted.size() / 100], hi = sorted[sorted.size() - 1 - sorted.size() / 200];
  const float height = hi - lo;
  if (!(height > 0.0f)) return head;
  const float radius = std::min(0.17f, 0.45f * height);
  std::vector<float> across[3];
  for (uint32_t i = 0; i < scene.count; ++i) {
    if (heights[i] < hi - 1.4f * radius || heights[i] > hi) continue;
    const Vec3 p(&scene.positions[size_t(i) * 3]);
    const Vec3 flat = p - up * heights[i];
    across[0].push_back(flat.x);
    across[1].push_back(flat.y);
    across[2].push_back(flat.z);
  }
  if (across[0].size() < 20) return head;
  const float middle = hi - 0.85f * radius;
  for (int k = 0; k < 3; ++k) {
    std::vector<float>& v = across[k];
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    head.center[k] = v[v.size() / 2] + middle * u[k];
  }
  head.radius = radius;
  head.subject_height = height;
  head.found = true;
  return head;
}

FaceViews render_face_views(const SplatScene& scene, const float up_in[3], int count, uint32_t size,
                            const float* toward) {
  FaceViews fv;
  const Vec3 up = normalize(Vec3(up_in));
  up.store(fv.up);
  fv.head = find_head(scene, fv.up);
  if (!fv.head.found) return fv;
  const Vec3 centre(fv.head.center);
  const float radius = fv.head.radius;
  for (uint32_t i = 0; i < scene.count; ++i) {
    if (length(Vec3(&scene.positions[size_t(i) * 3]) - centre) < 1.6f * radius) fv.subset.push_back(i);
  }
  const Vec3 any = std::abs(up.z) < 0.9f ? Vec3(0, 0, 1) : Vec3(1, 0, 0);
  const Vec3 h1 = normalize(any - up * dot(any, up));
  const Vec3 h2 = cross(up, h1);
  std::vector<float> azimuths;
  bool around = true;
  if (toward != nullptr) {
    const Vec3 t = Vec3(toward) - up * dot(Vec3(toward), up);
    if (length(t) > 1e-3f) {
      // The camera sits opposite the direction it looks.
      const Vec3 from = -normalize(t);
      const float base = std::atan2(dot(from, h2), dot(from, h1));
      for (float d : {0.0f, -20.0f, 20.0f}) azimuths.push_back(base + d * kPi / 180.0f);
      around = false;
    }
  }
  if (around) {
    for (int k = 0; k < count; ++k) azimuths.push_back(2.0f * kPi * float(k) / float(count));
  }
  const float background[3] = {0.82f, 0.82f, 0.82f};
  fv.views.resize(azimuths.size());
  fv.azimuth = azimuths;
  for (size_t k = 0; k < azimuths.size(); ++k) {
    OrthoView& v = fv.views[k];
    const Vec3 from = h1 * std::cos(azimuths[k]) + h2 * std::sin(azimuths[k]);
    const Vec3 forward = -from;
    const Vec3 down = -up;
    centre.store(v.origin);
    forward.store(v.forward);
    down.store(v.down);
    cross(down, forward).store(v.right);
    v.width = v.height = size;
    v.pixel = 2.6f * radius / float(size);
  }
  // The pictures are independent: one thread each, as many as there are cores.
  const size_t workers = std::clamp<size_t>(std::thread::hardware_concurrency(), 1, fv.views.size());
  std::vector<std::thread> threads;
  for (size_t w = 0; w < workers; ++w) {
    threads.emplace_back([&, w] {
      for (size_t k = w; k < fv.views.size(); k += workers) render_ortho(scene, fv.subset, fv.views[k], background);
    });
  }
  for (std::thread& t : threads) t.join();
  return fv;
}

std::vector<FaceLandmarks> parse_face_landmarks(std::string_view json) {
  const JsonValue root = parse_json(json);
  std::vector<FaceLandmarks> out;
  if (root.type != JsonValue::Type::Array) return out;
  auto point = [](const JsonValue* v, float xy[2]) {
    if (v == nullptr || v->type != JsonValue::Type::Array || v->array.size() < 2) return false;
    xy[0] = float(v->array[0].number);
    xy[1] = float(v->array[1].number);
    return true;
  };
  for (const JsonValue& item : root.array) {
    FaceLandmarks f;
    f.view = int(item.number_or("view", -1));
    const JsonValue* eyes = item.find("eyes");
    if (f.view < 0 || eyes == nullptr || eyes->type != JsonValue::Type::Array || eyes->array.size() < 2) continue;
    if (!point(&eyes->array[0], f.eyes[0]) || !point(&eyes->array[1], f.eyes[1])) continue;
    f.has_mouth = point(item.find("mouth"), f.mouth);
    out.push_back(f);
  }
  return out;
}

FacePlacement place_face(const SplatScene& scene, const FaceViews& views, const std::vector<FaceLandmarks>& faces) {
  (void)scene;
  FacePlacement out;
  if (!views.head.found) return out;
  // Every detection's eyes (and mouth) on the model, through its picture's depth.
  struct Lifted {
    Vec3 left, right, mouth;
    bool has_mouth = false;
    float px = 0;  // eye distance in the picture: wider is more frontal
    Vec3 mid() const { return (left + right) * 0.5f; }
    float spacing() const { return length(right - left); }
  };
  std::vector<Lifted> lifted;
  for (const FaceLandmarks& f : faces) {
    if (f.view < 0 || f.view >= int(views.views.size())) continue;
    const OrthoView& v = views.views[size_t(f.view)];
    const int reach = int(0.04f * float(v.width)) + 3;
    auto lift = [&](const float* xy, Vec3* out3) {
      const float d = v.depth_near(xy[0], xy[1], reach);
      if (std::isnan(d)) return false;
      float p[3];
      v.point(xy[0], xy[1], d, p);
      *out3 = Vec3(p);
      return true;
    };
    Lifted l;
    l.px = std::hypot(f.eyes[0][0] - f.eyes[1][0], f.eyes[0][1] - f.eyes[1][1]);
    // Image left first: facing the person, that is the viewer's left.
    const int first = f.eyes[0][0] <= f.eyes[1][0] ? 0 : 1;
    if (l.px <= 2.0f || !lift(f.eyes[first], &l.left) || !lift(f.eyes[1 - first], &l.right)) continue;
    l.has_mouth = f.has_mouth && lift(f.mouth, &l.mouth);
    lifted.push_back(l);
  }
  // Detectors also see faces in hair and the backs of heads, but only the real
  // face is found in the same place from several directions: take the
  // detection most others agree with (ties: the most frontal).
  auto agree = [](const Lifted& a, const Lifted& b) {
    const float e = std::max(a.spacing(), b.spacing());
    return length(a.mid() - b.mid()) < 0.35f * e && std::abs(a.spacing() - b.spacing()) < 0.35f * e;
  };
  int best = -1, best_support = 0;
  for (size_t i = 0; i < lifted.size(); ++i) {
    int support = 0;
    for (const Lifted& other : lifted) support += agree(lifted[i], other) ? 1 : 0;
    if (support > best_support || (support == best_support && lifted[i].px > lifted[size_t(best)].px)) {
      best = int(i);
      best_support = support;
    }
  }
  if (best < 0) return out;
  // Average the most frontal of those that agree.
  float widest = 0;
  for (const Lifted& l : lifted) {
    if (agree(lifted[size_t(best)], l)) widest = std::max(widest, l.px);
  }
  Vec3 left, right, mouth;
  int used = 0, mouths = 0;
  for (const Lifted& l : lifted) {
    if (!agree(lifted[size_t(best)], l) || l.px < 0.8f * widest) continue;
    left = left + l.left;
    right = right + l.right;
    ++used;
    if (l.has_mouth) {
      mouth = mouth + l.mouth;
      ++mouths;
    }
  }
  left = left * (1.0f / float(used));
  right = right * (1.0f / float(used));
  const Vec3 across = right - left;
  const float e = length(across);
  const float radius = views.head.radius;
  // A detection whose eyes are implausibly far apart or close for this head is not a face.
  if (!(e > 0.15f * radius && e < 0.8f * radius)) return out;
  const Vec3 r = normalize(across);
  const Vec3 centre = (left + right) * 0.5f;
  Vec3 u0 = Vec3(views.up);
  if (mouths > 0) {
    const Vec3 m = mouth * (1.0f / float(mouths));
    const Vec3 candidate = centre - m;
    // Use the mouth only when it is where a mouth can be (below the eyes).
    if (dot(normalize(candidate), Vec3(views.up)) > 0.5f) u0 = candidate;
  }
  const Vec3 u = normalize(u0 - r * dot(u0, r));
  const Vec3 f = cross(r, u);
  centre.store(out.center);
  r.store(out.right);
  u.store(out.up);
  f.store(out.forward);
  out.eye_distance = e;
  out.found = true;
  out.detected = true;
  out.views_used = used;
  out.views_agreeing = best_support;
  return out;
}

FacePlacement guess_face(const SplatScene& scene, const FaceViews& views, const float facing[3]) {
  FacePlacement out;
  if (!views.head.found) return out;
  const Vec3 up(views.up);
  Vec3 f = Vec3(facing) - up * dot(Vec3(facing), up);
  if (length(f) < 1e-3f) return out;
  f = normalize(f);
  const Vec3 r = cross(up, f);
  const float radius = views.head.radius;
  // Eyes a little above the head's centre: about 1 cm for a standing adult
  // with ordinary hair. Hair raises the top the centre is measured from, by
  // anything from a little to (with buns) several centimetres, so this is
  // where the cover starts; the user moves it if needed.
  const Vec3 eyes = Vec3(views.head.center) + up * (0.06f * radius);
  // Find the surface in front of the eyes.
  OrthoView v;
  (eyes + f * (2.0f * radius)).store(v.origin);
  r.store(v.right);
  (-up).store(v.down);
  (-f).store(v.forward);
  v.width = v.height = 64;
  v.pixel = 0.6f * radius / 64.0f;
  const float black[3] = {0, 0, 0};
  render_ortho(scene, views.subset, v, black);
  const float d = v.depth_near(32.0f, 32.0f, 24);
  Vec3 centre = eyes + f * (0.8f * radius);
  if (!std::isnan(d)) {
    float p[3];
    v.point(32.0f, 32.0f, d, p);
    centre = Vec3(p);
  }
  centre.store(out.center);
  r.store(out.right);
  up.store(out.up);
  f.store(out.forward);
  out.eye_distance = 0.37f * radius;
  out.found = true;
  out.detected = false;
  return out;
}

// --- Covering it ---------------------------------------------------------------------

bool parse_face_style(std::string_view name, FaceStyle* style) {
  static constexpr std::pair<std::string_view, FaceStyle> kNames[] = {
      {"mosaic", FaceStyle::Mosaic}, {"blur", FaceStyle::Blur},   {"sunglasses", FaceStyle::Sunglasses},
      {"mask", FaceStyle::Mask},     {"smile", FaceStyle::Smile},   {"eyes", FaceStyle::Eyes},
  };
  for (const auto& [n, s] : kNames) {
    if (n == name) {
      *style = s;
      return true;
    }
  }
  return false;
}

SplatScene cover_face(const SplatScene& scene, const FacePlacement& face, const FaceCoverOptions& options,
                      FaceCoverStats* stats) {
  if (!face.found || !(face.eye_distance > 0.0f)) throw std::runtime_error("얼굴 위치를 모름");
  FaceCoverStats local;
  Frame fr;
  fr.r = normalize(Vec3(face.right));
  fr.u = normalize(Vec3(face.up));
  fr.f = normalize(Vec3(face.forward));
  fr.o = Vec3(face.center) + fr.u * (options.raise * face.eye_distance);
  fr.e = face.eye_distance * std::clamp(options.size, 0.3f, 3.0f);

  SplatScene out = scene;
  const FaceField field = face_field(scene, fr);
  if (options.style == FaceStyle::Sunglasses) {
    local.added = sunglasses(scene, field, fr, out);
  } else if (options.style == FaceStyle::Eyes) {
    local.recoloured = blur_eyes(out, field, fr);
  } else {
    Layer layer;
    layer.init(-2.0f, 2.2f, 4.0f, 5.0f, 1.0f / 32.0f);
    switch (options.style) {
      case FaceStyle::Mosaic:
        mosaic(field, layer);
        break;
      case FaceStyle::Blur:
        blur_face(field, layer);
        break;
      case FaceStyle::Mask:
        mask(field, layer);
        break;
      case FaceStyle::Smile:
        smile(field, layer);
        break;
      case FaceStyle::Sunglasses:
      case FaceStyle::Eyes:
        break;
    }
    local.recoloured = recolour_under(out, layer, field, fr, 0.5f);
    local.added = emit_layer(layer, fr, out);
  }
  if (stats) *stats = local;
  return out;
}

}  // namespace gs
