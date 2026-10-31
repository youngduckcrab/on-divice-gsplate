#include "gs/capture.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace gs {
namespace {

// Box-filter taps for resizing `in_len` samples to `out_len` (area average;
// upscaling picks the covering sample).
struct Taps {
  std::vector<uint32_t> first;
  std::vector<uint32_t> offset;  // into weights, out_len + 1 entries
  std::vector<float> weights;
};

Taps area_taps(uint32_t in_len, uint32_t out_len) {
  Taps t;
  t.first.resize(out_len);
  t.offset.resize(out_len + 1);
  const double ratio = double(in_len) / out_len;
  for (uint32_t o = 0; o < out_len; ++o) {
    const double x0 = o * ratio, x1 = (o + 1) * ratio;
    const uint32_t i0 = std::min(in_len - 1, uint32_t(std::floor(x0)));
    const uint32_t i1 = std::max(i0 + 1, std::min(in_len, uint32_t(std::ceil(x1))));
    t.first[o] = i0;
    t.offset[o] = uint32_t(t.weights.size());
    for (uint32_t i = i0; i < i1; ++i) {
      const double w = std::min<double>(i + 1, x1) - std::max<double>(i, x0);
      t.weights.push_back(float(w / ratio));
    }
  }
  t.offset[out_len] = uint32_t(t.weights.size());
  return t;
}

// One channel of an image (`step` bytes between samples, `row_stride` between
// rows) resized to out_w x out_h, as floats.
std::vector<float> resize_plane(const uint8_t* src, uint32_t w, uint32_t h, size_t row_stride, size_t step,
                                uint32_t out_w, uint32_t out_h) {
  const Taps tx = area_taps(w, out_w), ty = area_taps(h, out_h);
  std::vector<float> rows(size_t(h) * out_w);
  for (uint32_t y = 0; y < h; ++y) {
    const uint8_t* line = src + size_t(y) * row_stride;
    float* out = &rows[size_t(y) * out_w];
    for (uint32_t o = 0; o < out_w; ++o) {
      const float* wt = &tx.weights[tx.offset[o]];
      const uint32_t n = tx.offset[o + 1] - tx.offset[o];
      const uint8_t* s = line + size_t(tx.first[o]) * step;
      float sum = 0.0f;
      for (uint32_t k = 0; k < n; ++k) sum += wt[k] * s[k * step];
      out[o] = sum;
    }
  }
  std::vector<float> out(size_t(out_w) * out_h, 0.0f);
  for (uint32_t o = 0; o < out_h; ++o) {
    float* line = &out[size_t(o) * out_w];
    for (uint32_t k = ty.offset[o]; k < ty.offset[o + 1]; ++k) {
      const float wt = ty.weights[k];
      const float* in = &rows[size_t(ty.first[o] + (k - ty.offset[o])) * out_w];
      for (uint32_t x = 0; x < out_w; ++x) line[x] += wt * in[x];
    }
  }
  return out;
}

uint8_t to_byte(float v) { return uint8_t(std::clamp(v + 0.5f, 0.0f, 255.0f)); }

float dot3(const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// Squared distance from (x, y) to the segment a-b.
float segment_distance2(float x, float y, const std::array<float, 2>& a, const std::array<float, 2>& b) {
  const float dx = b[0] - a[0], dy = b[1] - a[1];
  const float len2 = dx * dx + dy * dy;
  float t = len2 > 0.0f ? ((x - a[0]) * dx + (y - a[1]) * dy) / len2 : 0.0f;
  t = std::clamp(t, 0.0f, 1.0f);
  const float px = a[0] + t * dx - x, py = a[1] + t * dy - y;
  return px * px + py * py;
}

}  // namespace

// --- Images ------------------------------------------------------------------

std::vector<uint8_t> yuv420_to_rgb(const Yuv420View& im, uint32_t out_w, uint32_t out_h) {
  if (im.width < 2 || im.height < 2 || out_w == 0 || out_h == 0) throw std::invalid_argument("빈 카메라 이미지");
  const uint32_t cw = im.width / 2, ch = im.height / 2;
  const std::vector<float> y = resize_plane(im.y, im.width, im.height, im.y_row_stride, 1, out_w, out_h);
  const std::vector<float> u = resize_plane(im.u, cw, ch, im.uv_row_stride, im.uv_pixel_stride, out_w, out_h);
  const std::vector<float> v = resize_plane(im.v, cw, ch, im.uv_row_stride, im.uv_pixel_stride, out_w, out_h);
  std::vector<uint8_t> rgb(size_t(out_w) * out_h * 3);
  for (size_t i = 0; i < y.size(); ++i) {
    // JFIF: full-range BT.601.
    const float cb = u[i] - 128.0f, cr = v[i] - 128.0f;
    rgb[i * 3 + 0] = to_byte(y[i] + 1.402f * cr);
    rgb[i * 3 + 1] = to_byte(y[i] - 0.344136f * cb - 0.714136f * cr);
    rgb[i * 3 + 2] = to_byte(y[i] + 1.772f * cb);
  }
  return rgb;
}

std::vector<uint8_t> resize_box(const uint8_t* src, uint32_t w, uint32_t h, uint32_t channels, uint32_t out_w,
                                uint32_t out_h) {
  if (w == out_w && h == out_h) return std::vector<uint8_t>(src, src + size_t(w) * h * channels);
  std::vector<uint8_t> out(size_t(out_w) * out_h * channels);
  for (uint32_t c = 0; c < channels; ++c) {
    const std::vector<float> plane = resize_plane(src + c, w, h, size_t(w) * channels, channels, out_w, out_h);
    for (size_t i = 0; i < plane.size(); ++i) out[i * channels + c] = to_byte(plane[i]);
  }
  return out;
}

std::vector<uint8_t> rotate_image(const uint8_t* src, uint32_t w, uint32_t h, uint32_t channels, int degrees) {
  std::vector<uint8_t> dst(size_t(w) * h * channels);
  auto put = [&](size_t to, size_t from) { std::memcpy(&dst[to * channels], &src[from * channels], channels); };
  switch (degrees) {
    case 0:
      std::memcpy(dst.data(), src, dst.size());
      break;
    case 90:  // (x, y) -> (h - 1 - y, x) in an h x w image
      for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) put(size_t(x) * h + (h - 1 - y), size_t(y) * w + x);
      }
      break;
    case 180:
      for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) put(size_t(h - 1 - y) * w + (w - 1 - x), size_t(y) * w + x);
      }
      break;
    case 270:
      for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) put(size_t(w - 1 - x) * h + y, size_t(y) * w + x);
      }
      break;
    default:
      throw std::invalid_argument("회전은 0/90/180/270만 가능");
  }
  return dst;
}

double sharpness(const uint8_t* y, uint32_t row_stride, uint32_t w, uint32_t h, uint32_t step) {
  double sum = 0.0, sum_sq = 0.0;
  size_t n = 0;
  for (uint32_t py = step; py + step < h; py += step) {
    const uint8_t* row = y + size_t(py) * row_stride;
    const uint8_t* up = row - size_t(step) * row_stride;
    const uint8_t* down = row + size_t(step) * row_stride;
    for (uint32_t px = step; px + step < w; px += step) {
      const int l = 4 * row[px] - row[px - step] - row[px + step] - up[px] - down[px];
      sum += l;
      sum_sq += double(l) * l;
      ++n;
    }
  }
  if (n == 0) return 0.0;
  const double mean = sum / n;
  return sum_sq / n - mean * mean;
}

// --- Subject -------------------------------------------------------------------

SubjectSeeds subject_seeds(const float c2w[16], float fx, float fy, float cx, float cy, uint32_t width,
                           uint32_t height, const float anchor[3], const float up[3], float half_height,
                           float radius_m) {
  SubjectSeeds seeds;
  const float t[3] = {c2w[3], c2w[7], c2w[11]};
  auto project = [&](const float p[3], float* u, float* v, float* depth) {
    const float d[3] = {p[0] - t[0], p[1] - t[1], p[2] - t[2]};
    // World -> OpenGL camera is R^T; OpenCV flips y and z.
    const float xg = c2w[0] * d[0] + c2w[4] * d[1] + c2w[8] * d[2];
    const float yg = c2w[1] * d[0] + c2w[5] * d[1] + c2w[9] * d[2];
    const float zg = c2w[2] * d[0] + c2w[6] * d[1] + c2w[10] * d[2];
    const float z = -zg;
    if (z <= 0.05f) return false;
    *u = fx * xg / z + cx;
    *v = fy * -yg / z + cy;
    *depth = z;
    return true;
  };
  float depth = 0.0f, u = 0.0f, v = 0.0f;
  if (!project(anchor, &u, &v, &depth)) return seeds;
  seeds.radius = std::max(6.0f, std::max(fx, fy) * radius_m / depth);
  constexpr int kSteps = 6;
  for (int i = -kSteps; i <= kSteps; ++i) {
    const float s = half_height * float(i) / kSteps;
    const float p[3] = {anchor[0] + s * up[0], anchor[1] + s * up[1], anchor[2] + s * up[2]};
    if (project(p, &u, &v, &depth)) seeds.points.push_back({u, v});
  }
  (void)width;
  (void)height;
  return seeds;
}

SubjectSelection select_subject(uint8_t* mask, uint32_t w, uint32_t h, const SubjectSeeds& seeds) {
  SubjectSelection result;
  const size_t n = size_t(w) * h;
  // Regions of the solid mask, 4-connected.
  std::vector<int32_t> label(n, -1);
  std::vector<uint32_t> stack;
  std::vector<float> nearest;  // per region: squared distance to the seeds
  std::vector<uint32_t> area;
  auto seed_distance2 = [&](uint32_t x, uint32_t y) {
    float best = std::numeric_limits<float>::infinity();
    const float fx = x + 0.5f, fy = y + 0.5f;
    if (seeds.points.size() == 1) return segment_distance2(fx, fy, seeds.points[0], seeds.points[0]);
    for (size_t i = 0; i + 1 < seeds.points.size(); ++i) {
      best = std::min(best, segment_distance2(fx, fy, seeds.points[i], seeds.points[i + 1]));
    }
    return best;
  };
  for (size_t start = 0; start < n; ++start) {
    if (mask[start] < 128 || label[start] >= 0) continue;
    const int32_t id = int32_t(nearest.size());
    float best = std::numeric_limits<float>::infinity();
    uint32_t count = 0;
    label[start] = id;
    stack.push_back(uint32_t(start));
    while (!stack.empty()) {
      const uint32_t p = stack.back();
      stack.pop_back();
      const uint32_t x = p % w, y = p / w;
      ++count;
      if (!seeds.points.empty()) best = std::min(best, seed_distance2(x, y));
      auto visit = [&](uint32_t q) {
        if (mask[q] >= 128 && label[q] < 0) {
          label[q] = id;
          stack.push_back(q);
        }
      };
      if (x > 0) visit(p - 1);
      if (x + 1 < w) visit(p + 1);
      if (y > 0) visit(p - w);
      if (y + 1 < h) visit(p + w);
    }
    nearest.push_back(best);
    area.push_back(count);
  }
  result.regions = uint32_t(nearest.size());
  std::vector<bool> keep(nearest.size(), false);
  const float r2 = seeds.radius * seeds.radius;
  bool any = false;
  for (size_t i = 0; i < nearest.size(); ++i) {
    if (nearest[i] <= r2) keep[i] = any = true;
  }
  if (!any && !nearest.empty()) {
    const size_t i = size_t(std::min_element(nearest.begin(), nearest.end()) - nearest.begin());
    if (nearest[i] <= 16.0f * r2) keep[i] = any = true;
  }
  result.found = any;
  // The kept regions and the soft edge reachable from them (values 1..127, 8-connected).
  std::vector<uint8_t> kept(n, 0);
  for (size_t p = 0; p < n; ++p) {
    if (label[p] >= 0 && keep[size_t(label[p])]) {
      kept[p] = 1;
      stack.push_back(uint32_t(p));
    }
  }
  while (!stack.empty()) {
    const uint32_t p = stack.back();
    stack.pop_back();
    const int x = int(p % w), y = int(p / w);
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        const int qx = x + dx, qy = y + dy;
        if (qx < 0 || qy < 0 || qx >= int(w) || qy >= int(h)) continue;
        const size_t q = size_t(qy) * w + qx;
        if (kept[q] || mask[q] == 0 || mask[q] >= 128) continue;
        kept[q] = 1;
        stack.push_back(uint32_t(q));
      }
    }
  }
  double covered = 0.0, removed = 0.0;
  for (size_t p = 0; p < n; ++p) {
    if (kept[p]) {
      covered += mask[p];
    } else {
      removed += mask[p];
      mask[p] = 0;
    }
  }
  for (size_t i = 0; i < keep.size(); ++i) result.dropped += keep[i] ? 0 : 1;
  result.coverage = float(covered / (255.0 * n));
  result.removed = float(removed / (255.0 * n));
  return result;
}

float gate_by_depth(uint8_t* mask, uint32_t w, uint32_t h, const float* depth, const uint8_t* confidence,
                    uint32_t dw, uint32_t dh, const float c2w[16], float fx, float fy, float cx, float cy,
                    const float anchor[3], const float up[3], float radius, uint8_t min_confidence) {
  if (depth == nullptr || dw < 3 || dh < 3) return 0.0f;
  // Depth samples further than `radius` from the axis.
  std::vector<uint8_t> far(size_t(dw) * dh, 0);
  for (uint32_t v = 0; v < dh; ++v) {
    for (uint32_t u = 0; u < dw; ++u) {
      const size_t i = size_t(v) * dw + u;
      const float z = depth[i];
      if (!(z > 0.0f) || (confidence != nullptr && confidence[i] < min_confidence)) continue;
      const float px = (u + 0.5f) * w / dw, py = (v + 0.5f) * h / dh;
      // OpenCV camera (x, y, z) is OpenGL (x, -y, -z).
      const float xc = (px - cx) / fx * z, yc = -(py - cy) / fy * z, zc = -z;
      float d[3];
      for (int r = 0; r < 3; ++r) {
        d[r] = c2w[r * 4 + 0] * xc + c2w[r * 4 + 1] * yc + c2w[r * 4 + 2] * zc + c2w[r * 4 + 3] - anchor[r];
      }
      const float along = dot3(d, up);
      const float off[3] = {d[0] - along * up[0], d[1] - along * up[1], d[2] - along * up[2]};
      far[i] = dot3(off, off) > radius * radius ? 1 : 0;
    }
  }
  std::vector<uint8_t> sure(far.size(), 0);
  for (uint32_t v = 1; v + 1 < dh; ++v) {
    for (uint32_t u = 1; u + 1 < dw; ++u) {
      bool all = true;
      for (int dy = -1; dy <= 1 && all; ++dy) {
        for (int dx = -1; dx <= 1 && all; ++dx) all = far[size_t(v + dy) * dw + (u + dx)] != 0;
      }
      sure[size_t(v) * dw + u] = all ? 1 : 0;
    }
  }
  double cleared = 0.0;
  for (uint32_t y = 0; y < h; ++y) {
    const uint32_t v = std::min(dh - 1, uint32_t(uint64_t(y) * dh / h));
    for (uint32_t x = 0; x < w; ++x) {
      uint8_t& m = mask[size_t(y) * w + x];
      if (m == 0 || !sure[size_t(v) * dw + std::min(dw - 1, uint32_t(uint64_t(x) * dw / w))]) continue;
      cleared += m;
      m = 0;
    }
  }
  return float(cleared / (255.0 * w * h));
}

// --- Keyframes -----------------------------------------------------------------

void KeyframePolicy::set_subject(const float anchor[3]) {
  std::copy(anchor, anchor + 3, subject_);
  has_subject_ = true;
}

float KeyframePolicy::angle_degrees(const float a[4], const float b[4]) {
  const double dot = std::min(1.0, std::fabs(double(a[0]) * b[0] + double(a[1]) * b[1] + double(a[2]) * b[2] +
                                             double(a[3]) * b[3]));
  return float(2.0 * std::acos(dot) * 180.0 / M_PI);
}

int KeyframePolicy::azimuth_bin(const float camera[3], const float subject[3]) {
  const double dx = double(camera[0]) - subject[0], dz = double(camera[2]) - subject[2];
  if (dx * dx + dz * dz < 1e-6) return -1;
  // Degrees from +z towards +x, in double: a float -pi/2 lands a hair past -90.
  const double degrees = std::fmod(std::atan2(dx, dz) * 180.0 / M_PI + 360.0, 360.0);
  return std::min(kBins - 1, int(degrees / (360.0 / kBins)));
}

int KeyframePolicy::azimuth_bin() const {
  if (!has_subject_ || !tracking_ || !has_previous_) return -1;
  return azimuth_bin(previous_position_, subject_);
}

float KeyframePolicy::distance() const {
  if (!has_subject_ || !tracking_ || !has_previous_) return 0.0f;
  const float dx = previous_position_[0] - subject_[0], dz = previous_position_[2] - subject_[2];
  return std::sqrt(dx * dx + dz * dz);
}

bool KeyframePolicy::track(bool tracking, const float position[3], const float rotation[4], int64_t t) {
  if (recording_ && tracking_ && !tracking) ++counters_.tracking_lost;
  tracking_ = tracking;
  if (!tracking) return false;
  if (has_previous_) {
    const float dt = float(t - previous_time_) / 1e9f;
    if (dt > 0.0f) {
      const float d[3] = {position[0] - previous_position_[0], position[1] - previous_position_[1],
                          position[2] - previous_position_[2]};
      linear_speed_ = 0.8f * linear_speed_ + 0.2f * (std::sqrt(dot3(d, d)) / dt);
      angular_speed_ = 0.8f * angular_speed_ + 0.2f * (angle_degrees(rotation, previous_rotation_) / dt);
    }
  }
  std::copy(position, position + 3, previous_position_);
  std::copy(rotation, rotation + 4, previous_rotation_);
  previous_time_ = t;
  has_previous_ = true;
  if (!recording_ || !has_subject_ || keyframes_ >= options_.max_keyframes) return false;

  if (has_key_) {
    if (t - key_time_ < options_.min_interval_ns) return false;
    const float dx = position[0] - subject_[0], dz = position[2] - subject_[2];
    const float dist = std::max(std::sqrt(dx * dx + dz * dz), 0.3f);
    const float d[3] = {position[0] - key_position_[0], position[1] - key_position_[1],
                        position[2] - key_position_[2]};
    const float moved = std::sqrt(dot3(d, d));
    const float turned = angle_degrees(rotation, key_rotation_);
    if (moved < std::max(options_.min_move_m, options_.move_fraction * dist) && turned < options_.min_turn_deg) {
      return false;
    }
  }
  ++counters_.candidates;
  if (angular_speed_ > options_.max_angular_dps || linear_speed_ > options_.max_linear_mps) {
    ++counters_.rejected_fast;
    return false;
  }
  return true;
}

bool KeyframePolicy::accept(double sharp) {
  std::vector<double> sorted(recent_sharpness_.begin(), recent_sharpness_.end());
  std::sort(sorted.begin(), sorted.end());
  const double median = sorted.empty() ? 0.0 : sorted[sorted.size() / 2];
  recent_sharpness_.push_back(sharp);
  if (recent_sharpness_.size() > options_.blur_window) recent_sharpness_.pop_front();
  if (recent_sharpness_.size() >= options_.blur_min_samples && sharp < options_.blur_ratio * median) {
    ++counters_.rejected_blur;
    return false;
  }
  ++keyframes_;
  has_key_ = true;
  std::copy(previous_position_, previous_position_ + 3, key_position_);
  std::copy(previous_rotation_, previous_rotation_ + 4, key_rotation_);
  key_time_ = previous_time_;
  const int bin = azimuth_bin(previous_position_, subject_);
  if (bin >= 0) coverage_[size_t(bin)] = true;
  return true;
}

}  // namespace gs
