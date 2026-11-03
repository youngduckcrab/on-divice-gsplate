#include "gs/pose.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>

#include "gs/json_parse.h"

namespace gs {
namespace {

constexpr float kPi = 3.14159265358979f;

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
Vec3 mix(Vec3 a, Vec3 b, float t) { return a + (b - a) * t; }

// Row-major 3x3.
struct Mat3 {
  float m[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
};
Vec3 mul(const Mat3& a, Vec3 v) {
  return {a.m[0] * v.x + a.m[1] * v.y + a.m[2] * v.z, a.m[3] * v.x + a.m[4] * v.y + a.m[5] * v.z,
          a.m[6] * v.x + a.m[7] * v.y + a.m[8] * v.z};
}
Mat3 mul(const Mat3& a, const Mat3& b) {
  Mat3 r;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      r.m[i * 3 + j] = a.m[i * 3] * b.m[j] + a.m[i * 3 + 1] * b.m[3 + j] + a.m[i * 3 + 2] * b.m[6 + j];
    }
  }
  return r;
}
// Rotation by `angle` about unit `k` (Rodrigues).
Mat3 axis_angle(Vec3 k, float angle) {
  const float c = std::cos(angle), s = std::sin(angle), t = 1.0f - c;
  Mat3 r;
  r.m[0] = c + k.x * k.x * t;
  r.m[1] = k.x * k.y * t - k.z * s;
  r.m[2] = k.x * k.z * t + k.y * s;
  r.m[3] = k.y * k.x * t + k.z * s;
  r.m[4] = c + k.y * k.y * t;
  r.m[5] = k.y * k.z * t - k.x * s;
  r.m[6] = k.z * k.x * t - k.y * s;
  r.m[7] = k.z * k.y * t + k.x * s;
  r.m[8] = c + k.z * k.z * t;
  return r;
}
// The smallest rotation taking unit `a` to unit `b`; `spare` is the axis for a half turn.
Mat3 rotate_between(Vec3 a, Vec3 b, Vec3 spare) {
  const Vec3 axis = cross(a, b);
  const float s = length(axis), c = std::clamp(dot(a, b), -1.0f, 1.0f);
  if (s < 1e-6f) {
    if (c > 0.0f) return Mat3();
    Vec3 k = spare - a * dot(spare, a);
    if (length(k) < 1e-4f) k = cross(a, std::abs(a.x) < 0.9f ? Vec3(1, 0, 0) : Vec3(0, 1, 0));
    return axis_angle(normalize(k), kPi);
  }
  return axis_angle(axis * (1.0f / s), std::atan2(s, c));
}
// (w, x, y, z) of a rotation matrix.
std::array<float, 4> matrix_quat(const Mat3& r) {
  const float* m = r.m;
  std::array<float, 4> q{};
  const float trace = m[0] + m[4] + m[8];
  if (trace > 0.0f) {
    const float s = 0.5f / std::sqrt(trace + 1.0f);
    q = {0.25f / s, (m[7] - m[5]) * s, (m[2] - m[6]) * s, (m[3] - m[1]) * s};
  } else if (m[0] > m[4] && m[0] > m[8]) {
    const float s = 2.0f * std::sqrt(1.0f + m[0] - m[4] - m[8]);
    q = {(m[7] - m[5]) / s, 0.25f * s, (m[1] + m[3]) / s, (m[2] + m[6]) / s};
  } else if (m[4] > m[8]) {
    const float s = 2.0f * std::sqrt(1.0f + m[4] - m[0] - m[8]);
    q = {(m[2] - m[6]) / s, (m[1] + m[3]) / s, 0.25f * s, (m[5] + m[7]) / s};
  } else {
    const float s = 2.0f * std::sqrt(1.0f + m[8] - m[0] - m[4]);
    q = {(m[3] - m[1]) / s, (m[2] + m[6]) / s, (m[5] + m[7]) / s, 0.25f * s};
  }
  return q;
}
Mat3 quat_matrix(const float* q) {
  const float len = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  const float w = len > 0 ? q[0] / len : 1.0f, x = len > 0 ? q[1] / len : 0.0f;
  const float y = len > 0 ? q[2] / len : 0.0f, z = len > 0 ? q[3] / len : 0.0f;
  Mat3 r;
  r.m[0] = 1 - 2 * (y * y + z * z);
  r.m[1] = 2 * (x * y - w * z);
  r.m[2] = 2 * (x * z + w * y);
  r.m[3] = 2 * (x * y + w * z);
  r.m[4] = 1 - 2 * (x * x + z * z);
  r.m[5] = 2 * (y * z - w * x);
  r.m[6] = 2 * (x * z - w * y);
  r.m[7] = 2 * (y * z + w * x);
  r.m[8] = 1 - 2 * (x * x + y * y);
  return r;
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float logit(float p) { return std::log(p / (1.0f - p)); }

float percentile(std::vector<float> v, float q) {
  if (v.empty()) return 0.0f;
  const size_t k = std::min(v.size() - 1, size_t(q * float(v.size() - 1) + 0.5f));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}

// The person's left point for each point, and the other way round.
constexpr int kMirror[kPosePoints] = {0,  4,  5,  6,  1,  2,  3,  8,  7,  10, 9,  12, 11, 14, 13, 16, 15,
                                      18, 17, 20, 19, 22, 21, 24, 23, 26, 25, 28, 27, 30, 29, 32, 31};

struct Observation {
  const OrthoView* view;
  const PoseLandmarks* marks;
};

// Least-squares point for landmark `j` over the views (`swap[k]`: read view
// k's mirrored landmark), weighted by likelihood x `robust[k][j]`.
Vec3 solve_point(const std::vector<Observation>& obs, const std::vector<uint8_t>& swap, int j,
                 const std::vector<std::array<float, kPosePoints>>* robust, Vec3 prior, float* weight_sum) {
  double a[9] = {}, b[3] = {};
  double total = 0;
  for (size_t k = 0; k < obs.size(); ++k) {
    const OrthoView& v = *obs[k].view;
    const int src = swap[k] ? kMirror[j] : j;
    float w = obs[k].marks->likelihood[src];
    if (robust != nullptr) w *= (*robust)[k][j];
    if (w <= 1e-4f) continue;
    const Vec3 o(v.origin), r(v.right), d(v.down);
    const float u = (obs[k].marks->x[src] - 0.5f * float(v.width)) * v.pixel + dot(o, r);
    const float h = (obs[k].marks->y[src] - 0.5f * float(v.height)) * v.pixel + dot(o, d);
    for (const auto& [axis, value] : {std::pair<Vec3, float>{r, u}, std::pair<Vec3, float>{d, h}}) {
      const float e[3] = {axis.x, axis.y, axis.z};
      for (int p = 0; p < 3; ++p) {
        for (int q = 0; q < 3; ++q) a[p * 3 + q] += w * e[p] * e[q];
        b[p] += w * e[p] * value;
      }
    }
    total += w;
  }
  if (weight_sum != nullptr) *weight_sum = float(total);
  // A whisper of the prior keeps the solve defined when the views are parallel.
  const double lambda = 1e-4 * std::max(total, 1e-3);
  const float pr[3] = {prior.x, prior.y, prior.z};
  for (int p = 0; p < 3; ++p) {
    a[p * 3 + p] += lambda;
    b[p] += lambda * pr[p];
  }
  const double det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) +
                     a[2] * (a[3] * a[7] - a[4] * a[6]);
  if (std::abs(det) < 1e-18) return prior;
  const double inv = 1.0 / det;
  const double x = (b[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (b[1] * a[8] - a[5] * b[2]) +
                    a[2] * (b[1] * a[7] - a[4] * b[2])) * inv;
  const double y = (a[0] * (b[1] * a[8] - a[5] * b[2]) - b[0] * (a[3] * a[8] - a[5] * a[6]) +
                    a[2] * (a[3] * b[2] - b[1] * a[6])) * inv;
  const double z = (a[0] * (a[4] * b[2] - b[1] * a[7]) - a[1] * (a[3] * b[2] - b[1] * a[6]) +
                    b[0] * (a[3] * a[7] - a[4] * a[6])) * inv;
  return {float(x), float(y), float(z)};
}

// Squared distance of point `p` from view k's sight line of landmark j.
float residual2(const Observation& ob, bool swap, int j, Vec3 p) {
  const OrthoView& v = *ob.view;
  const int src = swap ? kMirror[j] : j;
  const Vec3 o(v.origin), r(v.right), d(v.down);
  const float u = (ob.marks->x[src] - 0.5f * float(v.width)) * v.pixel + dot(o, r);
  const float h = (ob.marks->y[src] - 0.5f * float(v.height)) * v.pixel + dot(o, d);
  const float du = dot(p, r) - u, dh = dot(p, d) - h;
  return du * du + dh * dh;
}

// Distance of p from the segment a-b; `t` is where along it (0..1) the nearest point is.
float segment_distance(Vec3 p, Vec3 a, Vec3 b, float* t, Vec3* nearest) {
  const Vec3 ab = b - a;
  const float len2 = std::max(dot(ab, ab), 1e-12f);
  const float s = std::clamp(dot(p - a, ab) / len2, 0.0f, 1.0f);
  const Vec3 c = a + ab * s;
  if (t != nullptr) *t = s;
  if (nearest != nullptr) *nearest = c;
  return length(p - c);
}

bool is_torso(int b) { return b == kPelvis || b == kChest; }

// How far p is from bone b, in units of the body's thickness there.
float bone_distance(const Skeleton& sk, int b, Vec3 p, float* raw = nullptr) {
  const Skeleton::BoneRest& bone = sk.bones[b];
  Vec3 c;
  // The head turns about the base of the neck but is measured from a third of
  // the way up: the shoulders right beside the neck's base are not the head's.
  const Vec3 start = b == kHead ? mix(Vec3(bone.pivot), Vec3(bone.end), 0.3f) : Vec3(bone.pivot);
  const float d = segment_distance(p, start, Vec3(bone.end), nullptr, &c);
  if (raw != nullptr) *raw = d;
  if (is_torso(b)) {
    const Vec3 off = p - c;
    const float lat = dot(off, Vec3(sk.left)) / bone.half_width;
    const float dep = dot(off, Vec3(sk.forward)) / bone.half_depth;
    // Past its ends the torso gives way quickly (to the head, the thighs).
    const float ver = dot(off, Vec3(sk.up)) / (0.5f * bone.half_depth);
    return std::sqrt(lat * lat + dep * dep + ver * ver);
  }
  return d / bone.radius;
}

}  // namespace

BodyViews render_body_views(const SplatScene& scene, const float up_in[3], int count, uint32_t size) {
  BodyViews bv;
  if (scene.count == 0 || count <= 0 || size == 0) return bv;
  const Vec3 up = normalize(Vec3(up_in));
  up.store(bv.up);
  const Vec3 any = std::abs(up.z) < 0.9f ? Vec3(0, 0, 1) : Vec3(1, 0, 0);
  const Vec3 h1 = normalize(any - up * dot(any, up));
  const Vec3 h2 = cross(up, h1);
  std::vector<float> hs, a1, a2;
  for (uint32_t i = 0; i < scene.count; ++i) {
    if (sigmoid(scene.opacity_logits[i]) < 0.3f) continue;
    const Vec3 p(&scene.positions[size_t(i) * 3]);
    hs.push_back(dot(p, up));
    a1.push_back(dot(p, h1));
    a2.push_back(dot(p, h2));
  }
  if (hs.size() < 100) return bv;
  const float low = percentile(hs, 0.005f), high = percentile(hs, 0.995f);
  const float c1 = percentile(a1, 0.5f), c2 = percentile(a2, 0.5f);
  std::vector<float> radial(a1.size());
  for (size_t i = 0; i < a1.size(); ++i) radial[i] = std::hypot(a1[i] - c1, a2[i] - c2);
  const float reach = percentile(radial, 0.99f);
  bv.height = high - low;
  if (bv.height <= 0.0f) return bv;
  const Vec3 centre = up * (0.5f * (low + high)) + h1 * c1 + h2 * c2;
  centre.store(bv.centre);
  const float side = 1.12f * std::max(bv.height, 2.0f * reach);
  const float background[3] = {0.82f, 0.82f, 0.82f};
  bv.views.resize(size_t(count));
  for (int k = 0; k < count; ++k) {
    OrthoView& v = bv.views[size_t(k)];
    const float az = 2.0f * kPi * float(k) / float(count);
    const Vec3 from = h1 * std::cos(az) + h2 * std::sin(az);
    const Vec3 forward = -from, down = -up;
    centre.store(v.origin);
    forward.store(v.forward);
    down.store(v.down);
    cross(down, forward).store(v.right);
    v.width = v.height = size;
    v.pixel = side / float(size);
  }
  const std::vector<uint32_t> all;
  const size_t workers = std::clamp<size_t>(std::thread::hardware_concurrency(), 1, bv.views.size());
  std::vector<std::thread> threads;
  for (size_t w = 0; w < workers; ++w) {
    threads.emplace_back([&, w] {
      for (size_t k = w; k < bv.views.size(); k += workers) render_ortho(scene, all, bv.views[k], background);
    });
  }
  for (std::thread& t : threads) t.join();
  bv.found = true;
  return bv;
}

std::vector<PoseLandmarks> parse_pose_landmarks(std::string_view json) {
  const JsonValue root = parse_json(json);
  std::vector<PoseLandmarks> out;
  if (root.type != JsonValue::Type::Array) return out;
  for (const JsonValue& item : root.array) {
    const JsonValue* points = item.find("points");
    if (points == nullptr || points->type != JsonValue::Type::Array || points->array.size() < kPosePoints) {
      continue;
    }
    PoseLandmarks m;
    m.view = int(item.number_or("view", -1));
    for (int j = 0; j < kPosePoints; ++j) {
      const JsonValue& p = points->array[size_t(j)];
      if (p.type != JsonValue::Type::Array || p.array.size() < 2) continue;
      m.x[j] = float(p.array[0].number);
      m.y[j] = float(p.array[1].number);
      m.likelihood[j] = p.array.size() > 2 ? std::clamp(float(p.array[2].number), 0.0f, 1.0f) : 1.0f;
    }
    out.push_back(m);
  }
  return out;
}

Skeleton fit_skeleton(const BodyViews& views, const std::vector<PoseLandmarks>& landmarks) {
  Skeleton sk;
  std::vector<Observation> obs;
  for (const PoseLandmarks& m : landmarks) {
    if (m.view < 0 || size_t(m.view) >= views.views.size()) continue;
    obs.push_back({&views.views[size_t(m.view)], &m});
  }
  sk.views_used = int(obs.size());
  if (obs.size() < 2 || views.height <= 0.0f) return sk;
  const size_t n = obs.size();
  const Vec3 prior(views.centre);
  const float scale = views.height / 1.7f;
  const float tau = 0.03f * scale;  // a point this far off its sight line is suspect

  auto solve_all = [&](const std::vector<uint8_t>& swap, const std::vector<std::array<float, kPosePoints>>* robust,
                       std::array<Vec3, kPosePoints>& pts, std::array<float, kPosePoints>* weights) {
    for (int j = 0; j < kPosePoints; ++j) {
      float w = 0;
      pts[size_t(j)] = solve_point(obs, swap, j, robust, prior, &w);
      if (weights != nullptr) (*weights)[size_t(j)] = w;
    }
  };
  auto score = [&](const std::vector<uint8_t>& swap, const std::array<Vec3, kPosePoints>& pts) {
    double s = 0;
    for (size_t k = 0; k < n; ++k) {
      for (int j = 0; j < kPosePoints; ++j) {
        const float w = obs[k].marks->likelihood[swap[k] ? kMirror[j] : j];
        s += w * std::min(residual2(obs[k], swap[k] != 0, j, pts[size_t(j)]), tau * tau);
      }
    }
    return s;
  };

  // A choice of swaps, solved robustly (views far off a point count less for
  // it: a detector's bad view must not drag the others).
  std::vector<std::array<float, kPosePoints>> robust(n);
  auto evaluate = [&](const std::vector<uint8_t>& swap, int iterations, std::array<Vec3, kPosePoints>& pts,
                      std::array<float, kPosePoints>* weights) {
    for (auto& r : robust) r.fill(1.0f);
    for (int it = 0; it < iterations; ++it) {
      solve_all(swap, &robust, pts, weights);
      for (size_t k = 0; k < n; ++k) {
        for (int j = 0; j < kPosePoints; ++j) {
          const float r = std::sqrt(residual2(obs[k], swap[k] != 0, j, pts[size_t(j)]));
          robust[k][size_t(j)] = r <= tau ? 1.0f : tau / r;
        }
      }
    }
    solve_all(swap, &robust, pts, weights);
    return score(swap, pts);
  };

  // Which views' left and right to swap: every choice when there are few
  // views (the first view as it is: the whole skeleton's left is settled
  // later), else one view at a time while it helps.
  std::vector<uint8_t> best(n, 0);
  std::array<Vec3, kPosePoints> pts;
  if (n <= 10) {
    double best_score = 1e30;
    std::vector<uint8_t> swap(n, 0);
    for (uint32_t mask = 0; mask < (1u << (n - 1)); ++mask) {
      for (size_t k = 1; k < n; ++k) swap[k] = uint8_t((mask >> (k - 1)) & 1u);
      const double s = evaluate(swap, 3, pts, nullptr);
      if (s < best_score) {
        best_score = s;
        best = swap;
      }
    }
  } else {
    double current = evaluate(best, 3, pts, nullptr);
    for (bool improved = true; improved;) {
      improved = false;
      for (size_t k = 1; k < n; ++k) {
        best[k] ^= 1u;
        const double s = evaluate(best, 3, pts, nullptr);
        if (s < current) {
          current = s;
          improved = true;
        } else {
          best[k] ^= 1u;
        }
      }
    }
  }

  std::array<float, kPosePoints> weights{};
  evaluate(best, 8, pts, &weights);
  // How far the agreeing views are from the points (what disagrees is left out).
  double err = 0, err_w = 0;
  for (size_t k = 0; k < n; ++k) {
    for (int j = 0; j < kPosePoints; ++j) {
      const float r2 = residual2(obs[k], best[k] != 0, j, pts[size_t(j)]);
      if (r2 > 9.0f * tau * tau) continue;
      const float w = obs[k].marks->likelihood[best[k] ? kMirror[j] : j];
      err += w * r2;
      err_w += w;
    }
  }
  sk.error = float(std::sqrt(err / std::max(err_w, 1e-9)));
  int flipped = 0;
  for (uint8_t s : best) flipped += s;
  for (int j = 0; j < kPosePoints; ++j) {
    pts[size_t(j)].store(sk.points[j]);
    sk.confidence[j] = weights[size_t(j)] / float(n);
  }

  // The body's frame. The nose is in front: the person's left is then
  // up x forward. If the labels say otherwise, every view was read mirrored.
  const Vec3 up(views.up);
  auto P = [&](int j) { return Vec3(sk.points[j]); };
  auto flat = [&](Vec3 v) { return v - up * dot(v, up); };
  const Vec3 neck = (P(11) + P(12)) * 0.5f;
  Vec3 front = flat(P(0) - neck);
  if (length(front) < 0.01f * scale) front = flat((P(9) + P(10)) * 0.5f - neck);
  const Vec3 across = flat(P(11) - P(12)) + flat(P(23) - P(24));
  if (dot(across, cross(up, front)) < 0.0f) {
    for (int j = 0; j < kPosePoints; ++j) {
      const int m = kMirror[j];
      if (m <= j) continue;
      for (int c = 0; c < 3; ++c) std::swap(sk.points[j][c], sk.points[m][c]);
      std::swap(sk.confidence[j], sk.confidence[m]);
    }
    flipped = int(n) - flipped;
  }
  sk.views_flipped = flipped;
  Vec3 left = normalize(flat(P(11) - P(12)) + flat(P(23) - P(24)));
  if (length(left) < 0.5f) left = normalize(cross(up, front));
  const Vec3 forward = normalize(cross(left, up));
  up.store(sk.up);
  left.store(sk.left);
  forward.store(sk.forward);
  sk.height = views.height;

  // Bones.
  const Vec3 pelvis = (P(23) + P(24)) * 0.5f;
  const Vec3 spine = mix(pelvis, neck, 0.5f);
  const Vec3 head = (P(7) + P(8)) * 0.5f;
  auto set = [&](int b, int parent, Vec3 pivot, Vec3 end, float radius) {
    Skeleton::BoneRest& bone = sk.bones[b];
    bone.parent = parent;
    pivot.store(bone.pivot);
    end.store(bone.end);
    bone.radius = radius * scale;
  };
  set(kPelvis, -1, pelvis, spine, 0.15f);
  set(kChest, kPelvis, spine, neck, 0.15f);
  set(kHead, kChest, neck, head + (head - neck) * 0.45f, 0.11f);
  for (int side = 0; side < 2; ++side) {
    const int o = side;  // 0: left (odd landmarks), 1: right
    const float sgn = side == 0 ? 1.0f : -1.0f;
    const int cl = side == 0 ? kLeftClavicle : kRightClavicle;
    const int th = side == 0 ? kLeftThigh : kRightThigh;
    const Vec3 shoulder = P(11 + o), elbow = P(13 + o), wrist = P(15 + o);
    const Vec3 knuckles = (P(17 + o) + P(19 + o)) * 0.5f;
    // The collarbone: from beside the base of the neck, a little low and forward, to the shoulder.
    const Vec3 inner = neck + left * (sgn * 0.15f * length(flat(shoulder - neck))) - up * (0.03f * scale) +
                       forward * (0.02f * scale);
    set(cl, kChest, inner, shoulder, 0.06f);
    set(cl + 1, cl, shoulder, elbow, 0.055f);
    set(cl + 2, cl + 1, elbow, wrist, 0.045f);
    set(cl + 3, cl + 2, wrist, wrist + (knuckles - wrist) * 1.6f, 0.045f);
    const Vec3 hip = P(23 + o), knee = P(25 + o), ankle = P(27 + o), toe = P(31 + o);
    set(th, kPelvis, hip, knee, 0.085f);
    set(th + 1, th, knee, ankle, 0.06f);
    set(th + 2, th + 1, ankle, toe, 0.05f);
  }
  const float shoulders = length(flat(P(11) - P(12)));
  const float hips = length(flat(P(23) - P(24)));
  sk.bones[kChest].half_width = std::max(0.5f * shoulders, 0.1f * scale);
  sk.bones[kChest].half_depth = 0.6f * sk.bones[kChest].half_width;
  sk.bones[kPelvis].half_width = std::max(0.8f * hips, 0.12f * scale);
  sk.bones[kPelvis].half_depth = 0.75f * sk.bones[kPelvis].half_width;
  sk.found = true;
  return sk;
}

// Joints blended along the limb: the limb bone, its parent, where along the
// limb (in its thicknesses from the joint) the blend starts and ends, and how
// far from the limb's line it holds fully and fades out.
const std::tuple<int, int, float, float, float, float> kJointBlends[] = {
    {kLeftUpperArm, kLeftClavicle, -1.0f, 1.2f, 1.3f, 2.2f},
    {kRightUpperArm, kRightClavicle, -1.0f, 1.2f, 1.3f, 2.2f},
    {kLeftThigh, kPelvis, -0.5f, 1.0f, 1.0f, 1.6f},
    {kRightThigh, kPelvis, -0.5f, 1.0f, 1.0f, 1.6f},
    {kLeftForearm, kLeftUpperArm, -0.9f, 0.9f, 1.3f, 2.0f},
    {kRightForearm, kRightUpperArm, -0.9f, 0.9f, 1.3f, 2.0f},
    {kLeftShin, kLeftThigh, -0.8f, 0.8f, 1.3f, 2.0f},
    {kRightShin, kRightThigh, -0.8f, 0.8f, 1.3f, 2.0f},
    {kLeftHand, kLeftForearm, -0.6f, 0.8f, 1.3f, 2.0f},
    {kRightHand, kRightForearm, -0.6f, 0.8f, 1.3f, 2.0f},
    {kLeftFoot, kLeftShin, -0.6f, 0.6f, 1.3f, 2.0f},
    {kRightFoot, kRightShin, -0.6f, 0.6f, 1.3f, 2.0f},
};

std::vector<uint32_t> Skin::packed() const {
  std::vector<uint32_t> out(size_t(count) * 2);
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t b = 0, w = 0;
    for (int k = 0; k < 4; ++k) {
      b |= uint32_t(bones[size_t(i) * 4 + k]) << (8 * k);
      w |= uint32_t(weights[size_t(i) * 4 + k]) << (8 * k);
    }
    out[size_t(i) * 2] = b;
    out[size_t(i) * 2 + 1] = w;
  }
  return out;
}

Skin skin_splats(const SplatScene& scene, const Skeleton& skeleton_in) {
  Skin skin;
  skin.count = scene.count;
  skin.bones.assign(size_t(scene.count) * 4, 0);
  skin.weights.assign(size_t(scene.count) * 4, 0);
  if (!skeleton_in.found) {
    for (uint32_t i = 0; i < scene.count; ++i) skin.weights[size_t(i) * 4] = 255;
    return skin;
  }
  Skeleton sk = skeleton_in;
  std::vector<std::vector<int>> children(kBoneCount);
  for (int b = 0; b < kBoneCount; ++b) {
    if (sk.bones[b].parent >= 0) children[size_t(sk.bones[b].parent)].push_back(b);
  }
  auto nearest = [&](Vec3 p, float* raw) {
    int best = 0;
    float best_d = 1e30f;
    for (int b = 0; b < kBoneCount; ++b) {
      float r = 0;
      const float d = bone_distance(sk, b, p, &r);
      if (d < best_d) {
        best_d = d;
        best = b;
        if (raw != nullptr) *raw = r;
      }
    }
    return best;
  };
  // The limbs' thickness from the splats nearest them.
  {
    std::vector<std::vector<float>> dist(kBoneCount);
    const uint32_t step = std::max<uint32_t>(1, scene.count / 60000);
    for (uint32_t i = 0; i < scene.count; i += step) {
      float raw = 0;
      const int b = nearest(Vec3(&scene.positions[size_t(i) * 3]), &raw);
      dist[size_t(b)].push_back(raw);
    }
    // Limbs only: the head's splats include the hair, however big it is.
    for (int b = 0; b < kBoneCount; ++b) {
      if (is_torso(b) || b == kHead || dist[size_t(b)].size() < 50) continue;
      const float r0 = sk.bones[b].radius;
      sk.bones[b].radius = std::clamp(percentile(dist[size_t(b)], 0.6f), 0.75f * r0, 1.5f * r0);
    }
  }
  constexpr float kBlend = 0.35f;  // how far (in thicknesses) the blend between bones reaches
  for (uint32_t i = 0; i < scene.count; ++i) {
    const Vec3 p(&scene.positions[size_t(i) * 3]);
    float d[kBoneCount];
    for (int b = 0; b < kBoneCount; ++b) d[b] = bone_distance(sk, b, p);
    int primary = 0;
    for (int b = 1; b < kBoneCount; ++b) {
      if (d[b] < d[primary]) primary = b;
    }
    // The bone and its neighbours in the hierarchy share the splat near joints.
    std::array<std::pair<float, int>, kBoneCount> cand;
    int nc = 0;
    auto add = [&](int b) {
      const float x = (d[b] - d[primary]) / kBlend;
      cand[size_t(nc++)] = {std::exp(-0.5f * x * x), b};
    };
    add(primary);
    if (sk.bones[primary].parent >= 0) add(sk.bones[primary].parent);
    for (int c : children[size_t(primary)]) add(c);
    float acc[kBoneCount] = {};
    float total = 0;
    for (int k = 0; k < nc; ++k) total += cand[size_t(k)].first;
    for (int k = 0; k < nc; ++k) acc[cand[size_t(k)].second] += cand[size_t(k)].first / total;

    // At the limbs' joints the limb takes over from its parent gradually
    // along it (over about two thicknesses either side at the shoulders and
    // hips, which turn far; one at elbows, knees, wrists and ankles), instead
    // of where the two are equally near (a crease). Away from the limb's
    // line (the side of the torso under the arm) this fades back to the
    // weights above.
    for (const auto& [limb, parent, from, to, near, far] : kJointBlends) {
      if (acc[limb] + acc[parent] < 0.3f) continue;  // the splat is not about this joint
      const Skeleton::BoneRest& bone = sk.bones[limb];
      const Vec3 joint(bone.pivot);
      const Vec3 axis = normalize(Vec3(bone.end) - joint);
      const float along = dot(p - joint, axis);
      const float across = length((p - joint) - axis * along) / bone.radius;
      const float t = along / bone.radius;
      if (across >= far || t < from - 1.0f || t > to + 1.0f) continue;
      const float x = std::clamp((t - from) / (to - from), 0.0f, 1.0f);
      const float w_limb = x * x * (3.0f - 2.0f * x);
      // Full strength near the limb's line and inside the band, fading out beyond either.
      float k = std::clamp((far - across) / (far - near), 0.0f, 1.0f);
      k *= std::clamp(1.0f - std::max(from - t, t - to), 0.0f, 1.0f);
      k = k * k * (3.0f - 2.0f * k);
      for (float& v : acc) v *= 1.0f - k;
      acc[limb] += k * w_limb;
      acc[parent] += k * (1.0f - w_limb);
      break;
    }

    std::array<std::pair<float, int>, kBoneCount> top;
    for (int b = 0; b < kBoneCount; ++b) top[size_t(b)] = {acc[b], b};
    std::partial_sort(top.begin(), top.begin() + 4, top.end(), [](auto a, auto b) { return a.first > b.first; });
    float kept = 0;
    for (int k = 0; k < 4; ++k) kept += top[size_t(k)].first;
    int sum = 0;
    for (int k = 0; k < 4; ++k) {
      const int w = kept > 0 ? int(std::lround(255.0f * top[size_t(k)].first / kept)) : (k == 0 ? 255 : 0);
      skin.bones[size_t(i) * 4 + k] = uint8_t(top[size_t(k)].second);
      skin.weights[size_t(i) * 4 + k] = uint8_t(std::clamp(w, 0, 255));
      sum += std::clamp(w, 0, 255);
    }
    skin.weights[size_t(i) * 4] = uint8_t(std::clamp(int(skin.weights[size_t(i) * 4]) + 255 - sum, 0, 255));
  }
  return skin;
}

SplatScene split_at_joints(const SplatScene& scene, const Skin& skin, const Skeleton& sk, uint32_t* added) {
  const float scale = std::max(sk.height, 0.1f) / 1.7f;
  const float longest = 0.006f * scale;  // longer than this (1 sigma) across a joint: split
  const float piece = 0.005f * scale;
  std::vector<uint8_t> pieces(scene.count, 1);
  uint32_t total = 0;
  for (uint32_t i = 0; i < scene.count; ++i) {
    if (i < skin.count && skin.weights[size_t(i) * 4] < 247) {
      const float* ls = &scene.log_scales[size_t(i) * 3];
      const float big = std::exp(std::max(ls[0], std::max(ls[1], ls[2])));
      if (big > longest) pieces[i] = uint8_t(std::clamp(int(std::ceil(big / piece)), 2, 6));
    }
    total += pieces[i];
  }
  if (added != nullptr) *added = total - scene.count;
  SplatScene out;
  out.resize(total, scene.sh_degree);
  out.antialiased = scene.antialiased;
  out.convention = scene.convention;
  const int coeffs = sh_rest_coeffs(scene.sh_degree);
  uint32_t o = 0;
  for (uint32_t i = 0; i < scene.count; ++i) {
    const int n = pieces[i];
    const float* ls = &scene.log_scales[size_t(i) * 3];
    const int major = ls[0] >= ls[1] ? (ls[0] >= ls[2] ? 0 : 2) : (ls[1] >= ls[2] ? 1 : 2);
    const float sigma = std::exp(ls[major]);
    const Mat3 R = quat_matrix(&scene.rotations[size_t(i) * 4]);
    const Vec3 axis(R.m[major], R.m[3 + major], R.m[6 + major]);
    const float opacity = sigmoid(scene.opacity_logits[i]);
    for (int k = 0; k < n; ++k, ++o) {
      // Pieces spread over +-1.5 sigma along the long axis, overlapping, thinning towards the ends.
      const float x = n == 1 ? 0.0f : sigma * 1.5f * ((2.0f * k + 1.0f) / float(n) - 1.0f);
      const Vec3 p = Vec3(&scene.positions[size_t(i) * 3]) + axis * x;
      p.store(&out.positions[size_t(o) * 3]);
      for (int c = 0; c < 3; ++c) out.log_scales[size_t(o) * 3 + c] = ls[c];
      if (n > 1) out.log_scales[size_t(o) * 3 + major] = std::log(1.8f * sigma / float(n));
      std::copy_n(&scene.rotations[size_t(i) * 4], 4, &out.rotations[size_t(o) * 4]);
      const float a = n == 1 ? opacity : opacity * std::exp(-0.5f * (x / sigma) * (x / sigma));
      out.opacity_logits[o] = n == 1 ? scene.opacity_logits[i] : logit(std::clamp(a, 1e-4f, 0.9999f));
      std::copy_n(&scene.sh_dc[size_t(i) * 3], 3, &out.sh_dc[size_t(o) * 3]);
      if (coeffs > 0) {
        std::copy_n(&scene.sh_rest[size_t(i) * coeffs * 3], coeffs * 3, &out.sh_rest[size_t(o) * coeffs * 3]);
      }
    }
  }
  return out;
}

BonePose::BonePose() {
  for (int b = 0; b < kBoneCount; ++b) {
    for (int k = 0; k < 9; ++k) rotation[b][k] = (k % 4 == 0) ? 1.0f : 0.0f;
    for (int k = 0; k < 3; ++k) translation[b][k] = 0.0f;
  }
}

namespace {
const char* const kPoses[] = {"rest", "attention", "hands_on_hips", "arms_up", "bow", "wave", "walk"};
}

bool known_pose(std::string_view name) {
  for (const char* p : kPoses) {
    if (name == p) return true;
  }
  return false;
}

bool pose_moves(std::string_view name) { return name == "wave" || name == "walk"; }

BonePose pose_skeleton(const Skeleton& sk, std::string_view name, float time) {
  BonePose pose;
  if (!sk.found || name == "rest" || !known_pose(name)) return pose;
  const Vec3 U(sk.up), F(sk.forward), L(sk.left);
  auto dir = [&](float l, float u, float f) { return normalize(L * l + U * u + F * f); };
  const float scale = sk.height / 1.7f;
  auto rest_dir = [&](int b) { return normalize(Vec3(sk.bones[b].end) - Vec3(sk.bones[b].pivot)); };
  auto length_of = [&](int b) { return length(Vec3(sk.bones[b].end) - Vec3(sk.bones[b].pivot)); };

  // Torso and legs: where each bone points (scene); bones without one turn with their parent.
  std::array<bool, kBoneCount> has{};
  std::array<Vec3, kBoneCount> target{};
  auto aim = [&](int b, Vec3 d) {
    has[size_t(b)] = true;
    target[size_t(b)] = d;
  };
  // Arms: where the bones point, or where the wrist goes (and which way the
  // elbow), and which way the palm faces. With `with_pelvis` the plan is in
  // the standing body's frame and turns with the pelvis (a bow).
  struct ArmPlan {
    bool active = false;
    Vec3 upper, fore, hand, palm;
    bool reach = false;
    Vec3 wrist, pole;  // wrist: offset from the pelvis's joint
    bool with_pelvis = false;
  };
  std::array<ArmPlan, 2> arm;  // [0] left, [1] right
  auto arms_down = [&](int side, float swing) {
    const float s = side == 0 ? 1.0f : -1.0f;
    ArmPlan& a = arm[size_t(side)];
    a.active = true;
    a.upper = dir(s * 0.17f, -std::cos(swing), std::sin(swing));
    a.fore = dir(s * 0.1f, -std::cos(swing + 0.3f), std::sin(swing + 0.3f));
    a.hand = dir(s * 0.05f, -std::cos(swing + 0.3f), std::sin(swing + 0.3f));
    a.palm = dir(-s, 0.0f, 0.15f);  // towards the thigh
  };
  const Skeleton::BoneRest& pel = sk.bones[kPelvis];
  if (name == "attention") {
    arms_down(0, 0.0f);
    arms_down(1, 0.0f);
  } else if (name == "hands_on_hips") {
    for (int side = 0; side < 2; ++side) {
      const float s = side == 0 ? 1.0f : -1.0f;
      ArmPlan& a = arm[size_t(side)];
      a.active = a.reach = a.with_pelvis = true;
      // The palm on the side of the waist, fingers forward and down; the elbow out and a little back.
      a.wrist = L * (s * (pel.half_width + 0.035f * scale)) + U * (0.07f * scale) - F * (0.01f * scale);
      a.pole = dir(s * 0.85f, 0.0f, -0.5f);
      a.hand = dir(-s * 0.12f, -0.5f, 0.85f);
      a.palm = dir(-s, 0.0f, 0.0f);
    }
  } else if (name == "arms_up") {
    for (int side = 0; side < 2; ++side) {
      const float s = side == 0 ? 1.0f : -1.0f;
      ArmPlan& a = arm[size_t(side)];
      a.active = true;
      a.upper = dir(s * 0.28f, 0.96f, 0.04f);
      a.fore = dir(s * 0.22f, 0.97f, 0.05f);
      a.hand = dir(s * 0.2f, 0.97f, 0.08f);
      a.palm = dir(-s * 0.45f, 0.0f, 0.9f);  // the palms to the front, a little towards each other
    }
  } else if (name == "bow") {
    // A polite bow: the back straight, bending at the hips, the head in line,
    // the hands together in front; the legs stay as they stand.
    aim(kPelvis, dir(0.0f, std::cos(0.55f), std::sin(0.55f)));
    aim(kChest, dir(0.0f, std::cos(0.6f), std::sin(0.6f)));
    aim(kHead, dir(0.0f, std::cos(0.72f), std::sin(0.72f)));
    for (int b : {kLeftThigh, kLeftShin, kRightThigh, kRightShin}) aim(b, rest_dir(b));
    for (int side = 0; side < 2; ++side) {
      const float s = side == 0 ? 1.0f : -1.0f;
      ArmPlan& a = arm[size_t(side)];
      a.active = a.reach = a.with_pelvis = true;
      a.wrist = F * (pel.half_depth + 0.06f * scale + 0.015f * s * scale) - U * (0.12f * scale) +
                L * (s * 0.03f * scale);
      a.pole = dir(s * 0.7f, 0.0f, -0.4f);
      a.hand = dir(-s * 0.7f, -0.6f, 0.15f);
      a.palm = dir(0.0f, 0.0f, -1.0f);  // towards the body
    }
  } else if (name == "wave") {
    arms_down(0, 0.0f);
    const float sway = std::sin(2.0f * kPi * 1.3f * time);
    ArmPlan& a = arm[1];
    a.active = true;
    a.upper = dir(-0.88f, 0.32f, 0.22f);
    a.fore = dir(-(0.12f + 0.42f * sway), 1.0f, 0.12f);
    a.hand = dir(-(0.12f + 0.5f * sway), 1.0f, 0.1f);
    a.palm = dir(0.0f, 0.0f, 1.0f);  // to the front
  } else if (name == "walk") {
    const float phase = 2.0f * kPi * time / 1.1f;
    for (int side = 0; side < 2; ++side) {
      const float s = side == 0 ? 1.0f : -1.0f;
      const int th = side == 0 ? kLeftThigh : kRightThigh;
      const float ph = phase + (side == 0 ? 0.0f : kPi);
      const float swing = 0.38f * std::sin(ph);                                // thigh forward (+) and back
      const float bend = 0.7f * std::pow(std::max(0.0f, std::cos(ph)), 1.5f);  // knee, while it comes forward
      aim(th, dir(s * 0.02f, -std::cos(swing), std::sin(swing)));
      aim(th + 1, dir(s * 0.02f, -std::cos(swing - bend), std::sin(swing - bend)));
      arms_down(side, -0.3f * std::sin(ph));
    }
  }

  // Two bones from `shoulder` reaching `wrist`, the elbow towards `pole`.
  auto reach = [&](Vec3 shoulder, Vec3 wrist, float a, float c, Vec3 pole, Vec3* upper, Vec3* fore) {
    const Vec3 to = wrist - shoulder;
    const float d = std::clamp(length(to), std::abs(a - c) + 1e-3f, a + c - 1e-3f);
    const Vec3 x = normalize(to);
    Vec3 y = pole - x * dot(pole, x);
    y = length(y) > 1e-4f ? normalize(y) : normalize(cross(x, F));
    const float cos_a = std::clamp((a * a + d * d - c * c) / (2.0f * a * d), -1.0f, 1.0f);
    const Vec3 elbow = shoulder + (x * cos_a + y * std::sqrt(1.0f - cos_a * cos_a)) * a;
    *upper = normalize(elbow - shoulder);
    *fore = normalize(shoulder + x * d - elbow);
  };

  std::array<Mat3, kBoneCount> G;
  std::array<Vec3, kBoneCount> pivot;
  std::array<Vec3, 2> fore_target;
  std::array<float, 2> fore_twist = {0.0f, 0.0f};
  for (int b = 0; b < kBoneCount; ++b) {
    const Skeleton::BoneRest& bone = sk.bones[b];
    const int p = bone.parent;
    const Mat3 Gp = p >= 0 ? G[size_t(p)] : Mat3();
    pivot[size_t(b)] = p >= 0 ? mul(Gp, Vec3(bone.pivot) - Vec3(sk.bones[p].pivot)) + pivot[size_t(p)]
                              : Vec3(bone.pivot);
    G[size_t(b)] = Gp;
    const Vec3 now = normalize(mul(Gp, rest_dir(b)));
    if (has[size_t(b)]) G[size_t(b)] = mul(rotate_between(now, target[size_t(b)], F), Gp);

    const int side = (b >= kLeftClavicle && b <= kLeftHand) ? 0 : (b >= kRightClavicle && b <= kRightHand) ? 1 : -1;
    if (side < 0 || !arm[size_t(side)].active) continue;
    const ArmPlan& plan = arm[size_t(side)];
    const float s = side == 0 ? 1.0f : -1.0f;
    const int cl = side == 0 ? kLeftClavicle : kRightClavicle;
    const int ua = cl + 1, fa = cl + 2, hd = cl + 3;
    const Mat3& frame = plan.with_pelvis ? G[kPelvis] : Mat3();
    auto in_frame = [&](Vec3 v) { return plan.with_pelvis ? normalize(mul(frame, v)) : v; };
    auto planned = [&](Vec3 shoulder, Vec3* upper, Vec3* fore) {
      if (plan.reach) {
        const Vec3 wrist = pivot[kPelvis] + mul(frame, plan.wrist);
        reach(shoulder, wrist, length_of(ua), length_of(fa), in_frame(plan.pole), upper, fore);
      } else {
        *upper = in_frame(plan.upper);
        *fore = in_frame(plan.fore);
      }
    };
    if (b == cl) {
      // The shoulder girdle rises a little with the arm (and drops a little with it).
      Vec3 upper, fore;
      const Vec3 shoulder = mul(Gp, Vec3(sk.bones[ua].pivot) - Vec3(bone.pivot)) + pivot[size_t(b)];
      planned(shoulder, &upper, &fore);
      const Vec3 up_now = normalize(mul(Gp, U));
      const float lift = std::asin(std::clamp(dot(upper, up_now), -1.0f, 1.0f)) -
                         std::asin(std::clamp(dot(rest_dir(ua), U), -1.0f, 1.0f));
      const float turn = std::clamp(lift * (lift > 0.0f ? 0.22f : 0.1f), -0.35f, 0.35f);
      const Vec3 axis = cross(now, up_now);
      if (length(axis) > 1e-4f) G[size_t(b)] = mul(axis_angle(normalize(axis), turn), Gp);
    } else if (b == ua) {
      Vec3 upper, fore;
      planned(pivot[size_t(b)], &upper, &fore);
      fore_target[size_t(side)] = fore;
      Mat3 g = mul(rotate_between(now, upper, F), Gp);
      // Turn the arm about itself so the elbow bends the way the forearm goes
      // (its crease faces forward in the standing pose).
      const float bend = std::acos(std::clamp(dot(upper, fore), -1.0f, 1.0f));
      const float k = std::clamp((bend - 0.15f) / 0.35f, 0.0f, 1.0f);
      if (k > 0.0f) {
        const Vec3 u0 = rest_dir(b);
        const Vec3 crease = mul(g, normalize(F - u0 * dot(F, u0)));
        const Vec3 want = normalize(fore - upper * dot(fore, upper));
        const float angle = std::atan2(dot(cross(crease, want), upper), dot(crease, want));
        // Half of it: a captured arm's light and shade turn with it, and a full
        // turn shows; the elbow bending a little across itself does not.
        g = mul(axis_angle(upper, angle * k * 0.5f), g);
      }
      // Then turn the palm the way it should face, as an arm does: the
      // forearm turns (as a wrist does), and while the elbow is straight the
      // upper arm takes half of it (the shoulder turns too).
      const float share = 0.5f * (1.0f - k);
      const Vec3 f0 = rest_dir(fa), h0 = rest_dir(hd);
      Vec3 palm0 = U * -1.0f - L * (0.3f * s);  // palms down and in: how people stand to be captured
      palm0 = palm0 - h0 * dot(palm0, h0);
      palm0 = length(palm0) > 1e-3f ? normalize(palm0) : U * -1.0f;
      const Vec3 hand_want = in_frame(plan.hand), palm_want = in_frame(plan.palm);
      float best = -1e9f, best_turn = 0.0f;
      for (int step = -36; step <= 36; ++step) {
        const float turn = float(step) * (5.0f * kPi / 180.0f);
        const Mat3 gu = mul(axis_angle(upper, share * turn), g);
        const Mat3 gf0 = mul(rotate_between(normalize(mul(gu, f0)), fore, F), gu);
        const Mat3 gf = mul(axis_angle(fore, (1.0f - share) * turn), gf0);
        const Mat3 gh = mul(rotate_between(normalize(mul(gf, h0)), hand_want, F), gf);
        const float score = dot(mul(gh, palm0), palm_want) - 0.1f * std::abs(turn) / kPi;
        if (score > best) {
          best = score;
          best_turn = turn;
        }
      }
      G[size_t(b)] = mul(axis_angle(upper, share * best_turn), g);
      fore_twist[size_t(side)] = (1.0f - share) * best_turn;
    } else if (b == fa) {
      const Vec3 fore = fore_target[size_t(side)];
      G[size_t(b)] = mul(axis_angle(fore, fore_twist[size_t(side)]), mul(rotate_between(now, fore, F), Gp));
    } else if (b == hd) {
      G[size_t(b)] = mul(rotate_between(now, in_frame(plan.hand), F), Gp);
    }
  }
  // Keep the feet on the floor: the lowest of the ankles, heels and toes
  // stays where it was.
  float rest_low = 1e30f, posed_low = 1e30f;
  for (int foot : {kLeftFoot, kRightFoot}) {
    const Skeleton::BoneRest& bone = sk.bones[foot];
    const int heel = foot == kLeftFoot ? 29 : 30;
    for (Vec3 q : {Vec3(bone.pivot), Vec3(bone.end), Vec3(sk.points[heel])}) {
      const Vec3 moved = mul(G[size_t(foot)], q - Vec3(bone.pivot)) + pivot[size_t(foot)];
      rest_low = std::min(rest_low, dot(q, U));
      posed_low = std::min(posed_low, dot(moved, U));
    }
  }
  const Vec3 lift = U * (rest_low - posed_low);
  for (int b = 0; b < kBoneCount; ++b) {
    const Vec3 t = pivot[size_t(b)] - mul(G[size_t(b)], Vec3(sk.bones[b].pivot)) + lift;
    std::copy(G[size_t(b)].m, G[size_t(b)].m + 9, pose.rotation[b]);
    t.store(pose.translation[b]);
  }
  return pose;
}

namespace {

// A bone's motion as a unit dual quaternion: rotation r (w, x, y, z), and
// d = t r / 2 for its translation t.
struct DualQuat {
  std::array<float, 4> r{1, 0, 0, 0};
  std::array<float, 4> d{0, 0, 0, 0};
};
std::array<float, 4> quat_mul(const std::array<float, 4>& a, const std::array<float, 4>& b) {
  return {a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
          a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]};
}
DualQuat bone_dual_quat(const BonePose& pose, int b) {
  Mat3 r;
  std::copy(pose.rotation[b], pose.rotation[b] + 9, r.m);
  DualQuat q;
  q.r = matrix_quat(r);
  const float n = std::sqrt(q.r[0] * q.r[0] + q.r[1] * q.r[1] + q.r[2] * q.r[2] + q.r[3] * q.r[3]);
  for (float& c : q.r) c /= n;
  const std::array<float, 4> t = {0.0f, pose.translation[b][0], pose.translation[b][1], pose.translation[b][2]};
  q.d = quat_mul(t, q.r);
  for (float& c : q.d) c *= 0.5f;
  return q;
}

}  // namespace

SplatSkinning skinning_for(const BonePose& pose, uint32_t count) {
  SplatSkinning s;
  s.count = count;
  s.bones = kBoneCount;
  for (int b = 0; b < kBoneCount; ++b) {
    const DualQuat q = bone_dual_quat(pose, b);
    for (int c = 0; c < 4; ++c) {
      s.dq[b * 2][c] = q.r[size_t(c)];
      s.dq[b * 2 + 1][c] = q.d[size_t(c)];
    }
  }
  return s;
}

SplatScene apply_pose(const SplatScene& scene, const Skin& skin, const BonePose& pose) {
  SplatScene out;
  const int degree = std::min(scene.sh_degree, 1);
  out.resize(scene.count, degree);
  out.antialiased = scene.antialiased;
  out.convention = scene.convention;
  out.log_scales = scene.log_scales;
  out.opacity_logits = scene.opacity_logits;
  out.sh_dc = scene.sh_dc;
  std::array<DualQuat, kBoneCount> bone_q;
  for (int b = 0; b < kBoneCount; ++b) bone_q[size_t(b)] = bone_dual_quat(pose, b);
  const int in_coeffs = sh_rest_coeffs(scene.sh_degree);
  for (uint32_t i = 0; i < scene.count; ++i) {
    const Vec3 p(&scene.positions[size_t(i) * 3]);
    std::array<float, 4> r = {1, 0, 0, 0}, d = {0, 0, 0, 0};
    if (i < skin.count) {
      r = {0, 0, 0, 0};
      const std::array<float, 4>* first = nullptr;
      for (int k = 0; k < 4; ++k) {
        const float w = skin.weights[size_t(i) * 4 + k] / 255.0f;
        if (w <= 0.0f) continue;
        const DualQuat& q = bone_q[skin.bones[size_t(i) * 4 + k]];
        if (first == nullptr) first = &q.r;
        const float sign = (q.r[0] * (*first)[0] + q.r[1] * (*first)[1] + q.r[2] * (*first)[2] +
                            q.r[3] * (*first)[3]) < 0.0f ? -w : w;
        for (int c = 0; c < 4; ++c) {
          r[size_t(c)] += sign * q.r[size_t(c)];
          d[size_t(c)] += sign * q.d[size_t(c)];
        }
      }
      const float n = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]);
      if (n < 1e-8f) {
        r = {1, 0, 0, 0};
        d = {0, 0, 0, 0};
      } else {
        for (int c = 0; c < 4; ++c) {
          r[size_t(c)] /= n;
          d[size_t(c)] /= n;
        }
      }
    }
    const Mat3 R = quat_matrix(r.data());
    // Translation: 2 (d r*).
    const std::array<float, 4> rc = {r[0], -r[1], -r[2], -r[3]};
    const std::array<float, 4> t = quat_mul(d, rc);
    const Vec3 moved = mul(R, p) + Vec3(2.0f * t[1], 2.0f * t[2], 2.0f * t[3]);
    moved.store(&out.positions[size_t(i) * 3]);
    // The splat's own rotation, turned by the blend.
    const float* q = &scene.rotations[size_t(i) * 4];
    const std::array<float, 4> turned = quat_mul(r, {q[0], q[1], q[2], q[3]});
    std::copy(turned.begin(), turned.end(), &out.rotations[size_t(i) * 4]);
    if (degree >= 1) {
      // Band 1 is c . (-y, z, -x): as a vector v = (-c2, -c0, c1), turned with the splat.
      for (int c = 0; c < 3; ++c) {
        const float* in = &scene.sh_rest[(size_t(i) * in_coeffs) * 3];
        const Vec3 v(-in[2 * 3 + c], -in[0 * 3 + c], in[1 * 3 + c]);
        const Vec3 tv = mul(R, v);
        float* dst = &out.sh_rest[(size_t(i) * 3) * 3];
        dst[0 * 3 + c] = -tv.y;
        dst[1 * 3 + c] = tv.z;
        dst[2 * 3 + c] = -tv.x;
      }
    }
  }
  return out;
}

}  // namespace gs
