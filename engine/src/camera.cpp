#include "gs/camera.h"

#include <algorithm>
#include <cmath>

namespace gs {
namespace {

constexpr float kPi = 3.14159265358979f;
constexpr float kMaxPitch = 89.0f * kPi / 180.0f;

void normalize(float v[3]) {
  float len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (len > 0.0f) {
    for (int i = 0; i < 3; ++i) v[i] /= len;
  }
}

void cross(const float a[3], const float b[3], float out[3]) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

}  // namespace

void OrbitCamera::reset(const SceneBounds& bounds, bool flip_y) {
  flip_y_ = flip_y;
  const float s = flip_y ? -1.0f : 1.0f;
  target[0] = bounds.center[0];
  target[1] = s * bounds.center[1];
  target[2] = s * bounds.center[2];
  distance = 2.2f * bounds.radius;
  yaw = 0.0f;
  pitch = 0.25f;
}

void OrbitCamera::orbit(float dx_px, float dy_px) {
  constexpr float kRadiansPerPixel = 0.006f;
  yaw -= dx_px * kRadiansPerPixel;
  pitch = std::clamp(pitch + dy_px * kRadiansPerPixel, -kMaxPitch, kMaxPitch);
}

void OrbitCamera::pan(float dx_px, float dy_px, uint32_t width, uint32_t height) {
  float right[3], down[3], forward[3], centre[3];
  basis(right, down, forward, centre);
  // Move the target so the scene follows the finger at the target's depth.
  const float units_per_px = distance / focal(width, height);
  for (int i = 0; i < 3; ++i) target[i] -= (right[i] * dx_px + down[i] * dy_px) * units_per_px;
}

void OrbitCamera::zoom(float factor) {
  if (factor > 0.0f) distance = std::clamp(distance / factor, 1e-3f, 1e4f);
}

void OrbitCamera::set_flip_y(bool flip_y) {
  if (flip_y == flip_y_) return;
  // Keep looking at the same scene point: the flip maps (x, y, z) -> (x, -y, -z).
  target[1] = -target[1];
  target[2] = -target[2];
  flip_y_ = flip_y;
}

void OrbitCamera::basis(float right[3], float down[3], float forward[3], float centre[3]) const {
  const float cp = std::cos(pitch), sp = std::sin(pitch);
  const float cy = std::cos(yaw), sy = std::sin(yaw);
  centre[0] = target[0] + distance * cp * sy;
  centre[1] = target[1] + distance * sp;
  centre[2] = target[2] + distance * cp * cy;
  for (int i = 0; i < 3; ++i) forward[i] = target[i] - centre[i];
  normalize(forward);
  const float up[3] = {0.0f, 1.0f, 0.0f};
  cross(forward, up, right);
  normalize(right);
  cross(forward, right, down);  // x right, y down, z forward is right-handed
}

float OrbitCamera::focal(uint32_t width, uint32_t height) const {
  const float short_side = static_cast<float>(std::min(width, height));
  return 0.5f * short_side / std::tan(0.5f * fov_short_side_deg * kPi / 180.0f);
}

Camera OrbitCamera::camera(uint32_t width, uint32_t height) const {
  float right[3], down[3], forward[3], centre[3];
  basis(right, down, forward, centre);
  const float* rows[3] = {right, down, forward};
  const float s = flip_y_ ? -1.0f : 1.0f;

  Camera cam{};
  for (int r = 0; r < 3; ++r) {
    // Scene -> camera = (display -> camera) * (scene -> display), where
    // scene -> display is diag(1, s, s).
    cam.rotation[r * 3 + 0] = rows[r][0];
    cam.rotation[r * 3 + 1] = rows[r][1] * s;
    cam.rotation[r * 3 + 2] = rows[r][2] * s;
    cam.translation[r] = -(rows[r][0] * centre[0] + rows[r][1] * centre[1] + rows[r][2] * centre[2]);
  }
  cam.position[0] = centre[0];
  cam.position[1] = s * centre[1];
  cam.position[2] = s * centre[2];
  cam.fx = cam.fy = focal(width, height);
  cam.cx = 0.5f * static_cast<float>(width);
  cam.cy = 0.5f * static_cast<float>(height);
  cam.width = width;
  cam.height = height;
  return cam;
}

}  // namespace gs
