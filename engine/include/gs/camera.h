#pragma once

#include <cstdint>

#include "gs/splat_scene.h"

namespace gs {

// Pinhole camera in scene coordinates, OpenCV convention (x right, y down, z forward).
struct Camera {
  float rotation[9];     // scene -> camera, row-major
  float translation[3];  // scene -> camera
  float position[3];     // camera centre in scene coordinates
  float fx, fy, cx, cy;
  uint32_t width, height;
};

// Orbit camera around a target. The orbit lives in a Y-up display space;
// `flip_y` rotates the scene 180 degrees about X so Y-down data (COLMAP / .ply)
// shows upright.
class OrbitCamera {
 public:
  void reset(const SceneBounds& bounds, bool flip_y);
  void orbit(float dx_px, float dy_px);
  void pan(float dx_px, float dy_px, uint32_t width, uint32_t height);
  void zoom(float factor);  // > 1 moves closer
  void set_flip_y(bool flip_y);
  bool flip_y() const { return flip_y_; }
  Camera camera(uint32_t width, uint32_t height) const;

  float yaw = 0.0f;    // radians
  float pitch = 0.0f;  // radians, positive looks down from above
  float distance = 3.0f;
  float target[3] = {0.0f, 0.0f, 0.0f};  // display space
  float fov_short_side_deg = 50.0f;

 private:
  void basis(float right[3], float down[3], float forward[3], float centre[3]) const;
  float focal(uint32_t width, uint32_t height) const;

  bool flip_y_ = false;
};

}  // namespace gs
