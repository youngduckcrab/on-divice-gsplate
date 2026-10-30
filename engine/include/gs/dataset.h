#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gs/camera.h"
#include "gs/splat_scene.h"

namespace gs {

struct DatasetFrame {
  std::string name;
  float c2w[16];              // camera-to-world, OpenGL convention, row-major
  std::vector<uint8_t> rgba;  // training resolution; alpha = subject mask (255 without masks)
  uint32_t depth_width = 0;
  uint32_t depth_height = 0;
  std::vector<float> depth;  // metres, 0 = missing (low-res, ARCore-like)
  std::vector<uint8_t> depth_confidence;
  bool holdout = false;  // reserved for evaluation
};

// A capture: posed images (nerfstudio transforms.json layout) at training resolution.
struct Dataset {
  std::string name;
  uint32_t width = 0;  // training resolution
  uint32_t height = 0;
  float fx = 0, fy = 0, cx = 0, cy = 0;  // at training resolution
  bool has_masks = false;
  bool has_depth = false;
  std::vector<DatasetFrame> frames;
  std::vector<float> points;          // optional sparse point cloud, xyz
  std::vector<uint8_t> point_colors;  // rgb per point
  float camera_center[3] = {0, 0, 0};
  float camera_extent = 1.0f;  // max camera distance from camera_center (3DGS "scene extent")
  std::vector<std::string> skipped;  // frames left out because their files could not be read

  size_t train_count() const;
  size_t holdout_count() const;
};

// Loads `directory`/transforms.json. Images are resized to `target_width`
// (aspect kept); every 8th frame is held out. Frames with unreadable files are
// skipped (listed in `skipped`); throws std::runtime_error if too few remain.
Dataset load_dataset(const std::string& directory, uint32_t target_width);

// Pinhole camera (scene -> camera, OpenCV convention) for a frame.
Camera dataset_camera(const Dataset& dataset, const DatasetFrame& frame);

struct InitOptions {
  uint32_t max_gaussians = 150000;
  bool subject_only = true;      // keep only points inside the subject mask (when masks exist)
  float min_confidence = 0.3f;   // depth confidence threshold, 0..1
  float initial_opacity = 0.2f;
  uint32_t seed = 1;
};

struct InitStats {
  std::string source;  // "depth", "points" or "random"
  size_t raw_points = 0;
  float voxel_size = 0;
};

// Initial gaussians (SH degree 0) from the dataset's depth maps, its sparse
// points, or random points around the cameras' target, in that order of preference.
SplatScene initialize_gaussians(const Dataset& dataset, const InitOptions& options, InitStats* stats = nullptr);

}  // namespace gs
