#include "gs/dataset.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "gs/json_parse.h"
#include "image_io.h"

namespace gs {
namespace {

constexpr float kShC0 = 0.28209479177387814f;

std::string join(const std::string& dir, const std::string& rel) {
  if (rel.empty() || rel[0] == '/') return rel;
  return dir + "/" + rel;
}

// c2w (OpenGL) applied to an OpenCV camera-space point: OpenCV (x, y, z) is OpenGL (x, -y, -z).
void camera_to_world(const float c2w[16], float x, float y, float z, float out[3]) {
  for (int r = 0; r < 3; ++r) out[r] = c2w[r * 4 + 0] * x - c2w[r * 4 + 1] * y - c2w[r * 4 + 2] * z + c2w[r * 4 + 3];
}

template <typename F>
void parallel_for(size_t n, F&& body) {
  const unsigned threads = std::clamp(std::thread::hardware_concurrency(), 1u, 4u);
  std::atomic<size_t> next{0};
  std::exception_ptr error;
  std::mutex error_mutex;
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < threads; ++t) {
    pool.emplace_back([&] {
      for (size_t i = next++; i < n; i = next++) {
        try {
          body(i);
        } catch (...) {
          std::lock_guard<std::mutex> lock(error_mutex);
          if (!error) error = std::current_exception();
        }
      }
    });
  }
  for (std::thread& t : pool) t.join();
  if (error) std::rethrow_exception(error);
}

uint64_t voxel_key(int64_t x, int64_t y, int64_t z) {
  return (uint64_t(x & 0x1fffff) << 42) | (uint64_t(y & 0x1fffff) << 21) | uint64_t(z & 0x1fffff);
}

}  // namespace

size_t Dataset::train_count() const {
  return std::count_if(frames.begin(), frames.end(), [](const DatasetFrame& f) { return !f.holdout; });
}

size_t Dataset::holdout_count() const { return frames.size() - train_count(); }

Dataset load_dataset(const std::string& directory, uint32_t target_width) {
  const std::vector<uint8_t> text = read_file(join(directory, "transforms.json"));
  const JsonValue root = parse_json(std::string_view(reinterpret_cast<const char*>(text.data()), text.size()));
  const JsonValue* frames = root.find("frames");
  if (frames == nullptr || frames->type != JsonValue::Type::Array || frames->array.empty()) {
    throw std::runtime_error("transforms.json에 frames가 없음");
  }
  const double src_w = root.number_or("w", 0), src_h = root.number_or("h", 0);
  const double fl_x = root.number_or("fl_x", 0), fl_y = root.number_or("fl_y", fl_x);
  if (src_w <= 0 || src_h <= 0 || fl_x <= 0) throw std::runtime_error("transforms.json에 w/h/fl_x가 없음");
  const double depth_scale = root.number_or("depth_unit_scale_factor", 0.001);

  Dataset ds;
  const size_t slash = directory.find_last_of('/');
  ds.name = slash == std::string::npos ? directory : directory.substr(slash + 1);
  ds.width = std::min<uint32_t>(target_width, static_cast<uint32_t>(src_w));
  ds.height = static_cast<uint32_t>(std::lround(src_h * ds.width / src_w));
  const double s = double(ds.width) / src_w;
  ds.fx = static_cast<float>(fl_x * s);
  ds.fy = static_cast<float>(fl_y * s);
  ds.cx = static_cast<float>(root.number_or("cx", src_w / 2) * s);
  ds.cy = static_cast<float>(root.number_or("cy", src_h / 2) * s);

  ds.frames.resize(frames->array.size());
  std::vector<std::string> mask_paths(ds.frames.size()), depth_paths(ds.frames.size()), conf_paths(ds.frames.size());
  std::vector<std::string> image_paths(ds.frames.size());
  for (size_t i = 0; i < ds.frames.size(); ++i) {
    const JsonValue& f = frames->array[i];
    DatasetFrame& frame = ds.frames[i];
    image_paths[i] = f.string_or("file_path", "");
    if (image_paths[i].empty()) throw std::runtime_error("frame에 file_path가 없음");
    frame.name = image_paths[i];
    const JsonValue* m = f.find("transform_matrix");
    if (m == nullptr || m->array.size() != 4) throw std::runtime_error("frame에 transform_matrix가 없음");
    for (int r = 0; r < 4; ++r) {
      if (m->array[r].array.size() != 4) throw std::runtime_error("transform_matrix가 4x4가 아님");
      for (int c = 0; c < 4; ++c) frame.c2w[r * 4 + c] = static_cast<float>(m->array[r].array[c].number);
    }
    mask_paths[i] = f.string_or("mask_path", "");
    depth_paths[i] = f.string_or("depth_file_path", "");
    conf_paths[i] = f.string_or("depth_confidence_path", "");
    frame.holdout = (i % 8) == 0;
  }
  ds.has_masks = std::all_of(mask_paths.begin(), mask_paths.end(), [](const std::string& p) { return !p.empty(); });
  ds.has_depth = std::all_of(depth_paths.begin(), depth_paths.end(), [](const std::string& p) { return !p.empty(); });

  parallel_for(ds.frames.size(), [&](size_t i) {
    DatasetFrame& frame = ds.frames[i];
    Image8 image = decode_image(read_file(join(directory, image_paths[i])), 4);
    if (image.width != uint32_t(src_w) || image.height != uint32_t(src_h)) {
      throw std::runtime_error(image_paths[i] + " 크기가 transforms.json과 다름");
    }
    frame.rgba = resize_area(image.data.data(), image.width, image.height, 4, ds.width, ds.height);
    if (ds.has_masks) {
      Image8 mask = decode_image(read_file(join(directory, mask_paths[i])), 1);
      std::vector<uint8_t> m = resize_area(mask.data.data(), mask.width, mask.height, 1, ds.width, ds.height);
      for (size_t p = 0; p < m.size(); ++p) frame.rgba[p * 4 + 3] = m[p];
    } else {
      for (size_t p = 0; p < size_t(ds.width) * ds.height; ++p) frame.rgba[p * 4 + 3] = 255;
    }
    if (ds.has_depth) {
      Image16 depth = decode_image16(read_file(join(directory, depth_paths[i])));
      frame.depth_width = depth.width;
      frame.depth_height = depth.height;
      frame.depth.resize(depth.data.size());
      for (size_t p = 0; p < depth.data.size(); ++p) frame.depth[p] = float(depth.data[p] * depth_scale);
      if (!conf_paths[i].empty()) {
        Image8 conf = decode_image(read_file(join(directory, conf_paths[i])), 1);
        if (conf.width == depth.width && conf.height == depth.height) frame.depth_confidence = std::move(conf.data);
      }
    }
  });

  const std::string ply = root.string_or("ply_file_path", "");
  if (!ply.empty()) {
    const std::vector<uint8_t> bytes = read_file(join(directory, ply));
    const PlyVertices v(bytes.data(), bytes.size());
    const int x = v.find("x"), y = v.find("y"), z = v.find("z");
    const int r = v.find("red"), g = v.find("green"), b = v.find("blue");
    if (x < 0 || y < 0 || z < 0) throw std::runtime_error("점 구름 PLY에 x/y/z가 없음");
    ds.points.resize(v.count() * 3);
    ds.point_colors.assign(v.count() * 3, 128);
    for (uint64_t i = 0; i < v.count(); ++i) {
      ds.points[i * 3 + 0] = v.value(i, x);
      ds.points[i * 3 + 1] = v.value(i, y);
      ds.points[i * 3 + 2] = v.value(i, z);
      if (r >= 0 && g >= 0 && b >= 0) {
        ds.point_colors[i * 3 + 0] = static_cast<uint8_t>(v.value(i, r));
        ds.point_colors[i * 3 + 1] = static_cast<uint8_t>(v.value(i, g));
        ds.point_colors[i * 3 + 2] = static_cast<uint8_t>(v.value(i, b));
      }
    }
  }

  double sum[3] = {0, 0, 0};
  for (const DatasetFrame& f : ds.frames) {
    for (int k = 0; k < 3; ++k) sum[k] += f.c2w[k * 4 + 3];
  }
  for (int k = 0; k < 3; ++k) ds.camera_center[k] = float(sum[k] / ds.frames.size());
  float max_dist = 0.0f;
  for (const DatasetFrame& f : ds.frames) {
    float d2 = 0.0f;
    for (int k = 0; k < 3; ++k) d2 += (f.c2w[k * 4 + 3] - ds.camera_center[k]) * (f.c2w[k * 4 + 3] - ds.camera_center[k]);
    max_dist = std::max(max_dist, std::sqrt(d2));
  }
  ds.camera_extent = std::max(1.1f * max_dist, 1e-3f);
  return ds;
}

Camera dataset_camera(const Dataset& ds, const DatasetFrame& f) {
  // c2w_cv = c2w_gl * diag(1, -1, -1); the world -> camera rotation is its transpose.
  float rot_cv[3][3];
  for (int r = 0; r < 3; ++r) {
    rot_cv[r][0] = f.c2w[r * 4 + 0];
    rot_cv[r][1] = -f.c2w[r * 4 + 1];
    rot_cv[r][2] = -f.c2w[r * 4 + 2];
  }
  Camera cam{};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) cam.rotation[r * 3 + c] = rot_cv[c][r];
    cam.position[r] = f.c2w[r * 4 + 3];
  }
  for (int r = 0; r < 3; ++r) {
    cam.translation[r] = -(cam.rotation[r * 3 + 0] * cam.position[0] + cam.rotation[r * 3 + 1] * cam.position[1] +
                           cam.rotation[r * 3 + 2] * cam.position[2]);
  }
  cam.fx = ds.fx;
  cam.fy = ds.fy;
  cam.cx = ds.cx;
  cam.cy = ds.cy;
  cam.width = ds.width;
  cam.height = ds.height;
  return cam;
}

SplatScene initialize_gaussians(const Dataset& ds, const InitOptions& options, InitStats* stats) {
  std::vector<float> points;
  std::vector<float> colors;  // 0..1
  std::string source;
  const bool use_mask = options.subject_only && ds.has_masks;

  if (ds.has_depth) {
    source = "depth";
    const uint8_t min_conf = static_cast<uint8_t>(std::clamp(options.min_confidence, 0.0f, 1.0f) * 255.0f);
    for (const DatasetFrame& f : ds.frames) {
      if (f.holdout || f.depth.empty()) continue;
      for (uint32_t v = 0; v < f.depth_height; ++v) {
        for (uint32_t u = 0; u < f.depth_width; ++u) {
          const size_t di = size_t(v) * f.depth_width + u;
          const float z = f.depth[di];
          if (z <= 0.0f) continue;
          if (!f.depth_confidence.empty() && f.depth_confidence[di] < min_conf) continue;
          const float px = (u + 0.5f) * ds.width / f.depth_width;
          const float py = (v + 0.5f) * ds.height / f.depth_height;
          const uint32_t ix = std::min(ds.width - 1, uint32_t(px)), iy = std::min(ds.height - 1, uint32_t(py));
          const uint8_t* rgba = &f.rgba[(size_t(iy) * ds.width + ix) * 4];
          if (use_mask && rgba[3] < 128) continue;
          float world[3];
          camera_to_world(f.c2w, (px - ds.cx) / ds.fx * z, (py - ds.cy) / ds.fy * z, z, world);
          points.insert(points.end(), world, world + 3);
          for (int c = 0; c < 3; ++c) colors.push_back(rgba[c] / 255.0f);
        }
      }
    }
  }
  if (points.empty() && !ds.points.empty()) {
    source = "points";
    points = ds.points;
    for (uint8_t c : ds.point_colors) colors.push_back(c / 255.0f);
  }
  std::mt19937 rng(options.seed);
  if (points.empty()) {
    source = "random";
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    const float radius = 0.5f * ds.camera_extent;
    for (uint32_t i = 0; i < options.max_gaussians; ++i) {
      float p[3];
      do {
        for (float& x : p) x = u(rng);
      } while (p[0] * p[0] + p[1] * p[1] + p[2] * p[2] > 1.0f);
      for (int k = 0; k < 3; ++k) points.push_back(ds.camera_center[k] + radius * p[k]);
      for (int k = 0; k < 3; ++k) colors.push_back(0.5f);
    }
  }
  const size_t raw = points.size() / 3;

  float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
  for (size_t i = 0; i < raw; ++i) {
    for (int k = 0; k < 3; ++k) {
      lo[k] = std::min(lo[k], points[i * 3 + k]);
      hi[k] = std::max(hi[k], points[i * 3 + k]);
    }
  }
  const float diag = std::sqrt((hi[0] - lo[0]) * (hi[0] - lo[0]) + (hi[1] - lo[1]) * (hi[1] - lo[1]) +
                               (hi[2] - lo[2]) * (hi[2] - lo[2])) + 1e-6f;
  auto occupied = [&](float voxel) {
    std::unordered_map<uint64_t, uint32_t> cells;
    cells.reserve(raw / 4 + 16);
    for (size_t i = 0; i < raw; ++i) {
      const int64_t x = int64_t(std::floor((points[i * 3 + 0] - lo[0]) / voxel));
      const int64_t y = int64_t(std::floor((points[i * 3 + 1] - lo[1]) / voxel));
      const int64_t z = int64_t(std::floor((points[i * 3 + 2] - lo[2]) / voxel));
      cells.emplace(voxel_key(x, y, z), 0);
    }
    return cells.size();
  };
  // Merge points per voxel only when over budget; the voxel size is binary
  // searched so the occupied voxel count lands just under the budget.
  float voxel = 0.0f;
  std::vector<float> merged;
  std::vector<float> merged_colors;
  if (raw <= options.max_gaussians) {
    merged = std::move(points);
    merged_colors = std::move(colors);
  } else {
    float voxel_lo = diag * 1e-5f, voxel_hi = diag;
    for (int it = 0; it < 16; ++it) {
      const float mid = std::sqrt(voxel_lo * voxel_hi);
      if (occupied(mid) > options.max_gaussians) voxel_lo = mid; else voxel_hi = mid;
    }
    voxel = voxel_hi;

    struct Cell {
      double p[3] = {0, 0, 0};
      double c[3] = {0, 0, 0};
      uint32_t n = 0;
    };
    std::unordered_map<uint64_t, Cell> cells;
    for (size_t i = 0; i < raw; ++i) {
      const int64_t x = int64_t(std::floor((points[i * 3 + 0] - lo[0]) / voxel));
      const int64_t y = int64_t(std::floor((points[i * 3 + 1] - lo[1]) / voxel));
      const int64_t z = int64_t(std::floor((points[i * 3 + 2] - lo[2]) / voxel));
      Cell& cell = cells[voxel_key(x, y, z)];
      for (int k = 0; k < 3; ++k) {
        cell.p[k] += points[i * 3 + k];
        cell.c[k] += colors[i * 3 + k];
      }
      ++cell.n;
    }
    merged.reserve(cells.size() * 3);
    merged_colors.reserve(cells.size() * 3);
    for (const auto& [key, cell] : cells) {
      for (int k = 0; k < 3; ++k) {
        merged.push_back(float(cell.p[k] / cell.n));
        merged_colors.push_back(float(cell.c[k] / cell.n));
      }
    }
  }
  const size_t n = merged.size() / 3;

  // Scale from the mean distance to the 3 nearest neighbours, searched in a
  // grid sized from the expected spacing of points on a surface.
  const float cell_size = std::max(2.0f * voxel, 2.0f * diag / std::sqrt(float(std::max<size_t>(n, 1))));
  std::unordered_map<uint64_t, std::vector<uint32_t>> grid;
  auto cell_of = [&](const float* p, int64_t out[3]) {
    for (int k = 0; k < 3; ++k) out[k] = int64_t(std::floor((p[k] - lo[k]) / cell_size));
  };
  for (uint32_t i = 0; i < n; ++i) {
    int64_t c[3];
    cell_of(&merged[i * 3], c);
    grid[voxel_key(c[0], c[1], c[2])].push_back(i);
  }
  SplatScene scene;
  scene.convention = AxisConvention::RUB;  // dataset worlds are Y-up
  scene.resize(static_cast<uint32_t>(n), 0);
  const float opacity_logit = std::log(options.initial_opacity / (1.0f - options.initial_opacity));
  for (uint32_t i = 0; i < n; ++i) {
    const float* p = &merged[i * 3];
    int64_t c[3];
    cell_of(p, c);
    float best[3] = {INFINITY, INFINITY, INFINITY};
    for (int dz = -1; dz <= 1; ++dz) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          auto it = grid.find(voxel_key(c[0] + dx, c[1] + dy, c[2] + dz));
          if (it == grid.end()) continue;
          for (uint32_t j : it->second) {
            if (j == i) continue;
            const float* q = &merged[j * 3];
            const float d2 = (p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]) + (p[2] - q[2]) * (p[2] - q[2]);
            if (d2 < best[2]) {
              best[2] = d2;
              std::sort(best, best + 3);
            }
          }
        }
      }
    }
    float mean = 0.0f;
    int found = 0;
    for (float d2 : best) {
      if (std::isfinite(d2)) {
        mean += std::sqrt(d2);
        ++found;
      }
    }
    const float dist = found > 0 ? mean / found : cell_size;
    const float scale = std::clamp(dist, 1e-4f * diag, 0.02f * diag);
    for (int k = 0; k < 3; ++k) {
      scene.positions[i * 3 + k] = p[k];
      scene.log_scales[i * 3 + k] = std::log(scale);
      scene.sh_dc[i * 3 + k] = (merged_colors[i * 3 + k] - 0.5f) / kShC0;
    }
    scene.rotations[i * 4] = 1.0f;
    scene.opacity_logits[i] = opacity_logit;
  }

  if (stats != nullptr) {
    stats->source = source;
    stats->raw_points = raw;
    stats->voxel_size = voxel;
  }
  return scene;
}

}  // namespace gs
