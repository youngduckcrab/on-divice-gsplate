#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gs {

// Axis convention of a scene's coordinates.
enum class AxisConvention {
  RDF,  // right, down, forward: COLMAP and the standard 3DGS .ply layout
  RUB,  // right, up, back: three.js, and the convention SPZ stores internally
};

// Gaussian splats in the parameterisation used by 3DGS training: log scales,
// opacity logits, unnormalised (w, x, y, z) quaternions and raw SH coefficients.
struct SplatScene {
  uint32_t count = 0;
  int sh_degree = 0;         // 0..3
  bool antialiased = false;  // render with the mip-splatting opacity compensation
  AxisConvention convention = AxisConvention::RDF;
  std::vector<float> positions;       // 3 per splat
  std::vector<float> log_scales;      // 3 per splat
  std::vector<float> rotations;       // 4 per splat, (w, x, y, z)
  std::vector<float> opacity_logits;  // 1 per splat
  std::vector<float> sh_dc;           // 3 per splat
  std::vector<float> sh_rest;         // sh_rest_coeffs(sh_degree) * 3 per splat, [coeff][rgb]

  void resize(uint32_t n, int degree);
};

// Number of SH coefficients beyond the DC term: 0, 3, 8, 15 for degrees 0..3.
int sh_rest_coeffs(int degree);

struct SceneBounds {
  float center[3];  // per-axis median of positions
  float radius;     // 90th percentile distance from the center
};
SceneBounds compute_bounds(const SplatScene& scene);

// The vertex element of a binary little-endian PLY, with typed property access
// (point clouds as well as splats). Keeps a pointer into `data`.
class PlyVertices {
 public:
  PlyVertices(const uint8_t* data, size_t size);  // throws std::runtime_error
  uint64_t count() const { return count_; }
  int find(const std::string& name) const;  // property index, -1 if absent
  float value(uint64_t vertex, int property) const;

 private:
  struct Property {
    std::string name;
    int type;
    size_t offset;
  };
  const uint8_t* base_ = nullptr;
  uint64_t count_ = 0;
  size_t stride_ = 0;
  std::vector<Property> properties_;
};

// Loaders throw std::runtime_error with a user-presentable message.
SplatScene load_ply(const uint8_t* data, size_t size);
SplatScene load_spz(const uint8_t* data, size_t size);  // SPZ v2/v3 (gzip)
SplatScene load_splat_file(const uint8_t* data, size_t size);  // detects PLY or SPZ

// Binary little-endian PLY with the standard 3DGS vertex properties.
std::vector<uint8_t> save_ply(const SplatScene& scene);

// A torus knot of anisotropic splats with degree-1 SH, in RUB convention.
SplatScene make_synthetic_scene(uint32_t count, uint32_t seed = 1);

}  // namespace gs
