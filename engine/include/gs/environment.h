#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gs/background.h"
#include "gs/splat_scene.h"

namespace gs {

// A captured 3D place a model can be put into (tools/environments): a splat
// scene in RDF axes (y down), SH degree 0. Throws std::runtime_error.
SplatScene read_gsenv(const uint8_t* data, size_t size);
// The same encoding (positions to 16 bits, scales and colours through
// 256-entry codebooks), for tests and tools.
std::vector<uint8_t> write_gsenv(const SplatScene& scene);

// Where in a place the model stands: a point on the place's floor (place
// coordinates; the floor height is found there), how far the place is turned
// around the model, and the place's scale to metres. `sky` colours (zenith,
// horizon) stand in for what the place does not cover when measuring its light.
struct Spot {
  float x = 0.0f;
  float z = 0.0f;
  float yaw_deg = 0.0f;
  float scale = 1.0f;
  float sky[2][3] = {{0.45f, 0.62f, 0.88f}, {0.80f, 0.86f, 0.93f}};
};

struct Staging {
  SplatScene scene;          // the model's splats first, then the place's
  uint32_t model_count = 0;  // relight only these
  EnvLight light;            // what falls on the model there
  float floor = 0.0f;        // the place's floor height found at the spot (place coordinates)
};

// The model, unchanged, with the place around it: the place is scaled, turned
// about the model's vertical through its middle and moved so the floor under
// the spot meets the model's feet, in the model's axes. Its view-dependent
// colour is dropped (it would have to turn with it). The floor around the
// model's feet darkens as a soft shadow, longer away from the main light.
Staging stage(const SplatScene& model, const SplatScene& place, const Spot& spot);

// The light at `eye` (scene coordinates of `scene`, which is shown flipped
// when `flip_y`): the scene's splats seen from there, nearest first, on a
// coarse panorama, `sky` where nothing is; as EnvLight.
EnvLight light_at(const SplatScene& scene, const float eye[3], bool flip_y, const float sky[2][3],
                  uint32_t first = 0);

}  // namespace gs
