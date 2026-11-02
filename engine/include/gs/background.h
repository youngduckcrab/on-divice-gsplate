#pragma once

#include <cstdint>
#include <functional>
#include <string_view>
#include <vector>

#include "gs/camera.h"
#include "gs/splat_renderer.h"
#include "gs/splat_scene.h"
#include "gs/vk_context.h"

namespace gs {

// What the viewer shows where no splat covers the screen.
enum class Backdrop : uint32_t {
  Plain = 0,  // the dark grey the viewer always had
  Sky,        // a sky gradient, for places open to the sky
};

// Where a model stands, for lighting it: its extent along scene y (ignoring
// stray splats) and its middle across, in scene coordinates.
struct Ground {
  bool found = false;
  float low = 0.0f, high = 0.0f;  // y percentiles
  float centre[2] = {0.0f, 0.0f};  // x, z medians
};
Ground find_ground(const SplatScene& scene);

// The light around a model, for relighting it to match where it is:
// irradiance as 9 real SH coefficients per colour (convolved with the cosine
// lobe, display space: y up, the default camera looking along -z), scaled so
// the average over all directions has the luminance of how bright the place
// is (about 1 for an ordinary day or room, less when dark); the direction
// most of the light comes from, the share of the light it carries and its
// colour (luminance 1).
struct EnvLight {
  bool on = false;
  float sh[27] = {};  // [coefficient][rgb]
  float key[3] = {0.0f, 1.0f, 0.0f};
  float key_share = 0.0f;
  float key_color[3] = {1.0f, 1.0f, 1.0f};
};
// From linear radiance over the sphere (display directions), normalised to
// mean luminance `brightness`.
EnvLight light_from_directions(const std::function<void(float x, float y, float z, float rgb[3])>& radiance,
                               float brightness);

struct BackdropView {
  Backdrop kind = Backdrop::Plain;
  float zenith[3] = {0.40f, 0.60f, 0.88f};
  float horizon[3] = {0.78f, 0.86f, 0.94f};
  // Relighting of the model (the scene's first `relight_count` splats; 0: all).
  EnvLight light;
  Ground ground;
  uint32_t relight_count = 0;
};

// Draws the backdrop with one full-screen triangle, before the splats,
// inside the same render pass (no blending: it replaces the clear colour).
class BackgroundRenderer {
 public:
  BackgroundRenderer(VkContext& ctx, VkRenderPass render_pass);
  ~BackgroundRenderer();
  BackgroundRenderer(const BackgroundRenderer&) = delete;
  BackgroundRenderer& operator=(const BackgroundRenderer&) = delete;

  void set_render_pass(VkRenderPass render_pass);
  // `camera` as the splats are drawn with, `flip_y` as the OrbitCamera's
  // (display = scene turned half a turn about x), `rotation` as for
  // SplatRenderer::record_draw.
  void record(VkCommandBuffer cmd, const BackdropView& view, const Camera& camera, bool flip_y,
              uint32_t rotation = 0);

 private:
  void create_pipeline(VkRenderPass render_pass);

  VkContext& ctx_;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
};

// How the splats take the light (SplatRenderer::set_lighting): `strength` 0
// leaves their colours as captured, 1 lights them fully by it; only the first
// `count` splats (0: all).
SplatLighting splat_lighting(const EnvLight& light, const Ground& ground, bool flip_y, float strength,
                             uint32_t count = 0);

}  // namespace gs
