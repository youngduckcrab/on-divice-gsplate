#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "gs/camera.h"
#include "gs/splat_renderer.h"
#include "gs/splat_scene.h"
#include "gs/vk_context.h"

namespace gs {

// What the viewer shows behind a model.
enum class Backdrop : uint32_t {
  Plain = 0,  // the dark grey the viewer always had
  Studio,     // white cyclorama, floor fading into the wall
  Photo,      // a picture behind, filling the screen, not turning with the model
  Panorama,   // a 360-degree (equirectangular) picture around, its floor under the model
};
// "plain", "studio", "photo", "panorama"
bool parse_backdrop(std::string_view name, Backdrop* backdrop);

// The light a backdrop casts, for relighting the model to match it and for
// its shadow: irradiance as 9 real SH coefficients per colour (convolved with
// the cosine lobe, display space: y up, the default camera looking along -z),
// scaled so the average over all directions has the luminance of how bright
// the place is (1 for an ordinary day or room, less at night); the direction
// most of the light comes from and the share of the light it carries.
struct EnvLight {
  bool on = false;
  float sh[27] = {};  // [coefficient][rgb]
  float key[3] = {0.0f, 1.0f, 0.0f};
  float key_share = 0.0f;
  float key_color[3] = {1.0f, 1.0f, 1.0f};  // luminance 1
};
// From a picture (RGBA8, sRGB): a 360-degree one turned by `yaw_deg` as the
// backdrop shows it, or any other photo, taken as roughly what surrounds it.
EnvLight light_from_picture(const uint8_t* rgba, uint32_t width, uint32_t height, bool panorama, float yaw_deg);
// The white studio's soft light from the upper left front.
EnvLight studio_light();

// Where a model stands, for a floor and a shadow under it: its extent along
// scene y (ignoring stray splats) and its middle across, in scene coordinates.
struct Ground {
  bool found = false;
  float low = 0.0f, high = 0.0f;  // y percentiles
  float centre[2] = {0.0f, 0.0f};  // x, z medians
};
Ground find_ground(const SplatScene& scene);

// What to draw: the kind, where the model stands, and for panoramas how the
// photo is placed (sizes in metres for a person-sized model, scaled with the
// model otherwise), how much the backdrop blurs away from the model's
// distance, and the light that decides the shadow.
struct BackdropView {
  Backdrop kind = Backdrop::Plain;
  Ground ground;
  float yaw_deg = 0.0f;
  float camera_height = 1.6f;
  float radius = 20.0f;
  float blur = 0.6f;
  EnvLight light;
};

// Draws a backdrop with one full-screen triangle, before the splats, inside
// the same render pass (no blending: it replaces the clear colour).
class BackgroundRenderer {
 public:
  BackgroundRenderer(VkContext& ctx, VkRenderPass render_pass);
  ~BackgroundRenderer();
  BackgroundRenderer(const BackgroundRenderer&) = delete;
  BackgroundRenderer& operator=(const BackgroundRenderer&) = delete;

  void set_render_pass(VkRenderPass render_pass);
  // RGBA8 picture for Photo and Panorama (rows from the top), with its mip
  // levels for the blur. Uploads with ctx.run(); the GPU must be done with
  // earlier frames.
  void set_picture(const uint8_t* rgba, uint32_t width, uint32_t height);

  // `camera` as the splats are drawn with, `flip_y` as the OrbitCamera's
  // (display = scene turned half a turn about x), `rotation` as for
  // SplatRenderer::record_draw.
  void record(VkCommandBuffer cmd, const BackdropView& view, const Camera& camera, bool flip_y,
              uint32_t rotation = 0);

 private:
  void create_pipeline(VkRenderPass render_pass);
  void destroy_picture();

  VkContext& ctx_;
  VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  VkDescriptorPool pool_ = VK_NULL_HANDLE;
  VkDescriptorSet set_ = VK_NULL_HANDLE;
  VkSampler sampler_ = VK_NULL_HANDLE;
  VkImage image_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  VkImageView view_ = VK_NULL_HANDLE;
  float picture_aspect_ = 1.0f;
};

// How the splats take a backdrop's light (SplatRenderer::set_lighting):
// `strength` 0 leaves their colours as captured, 1 lights them fully by it.
SplatLighting splat_lighting(const EnvLight& light, const Ground& ground, bool flip_y, float strength);

}  // namespace gs
