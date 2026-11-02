#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "gs/camera.h"
#include "gs/splat_scene.h"
#include "gs/vk_context.h"

namespace gs {

// What the viewer shows behind a model.
enum class Backdrop : uint32_t {
  Plain = 0,    // the dark grey the viewer always had
  Studio,       // white cyclorama, floor fading into the wall
  Dark,         // dark room with a pool of light on the floor
  Park,         // sky with a sun over grass
  Checker,      // checkered floor to the horizon
  Sunset,       // evening sky over dark ground
  Night,        // stars over a dark floor with a glow
  Photo,        // a picture behind, filling the screen, not turning with the model
  Panorama,     // a 360-degree (equirectangular) picture all around
};
// "plain", "studio", "dark", "park", "checker", "sunset", "night", "photo", "panorama"
bool parse_backdrop(std::string_view name, Backdrop* backdrop);

// Where a model stands, for a floor and a shadow under it: its extent along
// scene y (ignoring stray splats) and its middle across, in scene coordinates.
struct Ground {
  bool found = false;
  float low = 0.0f, high = 0.0f;  // y percentiles
  float centre[2] = {0.0f, 0.0f};  // x, z medians
};
Ground find_ground(const SplatScene& scene);

// Draws a backdrop with one full-screen triangle, before the splats, inside
// the same render pass (no blending: it replaces the clear colour).
class BackgroundRenderer {
 public:
  BackgroundRenderer(VkContext& ctx, VkRenderPass render_pass);
  ~BackgroundRenderer();
  BackgroundRenderer(const BackgroundRenderer&) = delete;
  BackgroundRenderer& operator=(const BackgroundRenderer&) = delete;

  void set_render_pass(VkRenderPass render_pass);
  // RGBA8 picture for Photo and Panorama (rows from the top). Uploads with
  // ctx.run(); the GPU must be done with earlier frames.
  void set_picture(const uint8_t* rgba, uint32_t width, uint32_t height);

  // `camera` as the splats are drawn with, `flip_y` as the OrbitCamera's
  // (display = scene turned half a turn about x), `rotation` as for
  // SplatRenderer::record_draw.
  void record(VkCommandBuffer cmd, Backdrop backdrop, const Ground& ground, const Camera& camera, bool flip_y,
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

}  // namespace gs
