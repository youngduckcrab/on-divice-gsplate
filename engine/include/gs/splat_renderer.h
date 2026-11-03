#pragma once

#include <memory>
#include <vector>

#include "gs/camera.h"
#include "gs/radix_sort.h"
#include "gs/splat_scene.h"
#include "gs/vk_context.h"

namespace gs {

// A scene resident on the GPU, plus the per-frame scratch buffers used to draw it.
struct GpuScene {
  uint32_t count = 0;
  int sh_degree = 0;
  bool antialiased = false;
  Buffer splats;     // 64 B per splat: position + opacity, rotation, scale, SH DC
  Buffer sh_rest;    // SH coefficients beyond DC, fp16
  Buffer projected;  // per frame: 32 B per splat (screen centre, ellipse axes, colour)
  Buffer keys;       // per frame: 16-bit back-to-front depth keys
  Buffer order;      // per frame: sorted splat indices
  Buffer draw_args;  // VkDrawIndirectCommand; instanceCount = visible splats (host visible)
  Buffer skin;       // 8 B per posable splat: 4 bone indices, 4 weights (a byte each); see SplatSkinning
  uint32_t skin_count = 0;  // splats with skin data (the first ones)
};

// Posing of the splats that have skin data (see gs::pose_skeleton): each
// follows the blend of up to 4 bones' rigid motions, blended as dual
// quaternions (position, the axes of its covariance, and the direction its
// view-dependent colour is seen from).
constexpr uint32_t kMaxBones = 32;
struct SplatSkinning {
  uint32_t count = 0;  // splats posed (the first ones; 0: off); no more than the scene's skin_count
  uint32_t bones = 0;
  uint32_t pad[2] = {0, 0};
  float dq[kMaxBones * 2][4] = {};  // per bone: rotation quaternion (w, x, y, z), then its dual part
};

// Relighting of the splats to match a backdrop (see gs::splat_lighting): each
// splat's captured colour is multiplied, in linear light, by the irradiance
// the backdrop casts on it, its normal taken from its thinnest axis (turned
// away from the model's middle) or, for round splats, from the middle out,
// plus a rim of the main light where the surface turns away from the camera
// towards that light (a backlit outline).
struct SplatLighting {
  float sh[9][4] = {};                    // irradiance SH, rgb + unused, display space
  float centre[4] = {0, 0, 0, 0};         // the model's middle in scene coordinates; w: strength (0: off)
  float display[4] = {1, 0, 0, 0};        // x: sign taking scene y and z to display (-1 when flipped)
  float key[4] = {0, 1, 0, 0};            // main light direction, display space
  float rim[4] = {0, 0, 0, 0};            // rim light colour and intensity (rgb)
  uint32_t limit[4] = {0, 0, 0, 0};       // x: relight only splats below this index (0: all)
  // See-through cut for a place around the model: splats past `limit` between
  // the camera and the model's middle, within `cut[1]` of that line and more
  // than a few centimetres above the feet (scene y `cut[0]`), are not drawn,
  // so walls and furniture behind the camera never hide the model. cut[2] > 0: on.
  float cut[4] = {0, 0, 0, 0};
};

// Throws std::runtime_error if the scene exceeds the device's buffer limits.
// Lowers the SH degree if the coefficients would not fit a single binding.
// `skin` (2 uint32 per splat, gs::Skin::packed) makes the first splats posable.
std::unique_ptr<GpuScene> upload_scene(VkContext& ctx, const SplatScene& scene,
                                       const std::vector<uint32_t>* skin = nullptr);

// Rasterises splats as depth-sorted, alpha-blended screen-space ellipses:
// a compute pass projects every splat (EWA 2D covariance, SH colour), a radix
// sort orders them back to front, and an indirect instanced draw blends them.
class SplatRenderer {
 public:
  SplatRenderer(VkContext& ctx, VkRenderPass render_pass);
  ~SplatRenderer();
  SplatRenderer(const SplatRenderer&) = delete;
  SplatRenderer& operator=(const SplatRenderer&) = delete;

  // Rebuilds the graphics pipeline for a different (incompatible) render pass.
  void set_render_pass(VkRenderPass render_pass);

  // Binds `scene` (may be null). The GPU must be done with the previous scene.
  void attach(GpuScene* scene);
  GpuScene* scene() const { return scene_; }

  // Takes effect from the next record_prepare.
  void set_lighting(const SplatLighting& lighting) { lighting_ = lighting; }
  void set_skinning(const SplatSkinning& skinning) { skinning_ = skinning; }

  // Projection + depth sort. Record outside a render pass.
  void record_prepare(VkCommandBuffer cmd, const Camera& camera);
  // Draws the sorted splats. `width` x `height` is the camera's (logical) size;
  // `rotation` is the surface pre-rotation in quarter turns, so the render pass
  // extent is height x width for odd rotations.
  void record_draw(VkCommandBuffer cmd, uint32_t width, uint32_t height, uint32_t rotation = 0);
  // Splats drawn by the most recently completed frame.
  uint32_t visible_count() const;

 private:
  void create_graphics_pipeline(VkRenderPass render_pass);

  VkContext& ctx_;
  ComputePipeline preprocess_;
  RadixSort sort_;
  VkDescriptorSetLayout draw_set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout draw_layout_ = VK_NULL_HANDLE;
  VkPipeline draw_pipeline_ = VK_NULL_HANDLE;
  VkDescriptorPool pool_ = VK_NULL_HANDLE;
  GpuScene* scene_ = nullptr;
  VkDescriptorSet preprocess_set_ = VK_NULL_HANDLE;
  VkDescriptorSet draw_set_ = VK_NULL_HANDLE;
  RadixSort::Binding sort_binding_;
  SplatLighting lighting_;
  Buffer lighting_buffer_;
  SplatSkinning skinning_;
  Buffer skinning_buffer_;
};

}  // namespace gs
