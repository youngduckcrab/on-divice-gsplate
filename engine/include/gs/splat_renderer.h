#pragma once

#include <memory>

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
};

// Throws std::runtime_error if the scene exceeds the device's buffer limits.
// Lowers the SH degree if the coefficients would not fit a single binding.
std::unique_ptr<GpuScene> upload_scene(VkContext& ctx, const SplatScene& scene);

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
};

}  // namespace gs
