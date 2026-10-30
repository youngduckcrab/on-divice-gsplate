#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "gs/vk_context.h"

namespace gs {

// An RGBA8 colour target with a host-readable copy, for rendering without a
// window (tests, desktop tools).
class OffscreenTarget {
 public:
  OffscreenTarget(VkContext& ctx, uint32_t width, uint32_t height);
  ~OffscreenTarget();
  OffscreenTarget(const OffscreenTarget&) = delete;
  OffscreenTarget& operator=(const OffscreenTarget&) = delete;

  VkRenderPass render_pass() const { return render_pass_; }
  uint32_t width() const { return width_; }
  uint32_t height() const { return height_; }

  // Clears to `clear`, records `draw` inside the render pass, then copies the
  // image to host memory.
  void record(VkCommandBuffer cmd, const float clear[4], const std::function<void(VkCommandBuffer)>& draw);
  // RGBA8 pixels, row-major from the top-left; valid once the recorded work completed.
  std::vector<uint8_t> pixels() const;

 private:
  VkContext& ctx_;
  uint32_t width_;
  uint32_t height_;
  VkImage image_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  VkImageView view_ = VK_NULL_HANDLE;
  VkRenderPass render_pass_ = VK_NULL_HANDLE;
  VkFramebuffer framebuffer_ = VK_NULL_HANDLE;
  Buffer readback_;
};

// Writes RGB (alpha dropped) as a binary PPM.
bool write_ppm(const std::string& path, const std::vector<uint8_t>& rgba, uint32_t width, uint32_t height);

}  // namespace gs
