#pragma once

#include <android/native_window.h>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "gs/camera.h"
#include "gs/splat_renderer.h"
#include "gs/vk_context.h"

// On-screen splat viewer. The scene and camera live for the whole process; the
// window (and with it the swapchain and render thread) comes and goes with the
// SurfaceView. Frames are drawn on demand (camera or scene changed) unless
// continuous rendering is switched on for measuring FPS.
class Viewer {
 public:
  explicit Viewer(std::shared_ptr<gs::VkContext> ctx);
  ~Viewer();

  void attach_window(ANativeWindow* window);  // takes over one reference
  void detach_window();
  void window_resized();

  // Replaces the scene and resets the camera to frame it.
  void set_scene(std::unique_ptr<gs::GpuScene> scene, const gs::SceneBounds& bounds, bool flip_y);
  uint32_t scene_count();

  void orbit(float dx_px, float dy_px);
  void pan(float dx_px, float dy_px);
  void zoom(float factor);
  void reset_view();
  void set_flip_y(bool flip_y);
  void set_continuous(bool continuous);

  std::string stats_json();

 private:
  struct Frame {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkSemaphore acquired = VK_NULL_HANDLE;
  };
  struct Stats {
    double fps = 0.0;
    double prepare_ms = 0.0;
    double draw_ms = 0.0;
    uint32_t visible = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t rotation = 0;
    std::string format;
    std::string error;
  };

  void request_redraw();  // caller holds mutex_
  void render_loop();
  void create_frame_resources();
  void destroy_frame_resources();
  void create_swapchain();
  void destroy_swapchain();
  // Returns false when the swapchain is out of date.
  bool draw_frame(const gs::OrbitCamera& orbit);
  void read_timestamps(uint32_t frame);

  std::shared_ptr<gs::VkContext> ctx_;

  // Shared with the UI / loader threads.
  std::mutex mutex_;
  std::condition_variable cv_;
  ANativeWindow* window_ = nullptr;
  std::thread thread_;
  bool thread_running_ = false;
  bool stop_ = false;
  bool redraw_ = false;
  bool resized_ = false;
  bool continuous_ = false;
  std::unique_ptr<gs::GpuScene> scene_;
  std::unique_ptr<gs::GpuScene> pending_scene_;
  gs::OrbitCamera orbit_;
  gs::SceneBounds bounds_{{0.0f, 0.0f, 0.0f}, 1.0f};
  Stats stats_;

  // Owned by the render thread while it runs.
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;
  VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
  VkFormat format_ = VK_FORMAT_UNDEFINED;
  VkColorSpaceKHR color_space_ = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  VkExtent2D extent_{0, 0};      // swapchain image size
  uint32_t rotation_ = 0;        // pre-rotation in quarter turns
  VkRenderPass render_pass_ = VK_NULL_HANDLE;
  std::vector<VkImageView> views_;
  std::vector<VkFramebuffer> framebuffers_;
  std::vector<VkSemaphore> render_done_;  // one per swapchain image
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  VkQueryPool query_pool_ = VK_NULL_HANDLE;
  Frame frames_[2];
  bool frame_used_[2] = {false, false};
  uint32_t frame_index_ = 0;
  std::unique_ptr<gs::SplatRenderer> renderer_;
};
