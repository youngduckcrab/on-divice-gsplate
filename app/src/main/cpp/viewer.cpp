#define VK_USE_PLATFORM_ANDROID_KHR
#include <vulkan/vulkan.h>

#include "viewer.h"

#include <algorithm>
#include <chrono>
#include <cstring>

#include "gs/json.h"
#include "gs/log.h"

namespace {

using Clock = std::chrono::steady_clock;

const char* format_name(VkFormat format) {
  switch (format) {
    case VK_FORMAT_R8G8B8A8_UNORM: return "R8G8B8A8_UNORM";
    case VK_FORMAT_B8G8R8A8_UNORM: return "B8G8R8A8_UNORM";
    case VK_FORMAT_R8G8B8A8_SRGB: return "R8G8B8A8_SRGB";
    case VK_FORMAT_B8G8R8A8_SRGB: return "B8G8R8A8_SRGB";
    case VK_FORMAT_R5G6B5_UNORM_PACK16: return "R5G6B5";
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return "A2B10G10R10";
    default: return "other";
  }
}

}  // namespace

Viewer::Viewer(std::shared_ptr<gs::VkContext> ctx) : ctx_(std::move(ctx)) {
  orbit_.yaw = 0.0f;
  orbit_.pitch = 0.25f;
}

Viewer::~Viewer() { detach_window(); }

void Viewer::attach_window(ANativeWindow* window) {
  detach_window();
  std::lock_guard<std::mutex> lock(mutex_);
  window_ = window;
  stop_ = false;
  redraw_ = true;
  resized_ = false;
  stats_.error.clear();
  thread_running_ = true;
  thread_ = std::thread(&Viewer::render_loop, this);
}

void Viewer::detach_window() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!thread_running_) return;
    stop_ = true;
  }
  cv_.notify_all();
  thread_.join();
  std::lock_guard<std::mutex> lock(mutex_);
  thread_running_ = false;
  // A scene handed over after the render thread stopped was never swapped in.
  if (pending_scene_) scene_ = std::move(pending_scene_);
  if (window_ != nullptr) ANativeWindow_release(window_);
  window_ = nullptr;
}

void Viewer::window_resized() {
  std::lock_guard<std::mutex> lock(mutex_);
  resized_ = true;
  cv_.notify_all();
}

void Viewer::set_scene(std::unique_ptr<gs::GpuScene> scene, const gs::SceneBounds& bounds, bool flip_y) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (thread_running_) {
    pending_scene_ = std::move(scene);  // swapped in by the render thread once the GPU is idle
  } else {
    scene_ = std::move(scene);
  }
  bounds_ = bounds;
  orbit_.reset(bounds, flip_y);
  request_redraw();
}

uint32_t Viewer::scene_count() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (pending_scene_) return pending_scene_->count;
  return scene_ ? scene_->count : 0;
}

void Viewer::orbit(float dx_px, float dy_px) {
  std::lock_guard<std::mutex> lock(mutex_);
  orbit_.orbit(dx_px, dy_px);
  request_redraw();
}

void Viewer::pan(float dx_px, float dy_px) {
  std::lock_guard<std::mutex> lock(mutex_);
  orbit_.pan(dx_px, dy_px, std::max(stats_.width, 1u), std::max(stats_.height, 1u));
  request_redraw();
}

void Viewer::zoom(float factor) {
  std::lock_guard<std::mutex> lock(mutex_);
  orbit_.zoom(factor);
  request_redraw();
}

void Viewer::reset_view() {
  std::lock_guard<std::mutex> lock(mutex_);
  orbit_.reset(bounds_, orbit_.flip_y());
  request_redraw();
}

void Viewer::set_flip_y(bool flip_y) {
  std::lock_guard<std::mutex> lock(mutex_);
  orbit_.set_flip_y(flip_y);
  request_redraw();
}

void Viewer::set_continuous(bool continuous) {
  std::lock_guard<std::mutex> lock(mutex_);
  continuous_ = continuous;
  request_redraw();
}

void Viewer::request_redraw() {
  redraw_ = true;
  cv_.notify_all();
}

std::string Viewer::stats_json() {
  std::lock_guard<std::mutex> lock(mutex_);
  gs::JsonWriter w;
  w.begin_object();
  w.field("fps", stats_.fps);
  w.field("prepare_ms", stats_.prepare_ms);
  w.field("draw_ms", stats_.draw_ms);
  w.field("visible", stats_.visible);
  w.field("count", scene_ ? scene_->count : 0u);
  w.field("sh_degree", scene_ ? scene_->sh_degree : 0);
  w.field("width", stats_.width);
  w.field("height", stats_.height);
  w.field("rotation", stats_.rotation);
  w.field("format", stats_.format);
  w.field("continuous", continuous_);
  w.field("flip_y", orbit_.flip_y());
  w.field("timestamps", ctx_->enabled().timestamps);
  if (!stats_.error.empty()) w.field("error", stats_.error);
  w.end_object();
  return w.str();
}

// --- Render thread -------------------------------------------------------------

void Viewer::render_loop() {
  try {
    VkAndroidSurfaceCreateInfoKHR surface_info{VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR};
    surface_info.window = window_;
    GS_VK_CHECK(vkCreateAndroidSurfaceKHR(ctx_->instance(), &surface_info, nullptr, &surface_));
    VkBool32 supported = VK_FALSE;
    GS_VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(ctx_->physical_device(), ctx_->queue_family(), surface_,
                                                     &supported));
    if (!supported) throw std::runtime_error("GPU 큐가 이 화면에 출력할 수 없음");

    create_frame_resources();
    create_swapchain();
    renderer_ = std::make_unique<gs::SplatRenderer>(*ctx_, render_pass_);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      renderer_->attach(scene_.get());
    }

    auto fps_start = Clock::now();
    int fps_frames = 0;
    while (true) {
      gs::OrbitCamera orbit;
      std::unique_ptr<gs::GpuScene> incoming;
      bool resized = false;
      bool continuous = false;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return stop_ || redraw_ || resized_ || continuous_ || pending_scene_ != nullptr; });
        if (stop_) break;
        incoming = std::move(pending_scene_);
        resized = resized_;
        resized_ = false;
        redraw_ = false;
        continuous = continuous_;
        orbit = orbit_;
      }

      if (incoming) {
        ctx_->queue_wait_idle();  // earlier frames may still read the old scene
        std::lock_guard<std::mutex> lock(mutex_);
        renderer_->attach(nullptr);
        scene_ = std::move(incoming);
        renderer_->attach(scene_.get());
      }
      if (resized || extent_.width == 0 || extent_.height == 0) {
        ctx_->queue_wait_idle();
        create_swapchain();
      }
      if (extent_.width == 0 || extent_.height == 0) {
        // Window has no area right now; poll until it gets one.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        continue;
      }

      if (!draw_frame(orbit)) {
        std::lock_guard<std::mutex> lock(mutex_);
        resized_ = true;
        redraw_ = true;
        continue;
      }

      ++fps_frames;
      const auto now = Clock::now();
      const double elapsed = std::chrono::duration<double>(now - fps_start).count();
      std::lock_guard<std::mutex> lock(mutex_);
      stats_.visible = renderer_->visible_count();
      if (elapsed >= 1.0 || !continuous) {
        stats_.fps = continuous ? fps_frames / elapsed : 0.0;
        fps_frames = 0;
        fps_start = now;
      }
    }
  } catch (const std::exception& e) {
    gs::log_error("viewer: %s", e.what());
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.error = e.what();
  }

  ctx_->queue_wait_idle();
  renderer_.reset();
  destroy_swapchain();
  if (render_pass_ != VK_NULL_HANDLE) vkDestroyRenderPass(ctx_->device(), render_pass_, nullptr);
  render_pass_ = VK_NULL_HANDLE;
  format_ = VK_FORMAT_UNDEFINED;
  destroy_frame_resources();
  if (surface_ != VK_NULL_HANDLE) vkDestroySurfaceKHR(ctx_->instance(), surface_, nullptr);
  surface_ = VK_NULL_HANDLE;
}

void Viewer::create_frame_resources() {
  VkDevice device = ctx_->device();
  VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_info.queueFamilyIndex = ctx_->queue_family();
  GS_VK_CHECK(vkCreateCommandPool(device, &pool_info, nullptr, &command_pool_));
  for (Frame& f : frames_) {
    VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = command_pool_;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    GS_VK_CHECK(vkAllocateCommandBuffers(device, &alloc, &f.cmd));
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    GS_VK_CHECK(vkCreateFence(device, &fence_info, nullptr, &f.fence));
    VkSemaphoreCreateInfo sem_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    GS_VK_CHECK(vkCreateSemaphore(device, &sem_info, nullptr, &f.acquired));
  }
  frame_used_[0] = frame_used_[1] = false;
  frame_index_ = 0;
  if (ctx_->enabled().timestamps) {
    VkQueryPoolCreateInfo query_info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    query_info.queryCount = 6;  // 3 per frame in flight
    GS_VK_CHECK(vkCreateQueryPool(device, &query_info, nullptr, &query_pool_));
  }
}

void Viewer::destroy_frame_resources() {
  VkDevice device = ctx_->device();
  for (Frame& f : frames_) {
    if (f.fence != VK_NULL_HANDLE) vkDestroyFence(device, f.fence, nullptr);
    if (f.acquired != VK_NULL_HANDLE) vkDestroySemaphore(device, f.acquired, nullptr);
    f = Frame{};
  }
  if (query_pool_ != VK_NULL_HANDLE) vkDestroyQueryPool(device, query_pool_, nullptr);
  query_pool_ = VK_NULL_HANDLE;
  if (command_pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device, command_pool_, nullptr);
  command_pool_ = VK_NULL_HANDLE;
}

void Viewer::create_swapchain() {
  VkPhysicalDevice physical = ctx_->physical_device();
  VkDevice device = ctx_->device();

  VkSurfaceCapabilitiesKHR caps;
  GS_VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface_, &caps));
  VkExtent2D extent = caps.currentExtent;
  if (extent.width == 0xffffffffu) {
    extent.width = static_cast<uint32_t>(ANativeWindow_getWidth(window_));
    extent.height = static_cast<uint32_t>(ANativeWindow_getHeight(window_));
  }
  extent.width = std::clamp(extent.width, caps.minImageExtent.width, caps.maxImageExtent.width);
  extent.height = std::clamp(extent.height, caps.minImageExtent.height, caps.maxImageExtent.height);

  // Views, framebuffers and semaphores belong to the old images.
  for (VkFramebuffer fb : framebuffers_) vkDestroyFramebuffer(device, fb, nullptr);
  for (VkImageView view : views_) vkDestroyImageView(device, view, nullptr);
  for (VkSemaphore sem : render_done_) vkDestroySemaphore(device, sem, nullptr);
  framebuffers_.clear();
  views_.clear();
  render_done_.clear();
  // The format (and so the render pass) is fixed for the life of the surface.
  if (format_ == VK_FORMAT_UNDEFINED) {
    uint32_t count = 0;
    GS_VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface_, &count, nullptr));
    std::vector<VkSurfaceFormatKHR> formats(count);
    GS_VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface_, &count, formats.data()));
    if (formats.empty()) throw std::runtime_error("화면 출력 형식이 없음");
    // Splat colours are already in display (sRGB-encoded) space, so prefer UNORM.
    VkSurfaceFormatKHR chosen = formats[0];
    for (VkFormat wanted : {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM}) {
      auto it = std::find_if(formats.begin(), formats.end(), [&](const VkSurfaceFormatKHR& f) {
        return f.format == wanted && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
      });
      if (it != formats.end()) {
        chosen = *it;
        break;
      }
    }
    format_ = chosen.format;
    color_space_ = chosen.colorSpace;
  }

  if (render_pass_ == VK_NULL_HANDLE) {
    VkAttachmentDescription attachment{};
    attachment.format = format_;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentReference color_ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_ref;
    // Order the image layout transition after the acquire semaphore wait.
    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    VkRenderPassCreateInfo pass_info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    pass_info.attachmentCount = 1;
    pass_info.pAttachments = &attachment;
    pass_info.subpassCount = 1;
    pass_info.pSubpasses = &subpass;
    pass_info.dependencyCount = 1;
    pass_info.pDependencies = &dep;
    GS_VK_CHECK(vkCreateRenderPass(device, &pass_info, nullptr, &render_pass_));
  }

  VkSwapchainKHR old = swapchain_;
  swapchain_ = VK_NULL_HANDLE;
  if (extent.width == 0 || extent.height == 0) {
    if (old != VK_NULL_HANDLE) vkDestroySwapchainKHR(device, old, nullptr);
    extent_ = {0, 0};
    return;
  }

  VkSurfaceTransformFlagBitsKHR transform = caps.currentTransform;
  switch (transform) {
    case VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR: rotation_ = 1; break;
    case VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR: rotation_ = 2; break;
    case VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR: rotation_ = 3; break;
    default:
      rotation_ = 0;
      if (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) {
        transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
      }
  }

  VkCompositeAlphaFlagBitsKHR composite = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  for (VkCompositeAlphaFlagBitsKHR candidate :
       {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
        VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR}) {
    if (caps.supportedCompositeAlpha & candidate) {
      composite = candidate;
      break;
    }
  }

  uint32_t image_count = caps.minImageCount + 1;
  if (caps.maxImageCount > 0) image_count = std::min(image_count, caps.maxImageCount);

  VkSwapchainCreateInfoKHR info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  info.surface = surface_;
  info.minImageCount = image_count;
  info.imageFormat = format_;
  info.imageColorSpace = color_space_;
  info.imageExtent = extent;
  info.imageArrayLayers = 1;
  info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  info.preTransform = transform;
  info.compositeAlpha = composite;
  info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  info.clipped = VK_TRUE;
  info.oldSwapchain = old;
  VkResult result = vkCreateSwapchainKHR(device, &info, nullptr, &swapchain_);
  if (old != VK_NULL_HANDLE) vkDestroySwapchainKHR(device, old, nullptr);
  if (result != VK_SUCCESS) throw gs::VulkanError("vkCreateSwapchainKHR", result);
  extent_ = extent;

  uint32_t count = 0;
  GS_VK_CHECK(vkGetSwapchainImagesKHR(device, swapchain_, &count, nullptr));
  std::vector<VkImage> images(count);
  GS_VK_CHECK(vkGetSwapchainImagesKHR(device, swapchain_, &count, images.data()));
  for (VkImage image : images) {
    VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = format_;
    view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView view = VK_NULL_HANDLE;
    GS_VK_CHECK(vkCreateImageView(device, &view_info, nullptr, &view));
    views_.push_back(view);

    VkFramebufferCreateInfo fb_info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fb_info.renderPass = render_pass_;
    fb_info.attachmentCount = 1;
    fb_info.pAttachments = &view;
    fb_info.width = extent.width;
    fb_info.height = extent.height;
    fb_info.layers = 1;
    VkFramebuffer fb = VK_NULL_HANDLE;
    GS_VK_CHECK(vkCreateFramebuffer(device, &fb_info, nullptr, &fb));
    framebuffers_.push_back(fb);

    VkSemaphoreCreateInfo sem_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkSemaphore sem = VK_NULL_HANDLE;
    GS_VK_CHECK(vkCreateSemaphore(device, &sem_info, nullptr, &sem));
    render_done_.push_back(sem);
  }

  std::lock_guard<std::mutex> lock(mutex_);
  const bool swap = (rotation_ & 1u) != 0;
  stats_.width = swap ? extent.height : extent.width;
  stats_.height = swap ? extent.width : extent.height;
  stats_.rotation = rotation_;
  stats_.format = format_name(format_);
}

void Viewer::destroy_swapchain() {
  VkDevice device = ctx_->device();
  for (VkFramebuffer fb : framebuffers_) vkDestroyFramebuffer(device, fb, nullptr);
  for (VkImageView view : views_) vkDestroyImageView(device, view, nullptr);
  for (VkSemaphore sem : render_done_) vkDestroySemaphore(device, sem, nullptr);
  framebuffers_.clear();
  views_.clear();
  render_done_.clear();
  if (swapchain_ != VK_NULL_HANDLE) vkDestroySwapchainKHR(device, swapchain_, nullptr);
  swapchain_ = VK_NULL_HANDLE;
  extent_ = {0, 0};
}

void Viewer::read_timestamps(uint32_t frame) {
  if (query_pool_ == VK_NULL_HANDLE) return;
  uint64_t ts[3] = {0, 0, 0};
  VkResult r = vkGetQueryPoolResults(ctx_->device(), query_pool_, frame * 3, 3, sizeof(ts), ts, sizeof(uint64_t),
                                     VK_QUERY_RESULT_64_BIT);
  if (r != VK_SUCCESS) return;
  const auto& info = ctx_->info();
  const uint32_t bits = info.queue_families[info.queue_family].timestampValidBits;
  const uint64_t mask = bits >= 64 ? ~0ull : ((1ull << bits) - 1);
  const double period_ms = info.properties.limits.timestampPeriod / 1e6;
  const double prepare = double((ts[1] - ts[0]) & mask) * period_ms;
  const double draw = double((ts[2] - ts[1]) & mask) * period_ms;
  std::lock_guard<std::mutex> lock(mutex_);
  // Exponential moving average keeps the HUD readable while orbiting.
  stats_.prepare_ms = stats_.prepare_ms == 0.0 ? prepare : 0.8 * stats_.prepare_ms + 0.2 * prepare;
  stats_.draw_ms = stats_.draw_ms == 0.0 ? draw : 0.8 * stats_.draw_ms + 0.2 * draw;
}

bool Viewer::draw_frame(const gs::OrbitCamera& orbit) {
  VkDevice device = ctx_->device();
  Frame& f = frames_[frame_index_];
  GS_VK_CHECK(vkWaitForFences(device, 1, &f.fence, VK_TRUE, UINT64_MAX));
  if (frame_used_[frame_index_]) read_timestamps(frame_index_);

  uint32_t image = 0;
  VkResult r = vkAcquireNextImageKHR(device, swapchain_, UINT64_MAX, f.acquired, VK_NULL_HANDLE, &image);
  if (r == VK_ERROR_OUT_OF_DATE_KHR) return false;
  if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) throw gs::VulkanError("vkAcquireNextImageKHR", r);
  GS_VK_CHECK(vkResetFences(device, 1, &f.fence));

  const bool swap = (rotation_ & 1u) != 0;
  const uint32_t width = swap ? extent_.height : extent_.width;
  const uint32_t height = swap ? extent_.width : extent_.height;
  const gs::Camera camera = orbit.camera(width, height);

  GS_VK_CHECK(vkResetCommandBuffer(f.cmd, 0));
  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  GS_VK_CHECK(vkBeginCommandBuffer(f.cmd, &begin));
  const uint32_t q = frame_index_ * 3;
  if (query_pool_ != VK_NULL_HANDLE) {
    vkCmdResetQueryPool(f.cmd, query_pool_, q, 3);
    vkCmdWriteTimestamp(f.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool_, q);
  }
  renderer_->record_prepare(f.cmd, camera);
  if (query_pool_ != VK_NULL_HANDLE) {
    vkCmdWriteTimestamp(f.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, q + 1);
  }

  VkClearValue clear{};
  clear.color.float32[0] = 0.08f;
  clear.color.float32[1] = 0.09f;
  clear.color.float32[2] = 0.11f;
  clear.color.float32[3] = 1.0f;
  VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  pass.renderPass = render_pass_;
  pass.framebuffer = framebuffers_[image];
  pass.renderArea = {{0, 0}, extent_};
  pass.clearValueCount = 1;
  pass.pClearValues = &clear;
  vkCmdBeginRenderPass(f.cmd, &pass, VK_SUBPASS_CONTENTS_INLINE);
  renderer_->record_draw(f.cmd, width, height, rotation_);
  vkCmdEndRenderPass(f.cmd);
  if (query_pool_ != VK_NULL_HANDLE) {
    vkCmdWriteTimestamp(f.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, q + 2);
  }
  GS_VK_CHECK(vkEndCommandBuffer(f.cmd));

  const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.waitSemaphoreCount = 1;
  submit.pWaitSemaphores = &f.acquired;
  submit.pWaitDstStageMask = &wait_stage;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &f.cmd;
  submit.signalSemaphoreCount = 1;
  submit.pSignalSemaphores = &render_done_[image];
  GS_VK_CHECK(ctx_->submit(submit, f.fence));
  frame_used_[frame_index_] = true;
  frame_index_ ^= 1u;

  VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
  present.waitSemaphoreCount = 1;
  present.pWaitSemaphores = &render_done_[image];
  present.swapchainCount = 1;
  present.pSwapchains = &swapchain_;
  present.pImageIndices = &image;
  r = ctx_->present(present);
  if (r == VK_ERROR_OUT_OF_DATE_KHR) return false;
  if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) throw gs::VulkanError("vkQueuePresentKHR", r);
  return true;
}
