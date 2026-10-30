#include "gs/offscreen.h"

#include <cstdio>
#include <cstring>

namespace gs {

OffscreenTarget::OffscreenTarget(VkContext& ctx, uint32_t width, uint32_t height)
    : ctx_(ctx), width_(width), height_(height) {
  VkDevice device = ctx_.device();
  const VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

  VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.format = format;
  image_info.extent = {width, height, 1};
  image_info.mipLevels = 1;
  image_info.arrayLayers = 1;
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  GS_VK_CHECK(vkCreateImage(device, &image_info, nullptr, &image_));

  VkMemoryRequirements req;
  vkGetImageMemoryRequirements(device, image_, &req);
  VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  alloc.allocationSize = req.size;
  alloc.memoryTypeIndex = ctx_.find_memory_type(req.memoryTypeBits, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  GS_VK_CHECK(vkAllocateMemory(device, &alloc, nullptr, &memory_));
  GS_VK_CHECK(vkBindImageMemory(device, image_, memory_, 0));

  VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  view_info.image = image_;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = format;
  view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  GS_VK_CHECK(vkCreateImageView(device, &view_info, nullptr, &view_));

  VkAttachmentDescription attachment{};
  attachment.format = format;
  attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  attachment.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  VkAttachmentReference color_ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &color_ref;
  VkSubpassDependency deps[2] = {};
  // Prior compute writes (projection/sort) feed the vertex shader.
  deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
  deps[0].dstSubpass = 0;
  deps[0].srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  deps[0].dstStageMask = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  deps[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  deps[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  // Colour writes must land before the copy to host memory.
  deps[1].srcSubpass = 0;
  deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
  deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  deps[1].dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
  deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  deps[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  VkRenderPassCreateInfo pass_info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
  pass_info.attachmentCount = 1;
  pass_info.pAttachments = &attachment;
  pass_info.subpassCount = 1;
  pass_info.pSubpasses = &subpass;
  pass_info.dependencyCount = 2;
  pass_info.pDependencies = deps;
  GS_VK_CHECK(vkCreateRenderPass(device, &pass_info, nullptr, &render_pass_));

  VkFramebufferCreateInfo fb_info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  fb_info.renderPass = render_pass_;
  fb_info.attachmentCount = 1;
  fb_info.pAttachments = &view_;
  fb_info.width = width;
  fb_info.height = height;
  fb_info.layers = 1;
  GS_VK_CHECK(vkCreateFramebuffer(device, &fb_info, nullptr, &framebuffer_));

  readback_ = ctx_.create_buffer(uint64_t(width) * height * 4, MemoryKind::Readback);
}

OffscreenTarget::~OffscreenTarget() {
  VkDevice device = ctx_.device();
  vkDeviceWaitIdle(device);
  if (framebuffer_ != VK_NULL_HANDLE) vkDestroyFramebuffer(device, framebuffer_, nullptr);
  if (render_pass_ != VK_NULL_HANDLE) vkDestroyRenderPass(device, render_pass_, nullptr);
  if (view_ != VK_NULL_HANDLE) vkDestroyImageView(device, view_, nullptr);
  if (image_ != VK_NULL_HANDLE) vkDestroyImage(device, image_, nullptr);
  if (memory_ != VK_NULL_HANDLE) vkFreeMemory(device, memory_, nullptr);
}

void OffscreenTarget::record(VkCommandBuffer cmd, const float clear[4],
                             const std::function<void(VkCommandBuffer)>& draw) {
  VkClearValue clear_value{};
  std::memcpy(clear_value.color.float32, clear, 4 * sizeof(float));
  VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  begin.renderPass = render_pass_;
  begin.framebuffer = framebuffer_;
  begin.renderArea = {{0, 0}, {width_, height_}};
  begin.clearValueCount = 1;
  begin.pClearValues = &clear_value;
  vkCmdBeginRenderPass(cmd, &begin, VK_SUBPASS_CONTENTS_INLINE);
  draw(cmd);
  vkCmdEndRenderPass(cmd);

  VkBufferImageCopy region{};
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.imageExtent = {width_, height_, 1};
  vkCmdCopyImageToBuffer(cmd, image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback_.handle(), 1, &region);

  VkMemoryBarrier to_host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  to_host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &to_host, 0,
                       nullptr, 0, nullptr);
}

std::vector<uint8_t> OffscreenTarget::pixels() const {
  const size_t size = size_t(width_) * height_ * 4;
  std::vector<uint8_t> out(size);
  std::memcpy(out.data(), readback_.mapped(), size);
  return out;
}

bool write_ppm(const std::string& path, const std::vector<uint8_t>& rgba, uint32_t width, uint32_t height) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) return false;
  std::fprintf(f, "P6\n%u %u\n255\n", width, height);
  for (size_t i = 0; i < size_t(width) * height; ++i) std::fwrite(&rgba[i * 4], 1, 3, f);
  return std::fclose(f) == 0;
}

}  // namespace gs
