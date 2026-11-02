#include "gs/background.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "shaders/background_frag.h"
#include "shaders/background_vert.h"

namespace gs {
namespace {

struct Params {
  float axis_x[4];
  float axis_y[4];
  float axis_z[4];
  float eye[4];
  float ground[4];
  float extra[4];
  uint32_t info[4];
};
static_assert(sizeof(Params) == 112, "push constants must stay within 128 bytes");

}  // namespace

bool parse_backdrop(std::string_view name, Backdrop* backdrop) {
  static constexpr std::pair<std::string_view, Backdrop> kNames[] = {
      {"plain", Backdrop::Plain},     {"studio", Backdrop::Studio}, {"dark", Backdrop::Dark},
      {"park", Backdrop::Park},       {"checker", Backdrop::Checker}, {"sunset", Backdrop::Sunset},
      {"night", Backdrop::Night},     {"photo", Backdrop::Photo},     {"panorama", Backdrop::Panorama},
  };
  for (const auto& [n, b] : kNames) {
    if (n == name) {
      *backdrop = b;
      return true;
    }
  }
  return false;
}

Ground find_ground(const SplatScene& scene) {
  Ground g;
  const size_t n = scene.count;
  if (n == 0) return g;
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = scene.positions[i * 3 + 1];
  const size_t lo = n / 200, hi = n - 1 - n / 200;  // ignore stray splats
  std::nth_element(v.begin(), v.begin() + lo, v.end());
  g.low = v[lo];
  std::nth_element(v.begin(), v.begin() + hi, v.end());
  g.high = v[hi];
  for (int axis = 0; axis < 2; ++axis) {
    for (size_t i = 0; i < n; ++i) v[i] = scene.positions[i * 3 + (axis == 0 ? 0 : 2)];
    std::nth_element(v.begin(), v.begin() + n / 2, v.end());
    g.centre[axis] = v[n / 2];
  }
  g.found = g.high > g.low;
  return g;
}

BackgroundRenderer::BackgroundRenderer(VkContext& ctx, VkRenderPass render_pass) : ctx_(ctx) {
  VkDevice device = ctx_.device();
  VkDescriptorSetLayoutBinding binding{};
  binding.binding = 0;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  binding.descriptorCount = 1;
  binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  set_info.bindingCount = 1;
  set_info.pBindings = &binding;
  GS_VK_CHECK(vkCreateDescriptorSetLayout(device, &set_info, nullptr, &set_layout_));

  VkPushConstantRange push{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(Params)};
  VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layout_info.setLayoutCount = 1;
  layout_info.pSetLayouts = &set_layout_;
  layout_info.pushConstantRangeCount = 1;
  layout_info.pPushConstantRanges = &push;
  GS_VK_CHECK(vkCreatePipelineLayout(device, &layout_info, nullptr, &layout_));

  VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
  VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pool_info.maxSets = 1;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes = &pool_size;
  GS_VK_CHECK(vkCreateDescriptorPool(device, &pool_info, nullptr, &pool_));
  VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  alloc.descriptorPool = pool_;
  alloc.descriptorSetCount = 1;
  alloc.pSetLayouts = &set_layout_;
  GS_VK_CHECK(vkAllocateDescriptorSets(device, &alloc, &set_));

  VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  sampler.magFilter = VK_FILTER_LINEAR;
  sampler.minFilter = VK_FILTER_LINEAR;
  sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;  // panoramas wrap around
  sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sampler.maxLod = 0.0f;
  GS_VK_CHECK(vkCreateSampler(device, &sampler, nullptr, &sampler_));

  // Something valid to bind until a picture is set.
  const uint8_t grey[4] = {20, 23, 28, 255};
  set_picture(grey, 1, 1);
  create_pipeline(render_pass);
}

BackgroundRenderer::~BackgroundRenderer() {
  VkDevice device = ctx_.device();
  if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline_, nullptr);
  destroy_picture();
  if (sampler_ != VK_NULL_HANDLE) vkDestroySampler(device, sampler_, nullptr);
  if (pool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, pool_, nullptr);
  if (layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, layout_, nullptr);
  if (set_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, set_layout_, nullptr);
}

void BackgroundRenderer::set_render_pass(VkRenderPass render_pass) {
  if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), pipeline_, nullptr);
  pipeline_ = VK_NULL_HANDLE;
  create_pipeline(render_pass);
}

void BackgroundRenderer::destroy_picture() {
  VkDevice device = ctx_.device();
  if (view_ != VK_NULL_HANDLE) vkDestroyImageView(device, view_, nullptr);
  if (image_ != VK_NULL_HANDLE) vkDestroyImage(device, image_, nullptr);
  if (memory_ != VK_NULL_HANDLE) vkFreeMemory(device, memory_, nullptr);
  view_ = VK_NULL_HANDLE;
  image_ = VK_NULL_HANDLE;
  memory_ = VK_NULL_HANDLE;
}

void BackgroundRenderer::set_picture(const uint8_t* rgba, uint32_t width, uint32_t height) {
  if (width == 0 || height == 0) throw std::runtime_error("빈 그림");
  VkDevice device = ctx_.device();
  destroy_picture();
  VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.format = VK_FORMAT_R8G8B8A8_UNORM;  // values as stored: the screen is UNORM too
  image_info.extent = {width, height, 1};
  image_info.mipLevels = 1;
  image_info.arrayLayers = 1;
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  GS_VK_CHECK(vkCreateImage(device, &image_info, nullptr, &image_));
  VkMemoryRequirements req;
  vkGetImageMemoryRequirements(device, image_, &req);
  VkMemoryAllocateInfo mem{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mem.allocationSize = req.size;
  mem.memoryTypeIndex = ctx_.find_memory_type(req.memoryTypeBits, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  GS_VK_CHECK(vkAllocateMemory(device, &mem, nullptr, &memory_));
  GS_VK_CHECK(vkBindImageMemory(device, image_, memory_, 0));

  const VkDeviceSize bytes = VkDeviceSize(width) * height * 4;
  Buffer staging = ctx_.create_buffer(bytes, MemoryKind::Upload);
  std::memcpy(staging.mapped(), rgba, size_t(bytes));
  ctx_.run([&](VkCommandBuffer cmd) {
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image_;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(cmd, staging.handle(), image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);
  });

  VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  view_info.image = image_;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
  view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  GS_VK_CHECK(vkCreateImageView(device, &view_info, nullptr, &view_));

  VkDescriptorImageInfo image_desc{sampler_, view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = set_;
  write.dstBinding = 0;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  write.pImageInfo = &image_desc;
  vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
  picture_aspect_ = float(width) / float(height);
}

void BackgroundRenderer::create_pipeline(VkRenderPass render_pass) {
  VkDevice device = ctx_.device();
  VkShaderModule vert = ctx_.create_shader_module(shaders::background_vert);
  VkShaderModule frag = VK_NULL_HANDLE;
  try {
    frag = ctx_.create_shader_module(shaders::background_frag);
  } catch (...) {
    vkDestroyShaderModule(device, vert, nullptr);
    throw;
  }
  VkPipelineShaderStageCreateInfo stages[2] = {};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vert;
  stages[0].pName = "main";
  stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = frag;
  stages[1].pName = "main";

  VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  VkPipelineInputAssemblyStateCreateInfo input_assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  viewport.viewportCount = 1;
  viewport.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  raster.polygonMode = VK_POLYGON_MODE_FILL;
  raster.cullMode = VK_CULL_MODE_NONE;
  raster.lineWidth = 1.0f;
  VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineColorBlendAttachmentState blend{};
  blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                         VK_COLOR_COMPONENT_A_BIT;
  VkPipelineColorBlendStateCreateInfo color_blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  color_blend.attachmentCount = 1;
  color_blend.pAttachments = &blend;
  const VkDynamicState dynamic_states[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dynamic.dynamicStateCount = 2;
  dynamic.pDynamicStates = dynamic_states;

  VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  info.stageCount = 2;
  info.pStages = stages;
  info.pVertexInputState = &vertex_input;
  info.pInputAssemblyState = &input_assembly;
  info.pViewportState = &viewport;
  info.pRasterizationState = &raster;
  info.pMultisampleState = &multisample;
  info.pColorBlendState = &color_blend;
  info.pDynamicState = &dynamic;
  info.layout = layout_;
  info.renderPass = render_pass;
  const VkResult result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline_);
  vkDestroyShaderModule(device, vert, nullptr);
  vkDestroyShaderModule(device, frag, nullptr);
  if (result != VK_SUCCESS) throw VulkanError("vkCreateGraphicsPipelines", result);
}

void BackgroundRenderer::record(VkCommandBuffer cmd, Backdrop backdrop, const Ground& ground, const Camera& camera,
                                bool flip_y, uint32_t rotation) {
  // Display = scene with y and z negated when flipped; camera -> display is
  // the transpose of display -> camera, whose rows are the camera rows with
  // their y and z columns negated.
  const float s = flip_y ? -1.0f : 1.0f;
  Params p{};
  float* axes[3] = {p.axis_x, p.axis_y, p.axis_z};
  for (int r = 0; r < 3; ++r) {
    axes[r][0] = camera.rotation[r * 3 + 0];
    axes[r][1] = camera.rotation[r * 3 + 1] * s;
    axes[r][2] = camera.rotation[r * 3 + 2] * s;
  }
  p.axis_x[3] = camera.fx;
  p.axis_y[3] = camera.fy;
  p.axis_z[3] = camera.cx;
  p.eye[0] = camera.position[0];
  p.eye[1] = camera.position[1] * s;
  p.eye[2] = camera.position[2] * s;
  p.eye[3] = camera.cy;
  // The floor just under the model's lowest point as displayed.
  const float height = ground.found ? ground.high - ground.low : 1.0f;
  const float bottom = ground.found ? (flip_y ? -ground.high : ground.low) : -1.0f;
  p.ground[0] = bottom - 0.005f * height;
  p.ground[1] = ground.centre[0];
  p.ground[2] = ground.centre[1] * s;
  p.ground[3] = 0.2f * height;
  p.extra[0] = picture_aspect_;
  p.extra[1] = height;
  p.info[0] = camera.width;
  p.info[1] = camera.height;
  p.info[2] = rotation & 3u;
  p.info[3] = static_cast<uint32_t>(backdrop);

  const bool swap = (rotation & 1u) != 0;
  const uint32_t fb_width = swap ? camera.height : camera.width;
  const uint32_t fb_height = swap ? camera.width : camera.height;
  VkViewport viewport{0.0f, 0.0f, float(fb_width), float(fb_height), 0.0f, 1.0f};
  VkRect2D scissor{{0, 0}, {fb_width, fb_height}};
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &set_, 0, nullptr);
  vkCmdSetViewport(cmd, 0, 1, &viewport);
  vkCmdSetScissor(cmd, 0, 1, &scissor);
  vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(p), &p);
  vkCmdDraw(cmd, 3, 1, 0, 0);
}

}  // namespace gs
