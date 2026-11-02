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
  float env[4];
  float key[4];
  uint32_t info[4];
};
static_assert(sizeof(Params) == 128, "push constants must stay within 128 bytes");

constexpr float kPi = 3.14159265358979f;
constexpr float kLuma[3] = {0.2126f, 0.7152f, 0.0722f};

float srgb_to_linear(float c) { return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f); }

// Irradiance SH of the radiance `sample(x, y, z, rgb)` over the sphere (display
// directions), normalised to mean luminance `brightness`; plus the main light.
template <typename F>
EnvLight project(F&& sample, float brightness) {
  constexpr int kW = 128, kH = 64;
  double coeffs[9][3] = {};
  struct Cell {
    float lum, solid, x, y, z;
    float rgb[3];
  };
  std::vector<Cell> cells;
  cells.reserve(kW * kH);
  double total = 0;
  for (int j = 0; j < kH; ++j) {
    const float theta = (float(j) + 0.5f) / kH * kPi;
    const float solid = (2.0f * kPi / kW) * (kPi / kH) * std::sin(theta);
    for (int i = 0; i < kW; ++i) {
      const float phi = ((float(i) + 0.5f) / kW - 0.5f) * 2.0f * kPi;
      const float x = std::sin(theta) * std::sin(phi), y = std::cos(theta), z = -std::sin(theta) * std::cos(phi);
      float rgb[3];
      sample(x, y, z, rgb);
      const float basis[9] = {0.282095f,         0.488603f * y,     0.488603f * z,
                              0.488603f * x,     1.092548f * x * y, 1.092548f * y * z,
                              0.315392f * (3 * z * z - 1), 1.092548f * x * z, 0.546274f * (x * x - y * y)};
      for (int k = 0; k < 9; ++k) {
        for (int c = 0; c < 3; ++c) coeffs[k][c] += double(basis[k]) * rgb[c] * solid;
      }
      const float lum = kLuma[0] * rgb[0] + kLuma[1] * rgb[1] + kLuma[2] * rgb[2];
      cells.push_back({lum, solid, x, y, z, {rgb[0], rgb[1], rgb[2]}});
      total += double(lum) * solid;
    }
  }
  EnvLight light;
  const float band[9] = {kPi, 2 * kPi / 3, 2 * kPi / 3, 2 * kPi / 3, kPi / 4, kPi / 4, kPi / 4, kPi / 4, kPi / 4};
  const double mean = (kLuma[0] * coeffs[0][0] + kLuma[1] * coeffs[0][1] + kLuma[2] * coeffs[0][2]) * kPi * 0.282095;
  const double scale = mean > 1e-9 ? brightness / mean : 0.0;
  for (int k = 0; k < 9; ++k) {
    for (int c = 0; c < 3; ++c) light.sh[k * 3 + c] = float(coeffs[k][c] * band[k] * scale);
  }
  // The brightest 2% of the sphere: where the main light comes from.
  std::sort(cells.begin(), cells.end(), [](const Cell& a, const Cell& b) { return a.lum > b.lum; });
  double area = 0, power = 0, dx = 0, dy = 0, dz = 0, rgb[3] = {0, 0, 0};
  for (const Cell& c : cells) {
    if (area > 0.02 * 4 * kPi) break;
    area += c.solid;
    const double p = double(c.lum) * c.solid;
    power += p;
    dx += c.x * p;
    dy += c.y * p;
    dz += c.z * p;
    for (int k = 0; k < 3; ++k) rgb[k] += double(c.rgb[k]) * c.solid;
  }
  const double rgb_lum = kLuma[0] * rgb[0] + kLuma[1] * rgb[1] + kLuma[2] * rgb[2];
  if (rgb_lum > 0) {
    for (int k = 0; k < 3; ++k) light.key_color[k] = float(rgb[k] / rgb_lum);
  }
  const double len = std::sqrt(dx * dx + dy * dy + dz * dz);
  if (len > 0) {
    light.key[0] = float(dx / len);
    light.key[1] = float(dy / len);
    light.key[2] = float(dz / len);
  }
  light.key_share = total > 0 ? float(power / total) : 0.0f;
  light.on = scale > 0;
  return light;
}

}  // namespace

bool parse_backdrop(std::string_view name, Backdrop* backdrop) {
  static constexpr std::pair<std::string_view, Backdrop> kNames[] = {
      {"plain", Backdrop::Plain},
      {"studio", Backdrop::Studio},
      {"photo", Backdrop::Photo},
      {"panorama", Backdrop::Panorama},
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
  sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;  // panoramas wrap around
  sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sampler.maxLod = VK_LOD_CLAMP_NONE;
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
  // Mip levels by 2x2 averages, for blurring what is out of focus.
  std::vector<std::vector<uint8_t>> levels;
  std::vector<std::pair<uint32_t, uint32_t>> sizes;
  levels.emplace_back(rgba, rgba + size_t(width) * height * 4);
  sizes.emplace_back(width, height);
  while ((sizes.back().first > 1 || sizes.back().second > 1) && levels.size() < 12) {
    const auto [w, h] = sizes.back();
    const uint32_t nw = std::max(1u, w / 2), nh = std::max(1u, h / 2);
    const std::vector<uint8_t>& src = levels.back();
    std::vector<uint8_t> dst(size_t(nw) * nh * 4);
    for (uint32_t y = 0; y < nh; ++y) {
      for (uint32_t x = 0; x < nw; ++x) {
        for (int c = 0; c < 4; ++c) {
          uint32_t sum = 0;
          for (uint32_t dy = 0; dy < 2; ++dy) {
            for (uint32_t dx = 0; dx < 2; ++dx) {
              const uint32_t sx = std::min(w - 1, x * 2 + dx), sy = std::min(h - 1, y * 2 + dy);
              sum += src[(size_t(sy) * w + sx) * 4 + c];
            }
          }
          dst[(size_t(y) * nw + x) * 4 + c] = uint8_t((sum + 2) / 4);
        }
      }
    }
    levels.push_back(std::move(dst));
    sizes.emplace_back(nw, nh);
  }
  const uint32_t mip_levels = uint32_t(levels.size());

  VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.format = VK_FORMAT_R8G8B8A8_UNORM;  // values as stored: the screen is UNORM too
  image_info.extent = {width, height, 1};
  image_info.mipLevels = mip_levels;
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

  VkDeviceSize bytes = 0;
  for (const auto& level : levels) bytes += level.size();
  Buffer staging = ctx_.create_buffer(bytes, MemoryKind::Upload);
  std::vector<VkBufferImageCopy> copies;
  VkDeviceSize offset = 0;
  for (uint32_t m = 0; m < mip_levels; ++m) {
    std::memcpy(static_cast<uint8_t*>(staging.mapped()) + offset, levels[m].data(), levels[m].size());
    VkBufferImageCopy copy{};
    copy.bufferOffset = offset;
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m, 0, 1};
    copy.imageExtent = {sizes[m].first, sizes[m].second, 1};
    copies.push_back(copy);
    offset += levels[m].size();
  }
  ctx_.run([&](VkCommandBuffer cmd) {
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image_;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mip_levels, 0, 1};
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);
    vkCmdCopyBufferToImage(cmd, staging.handle(), image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           uint32_t(copies.size()), copies.data());
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
  view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mip_levels, 0, 1};
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

void BackgroundRenderer::record(VkCommandBuffer cmd, const BackdropView& view, const Camera& camera, bool flip_y,
                                uint32_t rotation) {
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
  const Ground& ground = view.ground;
  const float height = ground.found ? ground.high - ground.low : 1.0f;
  const float bottom = ground.found ? (flip_y ? -ground.high : ground.low) : -1.0f;
  p.ground[0] = bottom - 0.005f * height;
  p.ground[1] = ground.centre[0];
  p.ground[2] = ground.centre[1] * s;
  p.ground[3] = height;
  // Metres for a person-sized model; otherwise the photo is scaled to the model.
  const float metre = height > 1.0f && height < 2.4f ? 1.0f : height / 1.7f;
  p.env[0] = view.yaw_deg * kPi / 180.0f;
  p.env[1] = view.camera_height * metre;
  p.env[2] = view.radius * metre;
  p.env[3] = view.blur;
  p.key[0] = view.light.key[0];
  p.key[1] = view.light.key[1];
  p.key[2] = view.light.key[2];
  p.key[3] = view.light.on ? std::clamp(1.1f * view.light.key_share - 0.1f, 0.0f, 0.75f) : 0.0f;
  p.info[0] = camera.width;
  p.info[1] = camera.height;
  p.info[2] = (rotation & 3u) | (static_cast<uint32_t>(view.kind) << 8);
  std::memcpy(&p.info[3], &picture_aspect_, 4);

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

EnvLight light_from_picture(const uint8_t* rgba, uint32_t width, uint32_t height, bool panorama, float yaw_deg) {
  // A small linear copy to sample from.
  constexpr uint32_t kW = 256, kH = 128;
  std::vector<float> small(size_t(kW) * kH * 3, 0.0f);
  std::vector<float> weight(size_t(kW) * kH, 0.0f);
  for (uint32_t y = 0; y < height; ++y) {
    const uint32_t sy = y * kH / height;
    for (uint32_t x = 0; x < width; ++x) {
      const uint32_t sx = x * kW / width;
      const uint8_t* p = &rgba[(size_t(y) * width + x) * 4];
      float* d = &small[(size_t(sy) * kW + sx) * 3];
      for (int c = 0; c < 3; ++c) d[c] += srgb_to_linear(p[c] / 255.0f);
      weight[size_t(sy) * kW + sx] += 1.0f;
    }
  }
  double mean = 0;
  for (size_t i = 0; i < weight.size(); ++i) {
    for (int c = 0; c < 3; ++c) small[i * 3 + c] /= std::max(weight[i], 1.0f);
    mean += kLuma[0] * small[i * 3] + kLuma[1] * small[i * 3 + 1] + kLuma[2] * small[i * 3 + 2];
  }
  mean /= double(weight.size());
  auto at = [&](float u, float v, float rgb[3]) {
    const uint32_t x = std::min(kW - 1, uint32_t(std::max(0.0f, u) * kW));
    const uint32_t y = std::min(kH - 1, uint32_t(std::max(0.0f, v) * kH));
    for (int c = 0; c < 3; ++c) rgb[c] = small[(size_t(y) * kW + x) * 3 + c];
  };
  // How bright the place is: dark pictures light the model less.
  const float brightness = std::clamp(float(std::pow(mean / 0.18, 0.6)), 0.45f, 1.15f);
  const float yaw = yaw_deg * kPi / 180.0f;
  if (panorama) {
    return project(
        [&](float x, float y, float z, float rgb[3]) {
          float u = std::atan2(x, -z) / (2 * kPi) + 0.5f + yaw / (2 * kPi);
          u -= std::floor(u);
          at(u, std::acos(std::clamp(y, -1.0f, 1.0f)) / kPi, rgb);
        },
        brightness);
  }
  // A flat photo: stretched over the sphere as if it surrounded the model.
  return project(
      [&](float x, float y, float z, float rgb[3]) {
        const float u = 0.5f + std::atan2(x, std::abs(z)) / kPi * 0.8f;
        at(u, 0.5f - std::asin(std::clamp(y, -1.0f, 1.0f)) / kPi, rgb);
      },
      brightness);
}

EnvLight studio_light() {
  // Soft white all round, a big softbox to the upper left in front.
  const float box[3] = {-0.45f, 0.55f, 0.70f};
  const float norm = std::sqrt(box[0] * box[0] + box[1] * box[1] + box[2] * box[2]);
  return project(
      [&](float x, float y, float z, float rgb[3]) {
        const float c = (x * box[0] + y * box[1] + z * box[2]) / norm;
        const float v = 0.55f + (c > 0.88f ? 9.0f : 0.0f) + 0.15f * std::max(y, 0.0f);
        rgb[0] = rgb[1] = rgb[2] = v;
      },
      1.0f);
}

SplatLighting splat_lighting(const EnvLight& light, const Ground& ground, bool flip_y, float strength) {
  SplatLighting l;
  if (!light.on || !ground.found || strength <= 0.0f) return l;
  for (int k = 0; k < 9; ++k) {
    for (int c = 0; c < 3; ++c) l.sh[k][c] = light.sh[k * 3 + c];
  }
  l.centre[0] = ground.centre[0];
  l.centre[1] = 0.5f * (ground.low + ground.high);
  l.centre[2] = ground.centre[1];
  l.centre[3] = std::clamp(strength, 0.0f, 1.0f);
  l.display[0] = flip_y ? -1.0f : 1.0f;
  for (int k = 0; k < 3; ++k) {
    l.key[k] = light.key[k];
    // A strong single light (sun, window) outlines the model; soft rooms hardly do.
    l.rim[k] = light.key_color[k] * 1.6f * std::clamp(light.key_share, 0.0f, 1.0f);
  }
  return l;
}

}  // namespace gs
