#include "gs/splat_renderer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "shaders/splat_frag.h"
#include "shaders/splat_preprocess.h"
#include "shaders/splat_vert.h"

namespace gs {
namespace {

// Mirrors the push-constant block of splat_preprocess.comp.
struct PreprocessParams {
  float rot0[4];  // scene -> camera rotation row 0, w = translation x
  float rot1[4];
  float rot2[4];
  float cam_pos[4];
  float intrinsics[4];  // fx, fy, cx, cy
  uint32_t info[4];     // width, height, count, sh_degree | antialiased << 8
};
static_assert(sizeof(PreprocessParams) == 96, "push constant layout");

// Mirrors the push-constant block of splat.vert.
struct DrawParams {
  float viewport[2];
  uint32_t rotation;
  uint32_t pad;
};

constexpr uint32_t kPreprocessGroup = 256;
constexpr uint32_t kDepthKeyBits = 16;

uint16_t float_to_half(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const uint32_t exponent = (x >> 23) & 0xffu;
  uint32_t mantissa = x & 0x7fffffu;
  if (exponent == 0xff) return static_cast<uint16_t>(sign | 0x7c00u | (mantissa ? 0x200u : 0u));
  const int e = static_cast<int>(exponent) - 127 + 15;
  if (e >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
  if (e <= 0) {
    if (e < -10) return static_cast<uint16_t>(sign);
    mantissa |= 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - e);
    uint32_t half = mantissa >> shift;
    const uint32_t rest = mantissa & ((1u << shift) - 1);
    const uint32_t halfway = 1u << (shift - 1);
    if (rest > halfway || (rest == halfway && (half & 1u))) ++half;
    return static_cast<uint16_t>(sign | half);
  }
  uint32_t half = sign | (static_cast<uint32_t>(e) << 10) | (mantissa >> 13);
  const uint32_t rest = mantissa & 0x1fffu;
  if (rest > 0x1000u || (rest == 0x1000u && (half & 1u))) ++half;  // carry into the exponent is correct
  return static_cast<uint16_t>(half);
}

uint64_t sh_uints_per_splat(int degree) { return (uint64_t(sh_rest_coeffs(degree)) * 3 + 1) / 2; }

}  // namespace

std::unique_ptr<GpuScene> upload_scene(VkContext& ctx, const SplatScene& scene) {
  if (scene.count == 0) throw std::runtime_error("장면이 비어 있음");
  const uint64_t max_range = ctx.info().properties.limits.maxStorageBufferRange;
  const uint64_t n = scene.count;
  if (n * 64 > max_range) {
    throw std::runtime_error("가우시안이 너무 많음: " + std::to_string(n) + "개 (이 기기 최대 " +
                             std::to_string(max_range / 64) + "개)");
  }
  int degree = std::min(scene.sh_degree, 3);
  while (degree > 0 && n * sh_uints_per_splat(degree) * 4 > max_range) --degree;

  auto gpu = std::make_unique<GpuScene>();
  gpu->count = scene.count;
  gpu->sh_degree = degree;
  gpu->antialiased = scene.antialiased;

  {
    std::vector<float> splats(n * 16, 0.0f);
    for (uint64_t i = 0; i < n; ++i) {
      float* d = &splats[i * 16];
      for (int k = 0; k < 3; ++k) d[k] = scene.positions[i * 3 + k];
      d[3] = 1.0f / (1.0f + std::exp(-scene.opacity_logits[i]));
      const float* q = &scene.rotations[i * 4];
      const float len = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
      if (len > 1e-12f) {
        for (int k = 0; k < 4; ++k) d[4 + k] = q[k] / len;
      } else {
        d[4] = 1.0f;
      }
      for (int k = 0; k < 3; ++k) d[8 + k] = std::exp(scene.log_scales[i * 3 + k]);
      for (int k = 0; k < 3; ++k) d[12 + k] = scene.sh_dc[i * 3 + k];
    }
    gpu->splats = ctx.create_buffer(n * 64, MemoryKind::Device);
    ctx.upload(gpu->splats, splats.data(), n * 64);
  }

  const uint64_t stride = sh_uints_per_splat(degree);
  if (stride > 0) {
    const int scene_coeffs = sh_rest_coeffs(scene.sh_degree);
    const int coeffs = sh_rest_coeffs(degree);
    std::vector<uint32_t> packed(n * stride, 0u);
    for (uint64_t i = 0; i < n; ++i) {
      for (int j = 0; j < coeffs; ++j) {
        for (int c = 0; c < 3; ++c) {
          const uint32_t h = static_cast<uint32_t>(j * 3 + c);
          const uint16_t half = float_to_half(scene.sh_rest[(i * scene_coeffs + j) * 3 + c]);
          packed[i * stride + h / 2] |= uint32_t(half) << (16 * (h & 1u));
        }
      }
    }
    gpu->sh_rest = ctx.create_buffer(n * stride * 4, MemoryKind::Device);
    ctx.upload(gpu->sh_rest, packed.data(), n * stride * 4);
  } else {
    gpu->sh_rest = ctx.create_buffer(4, MemoryKind::Device);
  }

  gpu->projected = ctx.create_buffer(n * 32, MemoryKind::Device);
  gpu->keys = ctx.create_buffer(n * 4, MemoryKind::Device);
  gpu->order = ctx.create_buffer(n * 4, MemoryKind::Device);
  gpu->draw_args = ctx.create_buffer(16, MemoryKind::Upload, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
  const uint32_t args[4] = {4, 0, 0, 0};  // vertexCount, instanceCount, firstVertex, firstInstance
  std::memcpy(gpu->draw_args.mapped(), args, sizeof(args));
  return gpu;
}

SplatRenderer::SplatRenderer(VkContext& ctx, VkRenderPass render_pass)
    : ctx_(ctx),
      preprocess_(ctx.create_compute_pipeline(shaders::splat_preprocess, 6, sizeof(PreprocessParams))),
      sort_(ctx) {
  VkDevice device = ctx_.device();
  pool_ = ctx_.create_descriptor_pool(16, 96);

  VkDescriptorSetLayoutBinding bindings[2] = {};
  for (uint32_t i = 0; i < 2; ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  }
  VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  set_info.bindingCount = 2;
  set_info.pBindings = bindings;
  GS_VK_CHECK(vkCreateDescriptorSetLayout(device, &set_info, nullptr, &draw_set_layout_));

  VkPushConstantRange range{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(DrawParams)};
  VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layout_info.setLayoutCount = 1;
  layout_info.pSetLayouts = &draw_set_layout_;
  layout_info.pushConstantRangeCount = 1;
  layout_info.pPushConstantRanges = &range;
  GS_VK_CHECK(vkCreatePipelineLayout(device, &layout_info, nullptr, &draw_layout_));

  create_graphics_pipeline(render_pass);
}

SplatRenderer::~SplatRenderer() {
  VkDevice device = ctx_.device();
  if (draw_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device, draw_pipeline_, nullptr);
  if (draw_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, draw_layout_, nullptr);
  if (draw_set_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, draw_set_layout_, nullptr);
  if (pool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, pool_, nullptr);
}

void SplatRenderer::set_render_pass(VkRenderPass render_pass) {
  if (draw_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), draw_pipeline_, nullptr);
  draw_pipeline_ = VK_NULL_HANDLE;
  create_graphics_pipeline(render_pass);
}

void SplatRenderer::create_graphics_pipeline(VkRenderPass render_pass) {
  VkDevice device = ctx_.device();
  VkShaderModule vert = ctx_.create_shader_module(shaders::splat_vert);
  VkShaderModule frag = VK_NULL_HANDLE;
  try {
    frag = ctx_.create_shader_module(shaders::splat_frag);
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
  VkPipelineInputAssemblyStateCreateInfo input_assembly{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
  VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  viewport.viewportCount = 1;
  viewport.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  raster.polygonMode = VK_POLYGON_MODE_FILL;
  raster.cullMode = VK_CULL_MODE_NONE;
  raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  raster.lineWidth = 1.0f;
  VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  // Premultiplied "over", drawn back to front.
  VkPipelineColorBlendAttachmentState blend{};
  blend.blendEnable = VK_TRUE;
  blend.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
  blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  blend.colorBlendOp = VK_BLEND_OP_ADD;
  blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  blend.alphaBlendOp = VK_BLEND_OP_ADD;
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
  info.layout = draw_layout_;
  info.renderPass = render_pass;
  info.subpass = 0;
  VkResult result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &draw_pipeline_);
  vkDestroyShaderModule(device, vert, nullptr);
  vkDestroyShaderModule(device, frag, nullptr);
  if (result != VK_SUCCESS) throw VulkanError("vkCreateGraphicsPipelines", result);
}

void SplatRenderer::attach(GpuScene* scene) {
  GS_VK_CHECK(vkResetDescriptorPool(ctx_.device(), pool_, 0));
  scene_ = scene;
  preprocess_set_ = VK_NULL_HANDLE;
  draw_set_ = VK_NULL_HANDLE;
  sort_binding_ = {};
  if (scene == nullptr) return;
  sort_.reserve(scene->count);
  preprocess_set_ = ctx_.allocate_set(
      pool_, preprocess_.set_layout(),
      {&scene->splats, &scene->sh_rest, &scene->projected, &scene->keys, &scene->order, &scene->draw_args});
  draw_set_ = ctx_.allocate_set(pool_, draw_set_layout_, {&scene->projected, &scene->order});
  sort_binding_ = sort_.bind(pool_, scene->keys, scene->order);
}

void SplatRenderer::record_prepare(VkCommandBuffer cmd, const Camera& camera) {
  if (scene_ == nullptr) return;

  // The previous frame's draw may still read these buffers: wait before overwriting.
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                       0, nullptr, 0, nullptr);
  vkCmdFillBuffer(cmd, scene_->draw_args.handle(), 4, 4, 0);  // instanceCount = 0
  VkContext::barrier(cmd);

  PreprocessParams p{};
  for (int k = 0; k < 3; ++k) {
    p.rot0[k] = camera.rotation[0 + k];
    p.rot1[k] = camera.rotation[3 + k];
    p.rot2[k] = camera.rotation[6 + k];
    p.cam_pos[k] = camera.position[k];
  }
  p.rot0[3] = camera.translation[0];
  p.rot1[3] = camera.translation[1];
  p.rot2[3] = camera.translation[2];
  p.intrinsics[0] = camera.fx;
  p.intrinsics[1] = camera.fy;
  p.intrinsics[2] = camera.cx;
  p.intrinsics[3] = camera.cy;
  p.info[0] = camera.width;
  p.info[1] = camera.height;
  p.info[2] = scene_->count;
  p.info[3] = static_cast<uint32_t>(scene_->sh_degree) | (scene_->antialiased ? 0x100u : 0u);
  ctx_.dispatch(cmd, preprocess_, preprocess_set_, p, div_up(scene_->count, kPreprocessGroup));
  VkContext::barrier(cmd);

  sort_.record(cmd, sort_binding_, scene_->keys, scene_->order, scene_->count, kDepthKeyBits);

  VkMemoryBarrier to_draw{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  to_draw.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  to_draw.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                       &to_draw, 0, nullptr, 0, nullptr);
}

void SplatRenderer::record_draw(VkCommandBuffer cmd, uint32_t width, uint32_t height, uint32_t rotation) {
  if (scene_ == nullptr) return;
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw_pipeline_);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw_layout_, 0, 1, &draw_set_, 0, nullptr);
  const bool swap = (rotation & 1u) != 0;
  const uint32_t fb_width = swap ? height : width;
  const uint32_t fb_height = swap ? width : height;
  VkViewport viewport{0.0f, 0.0f, float(fb_width), float(fb_height), 0.0f, 1.0f};
  VkRect2D scissor{{0, 0}, {fb_width, fb_height}};
  vkCmdSetViewport(cmd, 0, 1, &viewport);
  vkCmdSetScissor(cmd, 0, 1, &scissor);
  const DrawParams params{{float(width), float(height)}, rotation & 3u, 0};
  vkCmdPushConstants(cmd, draw_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(params), &params);
  vkCmdDrawIndirect(cmd, scene_->draw_args.handle(), 0, 1, sizeof(VkDrawIndirectCommand));
}

uint32_t SplatRenderer::visible_count() const {
  if (scene_ == nullptr) return 0;
  uint32_t args[4];
  std::memcpy(args, scene_->draw_args.mapped(), sizeof(args));
  return args[1];
}

}  // namespace gs
