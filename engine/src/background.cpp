#include "gs/background.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "shaders/background_frag.h"
#include "shaders/background_vert.h"

namespace gs {
namespace {

struct Params {
  float axis_x[4];
  float axis_y[4];
  float axis_z[4];
  float eye[4];
  float zenith[4];
  float horizon[4];
  uint32_t info[4];
};
static_assert(sizeof(Params) <= 128, "push constants must stay within 128 bytes");

constexpr float kPi = 3.14159265358979f;
constexpr float kLuma[3] = {0.2126f, 0.7152f, 0.0722f};

// Irradiance SH of the radiance `sample(x, y, z, rgb)` over the sphere (display
// directions), normalised to mean luminance `brightness`; plus the main light.
EnvLight project(const std::function<void(float, float, float, float*)>& sample, float brightness) {
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

EnvLight light_from_directions(const std::function<void(float x, float y, float z, float rgb[3])>& radiance,
                               float brightness) {
  return project(radiance, brightness);
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
  VkPushConstantRange push{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(Params)};
  VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layout_info.pushConstantRangeCount = 1;
  layout_info.pPushConstantRanges = &push;
  GS_VK_CHECK(vkCreatePipelineLayout(ctx_.device(), &layout_info, nullptr, &layout_));
  create_pipeline(render_pass);
}

BackgroundRenderer::~BackgroundRenderer() {
  VkDevice device = ctx_.device();
  if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline_, nullptr);
  if (layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, layout_, nullptr);
}

void BackgroundRenderer::set_render_pass(VkRenderPass render_pass) {
  if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), pipeline_, nullptr);
  pipeline_ = VK_NULL_HANDLE;
  create_pipeline(render_pass);
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
  p.eye[3] = camera.cy;
  for (int k = 0; k < 3; ++k) {
    p.zenith[k] = view.zenith[k];
    p.horizon[k] = view.horizon[k];
  }
  p.info[0] = camera.width;
  p.info[1] = camera.height;
  p.info[2] = (rotation & 3u) | (static_cast<uint32_t>(view.kind) << 8);

  const bool swap = (rotation & 1u) != 0;
  const uint32_t fb_width = swap ? camera.height : camera.width;
  const uint32_t fb_height = swap ? camera.width : camera.height;
  VkViewport viewport{0.0f, 0.0f, float(fb_width), float(fb_height), 0.0f, 1.0f};
  VkRect2D scissor{{0, 0}, {fb_width, fb_height}};
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
  vkCmdSetViewport(cmd, 0, 1, &viewport);
  vkCmdSetScissor(cmd, 0, 1, &scissor);
  vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(p), &p);
  vkCmdDraw(cmd, 3, 1, 0, 0);
}

SplatLighting splat_lighting(const EnvLight& light, const Ground& ground, bool flip_y, float strength,
                             uint32_t count) {
  SplatLighting l;
  if (!ground.found) return l;
  l.centre[0] = ground.centre[0];
  l.centre[1] = 0.5f * (ground.low + ground.high);
  l.centre[2] = ground.centre[1];
  l.display[0] = flip_y ? -1.0f : 1.0f;
  l.limit[0] = count;
  if (count > 0) {
    // A place around the model: cut what stands between camera and model.
    l.cut[0] = flip_y ? ground.high : ground.low;
    l.cut[1] = 0.6f;
    l.cut[2] = 1.0f;
  }
  if (!light.on || strength <= 0.0f) return l;
  for (int k = 0; k < 9; ++k) {
    for (int c = 0; c < 3; ++c) l.sh[k][c] = light.sh[k * 3 + c];
  }
  l.centre[3] = std::clamp(strength, 0.0f, 1.0f);
  for (int k = 0; k < 3; ++k) {
    l.key[k] = light.key[k];
    // A strong single light (sun, window) outlines the model; soft rooms hardly do.
    l.rim[k] = light.key_color[k] * 1.6f * std::clamp(light.key_share, 0.0f, 1.0f);
  }
  return l;
}

}  // namespace gs
