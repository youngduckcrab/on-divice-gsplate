#include "gs/trainer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "gs/log.h"
#include "shaders/scan_blocks.h"
#include "shaders/scan_add.h"
#include "shaders/scan_local.h"
#include "shaders/train_backward.h"
#include "shaders/train_backward_bucket.h"
#include "shaders/train_backward_bucket2.h"
#include "shaders/train_backward_cut.h"
#include "shaders/train_backward_cut2.h"
#include "shaders/train_backward_scalar.h"
#include "shaders/train_backward_serial.h"
#include "shaders/train_backward_sg1.h"
#include "shaders/train_backward_sg4.h"
#include "shaders/train_backward_shared.h"
#include "shaders/train_densify_mark.h"
#include "shaders/train_densify_spawn.h"
#include "shaders/train_dispatch_args.h"
#include "shaders/train_emit.h"
#include "shaders/train_forward.h"
#include "shaders/train_project.h"
#include "shaders/train_ranges.h"
#include "shaders/train_reset_opacity.h"
#include "shaders/train_pair_sum.h"
#include "shaders/train_ssim.h"
#include "shaders/train_ssim_grad.h"
#include "shaders/train_step.h"
#include "shaders/train_step_sh.h"

namespace gs {

// Mirrors FrameData in train_common.glsl (std430, all 16-byte members).
struct Trainer::FrameData {
  float rot[3][4];
  float cam_pos[4];
  float intrinsics[4];
  uint32_t image[4];
  uint32_t model[4];
  float background[4];
  float lr0[4];
  float lr1[4];
  float adam[4];
  float bias[4];
  float densify0[4];
  float densify1[4];
  uint32_t limits[4];
  float loss[4];
  float head[4];
};

namespace {

constexpr float kBeta1 = 0.9f;
constexpr float kBeta2 = 0.999f;
constexpr float kEpsilon = 1e-15f;
constexpr uint32_t kFlagMaskedTarget = 1;
constexpr uint32_t kFlagBucketState = 2;
constexpr uint32_t kFlagSsim = 4;
constexpr uint32_t kBucketLanes = 32;   // splats per bucket (train_backward_bucket.comp)
constexpr uint32_t kBucketGroups = 8;   // buckets per workgroup
constexpr uint32_t kScanBlock = 1024;
constexpr uint32_t kPairGradBytes = 48;
constexpr uint32_t kBackwardBindings = 12;  // the variants use up to 12 (bucket reads its state)

struct ScanPush {
  uint32_t count;
  uint32_t total_index;
};

uint32_t bits_for(uint32_t n) {
  uint32_t bits = 0;
  while ((1u << bits) < n) ++bits;
  return bits;
}

float logit(float p) { return std::log(p / (1.0f - p)); }

// Dispatch of a kernel without push constants.
void dispatch_kernel(VkContext& ctx, VkCommandBuffer cmd, const ComputePipeline& pipeline,
                     std::initializer_list<const Buffer*> buffers, uint32_t groups_x, uint32_t groups_y = 1) {
  ctx.dispatch(cmd, pipeline, buffers, static_cast<const void*>(nullptr), 0u, groups_x, groups_y);
}

}  // namespace

TrainOptions scaled_options(uint32_t iterations) {
  TrainOptions o;
  const double scale = double(iterations) / double(o.iterations);
  o.iterations = iterations;
  o.densify_from = uint32_t(o.densify_from * scale);
  o.densify_until = uint32_t(o.densify_until * scale);
  o.opacity_reset_interval = uint32_t(o.opacity_reset_interval * scale);
  o.sh_degree_interval = std::max(1u, uint32_t(o.sh_degree_interval * scale));
  return o;
}

Trainer::Trainer(VkContext& ctx, Dataset dataset, const SplatScene& initial, const TrainOptions& options)
    : ctx_(ctx), dataset_(std::move(dataset)), options_(options), rng_(options.seed) {
  const DeviceInfo& info = ctx.info();
  if (dataset_.frames.empty() || dataset_.width == 0) throw std::runtime_error("데이터셋이 비어 있음");
  options_.sh_degree = std::clamp(options_.sh_degree, 0, 3);
  options_.masked = options_.masked && dataset_.has_masks;

  width_ = dataset_.width;
  height_ = dataset_.height;
  tiles_x_ = div_up(width_, kTileSize);
  tiles_y_ = div_up(height_, kTileSize);
  if (uint64_t(tiles_x_) * tiles_y_ > 65535) throw std::runtime_error("학습 해상도가 너무 큼");
  capacity_ = std::max(options_.max_gaussians, initial.count);
  sh_stride_ = uint32_t(sh_rest_coeffs(options_.sh_degree)) * 3;
  const VkDeviceSize max_range = info.properties.limits.maxStorageBufferRange;
  max_pairs_ = max_range / kPairGradBytes;
  if (uint64_t(capacity_) * 64 > max_range || uint64_t(capacity_) * std::max(sh_stride_, 1u) * 4 > max_range) {
    throw std::runtime_error("가우시안 최대 개수가 GPU 버퍼 한도를 넘음");
  }

  for (size_t i = 0; i < dataset_.frames.size(); ++i) {
    if (!dataset_.frames[i].holdout) train_frames_.push_back(i);
  }
  if (train_frames_.empty()) {
    for (size_t i = 0; i < dataset_.frames.size(); ++i) train_frames_.push_back(i);
  }

  // Pipelines. Each failure is collected so one run names every shader the
  // driver rejects.
  std::vector<std::string> failures;
  auto make = [&](const char* name, std::span<const uint32_t> spirv, uint32_t bindings, uint32_t push) {
    try {
      return ctx.create_compute_pipeline(spirv, bindings, push);
    } catch (const VulkanError& e) {
      failures.push_back(std::string(name) + " (" + vk_result_name(e.result) + ")");
      log_error("trainer: cannot create %s: %s", name, e.what());
      return ComputePipeline{};
    }
  };
  project_ = make("train_project", shaders::train_project, 5, 0);
  scan_local_ = make("scan_local", shaders::scan_local, 4, sizeof(ScanPush));
  scan_blocks_ = make("scan_blocks", shaders::scan_blocks, 1, sizeof(uint32_t));
  scan_add_ = make("scan_add", shaders::scan_add, 4, sizeof(ScanPush));
  emit_ = make("train_emit", shaders::train_emit, 9, 0);
  dispatch_args_ = make("train_dispatch_args", shaders::train_dispatch_args, 2, sizeof(uint32_t));
  ranges_ = make("train_ranges", shaders::train_ranges, 3, sizeof(uint32_t));
  forward_ = make("train_forward", shaders::train_forward, 15, 0);
  ssim_ = make("train_ssim", shaders::train_ssim, 4, 0);
  ssim_grad_ = make("train_ssim_grad", shaders::train_ssim_grad, 5, 0);
  pair_sum_ = make("train_pair_sum", shaders::train_pair_sum, 6, 0);
  step_ = make("train_step", shaders::train_step, 11, 0);
  step_sh_ = make("train_step_sh", shaders::train_step_sh, 7, 0);
  densify_mark_ = make("train_densify_mark", shaders::train_densify_mark, 11, 0);
  densify_spawn_ = make("train_densify_spawn", shaders::train_densify_spawn, 11, sizeof(uint32_t));
  reset_opacity_ = make("train_reset_opacity", shaders::train_reset_opacity, 4, sizeof(float));

  // Backward pass: the requested variant (Auto: the first subgroup variant the
  // driver accepts, else serial_cut2), then the alternatives that compile.
  const VkSubgroupFeatureFlags needed = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
  const bool has_subgroups = (info.subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
                             (info.subgroup.supportedOperations & needed) == needed;
  uint32_t min_subgroup = info.subgroup.subgroupSize;
  if (info.has_subgroup_size_control && info.subgroup_size_control.minSubgroupSize > 0) {
    min_subgroup = std::min(min_subgroup, info.subgroup_size_control.minSubgroupSize);
  }
  // The reduction slots (512 vec4) hold batch * 3 per subgroup.
  const uint32_t max_subgroups = std::max(1u, 256u / std::max(min_subgroup, 1u));
  uint32_t batch = 32;
  while (batch > 1 && max_subgroups * batch * 3 > 512) batch /= 2;
  using Reduction = TrainOptions::Reduction;
  struct Variant {
    Reduction reduction;
    const char* name;
    std::span<const uint32_t> spirv;
    bool subgroup;
  };
  const Variant variants[] = {{Reduction::Subgroup, "subgroup", shaders::train_backward, true},
                              {Reduction::SubgroupScalar, "subgroup_scalar", shaders::train_backward_scalar, true},
                              {Reduction::SubgroupLane4, "subgroup_lane4", shaders::train_backward_sg4, true},
                              {Reduction::SubgroupLane1, "subgroup_lane1", shaders::train_backward_sg1, true},
                              {Reduction::Bucket, "bucket", shaders::train_backward_bucket, false},
                              {Reduction::Bucket2, "bucket2", shaders::train_backward_bucket2, false},
                              {Reduction::SerialCut2, "serial_cut2", shaders::train_backward_cut2, false},
                              {Reduction::SerialCut, "serial_cut", shaders::train_backward_cut, false},
                              {Reduction::Serial, "serial", shaders::train_backward_serial, false},
                              {Reduction::Shared, "shared", shaders::train_backward_shared, false}};
  auto variant = [&](Reduction r) -> const Variant* {
    for (const Variant& v : variants) {
      if (v.reduction == r) return &v;
    }
    return nullptr;
  };
  // Subgroup variants may be rejected (noted); the others must compile.
  auto add_backward = [&](const Variant& v) {
    for (const Backward& b : backwards_) {
      if (b.name == v.name) return true;
    }
    if (v.subgroup && (!has_subgroups || min_subgroup < 4)) {
      if (reduction_note_.find("미지원") == std::string::npos) {
        if (!reduction_note_.empty()) reduction_note_ += ", ";
        reduction_note_ += "subgroup 연산 미지원";
      }
      return false;
    }
    Backward b;
    b.name = v.name;
    b.bucket = v.reduction == Reduction::Bucket || v.reduction == Reduction::Bucket2;
    b.push[0] = v.subgroup ? batch : 32;
    b.push[1] = max_subgroups;
    if (v.subgroup) {
      try {
        b.pipeline = ctx.create_compute_pipeline(v.spirv, kBackwardBindings, sizeof(b.push));
      } catch (const VulkanError& e) {
        log_error("trainer: %s backward rejected (%s)", v.name, e.what());
        if (!reduction_note_.empty()) reduction_note_ += ", ";
        reduction_note_ += std::string(v.name) + " 실패: " + vk_result_name(e.result);
        return false;
      }
    } else {
      const std::string shader = std::string("train_backward_") + v.name;
      b.pipeline = make(shader.c_str(), v.spirv, kBackwardBindings, sizeof(b.push));
    }
    backwards_.push_back(std::move(b));
    return true;
  };
  if (options_.reduction == Reduction::Auto) {
    for (const Variant& v : variants) {
      if (v.subgroup && add_backward(v)) break;
    }
    if (backwards_.empty()) add_backward(*variant(Reduction::SerialCut2));
  } else if (!add_backward(*variant(options_.reduction))) {
    throw std::runtime_error("subgroup 역전파를 쓸 수 없음 (" + reduction_note_ + ")");
  }
  for (Reduction r : options_.alternatives) {
    if (r != Reduction::Auto) add_backward(*variant(r));
  }
  // The bucket variant needs memory per pair; keep another for when it does not fit.
  if (std::none_of(backwards_.begin(), backwards_.end(), [](const Backward& b) { return !b.bucket; })) {
    add_backward(*variant(Reduction::SerialCut));
  }
  if (!failures.empty()) {
    std::string message = "학습 셰이더를 만들 수 없음:";
    for (const std::string& f : failures) message += " " + f;
    throw std::runtime_error(message);
  }
  log_info("trainer: %ux%u, %u tiles, capacity %u, sh %d, subgroup %u..%u, backward %s batch %u (+%zu alternatives)",
           width_, height_, tiles_x_ * tiles_y_, capacity_, options_.sh_degree, min_subgroup,
           info.subgroup.subgroupSize, reduction().c_str(), backwards_[0].push[0], backwards_.size() - 1);
  sort_ = std::make_unique<RadixSort>(ctx);

  const uint64_t cap = capacity_;
  const uint64_t sh_bytes = std::max<uint64_t>(cap * sh_stride_ * 4, 16);
  const uint64_t pixels = uint64_t(width_) * height_;
  const uint64_t tiles = uint64_t(tiles_x_) * tiles_y_;
  for (uint32_t i = 0; i < kBatch; ++i) {
    frames_.push_back(ctx.create_buffer(sizeof(FrameData), MemoryKind::Upload));
    targets_.push_back(ctx.create_buffer(pixels * 4, MemoryKind::Upload));
    readback_smalls_.push_back(ctx.create_buffer(16, MemoryKind::Readback));
    readback_losses_.push_back(ctx.create_buffer(tiles * 32, MemoryKind::Readback));
  }
  // Persistent descriptor sets: per batch slot about 16 pipelines (and one
  // per backward variant), up to 14 buffers each.
  set_pool_ = ctx.create_descriptor_pool(64 + kBatch * 32, 512 + kBatch * 384);
  params_ = ctx.create_buffer(cap * 64, MemoryKind::Device);
  sh_ = ctx.create_buffer(sh_bytes, MemoryKind::Device);
  m_params_ = ctx.create_buffer(cap * 64, MemoryKind::Device);
  v_params_ = ctx.create_buffer(cap * 64, MemoryKind::Device);
  m_sh_ = ctx.create_buffer(sh_bytes, MemoryKind::Device);
  v_sh_ = ctx.create_buffer(sh_bytes, MemoryKind::Device);
  stats_buf_ = ctx.create_buffer(cap * 16, MemoryKind::Device);
  grad_sum_ = ctx.create_buffer(cap * 48, MemoryKind::Device);
  sh_input_ = ctx.create_buffer(cap * 32, MemoryKind::Device);
  dispatch_args_buf_ = ctx.create_buffer(32, MemoryKind::Device, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
  projected_ = ctx.create_buffer(cap * 64, MemoryKind::Device);
  tiles_ = ctx.create_buffer(cap * 4, MemoryKind::Device);
  offsets_ = ctx.create_buffer(cap * 4, MemoryKind::Device);
  spawn_ = ctx.create_buffer(cap * 4, MemoryKind::Device);
  free_list_ = ctx.create_buffer(cap * 4, MemoryKind::Device);
  counters_ = ctx.create_buffer(16, MemoryKind::Device);
  mismatch_ = ctx.create_buffer(16, MemoryKind::Device);
  block_sums_ = ctx.create_buffer(std::max<uint64_t>(div_up(cap, kScanBlock), 1) * 4, MemoryKind::Device);
  totals_ = ctx.create_buffer(16, MemoryKind::Device);
  pixel_grad_ = ctx.create_buffer(pixels * 16, MemoryKind::Device);
  pixel_count_ = ctx.create_buffer(pixels * 4, MemoryKind::Device);
  image_ = ctx.create_buffer(pixels * 4, MemoryKind::Device);
  ranges_buf_ = ctx.create_buffer((tiles + 1) * 8, MemoryKind::Device);  // + the tile emit pads into
  tile_loss_ = ctx.create_buffer(tiles * 32, MemoryKind::Device);
  tile_count_ = ctx.create_buffer(tiles * 4, MemoryKind::Device);
  pixel_dot_ = ctx.create_buffer(pixels * 4, MemoryKind::Device);
  ssim_input_ = ctx.create_buffer(pixels * 32, MemoryKind::Device);
  ssim_maps_ = ctx.create_buffer(pixels * 48, MemoryKind::Device);
  readback_image_ = ctx.create_buffer(pixels * 4, MemoryKind::Readback);

  // Initial parameters; unused slots stay zero (free).
  slots_ = initial.count;
  alive_ = initial.count;
  std::vector<float> p(size_t(slots_) * 16, 0.0f);
  std::vector<float> s(size_t(slots_) * sh_stride_, 0.0f);
  const int init_coeffs = sh_rest_coeffs(initial.sh_degree);
  const int coeffs = sh_rest_coeffs(options_.sh_degree);
  for (uint32_t i = 0; i < slots_; ++i) {
    float* d = &p[size_t(i) * 16];
    for (int k = 0; k < 3; ++k) {
      d[k] = initial.positions[i * 3 + k];
      d[4 + k] = initial.log_scales[i * 3 + k];
      d[12 + k] = initial.sh_dc[i * 3 + k];
    }
    d[3] = initial.opacity_logits[i];
    d[7] = 1.0f;
    for (int k = 0; k < 4; ++k) d[8 + k] = initial.rotations[i * 4 + k];
    for (int j = 0; j < std::min(coeffs, init_coeffs); ++j) {
      for (int c = 0; c < 3; ++c) {
        s[size_t(i) * sh_stride_ + j * 3 + c] = initial.sh_rest[(size_t(i) * init_coeffs + j) * 3 + c];
      }
    }
  }
  ctx.run([&](VkCommandBuffer cmd) {
    for (const Buffer* b : {&params_, &sh_, &m_params_, &v_params_, &m_sh_, &v_sh_, &stats_buf_, &mismatch_}) {
      vkCmdFillBuffer(cmd, b->handle(), 0, VK_WHOLE_SIZE, 0);
    }
  });
  if (!p.empty()) ctx.upload(params_, p.data(), p.size() * 4);
  if (!s.empty()) ctx.upload(sh_, s.data(), s.size() * 4);
  stats_.slots = slots_;
  stats_.gaussians = alive_;
}

Trainer::~Trainer() {
  if (set_pool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(ctx_.device(), set_pool_, nullptr);
}

float Trainer::lambda_at(uint32_t step) const {
  if (options_.lambda_dssim_end < 0.0f || options_.iterations <= 1) return options_.lambda_dssim;
  const float t = std::min(1.0f, float(step) / float(options_.iterations - 1));
  return options_.lambda_dssim + (options_.lambda_dssim_end - options_.lambda_dssim) * t;
}

bool Trainer::uses_ssim() const { return options_.lambda_dssim > 0.0f || options_.lambda_dssim_end > 0.0f; }

uint32_t Trainer::sh_degree_at(uint32_t step) const {
  return std::min<uint32_t>(options_.sh_degree, options_.sh_degree_interval > 0 ? step / options_.sh_degree_interval
                                                                              : options_.sh_degree);
}

void Trainer::write_frame(uint32_t slot, const Camera& camera, const float background[3], bool masked_target,
                          uint32_t step, bool bucket_state, bool ssim) {
  FrameData f{};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) f.rot[r][c] = camera.rotation[r * 3 + c];
    f.rot[r][3] = camera.translation[r];
    f.cam_pos[r] = camera.position[r];
  }
  f.intrinsics[0] = camera.fx;
  f.intrinsics[1] = camera.fy;
  f.intrinsics[2] = camera.cx;
  f.intrinsics[3] = camera.cy;
  f.image[0] = width_;
  f.image[1] = height_;
  f.image[2] = tiles_x_;
  f.image[3] = tiles_y_;
  f.model[0] = slots_;
  f.model[1] = sh_degree_at(step);
  f.model[2] = sh_stride_;
  f.model[3] = (masked_target ? kFlagMaskedTarget : 0) | (bucket_state ? kFlagBucketState : 0) |
               (ssim ? kFlagSsim : 0);
  for (int c = 0; c < 3; ++c) f.background[c] = background[c];
  f.background[3] = 1.0f / (3.0f * float(width_) * float(height_));

  const float t = options_.iterations > 0 ? std::min(1.0f, float(step) / float(options_.iterations)) : 1.0f;
  float lr_pos = options_.lr_position;
  if (lr_pos > 0.0f && options_.lr_position_final > 0.0f) {
    lr_pos = std::exp(std::log(lr_pos) * (1.0f - t) + std::log(options_.lr_position_final) * t);
  }
  f.lr0[0] = lr_pos * dataset_.camera_extent;
  f.lr0[1] = options_.lr_scale;
  f.lr0[2] = options_.lr_rotation;
  f.lr0[3] = options_.lr_opacity;
  f.lr1[0] = options_.lr_sh_dc;
  f.lr1[1] = options_.lr_sh_rest;
  f.adam[0] = kBeta1;
  f.adam[1] = kBeta2;
  f.adam[2] = kEpsilon;
  const double n = double(step) + 1.0;
  f.bias[0] = float(1.0 / (1.0 - std::pow(double(kBeta1), n)));
  f.bias[1] = float(1.0 / (1.0 - std::pow(double(kBeta2), n)));

  const bool prune_big = options_.opacity_reset_interval > 0 && step > options_.opacity_reset_interval;
  f.densify0[0] = options_.densify_grad_threshold;
  f.densify0[1] = options_.percent_dense * dataset_.camera_extent;
  f.densify0[2] = options_.prune_opacity;
  f.densify0[3] = prune_big ? 0.1f * dataset_.camera_extent : 0.0f;
  f.densify1[0] = prune_big ? 20.0f : 0.0f;
  const uint32_t seed = options_.seed * 9781u + step * 6271u;
  std::memcpy(&f.densify1[1], &seed, 4);
  f.densify1[2] = 1.6f;
  f.limits[0] = uint32_t(pair_capacity_);
  f.loss[0] = lambda_at(step);
  f.loss[1] = 0.01f * 0.01f;  // SSIM constants for colours in [0, 1]
  f.loss[2] = 0.03f * 0.03f;
  f.head[3] = 1.0f;
  if (options_.head.found) {
    const float* c = options_.head.center;
    float t[3];
    for (int r = 0; r < 3; ++r) {
      t[r] = camera.rotation[r * 3] * c[0] + camera.rotation[r * 3 + 1] * c[1] + camera.rotation[r * 3 + 2] * c[2] +
             camera.translation[r];
    }
    if (t[2] > 0.05f) {
      f.head[0] = camera.fx * t[0] / t[2] + camera.cx;
      f.head[1] = camera.fy * t[1] / t[2] + camera.cy;
      f.head[2] = camera.fx * options_.head.radius / t[2];
      // A head filling much of the view needs no extra weight (close-ups).
      f.head[3] = f.head[2] < 0.25f * float(height_) ? options_.head_weight : 1.0f;
    }
  }
  std::memcpy(frames_[slot].mapped(), &f, sizeof(f));
}

VkDescriptorSet Trainer::bound(const ComputePipeline& pipeline, std::initializer_list<const Buffer*> buffers) {
  auto bits = [](auto handle) {
    uint64_t v = 0;
    std::memcpy(&v, &handle, sizeof(handle));
    return v;
  };
  std::vector<uint64_t> key{bits(pipeline.set_layout())};
  for (const Buffer* b : buffers) key.push_back(bits(b->handle()));
  auto it = sets_.find(key);
  if (it != sets_.end()) return it->second;
  VkDescriptorSet set = ctx_.allocate_set(set_pool_, pipeline.set_layout(), buffers);
  sets_.emplace(std::move(key), set);
  return set;
}

void Trainer::kernel(VkCommandBuffer cmd, const ComputePipeline& pipeline, std::initializer_list<const Buffer*> buffers,
                     uint32_t groups_x, uint32_t groups_y, const void* push, uint32_t push_size) {
  ctx_.dispatch(cmd, pipeline, bound(pipeline, buffers), push, push_size, groups_x, groups_y);
}

void Trainer::reset_sets() {
  GS_VK_CHECK(vkResetDescriptorPool(ctx_.device(), set_pool_, 0));
  sets_.clear();
  sort_bound_ = false;
}

void Trainer::record_scan(VkCommandBuffer cmd, const Buffer& values, const Buffer& prefix, uint32_t count,
                          uint32_t total_index) {
  const uint32_t blocks = div_up(count, kScanBlock);
  ScanPush push{count, total_index};
  kernel(cmd, scan_local_, {&values, &prefix, &block_sums_, &totals_}, blocks, 1, &push, sizeof(push));
  VkContext::barrier(cmd);
  kernel(cmd, scan_blocks_, {&block_sums_}, 1, 1, &blocks, sizeof(blocks));
  VkContext::barrier(cmd);
  kernel(cmd, scan_add_, {&values, &prefix, &block_sums_, &totals_}, blocks, 1, &push, sizeof(push));
  VkContext::barrier(cmd);
}

void Trainer::reserve_pairs(uint64_t expected) {
  // Views differ in pair count, so keep headroom over the largest seen.
  const uint64_t wanted = expected + expected / 8;
  if (wanted <= pair_capacity_) return;
  if (expected > max_pairs_) {
    throw std::runtime_error("타일-가우시안 쌍이 GPU 버퍼 한도를 넘음 (" + std::to_string(expected) + ")");
  }
  const uint64_t n =
      std::min<uint64_t>(max_pairs_, std::max<uint64_t>(wanted + wanted / 4, options_.min_pair_capacity));
  reset_sets();  // the sets point at the buffers replaced below
  keys_ = ctx_.create_buffer(n * 4, MemoryKind::Device);
  values_ = ctx_.create_buffer(n * 4, MemoryKind::Device);
  pair_gaussian_ = ctx_.create_buffer(n * 4, MemoryKind::Device);
  pair_grads_ = ctx_.create_buffer(n * kPairGradBytes, MemoryKind::Device);
  sort_->reserve(static_cast<uint32_t>(n));
  pair_capacity_ = n;
  // Bucket state: every tile's buckets after the first fit below n / 32 + tiles + 1.
  const uint64_t buckets = n / kBucketLanes + uint64_t(tiles_x_) * tiles_y_ + 1;
  const uint64_t state_bytes = buckets * kTileSize * kTileSize * 16;
  uint64_t limit = ctx_.info().properties.limits.maxStorageBufferRange;
  if (options_.bucket_state_limit > 0) limit = std::min(limit, options_.bucket_state_limit);
  bucket_fits_ = state_bytes <= limit;
  bucket_state_ = ctx_.create_buffer(bucket_fits_ ? state_bytes : 16, MemoryKind::Device);
  bucket_list_ = ctx_.create_buffer((buckets + 1) * 4, MemoryKind::Device);
}

void Trainer::refit_pairs() {
  // The buffers only grow, sized for the most pairs any step had. Early on,
  // while the initial gaussians are large, that can be several times what
  // training settles at, and past what bucket state can cover; size them to
  // the steps since the last retune instead, so bucket can run again.
  const uint64_t recent = recent_pairs_max_;
  recent_pairs_max_ = 0;
  if (bucket_fits_ || recent == 0) return;
  const uint64_t wanted = recent + recent / 8;
  if (std::max<uint64_t>(wanted + wanted / 4, options_.min_pair_capacity) >= pair_capacity_) return;
  max_pairs_seen_ = recent;
  pair_capacity_ = 0;  // reallocate at the smaller size
  reserve_pairs(max_pairs_seen_);
}

bool Trainer::select_backward(const std::string& name) {
  for (size_t v = 0; v < backwards_.size(); ++v) {
    if (backwards_[v].name == name) {
      active_ = v;
      return true;
    }
  }
  return false;
}

size_t Trainer::backward_to_run() const {
  if (!backwards_[active_].bucket || bucket_fits_) return active_;
  for (size_t v = 0; v < backwards_.size(); ++v) {
    if (!backwards_[v].bucket) return v;
  }
  return active_;
}

bool Trainer::needs_bucket_state(bool train, bool retune) const {
  if (!train || !bucket_fits_) return false;
  if (backwards_[backward_to_run()].bucket) return true;
  return retune && std::any_of(backwards_.begin(), backwards_.end(), [](const Backward& b) { return b.bucket; });
}

void Trainer::record_project(VkCommandBuffer cmd, uint32_t slot) {
  if (slots_ > 0) {
    kernel(cmd, project_, {&frames_[slot], &params_, &sh_, &projected_, &tiles_}, div_up(slots_, 128));
    VkContext::barrier(cmd);
    record_scan(cmd, tiles_, offsets_, slots_, 0);
  } else {
    vkCmdFillBuffer(cmd, totals_.handle(), 0, 16, 0);
    VkContext::barrier(cmd);
  }
}

void Trainer::record_sort(VkCommandBuffer cmd, uint32_t slot) {
  // The pair count is on the GPU (totals_): the passes over pairs are
  // dispatched indirectly for it, bounded by the capacity.
  const uint32_t capacity = uint32_t(pair_capacity_);
  const uint32_t key_bits = 16 + bits_for(tiles_x_ * tiles_y_ + 1);  // + the padding tile
  vkCmdFillBuffer(cmd, ranges_buf_.handle(), 0, VK_WHOLE_SIZE, 0);
  if (slots_ > 0) {
    kernel(cmd, dispatch_args_, {&totals_, &dispatch_args_buf_}, 1, 1, &capacity, sizeof(capacity));
    kernel(cmd, emit_,
           {&frames_[slot], &projected_, &tiles_, &offsets_, &keys_, &values_, &pair_gaussian_, &pair_grads_,
            &mismatch_},
           div_up(slots_, 128));
    VkContext::barrier(cmd);
    if (!sort_bound_) {
      sort_binding_ = sort_->bind(set_pool_, keys_, values_, &totals_);
      sort_bound_ = true;
    }
    sort_->record_indirect(cmd, sort_binding_, keys_, values_, capacity, dispatch_args_buf_, 0, key_bits);
    ctx_.dispatch_indirect(cmd, ranges_, bound(ranges_, {&keys_, &ranges_buf_, &totals_}), &capacity,
                           sizeof(capacity), dispatch_args_buf_, 16);
  }
  VkContext::barrier(cmd);
}

void Trainer::record_rasterize(VkCommandBuffer cmd, uint32_t slot) {
  vkCmdFillBuffer(cmd, bucket_list_.handle(), 0, 4, 0);
  VkContext::barrier(cmd);
  kernel(cmd, forward_,
         {&frames_[slot], &projected_, &ranges_buf_, &values_, &pair_gaussian_, &targets_[slot], &pixel_grad_,
          &pixel_count_, &image_, &tile_loss_, &tile_count_, &bucket_state_, &bucket_list_, &pixel_dot_,
          &ssim_input_},
         tiles_x_, tiles_y_);
  VkContext::barrier(cmd);
}

void Trainer::record_loss(VkCommandBuffer cmd, uint32_t slot, bool ssim, bool grad) {
  if (ssim) {
    kernel(cmd, ssim_, {&frames_[slot], &ssim_input_, &ssim_maps_, &tile_loss_}, tiles_x_, tiles_y_);
    VkContext::barrier(cmd);
    if (grad) {
      kernel(cmd, ssim_grad_, {&frames_[slot], &ssim_input_, &ssim_maps_, &pixel_grad_, &pixel_dot_}, tiles_x_,
             tiles_y_);
      VkContext::barrier(cmd);
    }
  }
  const VkBufferCopy loss{0, 0, tile_loss_.size()};
  vkCmdCopyBuffer(cmd, tile_loss_.handle(), readback_losses_[slot].handle(), 1, &loss);
  const VkBufferCopy pairs{0, 0, 4}, mismatch{0, 4, 4};
  vkCmdCopyBuffer(cmd, totals_.handle(), readback_smalls_[slot].handle(), 1, &pairs);
  vkCmdCopyBuffer(cmd, mismatch_.handle(), readback_smalls_[slot].handle(), 1, &mismatch);
}

void Trainer::record_backward(VkCommandBuffer cmd, uint32_t slot, const Backward& b) {
  // Bucket: enough workgroups for every bucket there can be; the rest exit.
  const uint32_t groups_x =
      b.bucket ? div_up(pair_capacity_ / kBucketLanes + tiles_x_ * tiles_y_ + 1, kBucketGroups) : tiles_x_;
  kernel(cmd, b.pipeline,
         {&frames_[slot], &projected_, &ranges_buf_, &values_, &pair_gaussian_, &pixel_grad_, &pixel_count_,
          &pair_grads_, &tile_count_, &bucket_state_, &bucket_list_, &pixel_dot_},
         groups_x, b.bucket ? 1 : tiles_y_, b.push, sizeof(b.push));
  VkContext::barrier(cmd);
}

void Trainer::record_optimize(VkCommandBuffer cmd, uint32_t slot, int part) {
  if (slots_ == 0) return;
  if (part == 0 || part == 1) {
    kernel(cmd, pair_sum_, {&frames_[slot], &tiles_, &offsets_, &pair_grads_, &totals_, &grad_sum_},
           div_up(slots_, 128));
    VkContext::barrier(cmd);
  }
  if (part == 0 || part == 2) {
    kernel(cmd, step_,
           {&frames_[slot], &params_, &sh_, &m_params_, &v_params_, &tiles_, &grad_sum_, &projected_, &stats_buf_,
            &totals_, &sh_input_},
           div_up(slots_, 128));
    VkContext::barrier(cmd);
  }
  if ((part == 0 || part == 3) && sh_stride_ > 0) {
    kernel(cmd, step_sh_, {&frames_[slot], &params_, &sh_, &m_sh_, &v_sh_, &sh_input_, &totals_},
           div_up(uint64_t(slots_) * (sh_stride_ / 3), 256));
    VkContext::barrier(cmd);
  }
}

bool Trainer::read_slot(uint32_t slot) {
  uint32_t small[2];
  std::memcpy(small, readback_smalls_[slot].mapped(), sizeof(small));
  stats_.pairs = small[0];
  stats_.tile_mismatch = small[1];
  max_pairs_seen_ = std::max<uint64_t>(max_pairs_seen_, small[0]);
  recent_pairs_max_ = std::max<uint64_t>(recent_pairs_max_, small[0]);
  const float* v = static_cast<const float*>(readback_losses_[slot].mapped());
  // Per tile (train_forward): |e|, e^2, SSIM, w |e|, w SSIM, w, head e^2, head pixels.
  for (double& s : last_losses_) s = 0.0;
  for (uint32_t t = 0; t < tiles_x_ * tiles_y_; ++t) {
    for (int k = 0; k < 8; ++k) last_losses_[k] += v[t * 8 + k];
  }
  return small[0] > pair_capacity_;
}

void Trainer::set_view_stats() {
  const double n = 3.0 * width_ * height_;
  stats_.l1 = last_losses_[0] / n;
  stats_.psnr = -10.0 * std::log10(std::max(last_losses_[1] / n, 1e-12));
  const double lambda = lambda_at(stats_.step);
  stats_.ssim = uses_ssim() ? last_losses_[2] / n : 0.0;
  // The objective, with the head's weight: (1 - lambda) sum w |e| + lambda sum w (1 - SSIM), over 3 pixels.
  stats_.loss = ((1.0 - lambda) * last_losses_[3] + lambda * (last_losses_[5] - last_losses_[4])) / n;
}

void Trainer::run_view(const Camera& camera, const uint8_t* target, const float background[3], bool masked_target,
                       bool train, bool profile, bool retune, bool ssim) {
  stats_.sh_degree = int(sh_degree_at(stats_.step));
  const bool ssim_grad = train && uses_ssim();
  ssim = ssim || ssim_grad;
  if (target != nullptr) std::memcpy(targets_[0].mapped(), target, size_t(width_) * height_ * 4);
  for (;;) {
    // Before the first count, guess: a few tiles per gaussian.
    reserve_pairs(max_pairs_seen_ > 0          ? max_pairs_seen_
                  : options_.initial_pair_guess ? options_.initial_pair_guess
                                                : uint64_t(slots_) * 4);
    write_frame(0, camera, background, masked_target, stats_.step, needs_bucket_state(train, retune), ssim);

    if (!profile && !retune) {
      // Everything in one submission; the passes read the pair count on the GPU.
      const RunTiming time = ctx_.run([&](VkCommandBuffer cmd) {
        record_project(cmd, 0);
        record_sort(cmd, 0);
        record_rasterize(cmd, 0);
        record_loss(cmd, 0, ssim, ssim_grad);
        if (train) {
          record_backward(cmd, 0, backwards_[backward_to_run()]);
          record_optimize(cmd, 0);
        }
        VkContext::barrier_to_host(cmd);
      });
      // Too many pairs: the step left the model alone; grow and run it again.
      if (read_slot(0)) continue;
      if (train) {
        auto ema = [](double& avg, double v) { avg = avg == 0.0 ? v : 0.95 * avg + 0.05 * v; };
        ema(stats_.timing.wall_ms, time.cpu_ms);
        ema(stats_.timing.gpu_ms, std::max(time.gpu_ms, 0.0));
        ema(stats_.timing.batch, 1.0);
      }
      return;
    }

    // Profiling and retuning time the passes one submission each, on the full
    // pair count: read it first.
    const RunTiming project_time = ctx_.run([&](VkCommandBuffer cmd) {
      record_project(cmd, 0);
      const VkBufferCopy region{0, 0, 4};
      vkCmdCopyBuffer(cmd, totals_.handle(), readback_smalls_[0].handle(), 1, &region);
      VkContext::barrier_to_host(cmd);
    });
    uint32_t pairs = 0;
    std::memcpy(&pairs, readback_smalls_[0].mapped(), 4);
    if (pairs > pair_capacity_) {
      max_pairs_seen_ = std::max<uint64_t>(max_pairs_seen_, pairs);
      recent_pairs_max_ = std::max<uint64_t>(recent_pairs_max_, pairs);
      continue;
    }
    profile_ = PassTimes{};
    profile_.project = project_time.best_ms();
    profile_.sort = ctx_.run([&](VkCommandBuffer cmd) { record_sort(cmd, 0); }).best_ms();
    profile_.rasterize = ctx_.run([&](VkCommandBuffer cmd) { record_rasterize(cmd, 0); }).best_ms();
    profile_.ssim = ctx_.run([&](VkCommandBuffer cmd) {
                      record_loss(cmd, 0, ssim, ssim_grad);
                      VkContext::barrier_to_host(cmd);
                    }).best_ms();
    auto optimize = [&] {
      for (int part = 1; part <= 3; ++part) {
        const double ms = ctx_.run([&](VkCommandBuffer cmd) { record_optimize(cmd, 0, part); }).best_ms();
        (part == 1 ? profile_.optimize_sum : part == 2 ? profile_.optimize_geometry : profile_.optimize_sh) = ms;
      }
      return profile_.optimize_sum + profile_.optimize_geometry + profile_.optimize_sh;
    };
    if (train && retune && backwards_.size() > 1) {
      // Each variant twice, in mirrored order (A B C C B A), so a clock change
      // during the measurement favours none. The backward pass only writes
      // pair_grads, and every variant writes the same values.
      const size_t n = backwards_.size();
      std::vector<double> ms(n, 0.0);
      for (size_t k = 0; k < 2 * n; ++k) {
        const size_t v = k < n ? k : 2 * n - 1 - k;
        if (backwards_[v].bucket && !bucket_fits_) continue;  // no state this step
        ms[v] += 0.5 * ctx_.run([&](VkCommandBuffer cmd) { record_backward(cmd, 0, backwards_[v]); }).best_ms();
      }
      for (size_t v = 0; v < n; ++v) {
        if (backwards_[v].bucket && !bucket_fits_) ms[v] = 1e30;
      }
      RetuneResult r;
      r.step = stats_.step + 1;
      for (size_t v = 0; v < n; ++v) {
        r.ms.emplace_back(backwards_[v].name, ms[v]);
        if (ms[v] < ms[active_]) active_ = v;
      }
      r.chosen = backwards_[active_].name;
      retunes_.push_back(std::move(r));
      profile_.backward = ms[active_];
      profile_.optimize = optimize();
    } else if (train) {
      profile_.backward =
          ctx_.run([&](VkCommandBuffer cmd) { record_backward(cmd, 0, backwards_[backward_to_run()]); }).best_ms();
      profile_.optimize = optimize();
    }
    profile_.valid = true;
    read_slot(0);
    return;
  }
}

void Trainer::next_view(size_t& frame, float background[3]) {
  if (order_pos_ >= order_.size()) {
    order_ = train_frames_;
    std::shuffle(order_.begin(), order_.end(), rng_);
    order_pos_ = 0;
  }
  if (options_.random_background) {
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (int c = 0; c < 3; ++c) background[c] = u(rng_);
  } else {
    std::copy(options_.background, options_.background + 3, background);
  }
  frame = order_[order_pos_++];
}

void Trainer::step(bool profile) {
  size_t frame = 0;
  float bg[3];
  next_view(frame, bg);
  step_on(frame, bg, profile);
}

bool Trainer::housekeeping_after(uint32_t t) const {
  if (t >= options_.densify_until) return false;
  return (t > options_.densify_from && options_.densify_interval > 0 && t % options_.densify_interval == 0) ||
         (options_.opacity_reset_interval > 0 && t % options_.opacity_reset_interval == 0);
}

void Trainer::after_step() {
  const uint32_t t = stats_.step;
  if (t < options_.densify_until) {
    if (t > options_.densify_from && options_.densify_interval > 0 && t % options_.densify_interval == 0) densify();
    if (options_.opacity_reset_interval > 0 && t % options_.opacity_reset_interval == 0) reset_opacity();
  }
}

void Trainer::step_on(size_t frame_index, const float background[3], bool profile) {
  const auto start = std::chrono::steady_clock::now();
  const DatasetFrame& frame = dataset_.frames.at(frame_index);
  run_view(dataset_camera(dataset_, frame), frame.rgba.data(), background, options_.masked, true, profile,
           retune_due(stats_.step + 1));
  last_slot_ = 0;
  set_view_stats();
  ++stats_.step;
  after_step();
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  stats_.step_ms = stats_.step_ms == 0.0 ? ms : 0.9 * stats_.step_ms + 0.1 * ms;
}

bool Trainer::retune_due(uint32_t s) const {
  // Not on the last step: nothing would run with the choice.
  return backwards_.size() > 1 && s < options_.iterations &&
         (s == options_.retune_first || (options_.retune_interval > 0 && s % options_.retune_interval == 0));
}

uint32_t Trainer::steps(uint32_t count) {
  uint32_t done = 0;
  while (done < count && !finished()) {
    // Singly: the first step (no pair count yet) and retuning steps.
    if (max_pairs_seen_ == 0 || retune_due(stats_.step + 1)) {
      if (retune_due(stats_.step + 1)) refit_pairs();
      step();
      ++done;
      continue;
    }
    uint32_t n = 0;
    for (;;) {
      ++n;
      const uint32_t s = stats_.step + n;
      if (n == kBatch || done + n == count || s >= options_.iterations || housekeeping_after(s) ||
          retune_due(s + 1)) {
        break;
      }
    }
    run_batch(n);
    done += n;
  }
  return done;
}

void Trainer::run_batch(uint32_t count) {
  const auto start = std::chrono::steady_clock::now();
  reserve_pairs(max_pairs_seen_);
  const size_t backward = backward_to_run();
  const bool buckets = needs_bucket_state(true, false);
  const bool ssim = uses_ssim();
  for (uint32_t s = 0; s < count; ++s) {
    size_t frame = 0;
    float bg[3];
    next_view(frame, bg);
    const DatasetFrame& f = dataset_.frames.at(frame);
    write_frame(s, dataset_camera(dataset_, f), bg, options_.masked, stats_.step + s, buckets, ssim);
    std::memcpy(targets_[s].mapped(), f.rgba.data(), size_t(width_) * height_ * 4);
  }
  const RunTiming time = ctx_.run([&](VkCommandBuffer cmd) {
    for (uint32_t s = 0; s < count; ++s) {
      record_project(cmd, s);
      record_sort(cmd, s);
      record_rasterize(cmd, s);
      record_loss(cmd, s, ssim, ssim);
      record_backward(cmd, s, backwards_[backward]);
      record_optimize(cmd, s);
    }
    VkContext::barrier_to_host(cmd);
  });
  for (uint32_t s = 0; s < count; ++s) {
    // A step whose pairs did not fit left the model alone; later ones went on.
    if (read_slot(s)) ++stats_.overflow_steps;
    set_view_stats();
    stats_.sh_degree = int(sh_degree_at(stats_.step));
    ++stats_.step;
  }
  last_slot_ = count - 1;
  after_step();  // only the last step of a batch can have any
  // Moving averages as if each step had been timed alone.
  const double decay = std::pow(0.95, count);
  auto ema = [&](double& avg, double v) { avg = avg == 0.0 ? v : v + (avg - v) * decay; };
  ema(stats_.timing.wall_ms, time.cpu_ms / count);
  ema(stats_.timing.gpu_ms, std::max(time.gpu_ms, 0.0) / count);
  ema(stats_.timing.batch, double(count));
  const double ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / count;
  stats_.step_ms = stats_.step_ms == 0.0 ? ms : ms + (stats_.step_ms - ms) * std::pow(0.9, count);
}

void Trainer::densify() {
  // The frame data still describes the last step; refresh it for the densify thresholds.
  Buffer& frame = frames_[last_slot_];
  FrameData* f = static_cast<FrameData*>(frame.mapped());
  const bool prune_big = options_.opacity_reset_interval > 0 && stats_.step > options_.opacity_reset_interval;
  f->densify0[3] = prune_big ? 0.1f * dataset_.camera_extent : 0.0f;
  f->densify1[0] = prune_big ? 20.0f : 0.0f;
  f->model[0] = slots_;

  const uint32_t capacity = capacity_;
  ctx_.run([&](VkCommandBuffer cmd) {
    vkCmdFillBuffer(cmd, counters_.handle(), 0, 16, 0);
    VkContext::barrier(cmd);
    std::initializer_list<const Buffer*> buffers = {&frame, &params_,    &sh_,    &m_params_,  &v_params_, &m_sh_,
                                                    &v_sh_,  &stats_buf_, &spawn_, &free_list_, &counters_};
    dispatch_kernel(ctx_, cmd, densify_mark_, buffers, div_up(slots_, 128));
    VkContext::barrier(cmd);
    ctx_.dispatch(cmd, densify_spawn_, buffers, capacity, div_up(slots_, 128));
    VkContext::barrier(cmd);
    vkCmdFillBuffer(cmd, stats_buf_.handle(), 0, VK_WHOLE_SIZE, 0);
    VkBufferCopy region{0, 0, 16};
    vkCmdCopyBuffer(cmd, counters_.handle(), readback_smalls_[0].handle(), 1, &region);
    VkContext::barrier_to_host(cmd);
  });
  uint32_t counters[4];
  std::memcpy(counters, readback_smalls_[0].mapped(), sizeof(counters));
  const uint32_t free_slots = counters[0], spawned = counters[1], pruned = counters[2];
  const uint32_t room = free_slots + (capacity_ - slots_);
  const uint32_t placed = std::min(spawned, room);
  if (placed > free_slots) slots_ += placed - free_slots;
  alive_ = alive_ - pruned + placed;
  stats_.spawned += placed;
  stats_.pruned += pruned;
  stats_.slots = slots_;
  stats_.gaussians = alive_;
  ++stats_.densify_events;
}

void Trainer::reset_opacity() {
  const float max_logit = logit(0.01f);
  Buffer& frame = frames_[last_slot_];
  static_cast<FrameData*>(frame.mapped())->model[0] = slots_;
  ctx_.run([&](VkCommandBuffer cmd) {
    ctx_.dispatch(cmd, reset_opacity_, {&frame, &params_, &m_params_, &v_params_}, max_logit, div_up(slots_, 128));
  });
}

std::vector<uint8_t> Trainer::render(const Camera& camera, const float background[3]) {
  if (camera.width != width_ || camera.height != height_) {
    throw std::invalid_argument("Trainer::render: camera size must match the dataset");
  }
  run_view(camera, nullptr, background, false, false);
  ctx_.run([&](VkCommandBuffer cmd) {
    VkBufferCopy region{0, 0, image_.size()};
    vkCmdCopyBuffer(cmd, image_.handle(), readback_image_.handle(), 1, &region);
    VkContext::barrier_to_host(cmd);
  });
  std::vector<uint8_t> out(size_t(width_) * height_ * 4);
  std::memcpy(out.data(), readback_image_.mapped(), out.size());
  return out;
}

EvalResult Trainer::evaluate() {
  EvalResult result;
  const float black[3] = {0.0f, 0.0f, 0.0f};
  double psnr_sum = 0.0, l1_sum = 0.0, ssim_sum = 0.0, head_sum = 0.0;
  const double n = 3.0 * width_ * height_;
  for (const DatasetFrame& frame : dataset_.frames) {
    if (!frame.holdout) continue;
    run_view(dataset_camera(dataset_, frame), frame.rgba.data(), black, options_.masked, false, false, false, true);
    const double abs_sum = last_losses_[0], sq_sum = last_losses_[1];
    psnr_sum += -10.0 * std::log10(std::max(sq_sum / n, 1e-12));
    l1_sum += abs_sum / n;
    ssim_sum += last_losses_[2] / n;
    if (last_losses_[7] > 0.0) {
      head_sum += -10.0 * std::log10(std::max(last_losses_[6] / (3.0 * last_losses_[7]), 1e-12));
      ++result.head_views;
    }
    ++result.views;
  }
  if (result.views > 0) {
    result.psnr = psnr_sum / result.views;
    result.ssim = ssim_sum / result.views;
    result.head_psnr = result.head_views > 0 ? head_sum / result.head_views : 0.0;
    result.l1 = l1_sum / result.views;
  }
  return result;
}

SplatScene Trainer::export_scene() {
  std::vector<float> p(size_t(slots_) * 16);
  std::vector<float> s(size_t(slots_) * sh_stride_);
  if (!p.empty()) ctx_.download(params_, p.data(), p.size() * 4);
  if (!s.empty()) ctx_.download(sh_, s.data(), s.size() * 4);
  SplatScene scene;
  scene.convention = AxisConvention::RUB;
  uint32_t alive = 0;
  for (uint32_t i = 0; i < slots_; ++i) alive += p[size_t(i) * 16 + 7] > 0.5f ? 1 : 0;
  scene.resize(alive, options_.sh_degree);
  const int coeffs = sh_rest_coeffs(options_.sh_degree);
  uint32_t o = 0;
  for (uint32_t i = 0; i < slots_; ++i) {
    const float* d = &p[size_t(i) * 16];
    if (d[7] < 0.5f) continue;
    for (int k = 0; k < 3; ++k) {
      scene.positions[o * 3 + k] = d[k];
      scene.log_scales[o * 3 + k] = d[4 + k];
      scene.sh_dc[o * 3 + k] = d[12 + k];
    }
    scene.opacity_logits[o] = d[3];
    const float len = std::sqrt(d[8] * d[8] + d[9] * d[9] + d[10] * d[10] + d[11] * d[11]);
    for (int k = 0; k < 4; ++k) scene.rotations[o * 4 + k] = len > 0.0f ? d[8 + k] / len : (k == 0 ? 1.0f : 0.0f);
    for (int j = 0; j < coeffs * 3; ++j) scene.sh_rest[size_t(o) * coeffs * 3 + j] = s[size_t(i) * sh_stride_ + j];
    ++o;
  }
  return scene;
}

Trainer::Gradients Trainer::first_step_gradients() {
  Gradients g;
  g.params.resize(size_t(slots_) * 16);
  g.sh.resize(size_t(slots_) * sh_stride_);
  if (!g.params.empty()) ctx_.download(m_params_, g.params.data(), g.params.size() * 4);
  if (!g.sh.empty()) ctx_.download(m_sh_, g.sh.data(), g.sh.size() * 4);
  for (float& v : g.params) v /= (1.0f - kBeta1);
  for (float& v : g.sh) v /= (1.0f - kBeta1);
  return g;
}

Trainer::DebugSnapshot Trainer::debug_snapshot() {
  DebugSnapshot d;
  const size_t pixels = size_t(width_) * height_;
  d.pixel_count.resize(pixels);
  d.pixel_grad.resize(pixels * 4);
  d.projected.resize(size_t(slots_) * 16);
  d.tiles.resize(slots_);
  d.offsets.resize(slots_);
  d.pair_grads.resize(size_t(stats_.pairs) * 12);
  ctx_.download(pixel_count_, d.pixel_count.data(), d.pixel_count.size() * 4);
  ctx_.download(pixel_grad_, d.pixel_grad.data(), d.pixel_grad.size() * 4);
  if (slots_ > 0) {
    ctx_.download(projected_, d.projected.data(), d.projected.size() * 4);
    ctx_.download(tiles_, d.tiles.data(), d.tiles.size() * 4);
    ctx_.download(offsets_, d.offsets.data(), d.offsets.size() * 4);
  }
  if (!d.pair_grads.empty()) ctx_.download(pair_grads_, d.pair_grads.data(), d.pair_grads.size() * 4);
  if (stats_.pairs > 0) {
    // The sorted keys carry each pair's pre-sort index along.
    std::vector<uint32_t> keys(stats_.pairs), values(stats_.pairs);
    ctx_.download(keys_, keys.data(), keys.size() * 4);
    ctx_.download(values_, values.data(), values.size() * 4);
    d.pair_tile.assign(stats_.pairs, ~0u);
    for (size_t k = 0; k < keys.size(); ++k) {
      if (values[k] < d.pair_tile.size()) d.pair_tile[values[k]] = keys[k] >> 16;
    }
  }
  return d;
}

}  // namespace gs
