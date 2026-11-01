#include "gs/trainer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
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
#include "shaders/train_motion_bind.h"
#include "shaders/train_motion_sum.h"
#include "shaders/train_project.h"
#include "shaders/train_ranges.h"
#include "shaders/train_reset_opacity.h"
#include "shaders/train_pair_sum.h"
#include "shaders/train_ssim.h"
#include "shaders/train_ssim_grad.h"
#include "shaders/train_step.h"
#include "shaders/train_step_motion.h"
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
constexpr uint32_t kFlagPoseGrad = 8;
constexpr uint32_t kFlagPoseOnly = 16;
constexpr uint32_t kFlagMotion = 32;
constexpr uint32_t kFlagMotionGrad = 64;
constexpr uint32_t kMaxMotionNodes = 128;  // train_step: one thread per node
constexpr uint32_t kMotionNodeBytes = 80;  // train_motion.glsl: 5 vec4 per node
constexpr size_t kNone = ~size_t(0);
constexpr uint32_t kBucketLanes = 32;   // splats per bucket (train_backward_bucket.comp)
constexpr uint32_t kBucketGroups = 8;   // buckets per workgroup
constexpr uint32_t kScanBlock = 1024;
constexpr uint32_t kPairGradBytes = 48;
constexpr uint32_t kBackwardBindings = 13;  // the variants use up to 13 (bucket reads its state, two buffers)

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

// exp([w]x): rotation by |w| radians about w, row-major.
void rotation_from(const double w[3], double out[9]) {
  const double theta = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
  const double K[9] = {0, -w[2], w[1], w[2], 0, -w[0], -w[1], w[0], 0};
  double K2[9];
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) K2[r * 3 + c] = K[r * 3] * K[c] + K[r * 3 + 1] * K[3 + c] + K[r * 3 + 2] * K[6 + c];
  }
  const double a = theta > 1e-9 ? std::sin(theta) / theta : 1.0;
  const double b = theta > 1e-9 ? (1.0 - std::cos(theta)) / (theta * theta) : 0.5;
  for (int i = 0; i < 9; ++i) out[i] = (i % 4 == 0 ? 1.0 : 0.0) + a * K[i] + b * K2[i];
}

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
  o.pose_from = uint32_t(o.pose_from * scale);
  o.motion_from = uint32_t(o.motion_from * scale);
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
  project_ = make("train_project", shaders::train_project, 7, 0);
  scan_local_ = make("scan_local", shaders::scan_local, 4, sizeof(ScanPush));
  scan_blocks_ = make("scan_blocks", shaders::scan_blocks, 1, sizeof(uint32_t));
  scan_add_ = make("scan_add", shaders::scan_add, 4, sizeof(ScanPush));
  emit_ = make("train_emit", shaders::train_emit, 9, 0);
  dispatch_args_ = make("train_dispatch_args", shaders::train_dispatch_args, 2, sizeof(uint32_t));
  ranges_ = make("train_ranges", shaders::train_ranges, 3, sizeof(uint32_t));
  forward_ = make("train_forward", shaders::train_forward, 16, 0);
  ssim_ = make("train_ssim", shaders::train_ssim, 4, 0);
  ssim_grad_ = make("train_ssim_grad", shaders::train_ssim_grad, 5, 0);
  pair_sum_ = make("train_pair_sum", shaders::train_pair_sum, 6, 0);
  step_ = make("train_step", shaders::train_step, 15, 0);
  step_sh_ = make("train_step_sh", shaders::train_step_sh, 7, 0);
  densify_mark_ = make("train_densify_mark", shaders::train_densify_mark, 11, 0);
  densify_spawn_ = make("train_densify_spawn", shaders::train_densify_spawn, 11, sizeof(uint32_t));
  reset_opacity_ = make("train_reset_opacity", shaders::train_reset_opacity, 4, sizeof(float));
  if (options_.refine_motion) {
    motion_bind_ = make("train_motion_bind", shaders::train_motion_bind, 3, 2 * sizeof(uint32_t));
    motion_sum_ = make("train_motion_sum", shaders::train_motion_sum, 2, 2 * sizeof(uint32_t));
    step_motion_ = make("train_step_motion", shaders::train_step_motion, 15, 0);
  }

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
    pose_grads_.push_back(ctx.create_buffer(std::max<uint64_t>(div_up(cap, 128), 1) * 32, MemoryKind::Readback));
  }
  pose_fix_.resize(dataset_.frames.size());
  if (options_.refine_motion) place_nodes(initial);
  const uint64_t node_count = nodes_.size();
  for (uint32_t i = 0; i < kBatch; ++i) {
    motion_slots_.push_back(ctx.create_buffer(std::max<uint64_t>(node_count, 1) * kMotionNodeBytes, MemoryKind::Upload));
    motion_grads_.push_back(ctx.create_buffer(std::max<uint64_t>(node_count, 1) * 32, MemoryKind::Readback));
  }
  // Without nodes the shaders never read these; they only need binding.
  motion_nodes_buf_ = ctx.create_buffer(std::max<uint64_t>(node_count, 1) * 16, MemoryKind::Upload);
  motion_bind_buf_ = ctx.create_buffer(node_count > 0 ? cap * 32 : 32, MemoryKind::Device);
  motion_partials_ =
      ctx.create_buffer(node_count > 0 ? std::max<uint64_t>(div_up(cap, 128), 1) * node_count * 32 : 32,
                        MemoryKind::Device);
  if (node_count > 0) {
    float* n = static_cast<float*>(motion_nodes_buf_.mapped());
    for (size_t k = 0; k < nodes_.size(); ++k) {
      for (int c = 0; c < 3; ++c) n[k * 4 + c] = nodes_[k].position[c];
      n[k * 4 + 3] = 1.0f / (2.0f * nodes_[k].radius * nodes_[k].radius);
    }
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
  if (!nodes_.empty()) bind_motion();
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
                          uint32_t step, bool bucket_state, bool ssim, uint32_t flags, const ViewMotion* motion) {
  FrameData f{};
  if (motion == nullptr || nodes_.empty()) {
    flags &= ~kFlagMotionGrad;
  } else {
    // Each node's [R | b] (x' = R x + b, b = n + shift - R n), turn and where it moved to.
    flags |= kFlagMotion;
    f.limits[1] = uint32_t(nodes_.size());
    float* out = static_cast<float*>(motion_slots_[slot].mapped());
    for (size_t k = 0; k < nodes_.size(); ++k) {
      const NodeMotion& m = (*motion)[k];
      const double w[3] = {m.turn[0], m.turn[1], m.turn[2]};
      double R[9];
      rotation_from(w, R);
      const float* n = nodes_[k].position;
      float* o = out + k * 20;
      for (int r = 0; r < 3; ++r) {
        const double moved = n[r] + m.shift[r];
        double b = moved;
        for (int c = 0; c < 3; ++c) {
          o[r * 4 + c] = float(R[r * 3 + c]);
          b -= R[r * 3 + c] * n[c];
        }
        o[r * 4 + 3] = float(b);
        o[12 + r] = m.turn[r];
        o[16 + r] = float(moved);
      }
      o[15] = o[19] = 0.0f;
    }
  }
  motion_grad_slot_[slot] = (flags & kFlagMotionGrad) != 0;
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
               (ssim ? kFlagSsim : 0) | flags;
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
  // With bucket state, a step with more pairs than it covers must not train.
  f.limits[2] = bucket_state ? uint32_t(bucket_per_buffer_) : 0;
  f.limits[3] = uint32_t(bucket_state ? std::min(pair_capacity_, bucket_pairs_) : pair_capacity_);
  slot_limit_[slot] = f.limits[3];
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
  if (wanted > pair_capacity_) {
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
    // Bucket state: with p pairs, every tile's buckets after the first fit
    // below p / 32 + tiles + 1. It goes in two buffers of up to the binding
    // limit each, sized for the pairs the buffers hold (n) or as many as fit:
    // sizing it for n alone (headroom included) put it past the limit at
    // ~700k pairs on the phone, though the steps needed less.
    const uint64_t tiles = uint64_t(tiles_x_) * tiles_y_;
    const uint64_t bucket_bytes = uint64_t(kTileSize) * kTileSize * 16;
    uint64_t limit = ctx_.info().properties.limits.maxStorageBufferRange;
    if (options_.bucket_state_limit > 0) limit = std::min(limit, options_.bucket_state_limit);
    const uint64_t per_buffer = std::min<uint64_t>(limit / bucket_bytes, (n / kBucketLanes + tiles + 2) / 2);
    bucket_per_buffer_ = per_buffer;
    bucket_pairs_ =
        2 * per_buffer > tiles + 1 ? std::min<uint64_t>(n, (2 * per_buffer - tiles - 1) * kBucketLanes) : 0;
    bucket_state_ = ctx_.create_buffer(std::max<uint64_t>(per_buffer * bucket_bytes, 16), MemoryKind::Device);
    bucket_state2_ = ctx_.create_buffer(std::max<uint64_t>(per_buffer * bucket_bytes, 16), MemoryKind::Device);
    bucket_list_ = ctx_.create_buffer((n / kBucketLanes + tiles + 2) * 4, MemoryKind::Device);
  }
  // A step with more pairs than the state covers is skipped and run again
  // without bucket (write_frame, read_slot); with headroom that stays rare.
  bucket_fits_ = bucket_pairs_ > 0 && wanted <= bucket_pairs_;
}

void Trainer::refit_pairs() {
  // The largest pair count so far can be an early peak (while the initial
  // gaussians are large) past what bucket state covers. Judge by the steps
  // since the last retune instead, and size the pair buffers to them.
  const uint64_t recent = recent_pairs_max_;
  recent_pairs_max_ = 0;
  if (bucket_fits_ || recent == 0) return;
  max_pairs_seen_ = recent;
  const uint64_t wanted = recent + recent / 8;
  if (std::max<uint64_t>(wanted + wanted / 4, options_.min_pair_capacity) < pair_capacity_) {
    pair_capacity_ = 0;  // reallocate at the smaller size
  }
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
    kernel(cmd, project_,
           {&frames_[slot], &params_, &sh_, &projected_, &tiles_, &motion_slots_[slot], &motion_bind_buf_},
           div_up(slots_, 128));
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
          &ssim_input_, &bucket_state2_},
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
          &pair_grads_, &tile_count_, &bucket_state_, &bucket_list_, &pixel_dot_, &bucket_state2_},
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
    kernel(cmd, motion_grad_slot_[slot] ? step_motion_ : step_,
           {&frames_[slot], &params_, &sh_, &m_params_, &v_params_, &tiles_, &grad_sum_, &projected_, &stats_buf_,
            &totals_, &sh_input_, &pose_grads_[slot], &motion_slots_[slot], &motion_bind_buf_, &motion_partials_},
           div_up(slots_, 128));
    VkContext::barrier(cmd);
    if (motion_grad_slot_[slot]) {
      const uint32_t push[2] = {div_up(slots_, 128u), uint32_t(nodes_.size())};
      kernel(cmd, motion_sum_, {&motion_partials_, &motion_grads_[slot]}, 1, 1, push, sizeof(push));
      VkContext::barrier(cmd);
    }
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
  return small[0] > slot_limit_[slot];
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
                       bool train, bool profile, bool retune, bool ssim, uint32_t flags, const ViewMotion* motion) {
  stats_.sh_degree = int(sh_degree_at(stats_.step));
  const bool ssim_grad = train && uses_ssim();
  ssim = ssim || ssim_grad;
  if (target != nullptr) std::memcpy(targets_[0].mapped(), target, size_t(width_) * height_ * 4);
  for (;;) {
    // Before the first count, guess: a few tiles per gaussian.
    reserve_pairs(max_pairs_seen_ > 0          ? max_pairs_seen_
                  : options_.initial_pair_guess ? options_.initial_pair_guess
                                                : uint64_t(slots_) * 4);
    write_frame(0, camera, background, masked_target, stats_.step, needs_bucket_state(train, retune), ssim,
                train ? flags : 0, motion);

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
    if (pairs > slot_limit_[0]) {
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
  bool densified = false;
  if (t < options_.densify_until) {
    if (t > options_.densify_from && options_.densify_interval > 0 && t % options_.densify_interval == 0) {
      densify();
      densified = true;
    }
    if (options_.opacity_reset_interval > 0 && t % options_.opacity_reset_interval == 0) reset_opacity();
  }
  // New gaussians need nodes, and the others drift away from theirs.
  if (!nodes_.empty() && (densified || t >= bound_at_ + 100)) bind_motion();
}

void Trainer::step_on(size_t frame_index, const float background[3], bool profile) {
  const auto start = std::chrono::steady_clock::now();
  const DatasetFrame& frame = dataset_.frames.at(frame_index);
  const bool pose = refines_pose_at(stats_.step);
  const bool motion = refines_motion_at(stats_.step);
  if (motion) start_motion(frame_index);
  run_view(camera_for(frame_index), frame.rgba.data(), background, options_.masked, true, profile,
           retune_due(stats_.step + 1), false, (pose ? kFlagPoseGrad : 0) | (motion ? kFlagMotionGrad : 0),
           motion ? &motion_[frame_index] : nullptr);
  if (pose) {
    update_pose(0, frame_index, 1.0f);
    update_pose_stats();
  }
  if (motion) {
    update_motion(0, frame_index);
    update_motion_stats();
  }
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
  size_t frames[kBatch];
  bool pose[kBatch], motion[kBatch];
  for (uint32_t s = 0; s < count; ++s) {
    float bg[3];
    next_view(frames[s], bg);
    pose[s] = refines_pose_at(stats_.step + s);
    motion[s] = refines_motion_at(stats_.step + s);
    if (motion[s]) start_motion(frames[s]);
    const DatasetFrame& f = dataset_.frames.at(frames[s]);
    write_frame(s, camera_for(frames[s]), bg, options_.masked, stats_.step + s, buckets, ssim,
                (pose[s] ? kFlagPoseGrad : 0) | (motion[s] ? kFlagMotionGrad : 0),
                motion[s] ? &motion_[frames[s]] : nullptr);
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
  bool moved = false, fidgeted = false;
  for (uint32_t s = 0; s < count; ++s) {
    // A step whose pairs did not fit left the model alone; later ones went on.
    const bool overflow = read_slot(s);
    if (overflow) ++stats_.overflow_steps;
    if (pose[s] && !overflow) {
      update_pose(s, frames[s], 1.0f);
      moved = true;
    }
    if (motion[s] && !overflow) {
      update_motion(s, frames[s]);
      fidgeted = true;
    }
    set_view_stats();
    stats_.sh_degree = int(sh_degree_at(stats_.step));
    ++stats_.step;
  }
  if (moved) update_pose_stats();
  if (fidgeted) update_motion_stats();
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
  return render_with(camera, background, nullptr);
}

std::vector<uint8_t> Trainer::render_view(size_t frame, const float background[3]) {
  if (!uses_motion()) return render_with(camera_for(frame), background, nullptr);
  const ViewMotion motion = view_motion(frame);
  return render_with(camera_for(frame), background, &motion);
}

std::vector<uint8_t> Trainer::render_with(const Camera& camera, const float background[3], const ViewMotion* motion) {
  if (camera.width != width_ || camera.height != height_) {
    throw std::invalid_argument("Trainer::render: camera size must match the dataset");
  }
  run_view(camera, nullptr, background, false, false, false, false, false, 0, motion);
  ctx_.run([&](VkCommandBuffer cmd) {
    VkBufferCopy region{0, 0, image_.size()};
    vkCmdCopyBuffer(cmd, image_.handle(), readback_image_.handle(), 1, &region);
    VkContext::barrier_to_host(cmd);
  });
  std::vector<uint8_t> out(size_t(width_) * height_ * 4);
  std::memcpy(out.data(), readback_image_.mapped(), out.size());
  return out;
}

EvalResult Trainer::evaluate() { return evaluate_views(dataset_, true); }

EvalResult Trainer::evaluate_at_rest(const Dataset& other) {
  if (other.width != width_ || other.height != height_) {
    throw std::invalid_argument("Trainer::evaluate_at_rest: the dataset's resolution must match");
  }
  return evaluate_views(other, false);
}

Trainer::PoseFix Trainer::align_camera(const Camera& camera, const DatasetFrame& frame, const ViewMotion* motion) {
  // The model was fitted to corrected poses: give each held-out view the
  // same chance, moving only its camera. Gradient descent whose step grows
  // after a step that lowered the loss and halves (from the best pose) after
  // one that did not; turns count as the shift they cause at the cameras'
  // distance. The view keeps the best pose seen, so it is never worse than
  // given. (Adam moved poses that were already right.)
  const float black[3] = {0.0f, 0.0f, 0.0f};
  const double reach = dataset_.camera_extent;
  PoseFix best{}, current{};
  double best_loss = std::numeric_limits<double>::infinity(), best_grad[6] = {0, 0, 0, 0, 0, 0};
  double length = options_.eval_pose_step * reach;
  for (uint32_t k = 0; k < options_.eval_pose_steps; ++k) {
    run_view(with_fix(camera, current), frame.rgba.data(), black, options_.masked, true, false, false, uses_ssim(),
             kFlagPoseGrad | kFlagPoseOnly, motion);
    const double lambda = lambda_at(stats_.step);
    const double loss = (1.0 - lambda) * last_losses_[3] + lambda * (last_losses_[5] - last_losses_[4]);
    if (loss < best_loss) {
      best = current;
      best_loss = loss;
      pose_gradient(0, best_grad);
      if (k > 0) length *= 1.5;
    } else {
      length *= 0.5;
    }
    double g[6], norm = 0.0;
    for (int c = 0; c < 6; ++c) {
      g[c] = c < 3 ? best_grad[c] : best_grad[c] / reach;
      norm += g[c] * g[c];
    }
    if (!(norm > 0.0)) break;
    norm = std::sqrt(norm);
    double step[6];
    for (int c = 0; c < 6; ++c) step[c] = -length * g[c] / norm / (c < 3 ? 1.0 : reach);
    current = best;
    current.updates = 1;
    apply_pose_step(current, step);
  }
  return best;
}

EvalResult Trainer::evaluate_views(const Dataset& ds, bool own) {
  EvalResult result;
  const float black[3] = {0.0f, 0.0f, 0.0f};
  const double n = 3.0 * width_ * height_;
  auto measure = [&](const Camera& camera, const DatasetFrame& frame, const ViewMotion* motion, double sums[5]) {
    run_view(camera, frame.rgba.data(), black, options_.masked, false, false, false, true, 0, motion);
    sums[0] += -10.0 * std::log10(std::max(last_losses_[1] / n, 1e-12));
    sums[1] += last_losses_[0] / n;
    sums[2] += last_losses_[2] / n;
    if (last_losses_[7] > 0.0) {
      sums[3] += -10.0 * std::log10(std::max(last_losses_[6] / (3.0 * last_losses_[7]), 1e-12));
      sums[4] += 1.0;
    }
  };
  const bool align = options_.refine_poses && options_.eval_pose_steps > 0;
  const bool moving = own && uses_motion();
  double sums[5] = {0, 0, 0, 0, 0}, unaligned[5] = {0, 0, 0, 0, 0};
  for (size_t i = 0; i < ds.frames.size(); ++i) {
    const DatasetFrame& frame = ds.frames[i];
    if (!frame.holdout) continue;
    ViewMotion motion;
    if (moving) motion = view_motion(i);
    const ViewMotion* m = moving ? &motion : nullptr;
    const Camera given = dataset_camera(ds, frame);
    PoseFix fix{};
    if (align) {
      measure(given, frame, m, unaligned);
      fix = align_camera(given, frame, m);
      if (own) pose_fix_[i] = fix;
    }
    measure(with_fix(given, fix), frame, m, sums);
    ++result.views;
  }
  if (result.views > 0) {
    result.psnr = sums[0] / result.views;
    result.l1 = sums[1] / result.views;
    result.ssim = sums[2] / result.views;
    result.head_views = uint32_t(sums[4]);
    result.head_psnr = sums[4] > 0.0 ? sums[3] / sums[4] : 0.0;
    result.aligned = align;
    result.psnr_unaligned = align ? unaligned[0] / result.views : result.psnr;
  }
  return result;
}

Camera Trainer::with_fix(Camera cam, const PoseFix& p) {
  if (p.updates == 0) return cam;
  // World -> camera [R | t] becomes [E R | E t + d].
  float rot[9], t[3];
  for (int r = 0; r < 3; ++r) {
    t[r] = p.shift[r];
    for (int c = 0; c < 3; ++c) {
      rot[r * 3 + c] = 0.0f;
      for (int k = 0; k < 3; ++k) rot[r * 3 + c] += p.rotation[r * 3 + k] * cam.rotation[k * 3 + c];
      t[r] += p.rotation[r * 3 + c] * cam.translation[c];
    }
  }
  std::copy(rot, rot + 9, cam.rotation);
  std::copy(t, t + 3, cam.translation);
  for (int c = 0; c < 3; ++c) cam.position[c] = -(rot[c] * t[0] + rot[3 + c] * t[1] + rot[6 + c] * t[2]);
  return cam;
}

Camera Trainer::camera_for(size_t frame) const {
  const Camera cam = dataset_camera(dataset_, dataset_.frames.at(frame));
  return frame < pose_fix_.size() ? with_fix(cam, pose_fix_[frame]) : cam;
}

bool Trainer::refines_pose_at(uint32_t step) const { return options_.refine_poses && step >= options_.pose_from; }

void Trainer::pose_gradient(uint32_t slot, double grad[6]) const {
  // Sum the workgroups' shares (train_step): dL/dd, then dL/dw.
  const uint32_t groups = div_up(slots_, 128u);
  const float* g = static_cast<const float*>(pose_grads_[slot].mapped());
  for (int c = 0; c < 6; ++c) grad[c] = 0.0;
  for (uint32_t k = 0; k < groups; ++k) {
    for (int c = 0; c < 3; ++c) {
      grad[c] += g[k * 8 + c];
      grad[3 + c] += g[k * 8 + 4 + c];
    }
  }
}

void Trainer::apply_pose_step(PoseFix& p, const double step[6]) {
  // The step applies after the current correction: [R | s] <- [E | d] [R | s].
  double E[9];
  rotation_from(step + 3, E);
  float rot[9], shift[3];
  for (int r = 0; r < 3; ++r) {
    shift[r] = float(step[r]);
    for (int c = 0; c < 3; ++c) {
      double v = 0.0;
      for (int k = 0; k < 3; ++k) v += E[r * 3 + k] * p.rotation[k * 3 + c];
      rot[r * 3 + c] = float(v);
      shift[r] += float(E[r * 3 + c] * p.shift[c]);
    }
  }
  std::copy(rot, rot + 9, p.rotation);
  std::copy(shift, shift + 3, p.shift);
}

void Trainer::update_pose(uint32_t slot, size_t frame, float lr_scale) {
  double grad[6];
  pose_gradient(slot, grad);
  std::copy(grad, grad + 6, last_pose_grad_);
  PoseFix& p = pose_fix_.at(frame);
  ++p.updates;
  const double c1 = 1.0 / (1.0 - std::pow(double(kBeta1), double(p.updates)));
  const double c2 = 1.0 / (1.0 - std::pow(double(kBeta2), double(p.updates)));
  const double t = options_.iterations > 0 ? std::min(1.0, double(stats_.step) / options_.iterations) : 1.0;
  lr_scale *= float(std::pow(double(std::max(options_.lr_pose_final, 1e-6f)), t));
  double step[6];
  for (int i = 0; i < 6; ++i) {
    p.m[i] = kBeta1 * p.m[i] + (1.0 - kBeta1) * grad[i];
    p.v[i] = kBeta2 * p.v[i] + (1.0 - kBeta2) * grad[i] * grad[i];
    const double lr = (i < 3 ? options_.lr_pose_shift * dataset_.camera_extent : options_.lr_pose_turn) * lr_scale;
    step[i] = p.v[i] > 0.0 ? -lr * (p.m[i] * c1) / (std::sqrt(p.v[i] * c2) + 1e-30) : 0.0;
  }
  apply_pose_step(p, step);
}

void Trainer::update_pose_stats() {
  double shift_sum = 0.0, turn_sum = 0.0, shift_max = 0.0, turn_max = 0.0;
  uint32_t n = 0, updates = 0;
  for (size_t i : train_frames_) {
    const PoseFix& p = pose_fix_[i];
    if (p.updates == 0) continue;
    const Camera a = dataset_camera(dataset_, dataset_.frames[i]), b = camera_for(i);
    double d2 = 0.0;
    for (int k = 0; k < 3; ++k) d2 += double(b.position[k] - a.position[k]) * (b.position[k] - a.position[k]);
    const double shift = std::sqrt(d2);
    const double trace = double(p.rotation[0]) + p.rotation[4] + p.rotation[8];
    const double turn = std::acos(std::clamp((trace - 1.0) / 2.0, -1.0, 1.0)) * 180.0 / M_PI;
    shift_sum += shift;
    turn_sum += turn;
    shift_max = std::max(shift_max, shift);
    turn_max = std::max(turn_max, turn);
    updates += p.updates;
    ++n;
  }
  stats_.pose_shift_mean = n ? shift_sum / n : 0.0;
  stats_.pose_turn_mean = n ? turn_sum / n : 0.0;
  stats_.pose_shift_max = shift_max;
  stats_.pose_turn_max = turn_max;
  stats_.pose_updates = updates;
}

// --- Subject motion -------------------------------------------------------------

void Trainer::place_nodes(const SplatScene& initial) {
  // k-means (k-means++ seeds, Lloyd's iterations) over up to 20000 of the
  // initial gaussians: the nodes spread over the subject like its surface.
  if (initial.count == 0) return;
  const uint32_t stride = std::max(1u, initial.count / 20000);
  std::vector<std::array<float, 3>> pts;
  for (uint32_t i = 0; i < initial.count; i += stride) {
    pts.push_back({initial.positions[i * 3], initial.positions[i * 3 + 1], initial.positions[i * 3 + 2]});
  }
  const size_t k_count = std::min<size_t>({options_.motion_nodes, kMaxMotionNodes, pts.size()});
  if (k_count == 0) return;
  auto dist2 = [](const std::array<float, 3>& a, const std::array<float, 3>& b) {
    const float x = a[0] - b[0], y = a[1] - b[1], z = a[2] - b[2];
    return x * x + y * y + z * z;
  };
  std::mt19937 rng(options_.seed * 7919u + 17u);  // not rng_: the view order stays as without motion
  std::vector<std::array<float, 3>> centres;
  centres.push_back(pts[std::uniform_int_distribution<size_t>(0, pts.size() - 1)(rng)]);
  std::vector<float> nearest(pts.size(), std::numeric_limits<float>::max());
  while (centres.size() < k_count) {
    double total = 0.0;
    for (size_t i = 0; i < pts.size(); ++i) {
      nearest[i] = std::min(nearest[i], dist2(pts[i], centres.back()));
      total += nearest[i];
    }
    double pick = std::uniform_real_distribution<double>(0.0, total)(rng);
    size_t chosen = pts.size() - 1;
    for (size_t i = 0; i < pts.size(); ++i) {
      pick -= nearest[i];
      if (pick <= 0.0) {
        chosen = i;
        break;
      }
    }
    centres.push_back(pts[chosen]);
  }
  std::vector<uint32_t> owner(pts.size(), 0);
  for (int iteration = 0; iteration < 15; ++iteration) {
    std::vector<std::array<double, 4>> sum(k_count, {0, 0, 0, 0});
    for (size_t i = 0; i < pts.size(); ++i) {
      float best = std::numeric_limits<float>::max();
      for (size_t k = 0; k < k_count; ++k) {
        const float d = dist2(pts[i], centres[k]);
        if (d < best) {
          best = d;
          owner[i] = uint32_t(k);
        }
      }
      for (int c = 0; c < 3; ++c) sum[owner[i]][c] += pts[i][c];
      sum[owner[i]][3] += 1.0;
    }
    for (size_t k = 0; k < k_count; ++k) {
      if (sum[k][3] == 0.0) continue;  // keeps its place
      for (int c = 0; c < 3; ++c) centres[k][c] = float(sum[k][c] / sum[k][3]);
    }
  }
  // Radius: the mean distance to the 3 nearest other nodes; neighbours: the 6 nearest.
  nodes_.resize(k_count);
  node_neighbors_.assign(k_count, {});
  std::vector<float> radii;
  for (size_t k = 0; k < k_count; ++k) {
    std::vector<std::pair<float, uint32_t>> others;
    for (size_t j = 0; j < k_count; ++j) {
      if (j != k) others.emplace_back(std::sqrt(dist2(centres[k], centres[j])), uint32_t(j));
    }
    std::sort(others.begin(), others.end());
    double r = 0.0;
    const size_t near = std::min<size_t>(3, others.size());
    for (size_t j = 0; j < near; ++j) r += others[j].first;
    float radius = near > 0 ? float(r / near) : 0.0f;
    if (!(radius > 1e-4f)) radius = std::max(0.05f * dataset_.camera_extent, 1e-3f);
    for (size_t j = 0; j < std::min<size_t>(6, others.size()); ++j) node_neighbors_[k].push_back(others[j].second);
    for (int c = 0; c < 3; ++c) nodes_[k].position[c] = centres[k][c];
    nodes_[k].radius = radius;
    radii.push_back(radius);
  }
  std::nth_element(radii.begin(), radii.begin() + radii.size() / 2, radii.end());
  node_spacing_ = radii[radii.size() / 2];
  motion_.assign(dataset_.frames.size(), ViewMotion(k_count));
  motion_updates_.assign(dataset_.frames.size(), 0);
  train_before_.assign(dataset_.frames.size(), kNone);
  train_after_.assign(dataset_.frames.size(), kNone);
  size_t last = kNone;
  for (size_t f = 0; f < dataset_.frames.size(); ++f) {
    train_before_[f] = last;
    if (!dataset_.frames[f].holdout) last = f;
  }
  last = kNone;
  for (size_t f = dataset_.frames.size(); f-- > 0;) {
    train_after_[f] = last;
    if (!dataset_.frames[f].holdout) last = f;
  }
  stats_.motion_nodes = uint32_t(k_count);
  log_info("trainer: %zu motion nodes, spacing %.3f", k_count, node_spacing_);
}

void Trainer::bind_motion() {
  const uint32_t push[2] = {slots_, uint32_t(nodes_.size())};
  if (slots_ > 0) {
    ctx_.run([&](VkCommandBuffer cmd) {
      kernel(cmd, motion_bind_, {&params_, &motion_nodes_buf_, &motion_bind_buf_}, div_up(slots_, 128), 1, push,
             sizeof(push));
    });
  }
  bound_at_ = stats_.step;
}

bool Trainer::refines_motion_at(uint32_t step) const { return !nodes_.empty() && step >= options_.motion_from; }

bool Trainer::uses_motion() const {
  return std::any_of(motion_updates_.begin(), motion_updates_.end(), [](uint32_t u) { return u > 0; });
}

Trainer::ViewMotion Trainer::view_motion(size_t frame) const {
  if (motion_updates_.at(frame) > 0) return motion_[frame];
  ViewMotion out(nodes_.size());
  size_t a = train_before_[frame], b = train_after_[frame];
  if (a != kNone && motion_updates_[a] == 0) a = kNone;
  if (b != kNone && motion_updates_[b] == 0) b = kNone;
  if (a == kNone && b == kNone) return out;
  const double t = a == kNone ? 1.0 : b == kNone ? 0.0 : double(frame - a) / double(b - a);
  for (size_t k = 0; k < nodes_.size(); ++k) {
    for (int c = 0; c < 3; ++c) {
      const double ta = a == kNone ? 0.0 : motion_[a][k].turn[c], tb = b == kNone ? 0.0 : motion_[b][k].turn[c];
      const double sa = a == kNone ? 0.0 : motion_[a][k].shift[c], sb = b == kNone ? 0.0 : motion_[b][k].shift[c];
      out[k].turn[c] = float(a == kNone ? tb : b == kNone ? ta : ta + (tb - ta) * t);
      out[k].shift[c] = float(a == kNone ? sb : b == kNone ? sa : sa + (sb - sa) * t);
    }
  }
  return out;
}

void Trainer::start_motion(size_t frame) {
  if (motion_updates_.at(frame) > 0) return;
  // The subject moves smoothly: start where the views next to it got.
  const ViewMotion guess = view_motion(frame);
  for (size_t k = 0; k < nodes_.size(); ++k) {
    std::copy(guess[k].turn, guess[k].turn + 3, motion_[frame][k].turn);
    std::copy(guess[k].shift, guess[k].shift + 3, motion_[frame][k].shift);
  }
}

void Trainer::update_motion(uint32_t slot, size_t frame) {
  const size_t K = nodes_.size();
  ViewMotion& m = motion_.at(frame);
  // The image loss's gradient (train_motion_sum), then the penalties'.
  const float* g = static_cast<const float*>(motion_grads_[slot].mapped());
  std::vector<double> grad(K * 6);
  for (size_t k = 0; k < K; ++k) {
    for (int c = 0; c < 3; ++c) {
      grad[k * 6 + c] = g[k * 8 + c];
      grad[k * 6 + 3 + c] = g[k * 8 + 4 + c];
    }
  }
  last_motion_grad_ = grad;
  const double inv_s2 = 1.0 / (double(node_spacing_) * node_spacing_);
  const double decay = options_.motion_decay, rigid = options_.motion_rigidity;
  const double smooth = options_.motion_smoothness;
  for (size_t k = 0; k < K; ++k) {
    for (int c = 0; c < 3; ++c) {
      grad[k * 6 + c] += 2.0 * decay * m[k].shift[c] * inv_s2;
      grad[k * 6 + 3 + c] += 2.0 * decay * m[k].turn[c];
    }
  }
  if (rigid > 0.0) {
    // As rigid as possible: neighbour j should be where node k's motion takes
    // it, e = (n_j + s_j) - (n_k + s_k) - R_k (n_j - n_k). A turn u of R_k
    // changes e by (R_k d) x u.
    for (size_t k = 0; k < K; ++k) {
      const double w[3] = {m[k].turn[0], m[k].turn[1], m[k].turn[2]};
      double R[9];
      rotation_from(w, R);
      for (uint32_t j : node_neighbors_[k]) {
        double d[3], Rd[3], e[3];
        for (int c = 0; c < 3; ++c) d[c] = double(nodes_[j].position[c]) - nodes_[k].position[c];
        for (int r = 0; r < 3; ++r) {
          Rd[r] = R[r * 3] * d[0] + R[r * 3 + 1] * d[1] + R[r * 3 + 2] * d[2];
          e[r] = d[r] + m[j].shift[r] - m[k].shift[r] - Rd[r];
        }
        const double f = 2.0 * rigid * inv_s2;
        for (int c = 0; c < 3; ++c) {
          grad[j * 6 + c] += f * e[c];
          grad[k * 6 + c] -= f * e[c];
        }
        grad[k * 6 + 3] += f * (e[1] * Rd[2] - e[2] * Rd[1]);
        grad[k * 6 + 4] += f * (e[2] * Rd[0] - e[0] * Rd[2]);
        grad[k * 6 + 5] += f * (e[0] * Rd[1] - e[1] * Rd[0]);
      }
    }
  }
  if (options_.motion_center > 0.0f) {
    // The mean over the trained views (this one included), per node.
    std::vector<double> mean(K * 6, 0.0);
    double views = 0.0;
    for (size_t f : train_frames_) {
      if (motion_updates_[f] == 0 && f != frame) continue;
      views += 1.0;
      for (size_t k = 0; k < K; ++k) {
        for (int c = 0; c < 3; ++c) {
          mean[k * 6 + c] += motion_[f][k].shift[c];
          mean[k * 6 + 3 + c] += motion_[f][k].turn[c];
        }
      }
    }
    const double f = 2.0 * options_.motion_center / std::max(views, 1.0);
    for (size_t k = 0; k < K; ++k) {
      for (int c = 0; c < 3; ++c) {
        grad[k * 6 + c] += f * mean[k * 6 + c] * inv_s2;
        grad[k * 6 + 3 + c] += f * mean[k * 6 + 3 + c];
      }
    }
  }
  if (smooth > 0.0) {
    for (size_t other : {train_before_[frame], train_after_[frame]}) {
      if (other == kNone || motion_updates_[other] == 0) continue;
      for (size_t k = 0; k < K; ++k) {
        for (int c = 0; c < 3; ++c) {
          grad[k * 6 + c] += 2.0 * smooth * (m[k].shift[c] - motion_[other][k].shift[c]) * inv_s2;
          grad[k * 6 + 3 + c] += 2.0 * smooth * (m[k].turn[c] - motion_[other][k].turn[c]);
        }
      }
    }
  }
  // Adam with one second moment per vector (shift, turn), so a step keeps
  // the gradient's direction: directions the image barely constrains (along
  // the view) move little.
  const uint32_t n = ++motion_updates_[frame];
  const double c1 = 1.0 / (1.0 - std::pow(double(kBeta1), double(n)));
  const double c2 = 1.0 / (1.0 - std::pow(double(kBeta2), double(n)));
  const double t = options_.iterations > 0 ? std::min(1.0, double(stats_.step) / options_.iterations) : 1.0;
  const double fade = std::pow(double(std::max(options_.lr_motion_final, 1e-6f)), t);
  const double lr[2] = {options_.lr_motion_shift * dataset_.camera_extent * fade, options_.lr_motion_turn * fade};
  for (size_t k = 0; k < K; ++k) {
    NodeMotion& node = m[k];
    for (int part = 0; part < 2; ++part) {
      double sq = 0.0;
      for (int c = 0; c < 3; ++c) {
        const double gi = grad[k * 6 + part * 3 + c];
        node.m[part * 3 + c] = kBeta1 * node.m[part * 3 + c] + (1.0 - kBeta1) * gi;
        sq += gi * gi;
      }
      node.v[part] = kBeta2 * node.v[part] + (1.0 - kBeta2) * sq / 3.0;
      if (!(node.v[part] > 0.0)) continue;
      const double denom = std::sqrt(node.v[part] * c2) + 1e-30;
      float* value = part == 0 ? node.shift : node.turn;
      for (int c = 0; c < 3; ++c) value[c] -= float(lr[part] * node.m[part * 3 + c] * c1 / denom);
    }
  }
}

void Trainer::update_motion_stats() {
  double shift_sum = 0.0, turn_sum = 0.0, shift_max = 0.0, turn_max = 0.0;
  size_t n = 0;
  uint32_t updates = 0;
  for (size_t f : train_frames_) {
    if (motion_updates_[f] == 0) continue;
    updates += motion_updates_[f];
    for (const NodeMotion& m : motion_[f]) {
      const double shift = std::sqrt(double(m.shift[0]) * m.shift[0] + double(m.shift[1]) * m.shift[1] +
                                     double(m.shift[2]) * m.shift[2]);
      const double turn = std::sqrt(double(m.turn[0]) * m.turn[0] + double(m.turn[1]) * m.turn[1] +
                                    double(m.turn[2]) * m.turn[2]) * 180.0 / M_PI;
      shift_sum += shift;
      turn_sum += turn;
      shift_max = std::max(shift_max, shift);
      turn_max = std::max(turn_max, turn);
      ++n;
    }
  }
  stats_.motion_shift_mean = n ? shift_sum / n : 0.0;
  stats_.motion_turn_mean = n ? turn_sum / n : 0.0;
  stats_.motion_shift_max = shift_max;
  stats_.motion_turn_max = turn_max;
  stats_.motion_updates = updates;
}

void Trainer::node_motion(size_t frame, uint32_t node, float turn[3], float shift[3]) const {
  const NodeMotion& m = motion_.at(frame).at(node);
  std::copy(m.turn, m.turn + 3, turn);
  std::copy(m.shift, m.shift + 3, shift);
}

void Trainer::set_node_motion(size_t frame, uint32_t node, const float turn[3], const float shift[3]) {
  NodeMotion& m = motion_.at(frame).at(node);
  std::copy(turn, turn + 3, m.turn);
  std::copy(shift, shift + 3, m.shift);
  if (motion_updates_[frame] == 0) motion_updates_[frame] = 1;  // counts as trained: used as is
}

Trainer::MotionBinding Trainer::motion_binding() {
  MotionBinding b;
  if (nodes_.empty() || slots_ == 0) return b;
  std::vector<uint32_t> raw(size_t(slots_) * 8);
  ctx_.download(motion_bind_buf_, raw.data(), raw.size() * 4);
  b.nodes.resize(size_t(slots_) * 4);
  b.weights.resize(size_t(slots_) * 4);
  for (size_t i = 0; i < slots_; ++i) {
    for (int j = 0; j < 4; ++j) {
      b.nodes[i * 4 + j] = raw[i * 8 + j];
      std::memcpy(&b.weights[i * 4 + j], &raw[i * 8 + 4 + j], 4);
    }
  }
  return b;
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
