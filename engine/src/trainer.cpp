#include "gs/trainer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "gs/log.h"
#include "shaders/radix_scan.h"
#include "shaders/scan_add.h"
#include "shaders/scan_local.h"
#include "shaders/train_backward.h"
#include "shaders/train_backward_cut.h"
#include "shaders/train_backward_cut2.h"
#include "shaders/train_backward_scalar.h"
#include "shaders/train_backward_serial.h"
#include "shaders/train_backward_sg1.h"
#include "shaders/train_backward_sg4.h"
#include "shaders/train_backward_shared.h"
#include "shaders/train_densify_mark.h"
#include "shaders/train_densify_spawn.h"
#include "shaders/train_emit.h"
#include "shaders/train_forward.h"
#include "shaders/train_project.h"
#include "shaders/train_ranges.h"
#include "shaders/train_reset_opacity.h"
#include "shaders/train_step.h"

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
};

namespace {

constexpr float kBeta1 = 0.9f;
constexpr float kBeta2 = 0.999f;
constexpr float kEpsilon = 1e-15f;
constexpr uint32_t kFlagMaskedTarget = 1;
constexpr uint32_t kScanBlock = 1024;
constexpr uint32_t kPairGradBytes = 48;
constexpr uint32_t kBackwardBindings = 9;  // the variants use up to 9 (serial_cut reads tile counts)

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

Trainer::Trainer(VkContext& ctx, const Dataset& dataset, const SplatScene& initial, const TrainOptions& options)
    : ctx_(ctx), dataset_(dataset), options_(options), rng_(options.seed) {
  const DeviceInfo& info = ctx.info();
  if (dataset.frames.empty() || dataset.width == 0) throw std::runtime_error("데이터셋이 비어 있음");
  options_.sh_degree = std::clamp(options_.sh_degree, 0, 3);
  options_.masked = options_.masked && dataset.has_masks;

  width_ = dataset.width;
  height_ = dataset.height;
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

  for (size_t i = 0; i < dataset.frames.size(); ++i) {
    if (!dataset.frames[i].holdout) train_frames_.push_back(i);
  }
  if (train_frames_.empty()) {
    for (size_t i = 0; i < dataset.frames.size(); ++i) train_frames_.push_back(i);
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
  scan_blocks_ = make("radix_scan", shaders::radix_scan, 1, sizeof(uint32_t));
  scan_add_ = make("scan_add", shaders::scan_add, 4, sizeof(ScanPush));
  emit_ = make("train_emit", shaders::train_emit, 9, 0);
  ranges_ = make("train_ranges", shaders::train_ranges, 2, sizeof(uint32_t));
  forward_ = make("train_forward", shaders::train_forward, 11, 0);
  step_ = make("train_step", shaders::train_step, 12, 0);
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
  frame_ = ctx.create_buffer(sizeof(FrameData), MemoryKind::Upload);
  params_ = ctx.create_buffer(cap * 64, MemoryKind::Device);
  sh_ = ctx.create_buffer(sh_bytes, MemoryKind::Device);
  m_params_ = ctx.create_buffer(cap * 64, MemoryKind::Device);
  v_params_ = ctx.create_buffer(cap * 64, MemoryKind::Device);
  m_sh_ = ctx.create_buffer(sh_bytes, MemoryKind::Device);
  v_sh_ = ctx.create_buffer(sh_bytes, MemoryKind::Device);
  stats_buf_ = ctx.create_buffer(cap * 16, MemoryKind::Device);
  projected_ = ctx.create_buffer(cap * 64, MemoryKind::Device);
  tiles_ = ctx.create_buffer(cap * 4, MemoryKind::Device);
  offsets_ = ctx.create_buffer(cap * 4, MemoryKind::Device);
  spawn_ = ctx.create_buffer(cap * 4, MemoryKind::Device);
  free_list_ = ctx.create_buffer(cap * 4, MemoryKind::Device);
  counters_ = ctx.create_buffer(16, MemoryKind::Device);
  mismatch_ = ctx.create_buffer(16, MemoryKind::Device);
  block_sums_ = ctx.create_buffer(std::max<uint64_t>(div_up(cap, kScanBlock), 1) * 4, MemoryKind::Device);
  totals_ = ctx.create_buffer(16, MemoryKind::Device);
  target_ = ctx.create_buffer(pixels * 4, MemoryKind::Upload);
  pixel_grad_ = ctx.create_buffer(pixels * 16, MemoryKind::Device);
  pixel_count_ = ctx.create_buffer(pixels * 4, MemoryKind::Device);
  image_ = ctx.create_buffer(pixels * 4, MemoryKind::Device);
  ranges_buf_ = ctx.create_buffer((tiles + 1) * 8, MemoryKind::Device);  // + the tile emit pads into
  tile_loss_ = ctx.create_buffer(tiles * 16, MemoryKind::Device);
  tile_count_ = ctx.create_buffer(tiles * 4, MemoryKind::Device);
  readback_small_ = ctx.create_buffer(16, MemoryKind::Readback);
  readback_image_ = ctx.create_buffer(pixels * 4, MemoryKind::Readback);
  readback_loss_ = ctx.create_buffer(tiles * 16, MemoryKind::Readback);

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

Trainer::~Trainer() = default;

void Trainer::write_frame(const Camera& camera, const float background[3], bool masked_target, uint32_t sh_degree) {
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
  f.model[1] = sh_degree;
  f.model[2] = sh_stride_;
  f.model[3] = masked_target ? kFlagMaskedTarget : 0;
  for (int c = 0; c < 3; ++c) f.background[c] = background[c];
  f.background[3] = 1.0f / (3.0f * float(width_) * float(height_));

  const float t = options_.iterations > 0 ? std::min(1.0f, float(stats_.step) / float(options_.iterations)) : 1.0f;
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
  const double n = double(stats_.step) + 1.0;
  f.bias[0] = float(1.0 / (1.0 - std::pow(double(kBeta1), n)));
  f.bias[1] = float(1.0 / (1.0 - std::pow(double(kBeta2), n)));

  const bool prune_big = options_.opacity_reset_interval > 0 && stats_.step > options_.opacity_reset_interval;
  f.densify0[0] = options_.densify_grad_threshold;
  f.densify0[1] = options_.percent_dense * dataset_.camera_extent;
  f.densify0[2] = options_.prune_opacity;
  f.densify0[3] = prune_big ? 0.1f * dataset_.camera_extent : 0.0f;
  f.densify1[0] = prune_big ? 20.0f : 0.0f;
  const uint32_t seed = options_.seed * 9781u + stats_.step * 6271u;
  std::memcpy(&f.densify1[1], &seed, 4);
  f.densify1[2] = 1.6f;
  std::memcpy(frame_.mapped(), &f, sizeof(f));
}

void Trainer::record_scan(VkCommandBuffer cmd, const Buffer& values, const Buffer& prefix, uint32_t count,
                          uint32_t total_index) {
  const uint32_t blocks = div_up(count, kScanBlock);
  ScanPush push{count, total_index};
  ctx_.dispatch(cmd, scan_local_, {&values, &prefix, &block_sums_, &totals_}, push, blocks);
  VkContext::barrier(cmd);
  ctx_.dispatch(cmd, scan_blocks_, {&block_sums_}, blocks, 1);
  VkContext::barrier(cmd);
  ctx_.dispatch(cmd, scan_add_, {&values, &prefix, &block_sums_, &totals_}, push, blocks);
  VkContext::barrier(cmd);
}

void Trainer::ensure_pair_capacity(uint64_t pairs) {
  if (pairs <= pair_capacity_) return;
  if (pairs > max_pairs_) {
    throw std::runtime_error("타일-가우시안 쌍이 GPU 버퍼 한도를 넘음 (" + std::to_string(pairs) + ")");
  }
  const uint64_t n = std::min<uint64_t>(max_pairs_, std::max<uint64_t>(pairs + pairs / 4, 1 << 16));
  keys_ = ctx_.create_buffer(n * 4, MemoryKind::Device);
  values_ = ctx_.create_buffer(n * 4, MemoryKind::Device);
  pair_gaussian_ = ctx_.create_buffer(n * 4, MemoryKind::Device);
  pair_grads_ = ctx_.create_buffer(n * kPairGradBytes, MemoryKind::Device);
  sort_->reserve(static_cast<uint32_t>(n));
  pair_capacity_ = n;
}

void Trainer::run_view(const Camera& camera, const uint8_t* target, const float background[3], bool masked_target,
                       bool train, bool profile, bool retune) {
  const uint32_t degree =
      std::min<uint32_t>(options_.sh_degree, options_.sh_degree_interval > 0 ? stats_.step / options_.sh_degree_interval
                                                                             : options_.sh_degree);
  write_frame(camera, background, masked_target, degree);
  stats_.sh_degree = int(degree);
  if (target != nullptr) std::memcpy(target_.mapped(), target, size_t(width_) * height_ * 4);

  // Phase 1: project and count pairs.
  const RunTiming project_time = ctx_.run([&](VkCommandBuffer cmd) {
    if (slots_ > 0) {
      dispatch_kernel(ctx_, cmd, project_, {&frame_, &params_, &sh_, &projected_, &tiles_}, div_up(slots_, 128));
      VkContext::barrier(cmd);
      record_scan(cmd, tiles_, offsets_, slots_, 0);
    } else {
      vkCmdFillBuffer(cmd, totals_.handle(), 0, 16, 0);
      VkContext::barrier(cmd);
    }
    VkBufferCopy region{0, 0, 4};
    vkCmdCopyBuffer(cmd, totals_.handle(), readback_small_.handle(), 1, &region);
    VkBufferCopy mismatch{0, 4, 4};
    vkCmdCopyBuffer(cmd, mismatch_.handle(), readback_small_.handle(), 1, &mismatch);
    VkContext::barrier_to_host(cmd);
  });
  uint32_t pairs = 0;
  std::memcpy(&pairs, readback_small_.mapped(), 4);
  std::memcpy(&stats_.tile_mismatch, static_cast<const uint8_t*>(readback_small_.mapped()) + 4, 4);
  ensure_pair_capacity(std::max<uint32_t>(pairs, 1));
  stats_.pairs = pairs;

  // Phase 2: sort, rasterise, and (training) back-propagate and step.
  const uint32_t key_bits = 16 + bits_for(tiles_x_ * tiles_y_ + 1);  // + the padding tile
  auto sort = [&](VkCommandBuffer cmd) {
    vkCmdFillBuffer(cmd, ranges_buf_.handle(), 0, VK_WHOLE_SIZE, 0);
    if (pairs > 0) {
      dispatch_kernel(ctx_, cmd, emit_,
                      {&frame_, &projected_, &tiles_, &offsets_, &keys_, &values_, &pair_gaussian_, &pair_grads_,
                       &mismatch_},
                      div_up(slots_, 128));
      VkContext::barrier(cmd);
      sort_->record(cmd, keys_, values_, pairs, key_bits);
      ctx_.dispatch(cmd, ranges_, {&keys_, &ranges_buf_}, pairs, div_up(pairs, 256));
    }
    VkContext::barrier(cmd);
  };
  auto rasterize = [&](VkCommandBuffer cmd) {
    dispatch_kernel(ctx_, cmd, forward_,
                    {&frame_, &projected_, &ranges_buf_, &values_, &pair_gaussian_, &target_, &pixel_grad_,
                     &pixel_count_, &image_, &tile_loss_, &tile_count_},
                    tiles_x_, tiles_y_);
    VkContext::barrier(cmd);
    VkBufferCopy loss{0, 0, tile_loss_.size()};
    vkCmdCopyBuffer(cmd, tile_loss_.handle(), readback_loss_.handle(), 1, &loss);
    VkContext::barrier_to_host(cmd);
  };
  auto backward_with = [&](VkCommandBuffer cmd, const Backward& b) {
    ctx_.dispatch(cmd, b.pipeline,
                  {&frame_, &projected_, &ranges_buf_, &values_, &pair_gaussian_, &pixel_grad_, &pixel_count_,
                   &pair_grads_, &tile_count_},
                  b.push, tiles_x_, tiles_y_);
    VkContext::barrier(cmd);
  };
  auto backward = [&](VkCommandBuffer cmd) { backward_with(cmd, backwards_[active_]); };
  auto optimize = [&](VkCommandBuffer cmd) {
    if (slots_ == 0) return;
    dispatch_kernel(ctx_, cmd, step_,
                    {&frame_, &params_, &sh_, &m_params_, &v_params_, &m_sh_, &v_sh_, &tiles_, &offsets_,
                     &pair_grads_, &projected_, &stats_buf_},
                    div_up(slots_, 128));
    VkContext::barrier(cmd);
  };

  if (!profile && !retune) {
    const RunTiming main_time = ctx_.run([&](VkCommandBuffer cmd) {
      sort(cmd);
      rasterize(cmd);
      if (train) {
        backward(cmd);
        optimize(cmd);
      }
    });
    if (train) {
      auto ema = [](double& avg, double v) { avg = avg == 0.0 ? v : 0.95 * avg + 0.05 * v; };
      ema(stats_.timing.project_wall_ms, project_time.cpu_ms);
      ema(stats_.timing.project_gpu_ms, std::max(project_time.gpu_ms, 0.0));
      ema(stats_.timing.main_wall_ms, main_time.cpu_ms);
      ema(stats_.timing.main_gpu_ms, std::max(main_time.gpu_ms, 0.0));
    }
    return;
  }
  profile_ = PassTimes{};
  profile_.project = project_time.best_ms();
  profile_.sort = ctx_.run(sort).best_ms();
  profile_.rasterize = ctx_.run(rasterize).best_ms();
  if (train && retune && backwards_.size() > 1) {
    // Each variant twice, in mirrored order (A B C C B A), so a clock change
    // during the measurement favours none. The backward pass only writes
    // pair_grads, and every variant writes the same values.
    const size_t n = backwards_.size();
    std::vector<double> ms(n, 0.0);
    for (size_t k = 0; k < 2 * n; ++k) {
      const size_t v = k < n ? k : 2 * n - 1 - k;
      ms[v] += 0.5 * ctx_.run([&](VkCommandBuffer cmd) { backward_with(cmd, backwards_[v]); }).best_ms();
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
    profile_.optimize = ctx_.run(optimize).best_ms();
  } else if (train) {
    profile_.backward = ctx_.run(backward).best_ms();
    profile_.optimize = ctx_.run(optimize).best_ms();
  }
  profile_.valid = true;
}

std::pair<double, double> Trainer::read_losses() {
  const float* v = static_cast<const float*>(readback_loss_.mapped());
  double abs_sum = 0.0, sq_sum = 0.0;
  for (uint32_t t = 0; t < tiles_x_ * tiles_y_; ++t) {
    abs_sum += v[t * 4];
    sq_sum += v[t * 4 + 1];
  }
  return {abs_sum, sq_sum};
}

void Trainer::step(bool profile) {
  if (order_pos_ >= order_.size()) {
    order_ = train_frames_;
    std::shuffle(order_.begin(), order_.end(), rng_);
    order_pos_ = 0;
  }
  float bg[3];
  if (options_.random_background) {
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (float& c : bg) c = u(rng_);
  } else {
    std::copy(options_.background, options_.background + 3, bg);
  }
  step_on(order_[order_pos_++], bg, profile);
}

void Trainer::step_on(size_t frame_index, const float background[3], bool profile) {
  const auto start = std::chrono::steady_clock::now();
  const DatasetFrame& frame = dataset_.frames.at(frame_index);
  // Steps are numbered from 1; time the backward variants on the scheduled ones.
  const uint32_t next = stats_.step + 1;
  const bool retune = backwards_.size() > 1 &&
                      (next == options_.retune_first ||
                       (options_.retune_interval > 0 && next % options_.retune_interval == 0));
  run_view(dataset_camera(dataset_, frame), frame.rgba.data(), background, options_.masked, true, profile, retune);
  const auto [abs_sum, sq_sum] = read_losses();
  const double n = 3.0 * width_ * height_;
  stats_.l1 = abs_sum / n;
  stats_.psnr = -10.0 * std::log10(std::max(sq_sum / n, 1e-12));
  ++stats_.step;

  const uint32_t t = stats_.step;
  if (t < options_.densify_until) {
    if (t > options_.densify_from && options_.densify_interval > 0 && t % options_.densify_interval == 0) densify();
    if (options_.opacity_reset_interval > 0 && t % options_.opacity_reset_interval == 0) reset_opacity();
  }
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  stats_.step_ms = stats_.step_ms == 0.0 ? ms : 0.9 * stats_.step_ms + 0.1 * ms;
}

void Trainer::densify() {
  // The frame data still describes the last step; refresh it for the densify thresholds.
  FrameData* f = static_cast<FrameData*>(frame_.mapped());
  const bool prune_big = options_.opacity_reset_interval > 0 && stats_.step > options_.opacity_reset_interval;
  f->densify0[3] = prune_big ? 0.1f * dataset_.camera_extent : 0.0f;
  f->densify1[0] = prune_big ? 20.0f : 0.0f;
  f->model[0] = slots_;

  const uint32_t capacity = capacity_;
  ctx_.run([&](VkCommandBuffer cmd) {
    vkCmdFillBuffer(cmd, counters_.handle(), 0, 16, 0);
    VkContext::barrier(cmd);
    std::initializer_list<const Buffer*> buffers = {&frame_, &params_,    &sh_,    &m_params_,  &v_params_, &m_sh_,
                                                    &v_sh_,  &stats_buf_, &spawn_, &free_list_, &counters_};
    dispatch_kernel(ctx_, cmd, densify_mark_, buffers, div_up(slots_, 128));
    VkContext::barrier(cmd);
    ctx_.dispatch(cmd, densify_spawn_, buffers, capacity, div_up(slots_, 128));
    VkContext::barrier(cmd);
    vkCmdFillBuffer(cmd, stats_buf_.handle(), 0, VK_WHOLE_SIZE, 0);
    VkBufferCopy region{0, 0, 16};
    vkCmdCopyBuffer(cmd, counters_.handle(), readback_small_.handle(), 1, &region);
    VkContext::barrier_to_host(cmd);
  });
  uint32_t counters[4];
  std::memcpy(counters, readback_small_.mapped(), sizeof(counters));
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
  static_cast<FrameData*>(frame_.mapped())->model[0] = slots_;
  ctx_.run([&](VkCommandBuffer cmd) {
    ctx_.dispatch(cmd, reset_opacity_, {&frame_, &params_, &m_params_, &v_params_}, max_logit, div_up(slots_, 128));
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
  double psnr_sum = 0.0, l1_sum = 0.0;
  const double n = 3.0 * width_ * height_;
  for (const DatasetFrame& frame : dataset_.frames) {
    if (!frame.holdout) continue;
    run_view(dataset_camera(dataset_, frame), frame.rgba.data(), black, options_.masked, false);
    const auto [abs_sum, sq_sum] = read_losses();
    psnr_sum += -10.0 * std::log10(std::max(sq_sum / n, 1e-12));
    l1_sum += abs_sum / n;
    ++result.views;
  }
  if (result.views > 0) {
    result.psnr = psnr_sum / result.views;
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
