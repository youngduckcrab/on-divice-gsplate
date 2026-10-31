#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "gs/camera.h"
#include "gs/dataset.h"
#include "gs/radix_sort.h"
#include "gs/splat_scene.h"
#include "gs/vk_context.h"

namespace gs {

struct TrainOptions {
  // How the backward pass sums gradients over a tile's pixels.
  enum class Reduction {
    Auto,            // the first subgroup variant the driver accepts, else SerialCut2
    Bucket,          // no subgroups: one thread per splat sweeping the tile's pixels, no reduction
    Bucket2,         // the same, two pixels per step (half the barriers)
    Subgroup,        // vec4 subgroup adds (fails if unavailable)
    SubgroupScalar,  // scalar subgroup adds (fails if unavailable)
    SubgroupLane4,   // scalar adds without subgroup id/elect, 4 splats per round (fails if unavailable)
    SubgroupLane1,   // the same, one splat per round
    SerialCut2,      // no subgroups: like SerialCut, two splats per round of barriers
    SerialCut,       // no subgroups: like Serial, stopping at the tile's last contributor
    Serial,          // no subgroups: one splat at a time, two-stage shared-memory sum
    Shared,          // no subgroups: batched splats, shared-memory tree per splat
  };

  uint32_t iterations = 3000;
  uint32_t max_gaussians = 300000;  // slot capacity
  int sh_degree = 3;
  uint32_t sh_degree_interval = 500;  // the active SH degree goes up by one every this many steps
  bool masked = true;                 // composite masked targets over the background (datasets with masks)
  bool random_background = true;      // otherwise `background`
  float background[3] = {0.0f, 0.0f, 0.0f};

  // Learning rates, 3DGS defaults; the position rate is scaled by the camera extent
  // and decays exponentially to the final rate.
  float lr_position = 1.6e-4f;
  float lr_position_final = 1.6e-6f;
  float lr_scale = 5e-3f;
  float lr_rotation = 1e-3f;
  float lr_opacity = 2.5e-2f;
  float lr_sh_dc = 2.5e-3f;
  float lr_sh_rest = 2.5e-3f / 20.0f;
  // Loss: (1 - lambda) L1 + lambda (1 - SSIM), lambda ramping linearly from
  // lambda_dssim at the first step to lambda_dssim_end at the last (< 0: no
  // ramp). 3D Gaussian Splatting keeps 0.2; on body (360x640, 600 steps,
  // held-out) the ramp to 1.0 gave 37.22 dB / SSIM 0.9871 against 36.14 dB /
  // 0.9810 for 0.2 throughout (and 34.96 / 0.9756 for L1 alone).
  float lambda_dssim = 0.2f;
  float lambda_dssim_end = 1.0f;
  // Pixels in the head region (estimate_head) weigh this much in the loss.
  // Views where the head covers much of the frame are left alone. Off by
  // default: on body (360x640, 600 steps) 3x left the head PSNR where it was
  // (31.09 -> 31.05 dB) with 4% more gaussians; Adam rescales each
  // parameter's steps, so a uniform weight mostly changes densification.
  // The region still gives the held-out head PSNR.
  HeadRegion head;
  float head_weight = 1.0f;

  // Camera pose refinement. Phone poses (ARCore) are off by up to
  // centimetres, which blurs and doubles the model. Each training view gets
  // a small correction (a rotation and a shift in its camera's axes), taken
  // by Adam from the same backward pass whenever the view is trained on, from
  // step `pose_from`. Before evaluation each held-out view is aligned the same
  // way with `eval_pose_steps` steps that move only its camera.
  bool refine_poses = false;
  uint32_t pose_from = 300;
  float lr_pose_shift = 1e-3f;   // per update, times the camera extent
  float lr_pose_turn = 1e-3f;    // radians per update
  float lr_pose_final = 1.0f;    // the rates decay exponentially to this fraction by the last step
  uint32_t eval_pose_steps = 20;
  float eval_pose_step = 0.005f;  // the alignment's first step, times the camera extent

  // Adaptive density control.
  uint32_t densify_from = 300;
  uint32_t densify_until = 2000;
  uint32_t densify_interval = 100;
  // 3DGS resets opacities every 3000 of 30000 steps. In short runs it makes the
  // model vanish mid-training and did not improve held-out PSNR (lizard and
  // head, 600 steps), so it is off by default.
  uint32_t opacity_reset_interval = 0;  // 0 disables
  float densify_grad_threshold = 2e-4f;
  float percent_dense = 0.01f;  // clone below, split above this fraction of the extent
  float prune_opacity = 0.005f;

  Reduction reduction = Reduction::Auto;
  // More variants known to give correct gradients on this GPU (see
  // choose_reduction). Early on and then every `retune_interval` steps the
  // trainer times all of them on the same training step and keeps the
  // fastest; their gradients agree, so switching does not change training.
  std::vector<Reduction> alternatives;
  uint32_t retune_first = 20;
  uint32_t retune_interval = 500;
  // Smallest pair buffer allocation; tests lower it to exercise overflow.
  uint32_t min_pair_capacity = 1 << 16;
  // Upper bound on the bucket backward's state buffer (0: the device's
  // binding limit); tests lower it to exercise the limit.
  uint64_t bucket_state_limit = 0;
  // Pairs to allocate for before the first step counts them (0: four per
  // gaussian); tests raise it to start with oversized buffers.
  uint64_t initial_pair_guess = 0;
  uint32_t seed = 1;
};

// The defaults' step schedule (densification, opacity reset, SH degree) is
// tuned for 3000 steps; this scales it to `iterations`.
TrainOptions scaled_options(uint32_t iterations);

// GPU milliseconds of each pass in one profiled step.
struct PassTimes {
  double project = 0.0;    // projection + pair count
  double sort = 0.0;       // pair emission, radix sort, tile ranges
  double rasterize = 0.0;  // forward + L1 loss
  double ssim = 0.0;       // SSIM and its gradient
  double backward = 0.0;   // per-pair gradients
  double optimize = 0.0;   // the three below
  double optimize_sum = 0.0;       // per-gaussian sum of its pairs' gradients
  double optimize_geometry = 0.0;  // per-gaussian backprop + Adam on 16 parameters
  double optimize_sh = 0.0;        // Adam on the SH coefficients, per float
  bool valid = false;
};

struct TrainStats {
  uint32_t step = 0;
  uint32_t gaussians = 0;  // alive
  uint32_t slots = 0;      // slots in use (alive + free)
  uint64_t pairs = 0;      // (tile, gaussian) pairs of the last step
  double l1 = 0.0;         // mean absolute error of the last step's view
  double psnr = 0.0;       // of the last step's view
  double ssim = 0.0;       // of the last step's view (0 when not computed: lambda_dssim 0)
  double loss = 0.0;       // the training loss of the last step's view
  double step_ms = 0.0;    // exponential moving average of the wall-clock step time
  int sh_degree = 0;       // active
  uint32_t densify_events = 0;
  uint32_t spawned = 0;  // gaussians added by densification, in total
  uint32_t pruned = 0;   // and removed
  // Gaussians whose emitted tiles differed from the projected count, over the
  // whole run. Both passes compute the same footprint, so 0.
  uint32_t tile_mismatch = 0;
  // Steps in a batch whose pairs did not fit the buffers (sized from earlier
  // steps, with headroom): their update was skipped, and the buffers grew.
  uint32_t overflow_steps = 0;
  // Pose refinement: the corrections of the training views so far.
  double pose_shift_mean = 0.0, pose_shift_max = 0.0;  // dataset units (metres for captures)
  double pose_turn_mean = 0.0, pose_turn_max = 0.0;    // degrees
  uint32_t pose_updates = 0;
  // Moving averages per training step: wall clock around the submit + wait,
  // GPU time from timestamps, and steps per submission.
  struct Timing {
    double wall_ms = 0.0;
    double gpu_ms = 0.0;
    double batch = 0.0;
  } timing;
};

// One timing of the backward variants on a training step.
struct RetuneResult {
  uint32_t step = 0;
  std::vector<std::pair<std::string, double>> ms;  // per variant, GPU ms
  std::string chosen;
};

struct EvalResult {
  double psnr = 0.0;  // mean over the held-out views (aligned first when refining poses)
  bool aligned = false;
  double psnr_unaligned = 0.0;  // at the poses as given, when aligned
  double ssim = 0.0;
  double head_psnr = 0.0;  // within the head circle, over the views that show it
  uint32_t head_views = 0;
  double l1 = 0.0;
  uint32_t views = 0;
};

// Optimises gaussians against a dataset on the GPU (3DGS: tile rasteriser,
// L1 loss, Adam, adaptive density control). Uses no float atomics, so it runs
// on mobile GPUs; needs subgroup arithmetic in compute shaders.
class Trainer {
 public:
  // Throws std::runtime_error if the device cannot train.
  // Takes the dataset by value: move it in to avoid a copy of its images.
  Trainer(VkContext& ctx, Dataset dataset, const SplatScene& initial, const TrainOptions& options);
  ~Trainer();
  Trainer(const Trainer&) = delete;
  Trainer& operator=(const Trainer&) = delete;

  // One optimisation step on the next training view, then densification on
  // schedule. With `profile`, the passes are submitted separately and timed
  // (see last_profile()), which costs a little extra.
  void step(bool profile = false);
  // Up to `count` steps (fewer at the end), recorded kBatch at a time into one
  // GPU submission: the CPU waits once per batch, not twice per step. Steps
  // before which the trainer reads results back (the first, retuning,
  // densification) end a batch. Returns the steps run.
  uint32_t steps(uint32_t count);
  static constexpr uint32_t kBatch = 10;
  const PassTimes& last_profile() const { return profile_; }
  bool finished() const { return stats_.step >= options_.iterations; }
  const TrainStats& stats() const { return stats_; }
  const TrainOptions& options() const { return options_; }
  const Dataset& dataset() const { return dataset_; }

  // The current model from `camera` (must be the dataset's resolution) over
  // `background`: RGBA8, alpha = coverage.
  std::vector<uint8_t> render(const Camera& camera, const float background[3]);
  // Held-out views over a black background.
  EvalResult evaluate();
  // A view's camera with its pose correction (the dataset's pose without refinement).
  Camera camera_for(size_t frame) const;
  // The alive gaussians, in the dataset's world (Y-up, RUB).
  SplatScene export_scene();

  // Testing: one step on `frame` over `background`. With all learning rates
  // zero the parameters stay put, and after the first step Adam's first moment
  // is (1 - beta1) * gradient, which first_step_gradients() returns.
  void step_on(size_t frame, const float background[3], bool profile = false);
  struct Gradients {
    std::vector<float> params;  // 16 per slot, the parameter layout of train_common.glsl
    std::vector<float> sh;      // sh_stride() per slot
  };
  Gradients first_step_gradients();
  // Testing: intermediate buffers of the most recent step.
  struct DebugSnapshot {
    std::vector<uint32_t> pixel_count;  // pairs up to the last contributor, per pixel
    std::vector<float> pixel_grad;      // 4 per pixel: dL/dC rgb, final transmittance
    std::vector<float> projected;       // 16 per slot
    std::vector<uint32_t> tiles;        // tiles touched per slot
    std::vector<uint32_t> offsets;      // first pair per slot
    std::vector<float> pair_grads;      // 12 per pair, pre-sort order
    std::vector<uint32_t> pair_tile;    // per pair, pre-sort order
  };
  DebugSnapshot debug_snapshot();
  uint32_t sh_stride() const { return sh_stride_; }
  // The backward variant in use ("serial_cut", "subgroup_lane4", ...), and why
  // subgroup variants were not used, if they were not.
  const std::string& reduction() const { return backwards_[active_].name; }
  const std::string& reduction_note() const { return reduction_note_; }
  // Timings of the backward variants on training steps (TrainOptions::alternatives).
  const std::vector<RetuneResult>& retunes() const { return retunes_; }
  // Testing: makes the named variant (one of those created) the active one.
  bool select_backward(const std::string& name);
  // Testing: the pose gradient of the latest pose update: dL/dd xyz, dL/dw xyz.
  const double* last_pose_gradient() const { return last_pose_grad_; }

  static constexpr uint32_t kTileSize = 16;

 private:
  struct FrameData;
  struct Backward {
    std::string name;
    ComputePipeline pipeline;
    uint32_t push[2] = {32, 4};  // batch, max subgroups
    bool bucket = false;         // needs the forward pass's bucket state; one workgroup per 8 buckets
  };
  // The variant to run: the active one, unless it is the bucket variant and
  // its state does not fit (then the first other one).
  size_t backward_to_run() const;
  // Per-step inputs live in slot `slot` of frames_/targets_; `step` (0-based)
  // sets the learning-rate schedule, SH degree and seed.
  void write_frame(uint32_t slot, const Camera& camera, const float background[3], bool masked_target,
                   uint32_t step, bool bucket_state, bool ssim, uint32_t pose_flags = 0);
  uint32_t sh_degree_at(uint32_t step) const;
  float lambda_at(uint32_t step) const;  // the D-SSIM weight of a step
  bool uses_ssim() const;                // any step has a D-SSIM part
  // Renders `camera` into image_/tile losses; with `train`, also the backward pass and Adam step.
  // With `retune`, every backward variant is timed on this step (implies profiling).
  // With `ssim`, also measures SSIM (training does whenever lambda_dssim > 0).
  void run_view(const Camera& camera, const uint8_t* target, const float background[3], bool masked_target,
                bool train, bool profile = false, bool retune = false, bool ssim = false, uint32_t pose_flags = 0);
  bool refines_pose_at(uint32_t step) const;
  // Adam step on `frame`'s pose correction from the gradient slot `slot` left.
  void update_pose(uint32_t slot, size_t frame, float lr_scale);
  struct PoseFix;
  void pose_gradient(uint32_t slot, double grad[6]) const;  // dL/dd, dL/dw summed over workgroups
  static void apply_pose_step(PoseFix& fix, const double step[6]);
  void update_pose_stats();
  void run_batch(uint32_t count);
  void next_view(size_t& frame, float background[3]);
  bool housekeeping_after(uint32_t step) const;  // densification or opacity reset after this step
  bool retune_due(uint32_t step) const;          // time the backward variants on this (1-based) step
  void after_step();                              // runs them
  // Pair buffers for `expected` pairs plus headroom; grows, never shrinks.
  void reserve_pairs(uint64_t expected);
  void refit_pairs();  // shrink the pair buffers to recent counts when bucket state no longer fits
  bool needs_bucket_state(bool train, bool retune) const;
  // The passes of one view, reading slot `slot`.
  void record_project(VkCommandBuffer cmd, uint32_t slot);
  void record_sort(VkCommandBuffer cmd, uint32_t slot);
  void record_rasterize(VkCommandBuffer cmd, uint32_t slot);
  // SSIM (`ssim`) and its gradient (`grad`), then copies of losses and counts to the slot.
  void record_loss(VkCommandBuffer cmd, uint32_t slot, bool ssim, bool grad);
  void record_backward(VkCommandBuffer cmd, uint32_t slot, const Backward& b);
  // The three optimizer kernels; `part` 0 records all, 1..3 one of them (profiling).
  void record_optimize(VkCommandBuffer cmd, uint32_t slot, int part = 0);
  void record_scan(VkCommandBuffer cmd, const Buffer& values, const Buffer& prefix, uint32_t count,
                   uint32_t total_index);
  // Reads slot `slot`'s pair count, tile mismatches and losses into stats;
  // true if the pairs did not fit (the step then left the model unchanged).
  bool read_slot(uint32_t slot);
  void densify();
  void reset_opacity();
  // Descriptor sets that stay valid across submissions, cached per pipeline
  // and buffer list; dropped when buffers are reallocated.
  VkDescriptorSet bound(const ComputePipeline& pipeline, std::initializer_list<const Buffer*> buffers);
  void kernel(VkCommandBuffer cmd, const ComputePipeline& pipeline, std::initializer_list<const Buffer*> buffers,
              uint32_t groups_x, uint32_t groups_y = 1, const void* push = nullptr, uint32_t push_size = 0);
  void reset_sets();

  VkContext& ctx_;
  Dataset dataset_;
  TrainOptions options_;
  TrainStats stats_;
  PassTimes profile_;
  uint32_t width_ = 0, height_ = 0, tiles_x_ = 0, tiles_y_ = 0;
  uint32_t capacity_ = 0;
  std::vector<Backward> backwards_;  // the requested variant, then the alternatives that compiled
  size_t active_ = 0;
  std::vector<RetuneResult> retunes_;
  bool bucket_fits_ = true;  // bucket_state_ could be sized for the pair capacity
  // Of the last slot read: sum |e|, e^2, SSIM, w |e|, w SSIM, w, head e^2, head pixels.
  double last_losses_[8] = {};
  void set_view_stats();                       // l1, psnr, ssim, loss from last_losses_
  uint64_t max_pairs_seen_ = 0;
  uint64_t recent_pairs_max_ = 0;  // since the last retune
  uint32_t last_slot_ = 0;  // slot of the last step, whose frame data densify() reuses
  VkDescriptorPool set_pool_ = VK_NULL_HANDLE;
  std::map<std::vector<uint64_t>, VkDescriptorSet> sets_;
  RadixSort::Binding sort_binding_;
  bool sort_bound_ = false;
  std::string reduction_note_;
  uint32_t sh_stride_ = 0;  // floats per slot
  uint32_t slots_ = 0;
  uint32_t alive_ = 0;
  uint64_t pair_capacity_ = 0;
  uint64_t max_pairs_ = 0;
  std::mt19937 rng_;
  std::vector<size_t> train_frames_;
  std::vector<size_t> order_;
  size_t order_pos_ = 0;

  ComputePipeline project_, scan_local_, scan_blocks_, scan_add_, emit_, dispatch_args_, ranges_, forward_, ssim_,
      ssim_grad_, pair_sum_, step_, step_sh_;
  ComputePipeline densify_mark_, densify_spawn_, reset_opacity_;
  std::unique_ptr<RadixSort> sort_;

  std::vector<Buffer> frames_;  // FrameData per batch slot, host-visible
  Buffer params_, sh_, m_params_, v_params_, m_sh_, v_sh_, stats_buf_;
  Buffer projected_, tiles_, offsets_, spawn_, free_list_, counters_, mismatch_, block_sums_, totals_;
  Buffer keys_, values_, pair_gaussian_, pair_grads_;
  std::vector<Buffer> targets_;  // per batch slot
  Buffer pixel_grad_, pixel_count_, image_, ranges_buf_, tile_loss_, tile_count_;
  Buffer bucket_state_, bucket_list_, pixel_dot_;
  Buffer grad_sum_, sh_input_;
  Buffer ssim_input_, ssim_maps_;  // per pixel: render and target colour; SSIM gradient parts A, B, C
  Buffer dispatch_args_buf_;  // indirect group counts of the passes over pairs (train_dispatch_args)  // per slot: summed pair gradients; colour gradient and view direction
  std::vector<Buffer> readback_smalls_, readback_losses_;  // per batch slot
  std::vector<Buffer> pose_grads_;  // per batch slot: 2 vec4 per train_step workgroup
  struct PoseFix {
    float rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // applied after the dataset's rotation (camera axes)
    float shift[3] = {0, 0, 0};
    double m[6] = {}, v[6] = {};  // Adam moments: shift xyz, turn xyz
    uint32_t updates = 0;
  };
  std::vector<PoseFix> pose_fix_;  // per dataset frame
  double last_pose_grad_[6] = {0, 0, 0, 0, 0, 0};
  Buffer readback_image_;
};

}  // namespace gs
