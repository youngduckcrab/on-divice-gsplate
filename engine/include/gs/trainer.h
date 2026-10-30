#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <string>
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
    Auto,            // the first of the variants below the driver accepts
    Subgroup,        // vec4 subgroup adds (fails if unavailable)
    SubgroupScalar,  // scalar subgroup adds (fails if unavailable)
    Shared,          // shared-memory tree, no subgroup operations (slower)
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
  uint32_t seed = 1;
};

// The defaults' step schedule (densification, opacity reset, SH degree) is
// tuned for 3000 steps; this scales it to `iterations`.
TrainOptions scaled_options(uint32_t iterations);

// GPU milliseconds of each pass in one profiled step.
struct PassTimes {
  double project = 0.0;    // projection + pair count
  double sort = 0.0;       // pair emission, radix sort, tile ranges
  double rasterize = 0.0;  // forward + loss
  double backward = 0.0;   // per-pair gradients
  double optimize = 0.0;   // per-gaussian backprop + Adam
  bool valid = false;
};

struct TrainStats {
  uint32_t step = 0;
  uint32_t gaussians = 0;  // alive
  uint32_t slots = 0;      // slots in use (alive + free)
  uint64_t pairs = 0;      // (tile, gaussian) pairs of the last step
  double l1 = 0.0;         // mean absolute error of the last step's view
  double psnr = 0.0;       // of the last step's view
  double step_ms = 0.0;    // exponential moving average of the wall-clock step time
  int sh_degree = 0;       // active
  uint32_t densify_events = 0;
  uint32_t spawned = 0;  // gaussians added by densification, in total
  uint32_t pruned = 0;   // and removed
};

struct EvalResult {
  double psnr = 0.0;  // mean over the held-out views
  double l1 = 0.0;
  uint32_t views = 0;
};

// Optimises gaussians against a dataset on the GPU (3DGS: tile rasteriser,
// L1 loss, Adam, adaptive density control). Uses no float atomics, so it runs
// on mobile GPUs; needs subgroup arithmetic in compute shaders.
class Trainer {
 public:
  // Throws std::runtime_error if the device cannot train.
  Trainer(VkContext& ctx, const Dataset& dataset, const SplatScene& initial, const TrainOptions& options);
  ~Trainer();
  Trainer(const Trainer&) = delete;
  Trainer& operator=(const Trainer&) = delete;

  // One optimisation step on the next training view, then densification on
  // schedule. With `profile`, the passes are submitted separately and timed
  // (see last_profile()), which costs a little extra.
  void step(bool profile = false);
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
  };
  DebugSnapshot debug_snapshot();
  uint32_t sh_stride() const { return sh_stride_; }
  // "subgroup" or "shared", and why the subgroup variant was not used, if it was not.
  const std::string& reduction() const { return reduction_; }
  const std::string& reduction_note() const { return reduction_note_; }

  static constexpr uint32_t kTileSize = 16;

 private:
  struct FrameData;
  void write_frame(const Camera& camera, const float background[3], bool masked_target, uint32_t sh_degree);
  // Renders `camera` into image_/tile losses; with `train`, also the backward pass and Adam step.
  void run_view(const Camera& camera, const uint8_t* target, const float background[3], bool masked_target,
                bool train, bool profile = false);
  void ensure_pair_capacity(uint64_t pairs);
  void record_scan(VkCommandBuffer cmd, const Buffer& values, const Buffer& prefix, uint32_t count,
                   uint32_t total_index);
  void densify();
  void reset_opacity();
  std::pair<double, double> read_losses();  // sum |e|, sum e^2

  VkContext& ctx_;
  Dataset dataset_;
  TrainOptions options_;
  TrainStats stats_;
  PassTimes profile_;
  uint32_t width_ = 0, height_ = 0, tiles_x_ = 0, tiles_y_ = 0;
  uint32_t capacity_ = 0;
  uint32_t backward_push_[2] = {32, 4};  // batch, max subgroups
  std::string reduction_;
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

  ComputePipeline project_, scan_local_, scan_blocks_, scan_add_, emit_, ranges_, forward_, backward_, step_;
  ComputePipeline densify_mark_, densify_spawn_, reset_opacity_;
  std::unique_ptr<RadixSort> sort_;

  Buffer frame_;  // FrameData, host-visible
  Buffer params_, sh_, m_params_, v_params_, m_sh_, v_sh_, stats_buf_;
  Buffer projected_, tiles_, offsets_, spawn_, free_list_, counters_, block_sums_, totals_;
  Buffer keys_, values_, pair_gaussian_, pair_grads_;
  Buffer target_, pixel_grad_, pixel_count_, image_, ranges_buf_, tile_loss_;
  Buffer readback_small_, readback_image_, readback_loss_;
};

}  // namespace gs
