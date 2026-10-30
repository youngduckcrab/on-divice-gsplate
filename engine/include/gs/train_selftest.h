#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "gs/camera.h"
#include "gs/dataset.h"
#include "gs/trainer.h"
#include "gs/vk_context.h"

namespace gs {

// Checks of the trainer against a double-precision CPU reference. The unit
// tests run them on lavapipe; the app runs them on the phone's own GPU and
// driver (trainer_self_test_json) to tell driver problems from training ones.

// Small synthetic capture: cameras orbiting the origin at 3 m, random noise
// targets (with a random mask when `masks`); every 8th frame held out.
Dataset make_test_dataset(uint32_t width, uint32_t height, size_t frames, bool masks, uint32_t seed);
// `n` random anisotropic gaussians within 0.7 m of the origin.
SplatScene make_test_scene(uint32_t n, int sh_degree, uint32_t seed);
// All learning rates zero and no densification: a step only records gradients.
TrainOptions frozen_train_options(int sh_degree, bool masked);

// CPU render of the training rasteriser (train_project + train_forward) and its
// L1 loss against `target`. Parameters use the trainer's 16-float slot layout.
double reference_loss(const Camera& camera, const std::vector<double>& params, const std::vector<double>& sh,
                      int sh_stride, int sh_degree, const std::vector<uint8_t>& target, const float background[3],
                      bool masked, std::vector<double>* image = nullptr);

struct GradientGroup {
  std::string name;
  double max_ref = 0.0;  // largest finite-difference gradient
  double max_err = 0.0;  // largest |GPU - finite difference|
  int bad = 0;           // beyond 5% + 2e-6
  int checked = 0;
};

// Where a wrong gradient comes from: each stage of the step checked on its own.
struct StageReport {
  uint32_t covered = 0;           // pixels with final transmittance < 0.999
  uint32_t covered_no_count = 0;  // ...whose last-contributor count is 0 (forward bookkeeping lost)
  uint32_t max_count = 0;
  uint64_t pairs = 0;
  uint64_t pairs_nonzero = 0;     // pairs with any nonzero gradient (backward output)
  int colour_checked = 0;
  int backward_colour_bad = 0;    // per-gaussian sums of the pairs' dL/dcolour vs the reference
  int step_colour_bad = 0;        // the step kernel's SH DC gradient vs those sums
};

struct GradientReport {
  std::string reduction;  // backward variant that ran
  bool masked = false;
  double image_max_diff = 0.0;  // forward image vs reference, 0..1
  double loss_gpu = 0.0;
  double loss_ref = 0.0;
  std::vector<GradientGroup> groups;
  StageReport stages;
  bool ok() const;
};

// One step on a random 10-gaussian scene (64x48); every parameter's GPU
// gradient against central differences of the reference.
GradientReport check_gradients(VkContext& ctx, bool masked, TrainOptions::Reduction reduction, int sh_degree = 2);

struct FitReport {
  std::string reduction;
  double psnr_before = 0.0;  // held-out
  double psnr_after = 0.0;
  double ms_per_step = 0.0;
  uint32_t steps = 0;
  bool ok() const { return psnr_after > 35.0 && psnr_after > psnr_before + 8.0; }
};

// Trains a perturbed copy of a known scene back to it (96x72, 24 views).
FitReport fit_known_scene(VkContext& ctx, uint32_t steps = 600,
                          TrainOptions::Reduction reduction = TrainOptions::Reduction::Auto);

// The shared-memory and subgroup primitives the kernels use, on this GPU.
struct ProbeReport {
  uint32_t atomic_max = 0;  // expected 250
  uint32_t atomic_add = 0;  // expected 32640
  uint32_t tree_max = 0;    // expected 250
  bool shared_ok = false;
  bool subgroup_compiled = false;
  std::string subgroup_error;
  bool subgroup_ok = false;
  uint32_t subgroups = 0;     // per 256-invocation workgroup
  uint32_t subgroup_size = 0;
  // Which single subgroup features compile: basic builtins, elect, subgroup id,
  // float add, ballot ("ok" or the error).
  std::vector<std::pair<std::string, std::string>> subgroup_parts;
};
ProbeReport probe_gpu_ops(VkContext& ctx);

// GPU milliseconds of one backward pass on a mid-size synthetic load.
double benchmark_backward(VkContext& ctx, TrainOptions::Reduction reduction);

// The fastest backward variant (benchmark_backward) among those whose
// gradients match the CPU reference on this GPU; the slow fallbacks only if
// none does. `log` says what happened to each one tried.
struct ReductionChoice {
  bool found = false;
  TrainOptions::Reduction reduction = TrainOptions::Reduction::Serial;
  std::string name;
  std::string log;
  // The other correct, non-fallback variants, fastest first: candidates for
  // TrainOptions::alternatives.
  std::vector<TrainOptions::Reduction> others;
};
ReductionChoice choose_reduction(VkContext& ctx);

// GPU probes, gradient checks for each backward variant and the fit, as JSON with "ok".
std::string trainer_self_test_json(VkContext& ctx);

}  // namespace gs
