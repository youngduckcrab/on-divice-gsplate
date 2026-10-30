// Trainer tests: the GPU forward pass and gradients against a CPU reference
// (central finite differences), the dataset loader, and a small end-to-end
// training run. CI runs them on lavapipe.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "gs/dataset.h"
#include "gs/train_selftest.h"
#include "gs/trainer.h"
#include "gs/vk_context.h"
#include "test_util.h"

namespace {

constexpr uint32_t kWidth = 64;
constexpr uint32_t kHeight = 48;

void test_gradients(gs::VkContext& ctx, bool masked,
                    gs::TrainOptions::Reduction reduction = gs::TrainOptions::Reduction::Auto) {
  const gs::GradientReport r = gs::check_gradients(ctx, masked, reduction);
  EXPECT(r.image_max_diff < 2.5 / 255.0, "forward image differs from the reference by %.4f", r.image_max_diff);
  EXPECT(std::fabs(r.loss_gpu - r.loss_ref) < 1e-4, "loss %.6f, reference %.6f", r.loss_gpu, r.loss_ref);
  for (const gs::GradientGroup& g : r.groups) {
    std::printf("  %-9s %s %-8s: max |fd| %.3g, max err %.3g, %d/%d off\n", g.name.c_str(),
                masked ? "masked" : "plain ", r.reduction.c_str(), g.max_ref, g.max_err, g.bad, g.checked);
    EXPECT(g.bad <= g.checked / 20, "%s gradients: %d of %d off (masked %d, %s)", g.name.c_str(), g.bad, g.checked,
           int(masked), r.reduction.c_str());
  }
  const gs::StageReport& st = r.stages;
  std::printf("  stages %s %-15s: covered %u (no count %u), max count %u, pairs %llu (nonzero %llu), "
              "colour bad backward %d / step %d of %d\n",
              masked ? "masked" : "plain ", r.reduction.c_str(), st.covered, st.covered_no_count, st.max_count,
              (unsigned long long)st.pairs, (unsigned long long)st.pairs_nonzero, st.backward_colour_bad,
              st.step_colour_bad, st.colour_checked);
  EXPECT(st.covered > 0 && st.covered_no_count == 0, "forward bookkeeping: %u covered, %u without count",
         st.covered, st.covered_no_count);
  EXPECT(st.pairs_nonzero > 0, "backward wrote no gradients");
  EXPECT(st.backward_colour_bad == 0 && st.step_colour_bad == 0, "stage colour checks: backward %d, step %d",
         st.backward_colour_bad, st.step_colour_bad);
  EXPECT(r.ok(), "gradient report not ok");
}

void test_cutoff_matches_serial(gs::VkContext& ctx) {
  // Dense, nearly opaque gaussians: pixels saturate, so the cut variant skips
  // pairs behind them. Gradients must equal the full serial variant's.
  gs::Dataset ds = gs::make_test_dataset(kWidth, kHeight, 1, false, 7);
  gs::SplatScene scene = gs::make_test_scene(80, 1, 5);
  for (float& v : scene.opacity_logits) v += 4.0f;
  for (float& v : scene.log_scales) v += 0.7f;
  const float bg[3] = {0.1f, 0.2f, 0.3f};
  constexpr int kVariants = 5;
  std::vector<float> grads[kVariants];
  float min_t = 1.0f;
  const gs::TrainOptions::Reduction variants[kVariants] = {
      gs::TrainOptions::Reduction::Serial, gs::TrainOptions::Reduction::SerialCut,
      gs::TrainOptions::Reduction::SerialCut2, gs::TrainOptions::Reduction::SubgroupLane4,
      gs::TrainOptions::Reduction::SubgroupLane1};
  for (int v = 0; v < kVariants; ++v) {
    gs::TrainOptions o = gs::frozen_train_options(1, false);
    o.reduction = variants[v];
    gs::Trainer trainer(ctx, ds, scene, o);
    trainer.step_on(0, bg);
    grads[v] = trainer.first_step_gradients().params;
    const gs::Trainer::DebugSnapshot snap = trainer.debug_snapshot();
    for (size_t p = 0; p < snap.pixel_count.size(); ++p) min_t = std::min(min_t, snap.pixel_grad[p * 4 + 3]);
  }
  double max_diff = 0.0, max_abs = 0.0;
  for (size_t i = 0; i < grads[0].size(); ++i) {
    for (int v = 1; v < kVariants; ++v) {
      max_diff = std::max(max_diff, double(std::fabs(grads[0][i] - grads[v][i])));
    }
    max_abs = std::max(max_abs, double(std::fabs(grads[0][i])));
  }
  std::printf("  cutoff: min transmittance %.2g, max |grad| %.3g, max diff %.3g\n", min_t, max_abs, max_diff);
  EXPECT(min_t < 1e-3f, "test scene does not saturate (min T %.3g)", min_t);
  EXPECT(max_diff <= 1e-5 * max_abs, "serial_cut(2) gradients differ from serial by %.3g", max_diff);
}

void test_footprint(gs::VkContext& ctx) {
  // Each gaussian gets a pair in every tile holding a pixel it reaches
  // (alpha >= 1/255), once, and in few others; half of them stretched.
  gs::Dataset ds = gs::make_test_dataset(kWidth, kHeight, 1, false, 11);
  gs::SplatScene scene = gs::make_test_scene(200, 1, 13);
  for (uint32_t i = 0; i < scene.count; i += 2) scene.log_scales[i * 3] += 1.5f;
  gs::Trainer trainer(ctx, ds, scene, gs::frozen_train_options(1, false));
  const float bg[3] = {0, 0, 0};
  trainer.step_on(0, bg);
  trainer.step_on(0, bg);  // the mismatch counter is read back one step late
  const gs::Trainer::DebugSnapshot snap = trainer.debug_snapshot();
  constexpr uint32_t T = gs::Trainer::kTileSize;
  const int tiles_x = int((kWidth + T - 1) / T), tiles_y = int((kHeight + T - 1) / T);
  uint64_t needed = 0, emitted = 0, boxed = 0;
  int missing = 0, duplicate = 0;
  for (size_t i = 0; i < snap.tiles.size(); ++i) {
    if (snap.tiles[i] == 0) continue;
    const float* q = &snap.projected[i * 16];
    const double t = 2.0 * std::log(255.0 * q[3]);
    std::vector<int> want(size_t(tiles_x) * tiles_y, 0), got(want.size(), 0);
    for (uint32_t y = 0; y < kHeight; ++y) {
      for (uint32_t x = 0; x < kWidth; ++x) {
        const double dx = q[0] - (x + 0.5), dy = q[1] - (y + 0.5);
        if (q[4] * dx * dx + 2.0 * q[5] * dx * dy + q[6] * dy * dy <= t) want[(y / T) * tiles_x + x / T] = 1;
      }
    }
    for (uint32_t k = snap.offsets[i]; k < snap.offsets[i] + snap.tiles[i]; ++k) {
      const uint32_t tile = snap.pair_tile.at(k);
      if (tile < got.size() && got[tile]++ > 0) ++duplicate;
    }
    for (size_t k = 0; k < want.size(); ++k) {
      needed += uint64_t(want[k]);
      if (want[k] != 0 && got[k] == 0) ++missing;
    }
    emitted += snap.tiles[i];
    // The previous bound: a box around the circle of the largest axis.
    const double a = q[12], b = q[13], c = q[14], mid = 0.5 * (a + c);
    const double lambda = mid + std::sqrt(std::max(0.1, mid * mid - (a * c - b * b)));
    const double r = std::ceil(std::sqrt(t * lambda));
    const int x0 = std::clamp(int(std::floor((q[0] - r) / T)), 0, tiles_x);
    const int x1 = std::clamp(int(std::floor((q[0] + r) / T)) + 1, 0, tiles_x);
    const int y0 = std::clamp(int(std::floor((q[1] - r) / T)), 0, tiles_y);
    const int y1 = std::clamp(int(std::floor((q[1] + r) / T)) + 1, 0, tiles_y);
    boxed += uint64_t(std::max(x1 - x0, 0) * std::max(y1 - y0, 0));
  }
  std::printf("  footprint: %llu pairs needed, %llu emitted, %llu with the circle box; %d missing, %d duplicate, "
              "%u mismatched\n",
              (unsigned long long)needed, (unsigned long long)emitted, (unsigned long long)boxed, missing, duplicate,
              trainer.stats().tile_mismatch);
  EXPECT(missing == 0 && duplicate == 0, "footprint: %d tiles missing, %d duplicate", missing, duplicate);
  EXPECT(emitted <= needed + needed / 20, "footprint: %llu pairs for %llu needed", (unsigned long long)emitted,
         (unsigned long long)needed);
  EXPECT(trainer.stats().tile_mismatch == 0, "project and emit disagree on %u gaussians",
         trainer.stats().tile_mismatch);
}

void test_retune(gs::VkContext& ctx) {
  // Timing every variant on a step leaves the same gradients as one variant.
  using R = gs::TrainOptions::Reduction;
  gs::Dataset ds = gs::make_test_dataset(kWidth, kHeight, 2, false, 3);
  gs::SplatScene scene = gs::make_test_scene(120, 1, 9);
  const float bg[3] = {0.2f, 0.3f, 0.4f};
  gs::TrainOptions o = gs::frozen_train_options(1, false);
  o.iterations = 4;
  o.reduction = R::SerialCut;
  gs::Trainer single(ctx, ds, scene, o);
  o.reduction = R::Serial;
  o.alternatives = {R::SerialCut, R::SerialCut2, R::SubgroupLane4};
  o.retune_first = 1;
  o.retune_interval = 3;
  gs::Trainer tuned(ctx, ds, scene, o);
  for (int s = 0; s < 4; ++s) {
    single.step_on(size_t(s % 2), bg);
    tuned.step_on(size_t(s % 2), bg);
  }
  const std::vector<float> a = single.first_step_gradients().params, b = tuned.first_step_gradients().params;
  double max_diff = 0.0, max_abs = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    max_diff = std::max(max_diff, double(std::fabs(a[i] - b[i])));
    max_abs = std::max(max_abs, double(std::fabs(a[i])));
  }
  std::printf("  retune: %zu timings, last", tuned.retunes().size());
  if (!tuned.retunes().empty()) {
    for (const auto& [name, ms] : tuned.retunes().back().ms) std::printf(" %s %.2f", name.c_str(), ms);
    std::printf(" -> %s", tuned.retunes().back().chosen.c_str());
  }
  std::printf("; max |m| %.3g, max diff %.3g\n", max_abs, max_diff);
  EXPECT(tuned.retunes().size() == 2 && tuned.retunes()[0].step == 1 && tuned.retunes()[1].step == 3,
         "retunes at the wrong steps (%zu)", tuned.retunes().size());
  EXPECT(tuned.retunes().front().ms.size() == 4, "%zu variants timed", tuned.retunes().front().ms.size());
  EXPECT(tuned.reduction() == tuned.retunes().back().chosen, "using %s, chose %s", tuned.reduction().c_str(),
         tuned.retunes().back().chosen.c_str());
  EXPECT(max_diff <= 1e-5 * max_abs, "gradients with retuning differ by %.3g", max_diff);
}

void test_probes(gs::VkContext& ctx) {
  const gs::ProbeReport p = gs::probe_gpu_ops(ctx);
  EXPECT(p.shared_ok, "shared probe: max %u add %u tree %u", p.atomic_max, p.atomic_add, p.tree_max);
  EXPECT(p.subgroup_compiled && p.subgroup_ok, "subgroup probe: %s, %u subgroups of %u", p.subgroup_error.c_str(),
         p.subgroups, p.subgroup_size);
  for (const auto& [name, result] : p.subgroup_parts) EXPECT(result == "ok", "subgroup %s: %s", name.c_str(), result.c_str());
  const gs::ReductionChoice choice = gs::choose_reduction(ctx);
  std::printf("  backward choice: %s\n", choice.log.c_str());
  EXPECT(choice.found, "no backward variant chosen (%s)", choice.log.c_str());
}

void test_scan_and_densify(gs::VkContext& ctx) {
  // Densify everything: a tiny threshold clones or splits every visible gaussian.
  gs::Dataset ds = gs::make_test_dataset(kWidth, kHeight, 1, false, 5);
  gs::SplatScene scene = gs::make_test_scene(40, 0, 3);
  gs::TrainOptions o = gs::frozen_train_options(0, false);
  o.iterations = 10;
  o.densify_from = 0;
  o.densify_until = 100;
  o.densify_interval = 1;
  o.densify_grad_threshold = 0.0f;
  o.max_gaussians = 60;
  gs::Trainer trainer(ctx, ds, scene, o);
  const float bg[3] = {0, 0, 0};
  trainer.step_on(0, bg);
  // Every visible gaussian wants a copy; only 20 free slots exist.
  EXPECT(trainer.stats().slots == 60, "slots %u after densify (capacity 60)", trainer.stats().slots);
  EXPECT(trainer.stats().gaussians == 60, "alive %u after densify", trainer.stats().gaussians);
  const gs::SplatScene out = trainer.export_scene();
  EXPECT(out.count == 60, "exported %u", out.count);
  bool finite = true;
  for (float v : out.positions) finite = finite && std::isfinite(v);
  EXPECT(finite, "non-finite positions after densify");
}

void test_training_converges(gs::VkContext& ctx) {
  const gs::FitReport f = gs::fit_known_scene(ctx);
  std::printf("  training: held-out PSNR %.2f -> %.2f dB, %.1f ms/step (%s)\n", f.psnr_before, f.psnr_after,
              f.ms_per_step, f.reduction.c_str());
  EXPECT(f.psnr_after > f.psnr_before + 8.0, "training did not converge: %.2f -> %.2f dB", f.psnr_before,
         f.psnr_after);
  EXPECT(f.psnr_after > 35.0, "held-out PSNR %.2f dB is too low", f.psnr_after);
}

void test_self_test_json(gs::VkContext& ctx) {
  const std::string json = gs::trainer_self_test_json(ctx);
  EXPECT(test::contains(json, "\"fit\"") && test::contains(json, "\"ok\":true}"), "self test: %s", json.c_str());
}

void write_file(const std::filesystem::path& path, const std::string& text) {
  std::ofstream f(path, std::ios::binary);
  f << text;
}

void test_dataset_loader() {
  // transforms.json with PNG frames written by hand (stored, uncompressed zlib blocks).
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "gs_dataset_test";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir / "images");
  auto crc32 = [](const uint8_t* data, size_t n, uint32_t crc = 0) {
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) {
      crc ^= data[i];
      for (int k = 0; k < 8; ++k) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return ~crc;
  };
  auto be32 = [](std::vector<uint8_t>& v, uint32_t x) {
    for (int s = 24; s >= 0; s -= 8) v.push_back(uint8_t(x >> s));
  };
  auto png = [&](uint32_t w, uint32_t h, uint8_t value) {
    std::vector<uint8_t> raw;
    for (uint32_t y = 0; y < h; ++y) {
      raw.push_back(0);
      for (uint32_t x = 0; x < w * 3; ++x) raw.push_back(uint8_t(value + x));
    }
    std::vector<uint8_t> z = {0x78, 0x01, 1, uint8_t(raw.size()), uint8_t(raw.size() >> 8),
                              uint8_t(~raw.size()), uint8_t(~raw.size() >> 8)};
    z.insert(z.end(), raw.begin(), raw.end());
    uint32_t a = 1, b = 0;
    for (uint8_t c : raw) {
      a = (a + c) % 65521;
      b = (b + a) % 65521;
    }
    be32(z, (b << 16) | a);
    std::vector<uint8_t> out = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    auto chunk = [&](const char* type, const std::vector<uint8_t>& data) {
      be32(out, uint32_t(data.size()));
      std::vector<uint8_t> body(type, type + 4);
      body.insert(body.end(), data.begin(), data.end());
      out.insert(out.end(), body.begin(), body.end());
      be32(out, crc32(body.data(), body.size()));
    };
    std::vector<uint8_t> ihdr;
    be32(ihdr, w);
    be32(ihdr, h);
    ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});
    chunk("IHDR", ihdr);
    chunk("IDAT", z);
    chunk("IEND", {});
    return std::string(out.begin(), out.end());
  };
  std::string frames;
  for (int i = 0; i < 9; ++i) {
    char name[32];
    std::snprintf(name, sizeof(name), "images/f%d.png", i);
    write_file(dir / name, png(8, 4, uint8_t(10 * i)));
    frames += std::string(i ? "," : "") + "{\"file_path\": \"" + name +
              "\", \"transform_matrix\": [[1,0,0," + std::to_string(i) + "],[0,1,0,0],[0,0,1,2],[0,0,0,1]]}";
  }
  write_file(dir / "transforms.json", "{\"w\": 8, \"h\": 4, \"fl_x\": 10.0, \"cx\": 4, \"cy\": 2, \"frames\": [" +
                                          frames + "]}");
  gs::Dataset ds = gs::load_dataset(dir.string(), 4);
  EXPECT(ds.width == 4 && ds.height == 2, "dataset size %ux%u", ds.width, ds.height);
  EXPECT(std::fabs(ds.fx - 5.0f) < 1e-5f && std::fabs(ds.cx - 2.0f) < 1e-5f, "intrinsics not rescaled");
  EXPECT(ds.frames.size() == 9 && ds.holdout_count() == 2, "frames %zu, holdout %zu", ds.frames.size(),
         ds.holdout_count());
  EXPECT(!ds.has_masks && ds.frames[1].rgba[3] == 255, "frames without masks must be opaque");
  // Pixel (0,0) of frame 1 averages source pixels (0..1, 0..1): red values 10 and 13.
  EXPECT(std::abs(int(ds.frames[1].rgba[0]) - 12) <= 1, "resized red %d", ds.frames[1].rgba[0]);
  EXPECT(std::fabs(ds.camera_center[0] - 4.0f) < 1e-5f, "camera centre x %.3f", ds.camera_center[0]);
  const gs::Camera cam = gs::dataset_camera(ds, ds.frames[3]);
  EXPECT(std::fabs(cam.position[0] - 3.0f) < 1e-6f && std::fabs(cam.translation[0] + 3.0f) < 1e-6f,
         "camera translation wrong");
  // OpenGL camera looking down -z: world +z is behind it, so camera-space z = -(world z - 2).
  EXPECT(cam.rotation[8] == -1.0f && cam.rotation[4] == -1.0f, "OpenGL -> OpenCV axes not flipped");
  gs::InitStats stats;
  gs::InitOptions init;
  init.max_gaussians = 100;
  gs::SplatScene scene = gs::initialize_gaussians(ds, init, &stats);
  EXPECT(stats.source == "random" && scene.count == 100, "random init %s %u", stats.source.c_str(), scene.count);

  // A damaged image drops its frame instead of failing the dataset.
  write_file(dir / "images/f4.png", "not a png");
  gs::Dataset partial = gs::load_dataset(dir.string(), 4);
  EXPECT(partial.frames.size() == 8 && partial.skipped.size() == 1, "damaged frame: %zu frames, %zu skipped",
         partial.frames.size(), partial.skipped.size());
  EXPECT(partial.frames[4].name == "images/f5.png", "wrong frame dropped: %s", partial.frames[4].name.c_str());
  for (int i = 0; i < 6; ++i) {
    char name[32];
    std::snprintf(name, sizeof(name), "images/f%d.png", i);
    write_file(dir / name, "");
  }
  bool threw = false;
  try {
    gs::load_dataset(dir.string(), 4);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  EXPECT(threw, "a dataset with most frames unreadable should be rejected");
  std::filesystem::remove_all(dir);
}

}  // namespace

int main() {
  try {
    test_dataset_loader();
    gs::VkContext ctx;
    test_gradients(ctx, false);
    test_gradients(ctx, true);
    test_gradients(ctx, true, gs::TrainOptions::Reduction::Shared);
    test_gradients(ctx, true, gs::TrainOptions::Reduction::SubgroupScalar);
    test_gradients(ctx, false, gs::TrainOptions::Reduction::SubgroupLane4);
    test_gradients(ctx, true, gs::TrainOptions::Reduction::SubgroupLane4);
    test_gradients(ctx, true, gs::TrainOptions::Reduction::SubgroupLane1);
    test_gradients(ctx, false, gs::TrainOptions::Reduction::SerialCut2);
    test_gradients(ctx, true, gs::TrainOptions::Reduction::SerialCut2);
    test_gradients(ctx, false, gs::TrainOptions::Reduction::SerialCut);
    test_gradients(ctx, true, gs::TrainOptions::Reduction::SerialCut);
    test_gradients(ctx, false, gs::TrainOptions::Reduction::Serial);
    test_gradients(ctx, true, gs::TrainOptions::Reduction::Serial);
    test_cutoff_matches_serial(ctx);
    test_footprint(ctx);
    test_retune(ctx);
    test_probes(ctx);
    test_scan_and_densify(ctx);
    test_training_converges(ctx);
    test_self_test_json(ctx);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: exception %s\n", e.what());
    return 1;
  }
  if (test::failures > 0) {
    std::fprintf(stderr, "%d failure(s)\n", test::failures);
    return 1;
  }
  std::printf("all trainer tests passed\n");
  return 0;
}
