// Trains gaussians on a dataset folder (transforms.json) with the same engine
// the Android app uses, and writes a 3DGS .ply.
//
// Usage:
//   gs_train <dataset_dir> <out.ply> [--width 360] [--iters 3000] [--max-gaussians 300000]
//            [--sh 3] [--no-masks] [--preview <dir>] [--eval-every 1000]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "gs/dataset.h"
#include "gs/offscreen.h"
#include "gs/trainer.h"
#include "gs/vk_context.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s <dataset_dir> <out.ply> [--width 360] [--iters 3000] [--max-gaussians N] [--sh 3]\n"
                 "       [--no-masks] [--preview <dir>] [--eval-every N]\n",
                 argv[0]);
    return 2;
  }
  uint32_t width = 360, eval_every = 1000;
  std::string preview;
  gs::TrainOptions options;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) throw std::runtime_error(arg + " needs a value");
      return argv[++i];
    };
    try {
      if (arg == "--width") {
        width = std::atoi(next());
      } else if (arg == "--iters") {
        options.iterations = std::atoi(next());
      } else if (arg == "--max-gaussians") {
        options.max_gaussians = std::atoi(next());
      } else if (arg == "--sh") {
        options.sh_degree = std::atoi(next());
      } else if (arg == "--no-masks") {
        options.masked = false;
      } else if (arg == "--preview") {
        preview = next();
      } else if (arg == "--eval-every") {
        eval_every = std::atoi(next());
      } else {
        std::fprintf(stderr, "unknown option %s\n", arg.c_str());
        return 2;
      }
    } catch (const std::exception& e) {
      std::fprintf(stderr, "%s\n", e.what());
      return 2;
    }
  }
  // Keep the schedule proportional for short runs.
  const double scale = options.iterations / 3000.0;
  options.densify_from = uint32_t(options.densify_from * scale);
  options.densify_until = uint32_t(options.densify_until * scale);
  options.opacity_reset_interval = uint32_t(options.opacity_reset_interval * scale);
  options.sh_degree_interval = std::max(1u, uint32_t(options.sh_degree_interval * scale));

  try {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    gs::Dataset ds = gs::load_dataset(argv[1], width);
    const auto t1 = clock::now();
    std::printf("dataset %s: %zu frames (%zu held out), %ux%u, masks %d, depth %d, %zu points (%.1f s)\n",
                ds.name.c_str(), ds.frames.size(), ds.holdout_count(), ds.width, ds.height, ds.has_masks,
                ds.has_depth, ds.points.size() / 3, std::chrono::duration<double>(t1 - t0).count());
    gs::InitStats init_stats;
    gs::InitOptions init;
    init.subject_only = options.masked;
    gs::SplatScene scene = gs::initialize_gaussians(ds, init, &init_stats);
    std::printf("init from %s: %zu points -> %u gaussians (voxel %.4f)\n", init_stats.source.c_str(),
                init_stats.raw_points, scene.count, init_stats.voxel_size);

    gs::VkContext ctx;
    gs::Trainer trainer(ctx, ds, scene, options);
    const auto t2 = clock::now();
    while (!trainer.finished()) {
      trainer.step();
      const gs::TrainStats& s = trainer.stats();
      if (s.step % 100 == 0 || trainer.finished()) {
        std::printf("step %5u  l1 %.4f  psnr %5.2f  gaussians %7u  pairs %8llu  sh %d  %.1f ms/step\n", s.step,
                    s.l1, s.psnr, s.gaussians, (unsigned long long)s.pairs, s.sh_degree, s.step_ms);
        std::fflush(stdout);
      }
      if (eval_every > 0 && (s.step % eval_every == 0 || trainer.finished())) {
        const gs::EvalResult e = trainer.evaluate();
        std::printf("  eval: held-out PSNR %.2f dB over %u views\n", e.psnr, e.views);
        if (!preview.empty()) {
          for (size_t f = 0; f < ds.frames.size(); f += ds.frames.size() / 4) {
            const float black[3] = {0, 0, 0};
            std::vector<uint8_t> img = trainer.render(gs::dataset_camera(ds, ds.frames[f]), black);
            char name[64];
            std::snprintf(name, sizeof(name), "/step%05u_view%03zu.ppm", s.step, f);
            gs::write_ppm(preview + name, img, ds.width, ds.height);
          }
        }
      }
    }
    const double train_s = std::chrono::duration<double>(clock::now() - t2).count();
    gs::SplatScene out = trainer.export_scene();
    gs::convert_convention(out, gs::AxisConvention::RDF);  // what .ply readers expect
    std::vector<uint8_t> ply = gs::save_ply(out);
    std::ofstream file(argv[2], std::ios::binary);
    file.write(reinterpret_cast<const char*>(ply.data()), std::streamsize(ply.size()));
    std::printf("trained %u steps in %.1f s; wrote %u gaussians to %s\n", trainer.stats().step, train_s, out.count,
                argv[2]);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
