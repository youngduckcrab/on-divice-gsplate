// Renders a splat file (or a synthetic scene) to a PPM image with the same
// renderer the Android viewer uses.
// Usage: gs_render <scene.ply|scene.spz|synthetic:N> <out.ppm> [width height yaw_deg pitch_deg]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "gs/camera.h"
#include "gs/offscreen.h"
#include "gs/splat_renderer.h"
#include "gs/splat_scene.h"
#include "gs/vk_context.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <scene.ply|scene.spz|synthetic:N> <out.ppm> [w h yaw pitch]\n", argv[0]);
    return 2;
  }
  try {
    const std::string input = argv[1];
    const uint32_t width = argc > 3 ? std::atoi(argv[3]) : 800;
    const uint32_t height = argc > 4 ? std::atoi(argv[4]) : 600;
    const float yaw = argc > 5 ? std::atof(argv[5]) * 3.14159265f / 180.0f : 0.0f;
    const float pitch = argc > 6 ? std::atof(argv[6]) * 3.14159265f / 180.0f : 0.25f;

    auto t0 = std::chrono::steady_clock::now();
    gs::SplatScene scene;
    if (input.rfind("synthetic:", 0) == 0) {
      scene = gs::make_synthetic_scene(std::atoi(input.c_str() + 10));
    } else {
      std::ifstream file(input, std::ios::binary);
      if (!file) throw std::runtime_error("cannot open " + input);
      std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
      scene = gs::load_splat_file(bytes.data(), bytes.size());
    }
    auto t1 = std::chrono::steady_clock::now();
    std::printf("loaded %u splats, sh degree %d, %s, antialiased %d (%.0f ms)\n", scene.count, scene.sh_degree,
                scene.convention == gs::AxisConvention::RDF ? "RDF" : "RUB", scene.antialiased ? 1 : 0,
                std::chrono::duration<double, std::milli>(t1 - t0).count());

    gs::VkContext ctx;
    auto gpu = gs::upload_scene(ctx, scene);
    gs::OffscreenTarget target(ctx, width, height);
    gs::SplatRenderer renderer(ctx, target.render_pass());
    renderer.attach(gpu.get());

    gs::OrbitCamera orbit;
    orbit.reset(gs::compute_bounds(scene), scene.convention == gs::AxisConvention::RDF);
    orbit.yaw = yaw;
    orbit.pitch = pitch;
    const gs::Camera camera = orbit.camera(width, height);

    const float clear[4] = {0.08f, 0.09f, 0.11f, 1.0f};
    gs::RunTiming timing = ctx.run([&](VkCommandBuffer cmd) {
      renderer.record_prepare(cmd, camera);
      target.record(cmd, clear, [&](VkCommandBuffer c) { renderer.record_draw(c, width, height); });
    });
    std::printf("rendered %ux%u: %u visible, %.1f ms\n", width, height, renderer.visible_count(),
                timing.best_ms());
    if (!gs::write_ppm(argv[2], target.pixels(), width, height)) throw std::runtime_error("cannot write output");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
