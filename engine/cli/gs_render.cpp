// Renders a splat file (or a synthetic scene) with the same renderer the Android
// viewer uses.
//
// Usage:
//   gs_render <scene> <out.ppm> [width height yaw_deg pitch_deg]
//       one orbit view of the scene
//   gs_render <scene> --views <cameras.txt> <out_dir>
//       one image per line of cameras.txt, written as out_dir/view_00000.ppm, ...
//       Each line: width height fx fy cx cy, then the 3x4 world-to-camera matrix
//       (OpenCV convention: x right, y down, z forward) row by row.
//
// <scene> is a .ply/.spz file or synthetic:N.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gs/camera.h"
#include "gs/offscreen.h"
#include "gs/splat_renderer.h"
#include "gs/splat_scene.h"
#include "gs/vk_context.h"

namespace {

constexpr float kClear[4] = {0.08f, 0.09f, 0.11f, 1.0f};

gs::SplatScene load_scene(const std::string& input) {
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
  return scene;
}

std::vector<uint8_t> render(gs::VkContext& ctx, gs::GpuScene& gpu, const gs::Camera& camera,
                            std::unique_ptr<gs::OffscreenTarget>& target,
                            std::unique_ptr<gs::SplatRenderer>& renderer, uint32_t* visible) {
  if (!target || target->width() != camera.width || target->height() != camera.height) {
    renderer.reset();
    target = std::make_unique<gs::OffscreenTarget>(ctx, camera.width, camera.height);
    renderer = std::make_unique<gs::SplatRenderer>(ctx, target->render_pass());
    renderer->attach(&gpu);
  }
  ctx.run([&](VkCommandBuffer cmd) {
    renderer->record_prepare(cmd, camera);
    target->record(cmd, kClear, [&](VkCommandBuffer c) { renderer->record_draw(c, camera.width, camera.height); });
  });
  *visible = renderer->visible_count();
  return target->pixels();
}

std::vector<gs::Camera> read_cameras(const std::string& path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("cannot open " + path);
  std::vector<gs::Camera> cameras;
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream in(line);
    double v[18];
    int n = 0;
    while (n < 18 && in >> v[n]) ++n;
    if (n == 0) continue;
    if (n != 18) throw std::runtime_error("camera line needs 18 numbers: " + line);
    gs::Camera cam{};
    cam.width = static_cast<uint32_t>(v[0]);
    cam.height = static_cast<uint32_t>(v[1]);
    cam.fx = float(v[2]);
    cam.fy = float(v[3]);
    cam.cx = float(v[4]);
    cam.cy = float(v[5]);
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) cam.rotation[r * 3 + c] = float(v[6 + r * 4 + c]);
      cam.translation[r] = float(v[6 + r * 4 + 3]);
    }
    // Camera centre = -R^T t.
    for (int c = 0; c < 3; ++c) {
      cam.position[c] = -(cam.rotation[0 * 3 + c] * cam.translation[0] + cam.rotation[1 * 3 + c] * cam.translation[1] +
                          cam.rotation[2 * 3 + c] * cam.translation[2]);
    }
    cameras.push_back(cam);
  }
  return cameras;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s <scene> <out.ppm> [w h yaw pitch]\n"
                 "       %s <scene> --views <cameras.txt> <out_dir>\n",
                 argv[0], argv[0]);
    return 2;
  }
  try {
    const std::string mode = argv[2];
    gs::SplatScene scene = load_scene(argv[1]);
    gs::VkContext ctx;
    auto gpu = gs::upload_scene(ctx, scene);
    std::unique_ptr<gs::OffscreenTarget> target;
    std::unique_ptr<gs::SplatRenderer> renderer;
    uint32_t visible = 0;

    if (mode == "--views") {
      if (argc < 5) throw std::runtime_error("--views needs <cameras.txt> <out_dir>");
      const std::vector<gs::Camera> cameras = read_cameras(argv[3]);
      const std::string out_dir = argv[4];
      for (size_t i = 0; i < cameras.size(); ++i) {
        auto pixels = render(ctx, *gpu, cameras[i], target, renderer, &visible);
        char name[32];
        std::snprintf(name, sizeof(name), "/view_%05zu.ppm", i);
        if (!gs::write_ppm(out_dir + name, pixels, cameras[i].width, cameras[i].height)) {
          throw std::runtime_error("cannot write " + out_dir + name);
        }
        std::printf("\rview %zu/%zu (%u visible)", i + 1, cameras.size(), visible);
        std::fflush(stdout);
      }
      std::printf("\n");
      return 0;
    }

    const uint32_t width = argc > 3 ? std::atoi(argv[3]) : 800;
    const uint32_t height = argc > 4 ? std::atoi(argv[4]) : 600;
    gs::OrbitCamera orbit;
    orbit.reset(gs::compute_bounds(scene), scene.convention == gs::AxisConvention::RDF);
    orbit.yaw = argc > 5 ? std::atof(argv[5]) * 3.14159265f / 180.0f : 0.0f;
    orbit.pitch = argc > 6 ? std::atof(argv[6]) * 3.14159265f / 180.0f : 0.25f;
    const gs::Camera camera = orbit.camera(width, height);
    auto start = std::chrono::steady_clock::now();
    auto pixels = render(ctx, *gpu, camera, target, renderer, &visible);
    std::printf("rendered %ux%u: %u visible, %.1f ms\n", width, height, visible,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    if (!gs::write_ppm(argv[2], pixels, width, height)) throw std::runtime_error("cannot write output");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
