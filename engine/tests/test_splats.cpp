// Splat scene I/O and viewer renderer tests (renders offscreen; CI uses lavapipe).
#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "gs/camera.h"
#include "gs/offscreen.h"
#include "gs/splat_renderer.h"
#include "gs/splat_scene.h"
#include "gs/vk_context.h"
#include "test_util.h"

namespace {

constexpr float kShC0 = 0.28209479177387814f;

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return INFINITY;
  float m = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

void test_ply_roundtrip() {
  gs::SplatScene scene = gs::make_synthetic_scene(1000, 7);
  std::vector<uint8_t> bytes = gs::save_ply(scene);
  gs::SplatScene back = gs::load_ply(bytes.data(), bytes.size());
  EXPECT(back.count == scene.count, "ply count %u != %u", back.count, scene.count);
  EXPECT(back.sh_degree == scene.sh_degree, "ply sh degree %d != %d", back.sh_degree, scene.sh_degree);
  EXPECT(back.convention == gs::AxisConvention::RDF, "ply should be read as RDF");
  EXPECT(max_abs_diff(back.positions, scene.positions) == 0.0f, "ply positions differ");
  EXPECT(max_abs_diff(back.rotations, scene.rotations) == 0.0f, "ply rotations differ");
  EXPECT(max_abs_diff(back.log_scales, scene.log_scales) == 0.0f, "ply scales differ");
  EXPECT(max_abs_diff(back.opacity_logits, scene.opacity_logits) == 0.0f, "ply opacity differs");
  EXPECT(max_abs_diff(back.sh_dc, scene.sh_dc) == 0.0f, "ply sh dc differs");
  EXPECT(max_abs_diff(back.sh_rest, scene.sh_rest) == 0.0f, "ply sh rest differs");

  bool threw = false;
  try {
    const char junk[] = "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n";
    gs::load_ply(reinterpret_cast<const uint8_t*>(junk), sizeof(junk) - 1);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  EXPECT(threw, "ascii ply should be rejected");
}

std::vector<uint8_t> gzip(const std::vector<uint8_t>& data) {
  z_stream stream{};
  deflateInit2(&stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, 16 + MAX_WBITS, 8, Z_DEFAULT_STRATEGY);
  std::vector<uint8_t> out(deflateBound(&stream, data.size()) + 64);
  stream.next_in = const_cast<Bytef*>(data.data());
  stream.avail_in = static_cast<uInt>(data.size());
  stream.next_out = out.data();
  stream.avail_out = static_cast<uInt>(out.size());
  deflate(&stream, Z_FINISH);
  out.resize(stream.total_out);
  deflateEnd(&stream);
  return out;
}

void test_spz_v2() {
  // Two splats, SH degree 1, fixed-point positions with 12 fractional bits.
  const uint32_t n = 2;
  std::vector<uint8_t> raw;
  auto put32 = [&](uint32_t v) {
    for (int i = 0; i < 4; ++i) raw.push_back(uint8_t(v >> (8 * i)));
  };
  put32(0x5053474e);
  put32(2);
  put32(n);
  raw.push_back(1);   // sh degree
  raw.push_back(12);  // fractional bits
  raw.push_back(0);   // flags
  raw.push_back(0);
  const float positions[2][3] = {{1.5f, -2.25f, 0.125f}, {-0.5f, 0.0f, 3.0f}};
  for (const auto& p : positions) {
    for (float v : p) {
      int32_t fixed = int32_t(std::lround(v * 4096.0f));
      for (int i = 0; i < 3; ++i) raw.push_back(uint8_t(fixed >> (8 * i)));
    }
  }
  raw.push_back(255);  // alphas
  raw.push_back(128);
  for (int i = 0; i < 6; ++i) raw.push_back(uint8_t(64 + i * 20));  // colours
  for (int i = 0; i < 6; ++i) raw.push_back(uint8_t(100 + i));      // log scales * 16 + 160
  raw.push_back(128);  // rotation of splat 0: xyz ~ 0 -> identity
  raw.push_back(128);
  raw.push_back(128);
  raw.push_back(255);  // splat 1: x ~ 1
  raw.push_back(128);
  raw.push_back(128);
  for (int i = 0; i < 18; ++i) raw.push_back(uint8_t(128 + i));  // SH: 2 splats * 3 coeffs * rgb

  std::vector<uint8_t> file = gzip(raw);
  gs::SplatScene s = gs::load_spz(file.data(), file.size());
  EXPECT(s.count == 2 && s.sh_degree == 1, "spz header: count %u degree %d", s.count, s.sh_degree);
  EXPECT(s.convention == gs::AxisConvention::RUB, "spz should be RUB");
  for (int i = 0; i < 2; ++i) {
    for (int k = 0; k < 3; ++k) {
      EXPECT(std::fabs(s.positions[i * 3 + k] - positions[i][k]) < 1e-3f, "spz position %d.%d", i, k);
    }
  }
  EXPECT(s.opacity_logits[0] > 5.0f, "spz alpha 255 should be nearly opaque");
  EXPECT(std::fabs(s.opacity_logits[1] - std::log(128.0f / 127.0f)) < 1e-3f, "spz alpha 128");
  EXPECT(std::fabs(s.sh_dc[0] - (64 / 255.0f - 0.5f) / 0.15f) < 1e-5f, "spz colour");
  EXPECT(std::fabs(s.log_scales[0] - (100 / 16.0f - 10.0f)) < 1e-5f, "spz scale");
  EXPECT(std::fabs(s.rotations[0] - 1.0f) < 0.01f, "spz identity rotation w = %f", s.rotations[0]);
  EXPECT(std::fabs(s.rotations[4 + 1] - 1.0f) < 0.01f, "spz x rotation x = %f", s.rotations[5]);
  EXPECT(std::fabs(s.sh_rest[5] - (5.0f / 128.0f)) < 1e-6f, "spz sh layout");

  bool threw = false;
  try {
    const uint8_t v4[8] = {0x4e, 0x47, 0x53, 0x50, 4, 0, 0, 0};
    gs::load_splat_file(v4, sizeof(v4));
  } catch (const std::runtime_error& e) {
    threw = test::contains(e.what(), "v4");
  }
  EXPECT(threw, "spz v4 should be rejected with a clear message");
}

// --- Renderer -------------------------------------------------------------------

struct TestSplat {
  float pos[3];
  float scale;
  float rgb[3];
  float opacity;
};

gs::SplatScene make_scene(const std::vector<TestSplat>& splats) {
  gs::SplatScene scene;
  scene.convention = gs::AxisConvention::RUB;
  scene.resize(static_cast<uint32_t>(splats.size()), 0);
  for (size_t i = 0; i < splats.size(); ++i) {
    const TestSplat& t = splats[i];
    for (int k = 0; k < 3; ++k) {
      scene.positions[i * 3 + k] = t.pos[k];
      scene.log_scales[i * 3 + k] = std::log(t.scale);
      scene.sh_dc[i * 3 + k] = (t.rgb[k] - 0.5f) / kShC0;
    }
    scene.rotations[i * 4] = 1.0f;
    scene.opacity_logits[i] = std::log(t.opacity / (1.0f - t.opacity));
  }
  return scene;
}

gs::OrbitCamera front_camera() {
  gs::OrbitCamera orbit;
  orbit.yaw = 0.0f;
  orbit.pitch = 0.0f;
  orbit.distance = 3.0f;
  return orbit;
}

std::vector<uint8_t> render(gs::VkContext& ctx, const gs::SplatScene& scene, const gs::OrbitCamera& orbit,
                            uint32_t w, uint32_t h, uint32_t* visible = nullptr) {
  auto gpu = gs::upload_scene(ctx, scene);
  gs::OffscreenTarget target(ctx, w, h);
  gs::SplatRenderer renderer(ctx, target.render_pass());
  renderer.attach(gpu.get());
  const gs::Camera camera = orbit.camera(w, h);
  const float clear[4] = {0.0f, 0.0f, 0.0f, 1.0f};
  ctx.run([&](VkCommandBuffer cmd) {
    renderer.record_prepare(cmd, camera);
    target.record(cmd, clear, [&](VkCommandBuffer c) { renderer.record_draw(c, w, h); });
  });
  if (visible != nullptr) *visible = renderer.visible_count();
  return target.pixels();
}

const uint8_t* pixel(const std::vector<uint8_t>& img, uint32_t w, uint32_t x, uint32_t y) {
  return &img[(size_t(y) * w + x) * 4];
}

void test_single_splat(gs::VkContext& ctx) {
  const uint32_t w = 64, h = 64;
  const float opacity = 0.95f;
  gs::SplatScene scene = make_scene({{{0, 0, 0}, 0.1f, {1, 0, 0}, opacity}});
  uint32_t visible = 0;
  auto img = render(ctx, scene, front_camera(), w, h, &visible);
  EXPECT(visible == 1, "single splat: visible %u", visible);

  // Expected falloff: opacity * exp(-0.5 d^2 / (sigma_px^2 + 0.3)) at pixel centres.
  const float focal = 0.5f * 64.0f / std::tan(25.0f * 3.14159265f / 180.0f);
  const float sigma_px = focal * 0.1f / 3.0f;
  const float var = sigma_px * sigma_px + 0.3f;
  for (int dx : {0, 2, 4}) {
    const float px = 32 + dx + 0.5f - 32.0f, py = 0.5f;
    const float alpha = std::min(opacity * std::exp(-0.5f * (px * px + py * py) / var), 0.99f);
    const uint8_t* p = pixel(img, w, 32 + dx, 32);
    const float expected = 255.0f * alpha;
    EXPECT(std::fabs(p[0] - expected) <= 6.0f, "falloff at dx=%d: red %u, expected %.1f", dx, p[0], expected);
    EXPECT(p[1] <= 2 && p[2] <= 2, "single splat should be pure red (g %u b %u)", p[1], p[2]);
  }
  const uint8_t* corner = pixel(img, w, 2, 2);
  EXPECT(corner[0] == 0 && corner[1] == 0 && corner[2] == 0, "background should stay black");
}

void test_depth_order(gs::VkContext& ctx) {
  const uint32_t w = 64, h = 64;
  // The camera sits at +z looking towards -z, so larger z is nearer.
  for (bool red_in_front : {true, false}) {
    const float red_z = red_in_front ? 0.5f : -0.5f;
    gs::SplatScene scene = make_scene({
        {{0, 0, red_z}, 0.1f, {1, 0, 0}, 0.9f},
        {{0, 0, -red_z}, 0.1f, {0, 0, 1}, 0.9f},
    });
    auto img = render(ctx, scene, front_camera(), w, h);
    const uint8_t* p = pixel(img, w, 32, 32);
    if (red_in_front) {
      EXPECT(p[0] > 180 && p[2] < 60, "red in front: got r %u b %u", p[0], p[2]);
    } else {
      EXPECT(p[2] > 180 && p[0] < 60, "blue in front: got r %u b %u", p[0], p[2]);
    }
  }
}

// Rendering Y-down data with the display flip must match rendering the
// equivalent Y-up data without it, SH included.
void test_flip_equivalence(gs::VkContext& ctx) {
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  gs::SplatScene up;
  up.convention = gs::AxisConvention::RUB;
  up.resize(400, 3);
  for (uint32_t i = 0; i < up.count; ++i) {
    for (int k = 0; k < 3; ++k) {
      up.positions[i * 3 + k] = 0.8f * u(rng);
      up.log_scales[i * 3 + k] = std::log(0.03f + 0.05f * std::fabs(u(rng)));
      up.sh_dc[i * 3 + k] = 1.5f * u(rng);
    }
    for (int k = 0; k < 4; ++k) up.rotations[i * 4 + k] = u(rng);
    up.opacity_logits[i] = 2.0f * u(rng);
    for (int j = 0; j < 15 * 3; ++j) up.sh_rest[i * 45 + j] = 0.3f * u(rng);
  }

  // 180 degrees about X: (x, y, z) -> (x, -y, -z). Quaternion (w, x, y, z) -> (w, x, -y, -z).
  // SH basis functions change sign with their parity in y and z.
  const float sh_sign[15] = {-1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1};
  gs::SplatScene down = up;
  down.convention = gs::AxisConvention::RDF;
  for (uint32_t i = 0; i < down.count; ++i) {
    down.positions[i * 3 + 1] *= -1.0f;
    down.positions[i * 3 + 2] *= -1.0f;
    down.rotations[i * 4 + 2] *= -1.0f;
    down.rotations[i * 4 + 3] *= -1.0f;
    for (int j = 0; j < 15; ++j) {
      for (int c = 0; c < 3; ++c) down.sh_rest[(size_t(i) * 15 + j) * 3 + c] *= sh_sign[j];
    }
  }

  gs::OrbitCamera orbit;
  orbit.yaw = 0.7f;
  orbit.pitch = 0.4f;
  orbit.distance = 3.0f;
  auto img_up = render(ctx, up, orbit, 96, 96);
  orbit.set_flip_y(true);
  auto img_down = render(ctx, down, orbit, 96, 96);
  size_t lit = 0;
  for (size_t i = 0; i < img_up.size(); i += 4) lit += (img_up[i] + img_up[i + 1] + img_up[i + 2]) > 30;
  EXPECT(lit > 500, "flip equivalence: scene barely visible (%zu lit pixels)", lit);
  int worst = 0;
  for (size_t i = 0; i < img_up.size(); ++i) worst = std::max(worst, std::abs(int(img_up[i]) - int(img_down[i])));
  EXPECT(worst <= 3, "flip equivalence: max channel difference %d", worst);
}

void test_synthetic_render(gs::VkContext& ctx) {
  gs::SplatScene scene = gs::make_synthetic_scene(20000);
  gs::OrbitCamera orbit;
  orbit.reset(gs::compute_bounds(scene), false);
  uint32_t visible = 0;
  auto img = render(ctx, scene, orbit, 160, 120, &visible);
  EXPECT(visible > 15000, "synthetic: only %u of 20000 visible", visible);
  size_t lit = 0;
  for (size_t i = 0; i < img.size(); i += 4) lit += (img[i] + img[i + 1] + img[i + 2]) > 30;
  EXPECT(lit > img.size() / 4 / 10, "synthetic: only %zu lit pixels", lit);
  if (const char* out = std::getenv("GS_TEST_OUTPUT")) gs::write_ppm(std::string(out) + "/synthetic.ppm", img, 160, 120);
}

}  // namespace

int main() {
  test_ply_roundtrip();
  test_spz_v2();
  try {
    gs::VkContext ctx;
    test_single_splat(ctx);
    test_depth_order(ctx);
    test_flip_equivalence(ctx);
    test_synthetic_render(ctx);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: exception: %s\n", e.what());
    return 1;
  }
  if (test::failures > 0) {
    std::fprintf(stderr, "%d failure(s)\n", test::failures);
    return 1;
  }
  std::printf("all splat tests passed\n");
  return 0;
}
