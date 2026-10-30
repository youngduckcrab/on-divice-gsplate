#include "gs/train_selftest.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>

#include "gs/json.h"

namespace gs {
namespace {

struct Ref {
  bool visible = false;
  double mx = 0, my = 0, depth = 0, opacity = 0;
  double con[3] = {0, 0, 0};
  double color[3] = {0, 0, 0};
  uint32_t key = 0;
};

void quat_to_matrix(const double q[4], double R[3][3]) {
  const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  const double w = q[0] / n, x = q[1] / n, y = q[2] / n, z = q[3] / n;
  R[0][0] = 1 - 2 * (y * y + z * z);
  R[0][1] = 2 * (x * y - w * z);
  R[0][2] = 2 * (x * z + w * y);
  R[1][0] = 2 * (x * y + w * z);
  R[1][1] = 1 - 2 * (x * x + z * z);
  R[1][2] = 2 * (y * z - w * x);
  R[2][0] = 2 * (x * z - w * y);
  R[2][1] = 2 * (y * z + w * x);
  R[2][2] = 1 - 2 * (x * x + y * y);
}

void sh_basis(const double d[3], int degree, double b[16]) {
  const double x = d[0], y = d[1], z = d[2];
  b[0] = 0.28209479177387814;
  if (degree >= 1) {
    b[1] = -0.4886025119029199 * y;
    b[2] = 0.4886025119029199 * z;
    b[3] = -0.4886025119029199 * x;
  }
  if (degree >= 2) {
    const double xx = x * x, yy = y * y, zz = z * z;
    b[4] = 1.0925484305920792 * x * y;
    b[5] = -1.0925484305920792 * y * z;
    b[6] = 0.31539156525252005 * (2 * zz - xx - yy);
    b[7] = -1.0925484305920792 * x * z;
    b[8] = 0.5462742152960396 * (xx - yy);
    if (degree >= 3) {
      b[9] = -0.5900435899266435 * y * (3 * xx - yy);
      b[10] = 2.890611442640554 * x * y * z;
      b[11] = -0.4570457994644658 * y * (4 * zz - xx - yy);
      b[12] = 0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy);
      b[13] = -0.4570457994644658 * x * (4 * zz - xx - yy);
      b[14] = 1.445305721320277 * z * (xx - yy);
      b[15] = -0.5900435899266435 * x * (xx - 3 * yy);
    }
  }
}

Ref project(const Camera& cam, const double* p, const double* sh, int stride, int degree) {
  Ref r;
  double t[3];
  for (int k = 0; k < 3; ++k) {
    t[k] = cam.translation[k];
    for (int c = 0; c < 3; ++c) t[k] += cam.rotation[k * 3 + c] * p[c];
  }
  const double opacity = 1.0 / (1.0 + std::exp(-p[3]));
  if (p[7] < 0.5 || t[2] <= 0.02 || opacity < 1.0 / 255.0) return r;
  const double fx = cam.fx, fy = cam.fy, cx = cam.cx, cy = cam.cy, W = cam.width, H = cam.height;
  const double z = t[2];
  const double lim_x = 1.3 * std::max(cx, W - cx) / fx, lim_y = 1.3 * std::max(cy, H - cy) / fy;
  const double tx = std::clamp(t[0] / z, -lim_x, lim_x), ty = std::clamp(t[1] / z, -lim_y, lim_y);
  const double J[2][3] = {{fx / z, 0, -fx * tx / z}, {0, fy / z, -fy * ty / z}};
  double R[3][3];
  quat_to_matrix(p + 8, R);
  double M[3][3], Sigma[3][3], T[2][3];
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) M[a][b] = R[a][b] * std::exp(p[4 + b]);
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      Sigma[a][b] = 0;
      for (int k = 0; k < 3; ++k) Sigma[a][b] += M[a][k] * M[b][k];
    }
  }
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 3; ++b) {
      T[a][b] = 0;
      for (int k = 0; k < 3; ++k) T[a][b] += J[a][k] * cam.rotation[k * 3 + b];
    }
  }
  double cov[2][2];
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      cov[a][b] = 0;
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) cov[a][b] += T[a][k] * Sigma[k][l] * T[b][l];
      }
    }
  }
  const double A = cov[0][0] + 0.3, B = cov[0][1], C = cov[1][1] + 0.3;
  const double det = A * C - B * B;
  if (det <= 0) return r;
  r.visible = true;
  r.mx = fx * t[0] / z + cx;
  r.my = fy * t[1] / z + cy;
  r.depth = z;
  r.opacity = opacity;
  r.con[0] = C / det;
  r.con[1] = -B / det;
  r.con[2] = A / det;
  double dir[3];
  double len = 0;
  for (int k = 0; k < 3; ++k) {
    dir[k] = p[k] - cam.position[k];
    len += dir[k] * dir[k];
  }
  len = std::sqrt(len);
  for (double& v : dir) v /= len;
  double b[16];
  sh_basis(dir, degree, b);
  const int coeffs = degree == 0 ? 0 : (degree + 1) * (degree + 1) - 1;
  for (int c = 0; c < 3; ++c) {
    double v = b[0] * p[12 + c];
    for (int k = 0; k < coeffs; ++k) v += b[k + 1] * sh[k * 3 + c];
    r.color[c] = std::max(v + 0.5, 0.0);
  }
  (void)stride;
  r.key = uint32_t(std::clamp(std::log2(z / 0.02) / std::log2(1000.0 / 0.02), 0.0, 1.0) * 65535.0);
  return r;
}

void look_at_c2w_gl(const float eye[3], const float target[3], float c2w[16]) {
  float f[3], len = 0;
  for (int k = 0; k < 3; ++k) {
    f[k] = target[k] - eye[k];
    len += f[k] * f[k];
  }
  len = std::sqrt(len);
  for (float& v : f) v /= len;
  const float up[3] = {0, 1, 0};
  float r[3] = {f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0]};
  len = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
  for (float& v : r) v /= len;
  const float u[3] = {r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]};
  // OpenGL camera: x = right, y = up, z = -forward.
  for (int k = 0; k < 3; ++k) {
    c2w[k * 4 + 0] = r[k];
    c2w[k * 4 + 1] = u[k];
    c2w[k * 4 + 2] = -f[k];
    c2w[k * 4 + 3] = eye[k];
  }
  c2w[12] = c2w[13] = c2w[14] = 0;
  c2w[15] = 1;
}

}  // namespace

// Renders and returns the L1 loss against `target` (RGBA8), compositing like the GPU.
double reference_loss(const Camera& cam, const std::vector<double>& params, const std::vector<double>& sh,
                      int stride, int degree, const std::vector<uint8_t>& target, const float bg[3], bool masked,
                      std::vector<double>* image) {
  const size_t n = params.size() / 16;
  std::vector<Ref> refs(n);
  for (size_t i = 0; i < n; ++i) refs[i] = project(cam, &params[i * 16], &sh[i * stride], stride, degree);
  std::vector<size_t> order;
  for (size_t i = 0; i < n; ++i) {
    if (refs[i].visible) order.push_back(i);
  }
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return refs[a].key < refs[b].key; });
  if (image) image->assign(size_t(cam.width) * cam.height * 3, 0.0);
  double loss = 0;
  for (uint32_t y = 0; y < cam.height; ++y) {
    for (uint32_t x = 0; x < cam.width; ++x) {
      double T = 1, C[3] = {0, 0, 0};
      for (size_t i : order) {
        const Ref& g = refs[i];
        const double dx = g.mx - (x + 0.5), dy = g.my - (y + 0.5);
        const double power = -0.5 * (g.con[0] * dx * dx + g.con[2] * dy * dy) - g.con[1] * dx * dy;
        if (power > 0) continue;
        const double alpha = std::min(0.99, g.opacity * std::exp(power));
        if (alpha < 1.0 / 255.0) continue;
        if (T * (1 - alpha) < 1e-4) break;
        for (int c = 0; c < 3; ++c) C[c] += g.color[c] * alpha * T;
        T *= 1 - alpha;
      }
      const uint8_t* t = &target[(size_t(y) * cam.width + x) * 4];
      const double a = t[3] / 255.0;
      for (int c = 0; c < 3; ++c) {
        const double out = C[c] + T * bg[c];
        const double tc = masked ? bg[c] * (1 - a) + t[c] / 255.0 * a : t[c] / 255.0;
        loss += std::fabs(out - tc);
        if (image) (*image)[(size_t(y) * cam.width + x) * 3 + c] = out;
      }
    }
  }
  return loss / (3.0 * cam.width * cam.height);
}

Dataset make_test_dataset(uint32_t width, uint32_t height, size_t frames, bool masks, uint32_t seed) {
  Dataset ds;
  ds.name = "test";
  ds.width = width;
  ds.height = height;
  ds.fx = ds.fy = 0.9f * width;
  ds.cx = width * 0.5f;
  ds.cy = height * 0.5f;
  ds.has_masks = masks;
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> byte(0, 255);
  for (size_t i = 0; i < frames; ++i) {
    DatasetFrame f;
    const float angle = 6.2831853f * float(i) / float(std::max<size_t>(frames, 1));
    const float eye[3] = {3.0f * std::sin(angle), 0.6f + 0.3f * std::sin(3.0f * angle), 3.0f * std::cos(angle)};
    const float target[3] = {0.0f, 0.0f, 0.0f};
    look_at_c2w_gl(eye, target, f.c2w);
    f.rgba.resize(size_t(width) * height * 4);
    for (uint8_t& v : f.rgba) v = static_cast<uint8_t>(byte(rng));
    if (!masks) {
      for (size_t p = 0; p < size_t(width) * height; ++p) f.rgba[p * 4 + 3] = 255;
    }
    f.holdout = frames > 1 && i % 8 == 0;
    ds.frames.push_back(std::move(f));
  }
  ds.camera_extent = 3.3f;
  return ds;
}

SplatScene make_test_scene(uint32_t n, int degree, uint32_t seed) {
  SplatScene s;
  s.convention = AxisConvention::RUB;
  s.resize(n, degree);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  for (uint32_t i = 0; i < n; ++i) {
    for (int k = 0; k < 3; ++k) {
      s.positions[i * 3 + k] = 0.7f * u(rng);
      s.log_scales[i * 3 + k] = std::log(0.08f + 0.06f * u(rng));
      s.sh_dc[i * 3 + k] = 0.8f * u(rng);
    }
    for (int k = 0; k < 4; ++k) s.rotations[i * 4 + k] = u(rng);
    s.rotations[i * 4] += 1.5f;
    s.opacity_logits[i] = 0.6f * u(rng) - 0.2f;
  }
  for (float& v : s.sh_rest) v = 0.25f * u(rng);
  return s;
}

TrainOptions frozen_train_options(int degree, bool masked) {
  TrainOptions o;
  o.iterations = 1;
  o.sh_degree = degree;
  o.sh_degree_interval = 0;
  o.masked = masked;
  o.random_background = false;
  o.lr_position = o.lr_position_final = o.lr_scale = o.lr_rotation = o.lr_opacity = o.lr_sh_dc = o.lr_sh_rest = 0;
  o.densify_from = o.densify_until = 1000000;
  o.opacity_reset_interval = 0;
  return o;
}

bool GradientReport::ok() const {
  if (image_max_diff >= 2.5 / 255.0 || std::fabs(loss_gpu - loss_ref) >= 1e-4) return false;
  for (const GradientGroup& g : groups) {
    if (g.bad > g.checked / 20) return false;
  }
  return !groups.empty();
}

GradientReport check_gradients(VkContext& ctx, bool masked, TrainOptions::Reduction reduction, int degree) {
  constexpr uint32_t kWidth = 64, kHeight = 48, kCount = 10;
  Dataset ds = make_test_dataset(kWidth, kHeight, 1, masked, 3);
  SplatScene scene = make_test_scene(kCount, degree, 11);
  TrainOptions options = frozen_train_options(degree, masked);
  options.reduction = reduction;
  Trainer trainer(ctx, ds, scene, options);
  const float bg[3] = {0.2f, 0.5f, 0.7f};
  const Camera cam = dataset_camera(ds, ds.frames[0]);

  GradientReport report;
  report.reduction = trainer.reduction();
  report.masked = masked;

  // Reference parameters, in the trainer's layout.
  const int stride = int(trainer.sh_stride());
  std::vector<double> params(kCount * 16, 0.0), sh(size_t(kCount) * stride, 0.0);
  for (uint32_t i = 0; i < kCount; ++i) {
    for (int k = 0; k < 3; ++k) {
      params[i * 16 + k] = scene.positions[i * 3 + k];
      params[i * 16 + 4 + k] = scene.log_scales[i * 3 + k];
      params[i * 16 + 12 + k] = scene.sh_dc[i * 3 + k];
    }
    params[i * 16 + 3] = scene.opacity_logits[i];
    params[i * 16 + 7] = 1.0;
    for (int k = 0; k < 4; ++k) params[i * 16 + 8 + k] = scene.rotations[i * 4 + k];
    for (int k = 0; k < stride; ++k) sh[i * stride + k] = scene.sh_rest[i * stride + k];
  }

  std::vector<double> ref_image;
  report.loss_ref = reference_loss(cam, params, sh, stride, degree, ds.frames[0].rgba, bg, masked, &ref_image);
  const std::vector<uint8_t> gpu_image = trainer.render(cam, bg);
  for (size_t p = 0; p < size_t(kWidth) * kHeight; ++p) {
    for (int c = 0; c < 3; ++c) {
      const double ref = std::clamp(ref_image[p * 3 + c], 0.0, 1.0);
      report.image_max_diff = std::max(report.image_max_diff, std::fabs(ref - gpu_image[p * 4 + c] / 255.0));
    }
  }

  trainer.step_on(0, bg);
  report.loss_gpu = trainer.stats().l1;
  const Trainer::Gradients g = trainer.first_step_gradients();

  struct Group {
    const char* name;
    std::vector<int> offsets;  // into the 16-float slot, or -1 - k for SH coefficient k
  };
  std::vector<Group> groups = {{"position", {0, 1, 2}},     {"opacity", {3}},        {"log_scale", {4, 5, 6}},
                               {"rotation", {8, 9, 10, 11}}, {"sh_dc", {12, 13, 14}}, {"sh_rest", {}}};
  for (int k = 0; k < stride; ++k) groups.back().offsets.push_back(-1 - k);
  const double eps = 1e-5;
  for (const Group& group : groups) {
    GradientGroup out;
    out.name = group.name;
    for (uint32_t i = 0; i < kCount; ++i) {
      for (int off : group.offsets) {
        double& value = off >= 0 ? params[i * 16 + off] : sh[i * stride + (-1 - off)];
        const double saved = value;
        value = saved + eps;
        const double up = reference_loss(cam, params, sh, stride, degree, ds.frames[0].rgba, bg, masked);
        value = saved - eps;
        const double down = reference_loss(cam, params, sh, stride, degree, ds.frames[0].rgba, bg, masked);
        value = saved;
        const double fd = (up - down) / (2 * eps);
        const double gpu = off >= 0 ? g.params[i * 16 + off] : g.sh[i * stride + (-1 - off)];
        const double err = std::fabs(gpu - fd);
        out.max_err = std::max(out.max_err, std::isfinite(err) ? err : 1e30);
        out.max_ref = std::max(out.max_ref, std::fabs(fd));
        ++out.checked;
        if (!(err <= 0.05 * std::fabs(fd) + 2e-6)) ++out.bad;
      }
    }
    if (out.checked > 0) report.groups.push_back(out);
  }
  return report;
}

FitReport fit_known_scene(VkContext& ctx, uint32_t steps, TrainOptions::Reduction reduction) {
  // Targets rendered from a known scene; training from a perturbed copy must fit them.
  const int degree = 1;
  Dataset ds = make_test_dataset(96, 72, 24, true, 9);
  SplatScene truth = make_test_scene(60, degree, 21);
  {
    Trainer renderer(ctx, ds, truth, frozen_train_options(degree, false));
    const float black[3] = {0, 0, 0};
    for (DatasetFrame& f : ds.frames) {
      std::vector<uint8_t> img = renderer.render(dataset_camera(ds, f), black);
      // Premultiplied over black: un-premultiply into colour + mask.
      for (size_t p = 0; p < img.size() / 4; ++p) {
        const float a = img[p * 4 + 3] / 255.0f;
        for (int c = 0; c < 3; ++c) {
          f.rgba[p * 4 + c] = a > 0 ? uint8_t(std::min(255.0f, img[p * 4 + c] / a + 0.5f)) : 0;
        }
        f.rgba[p * 4 + 3] = img[p * 4 + 3];
      }
    }
  }
  SplatScene init = truth;
  std::mt19937 rng(4);
  std::normal_distribution<float> noise(0.0f, 1.0f);
  for (float& v : init.positions) v += 0.05f * noise(rng);
  for (float& v : init.sh_dc) v = 0.0f;
  for (float& v : init.sh_rest) v = 0.0f;
  for (float& v : init.opacity_logits) v = -1.0f;
  TrainOptions o;
  o.iterations = steps;
  o.sh_degree = degree;
  o.sh_degree_interval = std::max(1u, steps / 3);
  // The truth is representable without new gaussians.
  o.densify_from = o.densify_until = 1000000;
  o.opacity_reset_interval = 0;
  o.max_gaussians = 2000;
  o.reduction = reduction;
  Trainer trainer(ctx, ds, init, o);
  FitReport report;
  report.reduction = trainer.reduction();
  report.psnr_before = trainer.evaluate().psnr;
  const auto start = std::chrono::steady_clock::now();
  while (!trainer.finished()) trainer.step();
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  report.steps = trainer.stats().step;
  report.ms_per_step = report.steps > 0 ? ms / report.steps : 0.0;
  report.psnr_after = trainer.evaluate().psnr;
  return report;
}

std::string trainer_self_test_json(VkContext& ctx) {
  JsonWriter w;
  w.begin_object();
  bool ok = true;
  w.key("gradients").begin_array();
  struct Case {
    TrainOptions::Reduction reduction;
    bool masked;
  };
  const Case cases[] = {{TrainOptions::Reduction::Subgroup, false},
                        {TrainOptions::Reduction::Subgroup, true},
                        {TrainOptions::Reduction::Shared, false},
                        {TrainOptions::Reduction::Shared, true}};
  for (const Case& c : cases) {
    w.begin_object();
    w.field("requested", c.reduction == TrainOptions::Reduction::Subgroup ? "subgroup" : "shared");
    w.field("masked", c.masked);
    try {
      const GradientReport r = check_gradients(ctx, c.masked, c.reduction);
      w.field("reduction", r.reduction);
      w.field("ok", r.ok());
      w.field("image_max_diff", r.image_max_diff);
      w.field("loss_gpu", r.loss_gpu);
      w.field("loss_ref", r.loss_ref);
      w.key("groups").begin_array();
      for (const GradientGroup& g : r.groups) {
        w.begin_object();
        w.field("name", g.name);
        w.field("bad", g.bad);
        w.field("checked", g.checked);
        w.field("max_err", g.max_err);
        w.field("max_ref", g.max_ref);
        w.end_object();
      }
      w.end_array();
      ok = ok && r.ok();
    } catch (const std::exception& e) {
      // The subgroup variant may be unavailable; that alone is not a failure.
      w.field("ok", false);
      w.field("error", e.what());
      if (c.reduction == TrainOptions::Reduction::Shared) ok = false;
    }
    w.end_object();
  }
  w.end_array();
  try {
    const FitReport f = fit_known_scene(ctx);
    w.key("fit").begin_object();
    w.field("reduction", f.reduction);
    w.field("ok", f.ok());
    w.field("psnr_before", f.psnr_before);
    w.field("psnr_after", f.psnr_after);
    w.field("steps", f.steps);
    w.field("ms_per_step", f.ms_per_step);
    w.end_object();
    ok = ok && f.ok();
  } catch (const std::exception& e) {
    w.key("fit").begin_object().field("ok", false).field("error", e.what()).end_object();
    ok = false;
  }
  w.field("ok", ok);
  w.end_object();
  return w.str();
}

}  // namespace gs
