#include "gs/diagnose.h"

#include <algorithm>
#include <cmath>

#include "gs/camera.h"
#include "gs/dataset.h"

namespace gs {

namespace {

constexpr int kEdgeRadius = 3;          // px around the mask's edge
constexpr size_t kMinInsidePixels = 500;  // for a view's gain

double psnr_of(double sse, double values) { return values > 0.0 ? -10.0 * std::log10(std::max(sse / values, 1e-12)) : 0.0; }

// Running max (dilate) or min (erode) of a binary image over a (2r+1)^2 square, separably.
std::vector<uint8_t> morph(const std::vector<uint8_t>& in, uint32_t w, uint32_t h, int r, bool dilate) {
  std::vector<uint8_t> tmp(in.size()), out(in.size());
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      uint8_t v = dilate ? 0 : 1;
      for (int d = -r; d <= r; ++d) {
        const int xx = std::clamp(int(x) + d, 0, int(w) - 1);
        const uint8_t s = in[size_t(y) * w + xx];
        v = dilate ? std::max(v, s) : std::min(v, s);
      }
      tmp[size_t(y) * w + x] = v;
    }
  }
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      uint8_t v = dilate ? 0 : 1;
      for (int d = -r; d <= r; ++d) {
        const int yy = std::clamp(int(y) + d, 0, int(h) - 1);
        const uint8_t s = tmp[size_t(yy) * w + x];
        v = dilate ? std::max(v, s) : std::min(v, s);
      }
      out[size_t(y) * w + x] = v;
    }
  }
  return out;
}

double percentile(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double pos = q * double(v.size() - 1);
  const size_t i = size_t(pos);
  const double t = pos - double(i);
  return i + 1 < v.size() ? v[i] * (1.0 - t) + v[i + 1] * t : v[i];
}

double stddev(const std::vector<double>& v) {
  if (v.size() < 2) return 0.0;
  double mean = 0.0, sq = 0.0;
  for (double x : v) mean += x;
  mean /= double(v.size());
  for (double x : v) sq += (x - mean) * (x - mean);
  return std::sqrt(sq / double(v.size() - 1));
}

}  // namespace

Diagnosis diagnose(Trainer& trainer) {
  Diagnosis d;
  const Dataset& ds = trainer.dataset();
  const uint32_t w = ds.width, h = ds.height;
  const size_t n = size_t(w) * h;
  const float black[3] = {0.0f, 0.0f, 0.0f};
  const HeadRegion& head = trainer.options().head;
  const double luma[3] = {0.299, 0.587, 0.114};

  double region[5] = {0, 0, 0, 0, 0};  // edge, outline, inside, head, outside
  double inside_sse = 0.0, inside_fitted = 0.0, inside_values = 0.0, psnr_sum = 0.0, edge_pixels = 0.0;
  std::vector<double> gains, tints, sharpness;
  std::vector<Diagnosis::View> views;

  for (size_t f = 0; f < ds.frames.size(); ++f) {
    const DatasetFrame& frame = ds.frames[f];
    const std::vector<uint8_t> render = trainer.render_view(f, black);
    const std::vector<uint8_t>& target = frame.rgba;
    float gain[3];  // the trainer's exposure gains: the photo as the model was fitted to it
    trainer.view_gain(f, gain);
    std::vector<uint8_t> mask(n);
    for (size_t p = 0; p < n; ++p) mask[p] = target[p * 4 + 3] >= 128 ? 1 : 0;
    const std::vector<uint8_t> grown = morph(mask, w, h, kEdgeRadius, true);
    const std::vector<uint8_t> shrunk = morph(mask, w, h, kEdgeRadius, false);

    // The head circle in this view, as the trainer projects it.
    double hx = 0, hy = 0, hr = 0;
    if (head.found) {
      const Camera cam = trainer.camera_for(f);
      double t[3];
      for (int r = 0; r < 3; ++r) {
        t[r] = cam.translation[r];
        for (int c = 0; c < 3; ++c) t[r] += double(cam.rotation[r * 3 + c]) * head.center[c];
      }
      if (t[2] > 0.05) {
        hx = cam.fx * t[0] / t[2] + cam.cx;
        hy = cam.fy * t[1] / t[2] + cam.cy;
        hr = cam.fx * head.radius / t[2];
      }
    }

    double sse = 0.0;
    // Inside pixels: sums per channel of render R, photo P, R^2, R P, P^2.
    double s_r[3] = {0, 0, 0}, s_p[3] = {0, 0, 0}, s_rr[3] = {0, 0, 0}, s_rp[3] = {0, 0, 0}, s_pp[3] = {0, 0, 0};
    size_t inside_count = 0;
    double lap_sum = 0.0, lap_sq = 0.0;
    size_t lap_count = 0;
    for (uint32_t y = 0; y < h; ++y) {
      for (uint32_t x = 0; x < w; ++x) {
        const size_t p = size_t(y) * w + x;
        const double m = target[p * 4 + 3] / 255.0, a = render[p * 4 + 3] / 255.0;
        double rgb_r[3], rgb_p[3], e2 = 0.0;
        for (int c = 0; c < 3; ++c) {
          rgb_r[c] = render[p * 4 + c] / 255.0;  // over black: already times coverage
          rgb_p[c] = target[p * 4 + c] / 255.0 * m / gain[c];
          e2 += (rgb_r[c] - rgb_p[c]) * (rgb_r[c] - rgb_p[c]);
        }
        sse += e2;
        const bool edge = grown[p] != shrunk[p];
        if (edge) {
          region[0] += e2;
          edge_pixels += 1.0;
        } else if (std::fabs(a - m) > 0.5) {
          region[1] += e2;
        } else if (mask[p]) {
          region[2] += e2;
          if (hr > 0.0 && std::hypot(x + 0.5 - hx, y + 0.5 - hy) <= hr) region[3] += e2;
        } else {
          region[4] += e2;
        }
        if (!edge && mask[p] && m > 0.95 && a > 0.95) {
          for (int c = 0; c < 3; ++c) {
            s_r[c] += rgb_r[c];
            s_p[c] += rgb_p[c];
            s_rr[c] += rgb_r[c] * rgb_r[c];
            s_rp[c] += rgb_r[c] * rgb_p[c];
            s_pp[c] += rgb_p[c] * rgb_p[c];
          }
          ++inside_count;
          // Laplacian of the photo's brightness, where the 4 neighbours are inside too.
          if (x > 0 && y > 0 && x + 1 < w && y + 1 < h && shrunk[p - 1] && shrunk[p + 1] && shrunk[p - w] &&
              shrunk[p + w]) {
            auto lum = [&](size_t q) {
              return luma[0] * target[q * 4] + luma[1] * target[q * 4 + 1] + luma[2] * target[q * 4 + 2];
            };
            const double lap = (lum(p - 1) + lum(p + 1) + lum(p - w) + lum(p + w) - 4.0 * lum(p)) / 255.0;
            lap_sum += lap;
            lap_sq += lap * lap;
            ++lap_count;
          }
        }
      }
    }

    Diagnosis::View view;
    view.frame = f;
    view.holdout = frame.holdout;
    view.psnr = psnr_of(sse, 3.0 * double(n));
    psnr_sum += view.psnr;
    if (inside_count >= kMinInsidePixels) {
      double gain[3];
      for (int c = 0; c < 3; ++c) {
        gain[c] = s_rr[c] > 0.0 ? s_rp[c] / s_rr[c] : 1.0;
        const double k = double(inside_count);
        const double plain = s_pp[c] - 2.0 * s_rp[c] + s_rr[c];  // sum (P - R)^2
        // Best P ~ a R + b: what remains is var(P) (1 - corr^2) times the count.
        const double var_r = s_rr[c] - s_r[c] * s_r[c] / k, var_p = s_pp[c] - s_p[c] * s_p[c] / k;
        const double cov = s_rp[c] - s_r[c] * s_p[c] / k;
        const double fitted = var_r > 1e-12 ? std::max(var_p - cov * cov / var_r, 0.0) : var_p;
        inside_sse += plain;
        inside_fitted += std::min(fitted, plain);
      }
      inside_values += 3.0 * double(inside_count);
      view.gain = luma[0] * gain[0] + luma[1] * gain[1] + luma[2] * gain[2];
      gains.push_back(view.gain);
      if (gain[1] > 1e-6) tints.push_back(0.5 * (gain[0] / gain[1] + gain[2] / gain[1]));
    }
    if (lap_count > 0) {
      const double mean = lap_sum / double(lap_count);
      view.sharpness = lap_sq / double(lap_count) - mean * mean;
    } else {
      view.sharpness = 0.0;
    }
    sharpness.push_back(view.sharpness);
    views.push_back(view);
  }

  d.views = uint32_t(views.size());
  if (views.empty()) return d;
  d.psnr = psnr_sum / double(views.size());
  const double total = region[0] + region[1] + region[2] + region[4];
  if (total > 0.0) {
    d.share_edge = region[0] / total;
    d.share_outline = region[1] / total;
    d.share_inside = region[2] / total;
    d.share_head = region[3] / total;
    d.share_outside = region[4] / total;
  }
  d.edge_pixels = edge_pixels / (double(n) * double(views.size()));
  d.gain_views = uint32_t(gains.size());
  if (!gains.empty()) {
    d.gain_low = percentile(gains, 0.05);
    d.gain_high = percentile(gains, 0.95);
    d.gain_spread = stddev(gains);
    d.tint_spread = stddev(tints);
    d.inside_psnr = psnr_of(inside_sse, inside_values);
    d.inside_psnr_fitted = psnr_of(inside_fitted, inside_values);
  }
  const double median = percentile(sharpness, 0.5);
  size_t blurry = 0;
  for (Diagnosis::View& v : views) {
    v.sharpness = median > 0.0 ? v.sharpness / median : 1.0;
    if (v.sharpness < 0.5) ++blurry;
  }
  d.blurry_fraction = double(blurry) / double(views.size());
  std::sort(views.begin(), views.end(), [](const Diagnosis::View& a, const Diagnosis::View& b) { return a.psnr < b.psnr; });
  views.resize(std::min<size_t>(views.size(), 3));
  d.worst = views;
  return d;
}

}  // namespace gs
