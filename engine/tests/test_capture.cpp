// Live capture (gs/capture.h): image conversion, picking the subject out of a
// person mask, keyframe choice, and a capture built in memory matching the
// same capture read back from its files. No GPU needed.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <vector>

#include "gs/capture.h"
#include "gs/dataset.h"
#include "test_util.h"

namespace {

// Full-range BT.601 YUV 4:2:0 of an RGB image, chroma averaged over 2x2;
// interleaved V/U (NV21-like: pixel stride 2) with padded rows.
struct Yuv {
  uint32_t w, h, y_stride, uv_stride;
  std::vector<uint8_t> y, uv;  // uv: V U V U ...
  gs::Yuv420View view() const {
    gs::Yuv420View v;
    v.width = w;
    v.height = h;
    v.y = y.data();
    v.y_row_stride = y_stride;
    v.v = uv.data();
    v.u = uv.data() + 1;
    v.uv_row_stride = uv_stride;
    v.uv_pixel_stride = 2;
    return v;
  }
};

Yuv to_yuv(const std::vector<uint8_t>& rgb, uint32_t w, uint32_t h) {
  Yuv out{w, h, w + 6, w + 4, {}, {}};
  out.y.assign(size_t(out.y_stride) * h, 0);
  out.uv.assign(size_t(out.uv_stride) * (h / 2), 0);
  auto clamp8 = [](double v) { return uint8_t(std::lround(std::fmin(255.0, std::fmax(0.0, v)))); };
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      const uint8_t* p = &rgb[(size_t(y) * w + x) * 3];
      out.y[size_t(y) * out.y_stride + x] = clamp8(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]);
    }
  }
  for (uint32_t y = 0; y < h / 2; ++y) {
    for (uint32_t x = 0; x < w / 2; ++x) {
      double cb = 0, cr = 0;
      for (int k = 0; k < 4; ++k) {
        const uint8_t* p = &rgb[(size_t(2 * y + k / 2) * w + 2 * x + k % 2) * 3];
        cb += 128.0 - 0.168736 * p[0] - 0.331264 * p[1] + 0.5 * p[2];
        cr += 128.0 + 0.5 * p[0] - 0.418688 * p[1] - 0.081312 * p[2];
      }
      out.uv[size_t(y) * out.uv_stride + 2 * x + 0] = clamp8(cr / 4);
      out.uv[size_t(y) * out.uv_stride + 2 * x + 1] = clamp8(cb / 4);
    }
  }
  return out;
}

std::vector<uint8_t> smooth_rgb(uint32_t w, uint32_t h, int seed) {
  std::vector<uint8_t> rgb(size_t(w) * h * 3);
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      uint8_t* p = &rgb[(size_t(y) * w + x) * 3];
      p[0] = uint8_t(40 + (x * 150) / w + seed);
      p[1] = uint8_t(60 + (y * 120) / h);
      p[2] = uint8_t(200 - (x * 80) / w - (y * 40) / h);
    }
  }
  return rgb;
}

void test_yuv_and_rotation() {
  const uint32_t w = 64, h = 32;
  const std::vector<uint8_t> rgb = smooth_rgb(w, h, 0);
  const Yuv yuv = to_yuv(rgb, w, h);
  const std::vector<uint8_t> same = gs::yuv420_to_rgb(yuv.view(), w, h);
  int worst = 0;
  for (size_t i = 0; i < rgb.size(); ++i) worst = std::max(worst, std::abs(int(same[i]) - int(rgb[i])));
  EXPECT(worst <= 4, "YUV round trip off by %d", worst);
  // Half size: the average of each 2x2 block.
  const std::vector<uint8_t> half = gs::yuv420_to_rgb(yuv.view(), w / 2, h / 2);
  worst = 0;
  for (uint32_t y = 0; y < h / 2; ++y) {
    for (uint32_t x = 0; x < w / 2; ++x) {
      for (int c = 0; c < 3; ++c) {
        int sum = 0;
        for (int k = 0; k < 4; ++k) sum += rgb[(size_t(2 * y + k / 2) * w + 2 * x + k % 2) * 3 + c];
        worst = std::max(worst, std::abs(int(half[(size_t(y) * (w / 2) + x) * 3 + c]) - (sum + 2) / 4));
      }
    }
  }
  EXPECT(worst <= 4, "YUV half size off by %d", worst);

  // 3 x 2:  0 1 2      90 clockwise, 2 x 3:  3 0
  //         3 4 5                            4 1
  //                                          5 2   (as Android's Matrix.postRotate(90) turns a bitmap)
  const uint8_t src[6] = {0, 1, 2, 3, 4, 5};
  EXPECT((gs::rotate_image(src, 3, 2, 1, 90) == std::vector<uint8_t>{3, 0, 4, 1, 5, 2}), "rotate 90");
  EXPECT((gs::rotate_image(src, 3, 2, 1, 180) == std::vector<uint8_t>{5, 4, 3, 2, 1, 0}), "rotate 180");
  std::vector<uint8_t> rgba(4 * 3 * 4);
  for (size_t i = 0; i < rgba.size(); ++i) rgba[i] = uint8_t(i * 7);
  for (int upright : {0, 90, 180, 270}) {
    const std::vector<uint8_t> u = gs::rotate_image(rgba.data(), 4, 3, 4, upright);
    const uint32_t uw = upright % 180 == 0 ? 4 : 3, uh = upright % 180 == 0 ? 3 : 4;
    EXPECT(gs::rotate_image(u.data(), uw, uh, 4, (360 - upright) % 360) == rgba, "upright %d and back", upright);
  }
  const std::vector<uint8_t> box = gs::resize_box(rgb.data(), w, h, 3, w / 2, h / 2);
  EXPECT(std::abs(int(box[0]) - (rgb[0] + rgb[3] + rgb[w * 3] + rgb[w * 3 + 3] + 2) / 4) <= 1, "box resize");
}

void test_subject_selection() {
  // Camera at z = 2 looking down -z (identity OpenGL rotation), subject at the origin.
  const float c2w[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1};
  const float anchor[3] = {0, 0, 0}, up[3] = {0, 1, 0};
  const gs::SubjectSeeds seeds = gs::subject_seeds(c2w, 100, 100, 50, 30, 100, 60, anchor, up, 0.3f, 0.1f);
  EXPECT(seeds.points.size() == 13, "%zu seeds", seeds.points.size());
  EXPECT(std::fabs(seeds.points[6][0] - 50) < 1e-4f && std::fabs(seeds.points[6][1] - 30) < 1e-4f,
         "anchor projects to %.2f %.2f", seeds.points[6][0], seeds.points[6][1]);
  // +y (up) is towards the top of the image: 0.3 m up at 2 m is 15 px above.
  EXPECT(std::fabs(seeds.points[12][1] - 15) < 1e-3f, "top seed v %.3f", seeds.points[12][1]);
  EXPECT(std::fabs(seeds.radius - 6.0f) < 1e-4f, "radius %.2f (10 cm at 2 m is 5 px, at least 6)", seeds.radius);

  // The subject (round the seeds) with a soft edge, another person on the
  // right, and a speck far away.
  const uint32_t w = 100, h = 60;
  std::vector<uint8_t> mask(w * h, 0);
  auto fill = [&](int x0, int y0, int x1, int y1, uint8_t v) {
    for (int y = y0; y < y1; ++y) {
      for (int x = x0; x < x1; ++x) mask[size_t(y) * w + x] = v;
    }
  };
  fill(43, 10, 58, 55, 255);
  fill(42, 10, 43, 55, 90);  // soft edge
  fill(41, 10, 42, 55, 30);
  fill(80, 5, 95, 58, 255);  // someone else
  fill(79, 5, 80, 58, 60);
  fill(5, 5, 8, 8, 200);
  std::vector<uint8_t> m = mask;
  const gs::SubjectSelection s = gs::select_subject(m.data(), w, h, seeds);
  EXPECT(s.found && s.regions == 3 && s.dropped == 2, "found %d regions %u dropped %u", int(s.found), s.regions,
         s.dropped);
  EXPECT(m[20 * w + 50] == 255 && m[20 * w + 42] == 90 && m[20 * w + 41] == 30, "subject or its edge removed");
  EXPECT(m[20 * w + 85] == 0 && m[20 * w + 79] == 0 && m[6 * w + 6] == 0, "other person or speck kept");
  EXPECT(std::fabs(s.coverage - (15 * 45 + 45 * (90 + 30) / 255.0f) / (w * h)) < 1e-4f, "coverage %.4f", s.coverage);

  // The seeds miss every region by a little: the nearest within 4 radii is kept.
  m = mask;
  for (int y = 10; y < 55; ++y) {
    for (int x = 41; x < 58; ++x) m[size_t(y) * w + x] = 0;
  }
  for (int y = 10; y < 55; ++y) {
    for (int x = 62; x < 70; ++x) m[size_t(y) * w + x] = 255;  // 12 px from the seed line, radius 6
  }
  gs::SubjectSelection near = gs::select_subject(m.data(), w, h, seeds);
  EXPECT(near.found && m[20 * w + 65] == 255 && m[20 * w + 85] == 0, "nearest region not kept alone");
  // Nothing near: cleared.
  std::vector<uint8_t> far(w * h, 0);
  for (int y = 0; y < 10; ++y) {
    for (int x = 90; x < 100; ++x) far[size_t(y) * w + x] = 255;
  }
  gs::SubjectSelection none = gs::select_subject(far.data(), w, h, seeds);
  EXPECT(!none.found && far[5 * w + 95] == 0, "a mask without the subject must be cleared");
}

void test_depth_gate() {
  // Camera at z = 2 looking down -z, subject's axis through the origin. The
  // subject (x 40..60) is at 2 m, someone touching them in the image (x 70..95)
  // at 3.5 m, the wall at 4 m; depth is 5 px per sample, and the subject's
  // left edge column has the wall's depth (ARCore blurs edges).
  const float c2w[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1};
  const float anchor[3] = {0, 0, 0}, up[3] = {0, 1, 0};
  const uint32_t w = 100, h = 60, dw = 20, dh = 12;
  std::vector<float> depth(dw * dh, 4.0f);
  for (uint32_t v = 0; v < dh; ++v) {
    for (uint32_t u = 9; u < 12; ++u) depth[v * dw + u] = 2.0f;
    for (uint32_t u = 14; u < 19; ++u) depth[v * dw + u] = 3.5f;
  }
  std::vector<uint8_t> mask(w * h, 0);
  for (uint32_t y = 10; y < 50; ++y) {
    for (uint32_t x = 40; x < 60; ++x) mask[y * w + x] = 255;
    for (uint32_t x = 70; x < 95; ++x) mask[y * w + x] = 255;
  }
  for (uint32_t y = 28; y < 32; ++y) {
    for (uint32_t x = 60; x < 70; ++x) mask[y * w + x] = 255;  // joined in the image
  }
  const float cut = gs::gate_by_depth(mask.data(), w, h, depth.data(), nullptr, dw, dh, c2w, 100, 100, 50, 30, anchor, up);
  EXPECT(cut > 0.05f, "depth gate cut %.4f", cut);
  EXPECT(mask[20 * w + 50] == 255 && mask[20 * w + 41] == 255, "subject or its blurred edge cut");
  EXPECT(mask[20 * w + 85] == 0 && mask[40 * w + 80] == 0, "the person behind was not cut");
  // With the wall's depth missing (too far for ARCore), the person behind is
  // still cut up to their outline; the subject's blurred edge still stays.
  std::vector<float> holes = depth;
  for (float& z : holes) z = z == 4.0f ? 0.0f : z;
  for (uint32_t v = 0; v < dh; ++v) holes[v * dw + 8] = 4.0f;  // the subject's edge, as before
  std::vector<uint8_t> m3(w * h, 0);
  for (uint32_t y = 10; y < 50; ++y) {
    for (uint32_t x = 40; x < 60; ++x) m3[y * w + x] = 255;
    for (uint32_t x = 70; x < 95; ++x) m3[y * w + x] = 255;
  }
  gs::gate_by_depth(m3.data(), w, h, holes.data(), nullptr, dw, dh, c2w, 100, 100, 50, 30, anchor, up);
  EXPECT(m3[20 * w + 41] == 255 && m3[20 * w + 50] == 255, "subject edge cut with the wall unknown");
  EXPECT(m3[20 * w + 71] == 0 && m3[20 * w + 93] == 0 && m3[11 * w + 85] == 0, "person behind kept at their outline");
  // Missing depth decides nothing.
  std::vector<float> none(dw * dh, 0.0f);
  std::vector<uint8_t> m2(w * h, 255);
  EXPECT(gs::gate_by_depth(m2.data(), w, h, none.data(), nullptr, dw, dh, c2w, 100, 100, 50, 30, anchor, up) == 0.0f,
         "no depth must cut nothing");
}

void test_keyframe_policy() {
  using P = gs::KeyframePolicy;
  const float subject[3] = {0, 1, 0};
  const float a[3] = {0, 1.5f, 1}, b[3] = {1, 1.5f, 0}, c[3] = {0, 1.5f, -1}, d[3] = {-1, 1.5f, 0}, e[3] = {0, 2, 0};
  EXPECT(P::azimuth_bin(a, subject) == 0 && P::azimuth_bin(b, subject) == 9 && P::azimuth_bin(c, subject) == 18 &&
             P::azimuth_bin(d, subject) == 27 && P::azimuth_bin(e, subject) == -1,
         "azimuth bins %d %d %d %d %d", P::azimuth_bin(a, subject), P::azimuth_bin(b, subject),
         P::azimuth_bin(c, subject), P::azimuth_bin(d, subject), P::azimuth_bin(e, subject));
  const float identity[4] = {0, 0, 0, 1};
  const float s = std::sqrt(0.5f);
  const float quarter[4] = {0, s, 0, s};
  EXPECT(std::fabs(P::angle_degrees(identity, quarter) - 90) < 1e-3f && P::angle_degrees(quarter, quarter) < 0.1f,
         "quaternion angles");

  // A slow lap (radius 1.2 m, 40 s, 30 fps) facing the subject.
  P policy;
  const float anchor[3] = {0, 1, 0};
  policy.set_subject(anchor);
  policy.set_recording(true);
  int taken = 0, candidates = 0;
  const int frames = 40 * 30;
  for (int i = 0; i <= frames; ++i) {
    const double t = i / 30.0;
    const double angle = 2 * M_PI * i / frames;
    const float pos[3] = {float(1.2 * std::sin(angle)), 1.4f, float(1.2 * std::cos(angle))};
    // Yaw so the camera's -z points at the subject.
    const float rot[4] = {0, float(std::sin(angle / 2)), 0, float(std::cos(angle / 2))};
    if (policy.track(true, pos, rot, int64_t(t * 1e9))) {
      ++candidates;
      taken += policy.accept(100.0 + (i % 7)) ? 1 : 0;
    }
  }
  int covered = 0;
  for (bool bin : policy.coverage()) covered += bin ? 1 : 0;
  // 7.5 m of path, a keyframe every 6 cm (5% of 1.2 m) or so.
  EXPECT(taken == int(policy.keyframes()) && taken > 90 && taken < 150 && covered == P::kBins,
         "lap: %d keyframes of %d candidates, %d/%d bins", taken, candidates, covered, P::kBins);
  EXPECT(policy.counters().rejected_fast == 0 && policy.counters().rejected_blur == 0, "slow sharp lap rejected some");
  EXPECT(std::fabs(policy.linear_speed() - 0.19f) < 0.02f, "speed %.3f m/s", policy.linear_speed());

  // Too fast, then blurred, then tracking lost.
  P fast;
  fast.set_subject(anchor);
  fast.set_recording(true);
  int fast_taken = 0;
  for (int i = 0; i < 60; ++i) {
    const float pos[3] = {0.05f * i, 1.4f, 1.2f};  // 1.5 m/s
    if (fast.track(true, pos, identity, int64_t(i * 1e9 / 30))) fast_taken += fast.accept(100) ? 1 : 0;
  }
  EXPECT(fast_taken <= 1 && fast.counters().rejected_fast > 5, "fast: %d taken, %u rejected", fast_taken,
         fast.counters().rejected_fast);
  P blur;
  blur.set_subject(anchor);
  blur.set_recording(true);
  int blurred = 0;
  for (int i = 0; i < 300; ++i) {
    const double angle = 2 * M_PI * i / 1200;
    const float pos[3] = {float(1.2 * std::sin(angle)), 1.4f, float(1.2 * std::cos(angle))};
    const float rot[4] = {0, float(std::sin(angle / 2)), 0, float(std::cos(angle / 2))};
    if (blur.track(true, pos, rot, int64_t(i * 1e9 / 30))) {
      const bool sharp = blur.counters().candidates % 4 != 0;
      if (!blur.accept(sharp ? 100.0 : 30.0)) ++blurred;
    }
  }
  EXPECT(blurred > 3 && blur.counters().rejected_blur == uint32_t(blurred), "blurred frames %d", blurred);
  blur.track(false, a, identity, 0);
  EXPECT(blur.counters().tracking_lost == 1, "tracking lost %u", blur.counters().tracking_lost);

  // Sharpness: a checkerboard is sharper than a flat image.
  std::vector<uint8_t> flat(64 * 64, 128), checker(64 * 64);
  for (size_t i = 0; i < checker.size(); ++i) checker[i] = ((i % 64) / 4 + (i / 64) / 4) % 2 ? 200 : 50;
  EXPECT(gs::sharpness(flat.data(), 64, 64, 64) == 0.0 && gs::sharpness(checker.data(), 64, 64, 64) > 1000.0,
         "sharpness");
}

void test_live_capture() {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "gs_live_capture_test";
  std::filesystem::remove_all(dir);
  gs::LiveCaptureConfig config;
  config.name = "scan-test";
  config.archive_dir = dir.string();
  config.sensor_width = 96;  // landscape sensor image
  config.sensor_height = 48;
  config.fx = config.fy = 80;
  config.cx = 48;
  config.cy = 24;
  config.train_short_side = 24;
  config.archive_short_side = 36;
  config.upright_degrees = 90;
  config.source = "test";
  gs::LiveCapture capture(config);
  EXPECT(capture.width() == 48 && capture.height() == 24 && capture.upright_width() == 24 &&
             capture.upright_height() == 48,
         "sizes %ux%u upright %ux%u", capture.width(), capture.height(), capture.upright_width(),
         capture.upright_height());
  const int count = 12;
  for (int i = 0; i < count; ++i) {
    const Yuv yuv = to_yuv(smooth_rgb(96, 48, i), 96, 48);
    const float c2w[16] = {1, 0, 0, 0.1f * i, 0, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1};
    std::vector<uint16_t> depth(8 * 4 + 2 * 4, uint16_t(2000));  // rows of 10 samples, 8 used
    std::vector<uint8_t> conf(8 * 4 + 2 * 4, 255);
    const bool with_depth = i % 3 != 0;
    const uint32_t index = capture.submit(yuv.view(), c2w, 1000 * i, with_depth ? depth.data() : nullptr, 8, 4, 10,
                                          with_depth ? conf.data() : nullptr, 10);
    EXPECT(index == uint32_t(i), "index %u", index);
    const std::vector<uint8_t> upright = capture.prepare(index);
    EXPECT(upright.size() == 24 * 48 * 4, "upright size %zu", upright.size());
    if (i == 5) {
      capture.drop(index);  // the segmenter failed
      continue;
    }
    // Upright mask (24 x 48): the person in the middle; frame 7 has nobody.
    std::vector<float> mask(24 * 48, 0.0f);
    if (i != 7) {
      for (int y = 8; y < 44; ++y) {
        for (int x = 8; x < 16; ++x) mask[size_t(y) * 24 + x] = 1.0f;
      }
    }
    const gs::LiveFrameReport r = capture.add_mask(index, mask.data(), 24, 48);
    EXPECT(r.kept == (i != 7), "frame %d kept %d", i, int(r.kept));
  }
  const gs::LiveCaptureStats stats = capture.stats();
  EXPECT(stats.submitted == count && stats.kept == count - 2 && stats.without_subject == 1 &&
             stats.without_depth == 4,
         "stats: %u submitted, %u kept, %u without subject, %u without depth", stats.submitted, stats.kept,
         stats.without_subject, stats.without_depth);
  gs::Dataset memory = capture.take_dataset();
  capture.complete_archive();
  EXPECT(capture.stats().archive_written == uint32_t(count - 2) && capture.stats().archive_failures == 0,
         "archive: %u written, %u failed (%s)", capture.stats().archive_written, capture.stats().archive_failures,
         capture.stats().archive_error.c_str());
  gs::Dataset disk = gs::load_dataset(dir.string(), 24);
  EXPECT(memory.frames.size() == size_t(count - 2) && disk.frames.size() == memory.frames.size(),
         "frames: memory %zu, disk %zu", memory.frames.size(), disk.frames.size());
  EXPECT(memory.width == disk.width && memory.height == disk.height && std::fabs(memory.fx - disk.fx) < 1e-4f &&
             std::fabs(memory.cx - disk.cx) < 1e-4f && std::fabs(memory.cy - disk.cy) < 1e-4f,
         "intrinsics: memory %ux%u fx %.4f cx %.4f, disk %ux%u fx %.4f cx %.4f", memory.width, memory.height,
         memory.fx, memory.cx, disk.width, disk.height, disk.fx, disk.cx);
  EXPECT(memory.has_masks && disk.has_masks && memory.has_depth && disk.has_depth, "masks/depth flags");
  EXPECT(memory.holdout_count() == disk.holdout_count() && memory.holdout_count() == 2, "holdout %zu / %zu",
         memory.holdout_count(), disk.holdout_count());
  EXPECT(std::fabs(memory.camera_center[0] - disk.camera_center[0]) < 1e-5f &&
             std::fabs(memory.camera_extent - disk.camera_extent) < 1e-5f,
         "camera bounds differ");
  int worst = 0;
  bool same = true;
  for (size_t f = 0; f < memory.frames.size() && f < disk.frames.size(); ++f) {
    const gs::DatasetFrame &a = memory.frames[f], &b = disk.frames[f];
    same = same && a.name == b.name && a.holdout == b.holdout && a.depth.size() == b.depth.size() &&
           a.depth_confidence.size() == b.depth_confidence.size();
    for (int k = 0; k < 16; ++k) same = same && std::fabs(a.c2w[k] - b.c2w[k]) < 1e-6f;
    for (size_t i = 0; i < a.depth.size() && i < b.depth.size(); ++i) same = same && std::fabs(a.depth[i] - b.depth[i]) < 1e-6f;
    for (size_t i = 0; i < a.rgba.size() && i < b.rgba.size(); ++i) {
      const int diff = std::abs(int(a.rgba[i]) - int(b.rgba[i]));
      if (i % 4 == 3) {
        same = same && diff == 0;  // the mask is stored losslessly at this resolution
      } else {
        worst = std::max(worst, diff);  // JPEG
      }
    }
  }
  EXPECT(same, "memory and disk frames differ (names, poses, depth or masks)");
  EXPECT(worst <= 12, "image differs from its JPEG by %d", worst);
  // The mask came back to the sensor orientation: upright (x, y) is sensor (y, 23 - x) for 90 degrees.
  const gs::DatasetFrame& f0 = memory.frames[0];
  EXPECT(f0.rgba[(size_t(23 - 10) * 48 + 20) * 4 + 3] == 255 && f0.rgba[(size_t(23 - 20) * 48 + 20) * 4 + 3] == 0,
         "mask orientation");
  std::filesystem::remove_all(dir);
}

}  // namespace

int main() {
  try {
    test_yuv_and_rotation();
    test_subject_selection();
    test_depth_gate();
    test_keyframe_policy();
    test_live_capture();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: exception %s\n", e.what());
    return 1;
  }
  if (test::failures > 0) {
    std::fprintf(stderr, "%d failure(s)\n", test::failures);
    return 1;
  }
  std::printf("all capture tests passed\n");
  return 0;
}
