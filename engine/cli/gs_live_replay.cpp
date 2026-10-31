// Replays a capture dataset (sensor-oriented, like the app's) through the
// live capture path as the app runs it: each frame becomes a YUV camera
// image, its mask the segmenter's upright output, then submit / prepare /
// add_mask, and the files are written to <out_dir>. Prints per-frame times
// and what subject selection did.
//
//   gs_live_replay <capture_dir> <out_dir> [--anchor x,y,z] [--bystander] [--keep-mask]
//                  [--train-short 540] [--archive-short 720]
//
// --bystander adds a second person to every mask (the subject's mask, smaller,
// at the side of the image) to check it is removed; --keep-mask turns the
// subject selection off, as before it existed.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "gs/capture.h"
#include "gs/json_parse.h"
#include "image_io.h"

namespace {

std::vector<uint8_t> yuv_planes(const std::vector<uint8_t>& rgb, uint32_t w, uint32_t h, gs::Yuv420View* view) {
  // Y plane, then interleaved V/U (NV21), full-range BT.601.
  std::vector<uint8_t> out(size_t(w) * h + size_t(w) * (h / 2));
  auto clamp8 = [](float v) { return uint8_t(std::lround(std::fmin(255.0f, std::fmax(0.0f, v)))); };
  for (size_t i = 0; i < size_t(w) * h; ++i) {
    const uint8_t* p = &rgb[i * 3];
    out[i] = clamp8(0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]);
  }
  uint8_t* uv = out.data() + size_t(w) * h;
  for (uint32_t y = 0; y < h / 2; ++y) {
    for (uint32_t x = 0; x < w / 2; ++x) {
      float cb = 0, cr = 0;
      for (int k = 0; k < 4; ++k) {
        const uint8_t* p = &rgb[(size_t(2 * y + k / 2) * w + 2 * x + k % 2) * 3];
        cb += 128.0f - 0.168736f * p[0] - 0.331264f * p[1] + 0.5f * p[2];
        cr += 128.0f + 0.5f * p[0] - 0.418688f * p[1] - 0.081312f * p[2];
      }
      uv[size_t(y) * w + 2 * x] = clamp8(cr / 4);
      uv[size_t(y) * w + 2 * x + 1] = clamp8(cb / 4);
    }
  }
  view->width = w;
  view->height = h;
  view->y = out.data();
  view->y_row_stride = w;
  view->v = uv;
  view->u = uv + 1;
  view->uv_row_stride = w;
  view->uv_pixel_stride = 2;
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <capture_dir> <out_dir> [--anchor x,y,z] [--bystander] [--train-short N] "
                         "[--archive-short N]\n", argv[0]);
    return 2;
  }
  const std::string in = argv[1], out = argv[2];
  float anchor[3] = {0, 1, 0};
  bool bystander = false, keep_mask = false;
  uint32_t train_short = 540, archive_short = 720;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--anchor" && i + 1 < argc) {
      std::sscanf(argv[++i], "%f,%f,%f", &anchor[0], &anchor[1], &anchor[2]);
    } else if (a == "--bystander") {
      bystander = true;
    } else if (a == "--keep-mask") {
      keep_mask = true;
    } else if (a == "--train-short" && i + 1 < argc) {
      train_short = uint32_t(std::atoi(argv[++i]));
    } else if (a == "--archive-short" && i + 1 < argc) {
      archive_short = uint32_t(std::atoi(argv[++i]));
    }
  }
  try {
    const std::vector<uint8_t> text = gs::read_file(in + "/transforms.json");
    const gs::JsonValue root = gs::parse_json(std::string_view(reinterpret_cast<const char*>(text.data()), text.size()));
    gs::LiveCaptureConfig config;
    config.name = std::filesystem::path(out).filename().string();
    config.archive_dir = out;
    config.sensor_width = uint32_t(root.number_or("w", 0));
    config.sensor_height = uint32_t(root.number_or("h", 0));
    config.fx = float(root.number_or("fl_x", 0));
    config.fy = float(root.number_or("fl_y", config.fx));
    config.cx = float(root.number_or("cx", config.sensor_width / 2.0));
    config.cy = float(root.number_or("cy", config.sensor_height / 2.0));
    config.train_short_side = train_short;
    config.archive_short_side = archive_short;
    config.depth_scale = float(root.number_or("depth_unit_scale_factor", 0.001));
    if (const gs::JsonValue* lab = root.find("gsplat_lab")) config.upright_degrees = int(lab->number_or("upright_degrees", 90));
    std::copy(anchor, anchor + 3, config.anchor);
    config.source = "replay";
    config.pick_subject = !keep_mask;
    std::filesystem::remove_all(out);
    gs::LiveCapture capture(config);

    const gs::JsonValue* frames = root.find("frames");
    double decode_ms = 0, prepare_ms = 0, add_ms = 0, submit_ms = 0;
    using Clock = std::chrono::steady_clock;
    auto ms = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    uint32_t n = 0;
    double extra_area[2] = {0, 0}, extra_left[2] = {0, 0};  // bystander, frames without / with depth
    for (const gs::JsonValue& f : frames->array) {
      const auto t0 = Clock::now();
      gs::Image8 image = gs::decode_image(gs::read_file(in + "/" + f.string_or("file_path", "")), 3);
      gs::Image8 mask = gs::decode_image(gs::read_file(in + "/" + f.string_or("mask_path", "")), 1);
      std::vector<uint16_t> depth;
      std::vector<uint8_t> conf;
      uint32_t dw = 0, dh = 0;
      const std::string dpath = f.string_or("depth_file_path", "");
      if (!dpath.empty()) {
        gs::Image16 d = gs::decode_image16(gs::read_file(in + "/" + dpath));
        depth = d.data;
        dw = d.width;
        dh = d.height;
        const std::string cpath = f.string_or("depth_confidence_path", "");
        if (!cpath.empty()) conf = gs::decode_image(gs::read_file(in + "/" + cpath), 1).data;
      }
      float c2w[16];
      const gs::JsonValue* m = f.find("transform_matrix");
      for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) c2w[r * 4 + c] = float(m->array[r].array[c].number);
      }
      // What the segmenter would return: the mask, upright, 0..1.
      std::vector<uint8_t> up = gs::rotate_image(mask.data.data(), mask.width, mask.height, 1, config.upright_degrees);
      const uint32_t uw = config.upright_degrees % 180 == 0 ? mask.width : mask.height;
      const uint32_t uh = config.upright_degrees % 180 == 0 ? mask.height : mask.width;
      std::vector<float> seg(up.size());
      for (size_t i = 0; i < up.size(); ++i) seg[i] = up[i] / 255.0f;
      if (bystander) {
        // Someone further back: the subject's mask at 60% size, on the side
        // of the image away from the subject.
        double sum = 0, mean_x = 0;
        for (uint32_t y = 0; y < uh; ++y) {
          for (uint32_t x = 0; x < uw; ++x) {
            sum += up[size_t(y) * uw + x];
            mean_x += double(x) * up[size_t(y) * uw + x];
          }
        }
        mean_x = sum > 0 ? mean_x / sum : uw / 2.0;
        const double centre = mean_x < uw / 2.0 ? 0.85 * uw : 0.15 * uw;
        for (uint32_t y = 0; y < uh; ++y) {
          for (uint32_t x = 0; x < uw; ++x) {
            const double sx = mean_x + (x - centre) / 0.6, sy = uh / 2.0 + (y - uh * 0.45) / 0.6;
            if (sx < 0 || sy < 0 || sx >= uw || sy >= uh) continue;
            const float v = up[size_t(sy) * uw + size_t(sx)] / 255.0f;
            seg[size_t(y) * uw + x] = std::fmax(seg[size_t(y) * uw + x], v);
          }
        }
      }
      if (bystander && !depth.empty()) {
        // The bystander's own depth: 3.5 m, over a metre behind the subject (where the subject does not hide them).
        std::vector<uint8_t> extra(up.size());
        for (size_t i = 0; i < up.size(); ++i) extra[i] = seg[i] > up[i] / 255.0f + 0.5f ? 255 : 0;
        extra = gs::rotate_image(extra.data(), uw, uh, 1, (360 - config.upright_degrees) % 360);
        for (uint32_t v = 0; v < dh; ++v) {
          for (uint32_t u = 0; u < dw; ++u) {
            const size_t mx = size_t((u + 0.5) * mask.width / dw), my = size_t((v + 0.5) * mask.height / dh);
            if (extra[my * mask.width + mx] == 0) continue;
            depth[size_t(v) * dw + u] = 3500;
            if (!conf.empty()) conf[size_t(v) * dw + u] = 40;
          }
        }
      }
      gs::Yuv420View view;
      const std::vector<uint8_t> planes = yuv_planes(image.data, image.width, image.height, &view);
      const auto t1 = Clock::now();
      const uint32_t index = capture.submit(view, c2w, int64_t(n) * 33'000'000, depth.empty() ? nullptr : depth.data(),
                                            dw, dh, dw, conf.empty() ? nullptr : conf.data(), dw);
      const auto t2 = Clock::now();
      capture.prepare(index);
      const auto t3 = Clock::now();
      const gs::LiveFrameReport r = capture.add_mask(index, seg.data(), uw, uh);
      if (bystander) {
        // The same selection without the bystander: more kept means it was joined to the subject.
        std::vector<uint8_t> alone = gs::rotate_image(up.data(), uw, uh, 1, (360 - config.upright_degrees) % 360);
        const uint32_t sw = config.upright_degrees % 180 == 0 ? uw : uh, sh = config.upright_degrees % 180 == 0 ? uh : uw;
        alone = gs::resize_box(alone.data(), sw, sh, 1, capture.width(), capture.height());
        std::vector<uint8_t> both(seg.size());
        for (size_t i = 0; i < seg.size(); ++i) both[i] = uint8_t(std::fmin(1.0f, seg[i]) * 255.0f + 0.5f);
        both = gs::rotate_image(both.data(), uw, uh, 1, (360 - config.upright_degrees) % 360);
        both = gs::resize_box(both.data(), sw, sh, 1, capture.width(), capture.height());
        double area = 0;
        for (size_t i = 0; i < both.size(); ++i) area += std::fmax(0.0, double(both[i]) - alone[i]);
        area /= 255.0 * both.size();
        const float s = capture.width() / float(config.sensor_width);
        const gs::SubjectSelection base = gs::select_subject(
            alone.data(), capture.width(), capture.height(),
            gs::subject_seeds(c2w, config.fx * s, config.fy * s, config.cx * s, config.cy * s, capture.width(),
                              capture.height(), config.anchor, config.world_up));
        const int d = depth.empty() ? 0 : 1;
        extra_area[d] += area;
        extra_left[d] += std::fmax(0.0, double(r.selection.coverage) - base.coverage);
      }
      const auto t4 = Clock::now();
      decode_ms += ms(t0, t1);
      submit_ms += ms(t1, t2);
      prepare_ms += ms(t2, t3);
      add_ms += ms(t3, t4);
      if (n % 30 == 0) {
        std::printf("frame %3u: kept %d, regions %u, dropped %u, coverage %.3f, removed %.3f (depth %.3f)\n", n,
                    int(r.kept), r.selection.regions, r.selection.dropped, r.selection.coverage, r.selection.removed,
                    r.depth_removed);
      }
      ++n;
    }
    const auto t5 = Clock::now();
    gs::Dataset ds = capture.take_dataset();
    const double take_ms = ms(t5, Clock::now());
    const auto t6 = Clock::now();
    capture.complete_archive();
    const double drain_ms = ms(t6, Clock::now());
    const gs::LiveCaptureStats s = capture.stats();
    std::printf("frames %u: kept %u, without subject %u, without depth %u, regions dropped %u, mean coverage %.3f\n",
                s.submitted, s.kept, s.without_subject, s.without_depth, s.regions_dropped,
                s.kept ? s.coverage_sum / s.kept : 0.0);
    std::printf("depth gate: cut %u frames, %.4f of the image on average in those\n", s.depth_gated,
                s.depth_gated ? s.depth_removed_sum / s.depth_gated : 0.0);
    if (bystander) {
      auto pct = [](double left, double area) { return area > 0 ? 100.0 * left / area : 0.0; };
      std::printf("bystander left in the masks: %.1f%% (frames with depth %.1f%%, without %.1f%%)\n",
                  pct(extra_left[0] + extra_left[1], extra_area[0] + extra_area[1]), pct(extra_left[1], extra_area[1]),
                  pct(extra_left[0], extra_area[0]));
    }
    std::printf("in-memory dataset: %zu frames %ux%u (%zu held out), ready %.1f ms after the last mask\n",
                ds.frames.size(), ds.width, ds.height, ds.holdout_count(), take_ms);
    std::printf("per frame ms: submit (copy) %.2f, prepare (convert) %.2f, add_mask (select) %.2f, "
                "files %.2f (background); archive drained %.0f ms after the end\n",
                submit_ms / n, prepare_ms / n, add_ms / n, s.archive_ms / std::max(1u, s.archive_written), drain_ms);
    std::printf("(replay decode %.1f ms per frame, not part of the app)\n", decode_ms / n);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
