#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

#include "gs/capture.h"
#include "gs/json.h"
#include "gs/png_write.h"

namespace gs {
namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

void scaled_size(uint32_t w, uint32_t h, uint32_t short_side, uint32_t* out_w, uint32_t* out_h, double* scale) {
  const double s = std::min(1.0, double(short_side) / std::min(w, h));
  *out_w = uint32_t(std::lround(w * s));
  *out_h = uint32_t(std::lround(h * s));
  *scale = s;
}

std::string frame_name(const char* dir, uint32_t index, const char* ext) {
  char name[64];
  std::snprintf(name, sizeof(name), "%s/frame_%05u.%s", dir, index, ext);
  return name;
}

std::string number(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.9g", v);
  return buf;
}

}  // namespace

struct LiveCapture::Pending {
  float c2w[16];
  int64_t timestamp = 0;
  uint32_t w = 0, h = 0;
  std::vector<uint8_t> y, u, v;  // contiguous; chroma (w/2) x (h/2)
  std::vector<uint16_t> depth;
  uint32_t depth_w = 0, depth_h = 0;
  std::vector<uint8_t> confidence;
  // From prepare():
  bool prepared = false;
  std::vector<uint8_t> archive_rgb;  // archive resolution, when files are written
  std::vector<uint8_t> rgba;         // training resolution
  double convert_ms = 0;
};

struct LiveCapture::ArchiveJob {
  uint32_t index = 0;
  float c2w[16];
  int64_t timestamp = 0;
  std::vector<uint8_t> rgb;
  std::vector<uint8_t> mask;  // training resolution
  std::vector<uint16_t> depth;
  uint32_t depth_w = 0, depth_h = 0;
  std::vector<uint8_t> confidence;
};

struct LiveCapture::Written {
  float c2w[16];
  int64_t timestamp = 0;
  bool depth = false;
  bool confidence = false;
};

LiveCapture::LiveCapture(const LiveCaptureConfig& config) : config_(config) {
  if (config_.sensor_width < 2 || config_.sensor_height < 2 || config_.fx <= 0.0f) {
    throw std::invalid_argument("카메라 이미지 크기나 내부 파라미터가 없음");
  }
  const int r = config_.upright_degrees;
  if (r != 0 && r != 90 && r != 180 && r != 270) throw std::invalid_argument("세움 회전은 0/90/180/270만 가능");
  scaled_size(config_.sensor_width, config_.sensor_height, config_.train_short_side, &train_w_, &train_h_,
              &train_scale_);
  scaled_size(config_.sensor_width, config_.sensor_height, std::max(config_.archive_short_side, config_.train_short_side),
              &archive_w_, &archive_h_, &archive_scale_);
  if (!config_.archive_dir.empty()) {
    for (const char* sub : {"images", "masks", "depth", "confidence"}) {
      std::filesystem::create_directories(std::filesystem::path(config_.archive_dir) / sub);
    }
    archive_thread_ = std::thread([this] { archive_loop(); });
  }
}

LiveCapture::~LiveCapture() {
  {
    std::lock_guard<std::mutex> lock(archive_mutex_);
    archive_stop_ = true;
    archive_queue_.clear();
  }
  archive_cv_.notify_all();
  if (archive_thread_.joinable()) archive_thread_.join();
}

uint32_t LiveCapture::upright_width() const {
  return config_.upright_degrees % 180 == 0 ? train_w_ : train_h_;
}

uint32_t LiveCapture::upright_height() const {
  return config_.upright_degrees % 180 == 0 ? train_h_ : train_w_;
}

uint32_t LiveCapture::submit(const Yuv420View& image, const float c2w[16], int64_t timestamp_ns,
                             const uint16_t* depth, uint32_t depth_w, uint32_t depth_h, uint32_t depth_row_stride,
                             const uint8_t* confidence, uint32_t confidence_row_stride) {
  if (image.width != config_.sensor_width || image.height != config_.sensor_height) {
    throw std::invalid_argument("카메라 이미지 크기가 촬영 시작 때와 다름");
  }
  auto p = std::make_unique<Pending>();
  std::copy(c2w, c2w + 16, p->c2w);
  p->timestamp = timestamp_ns;
  p->w = image.width;
  p->h = image.height;
  p->y.resize(size_t(p->w) * p->h);
  for (uint32_t row = 0; row < p->h; ++row) {
    std::copy_n(image.y + size_t(row) * image.y_row_stride, p->w, &p->y[size_t(row) * p->w]);
  }
  const uint32_t cw = p->w / 2, ch = p->h / 2;
  p->u.resize(size_t(cw) * ch);
  p->v.resize(size_t(cw) * ch);
  for (uint32_t row = 0; row < ch; ++row) {
    const uint8_t* u = image.u + size_t(row) * image.uv_row_stride;
    const uint8_t* v = image.v + size_t(row) * image.uv_row_stride;
    for (uint32_t col = 0; col < cw; ++col) {
      p->u[size_t(row) * cw + col] = u[size_t(col) * image.uv_pixel_stride];
      p->v[size_t(row) * cw + col] = v[size_t(col) * image.uv_pixel_stride];
    }
  }
  if (depth != nullptr && depth_w > 0 && depth_h > 0) {
    p->depth_w = depth_w;
    p->depth_h = depth_h;
    p->depth.resize(size_t(depth_w) * depth_h);
    for (uint32_t row = 0; row < depth_h; ++row) {
      std::copy_n(depth + size_t(row) * depth_row_stride, depth_w, &p->depth[size_t(row) * depth_w]);
    }
    if (confidence != nullptr) {
      p->confidence.resize(p->depth.size());
      for (uint32_t row = 0; row < depth_h; ++row) {
        std::copy_n(confidence + size_t(row) * confidence_row_stride, depth_w, &p->confidence[size_t(row) * depth_w]);
      }
    }
  }
  std::lock_guard<std::mutex> lock(mutex_);
  const uint32_t index = next_index_++;
  ++stats_.submitted;
  if (p->depth.empty()) ++stats_.without_depth;
  pending_[index] = std::move(p);
  return index;
}

std::vector<uint8_t> LiveCapture::prepare(uint32_t index) {
  std::unique_ptr<Pending> p;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = pending_.find(index);
    if (it == pending_.end()) throw std::invalid_argument("없는 키프레임");
    p = std::move(it->second);
    pending_.erase(it);
  }
  const auto t0 = Clock::now();
  Yuv420View view;
  view.width = p->w;
  view.height = p->h;
  view.y = p->y.data();
  view.y_row_stride = p->w;
  view.u = p->u.data();
  view.v = p->v.data();
  view.uv_row_stride = p->w / 2;
  view.uv_pixel_stride = 1;
  std::vector<uint8_t> rgb;
  if (!config_.archive_dir.empty()) {
    // The training image comes from the saved one, as it would when loaded from disk.
    p->archive_rgb = yuv420_to_rgb(view, archive_w_, archive_h_);
    rgb = resize_box(p->archive_rgb.data(), archive_w_, archive_h_, 3, train_w_, train_h_);
  } else {
    rgb = yuv420_to_rgb(view, train_w_, train_h_);
  }
  p->y = {};
  p->u = {};
  p->v = {};
  p->rgba.resize(size_t(train_w_) * train_h_ * 4);
  for (size_t i = 0; i < size_t(train_w_) * train_h_; ++i) {
    p->rgba[i * 4 + 0] = rgb[i * 3 + 0];
    p->rgba[i * 4 + 1] = rgb[i * 3 + 1];
    p->rgba[i * 4 + 2] = rgb[i * 3 + 2];
    p->rgba[i * 4 + 3] = 255;
  }
  std::vector<uint8_t> upright = rotate_image(p->rgba.data(), train_w_, train_h_, 4, config_.upright_degrees);
  p->prepared = true;
  p->convert_ms = ms_since(t0);
  std::lock_guard<std::mutex> lock(mutex_);
  pending_[index] = std::move(p);
  return upright;
}

void LiveCapture::drop(uint32_t index) {
  std::lock_guard<std::mutex> lock(mutex_);
  pending_.erase(index);
}

LiveFrameReport LiveCapture::add_mask(uint32_t index, const float* mask, uint32_t mask_w, uint32_t mask_h) {
  std::unique_ptr<Pending> p;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = pending_.find(index);
    if (it == pending_.end() || !it->second->prepared) throw std::invalid_argument("준비되지 않은 키프레임");
    p = std::move(it->second);
    pending_.erase(it);
  }
  LiveFrameReport report;
  report.convert_ms = p->convert_ms;
  const auto t0 = Clock::now();
  std::vector<uint8_t> m(size_t(mask_w) * mask_h);
  for (size_t i = 0; i < m.size(); ++i) m[i] = uint8_t(std::clamp(mask[i], 0.0f, 1.0f) * 255.0f + 0.5f);
  // Back to the sensor's orientation: the rest of the way round.
  const int back = (360 - config_.upright_degrees) % 360;
  m = rotate_image(m.data(), mask_w, mask_h, 1, back);
  const uint32_t sw = back % 180 == 0 ? mask_w : mask_h, sh = back % 180 == 0 ? mask_h : mask_w;
  if (sw != train_w_ || sh != train_h_) m = resize_box(m.data(), sw, sh, 1, train_w_, train_h_);
  const float s = float(train_scale_);
  const SubjectSeeds seeds = subject_seeds(p->c2w, config_.fx * s, config_.fy * s, config_.cx * s, config_.cy * s,
                                           train_w_, train_h_, config_.anchor, config_.world_up);
  if (config_.pick_subject) {
    report.selection = select_subject(m.data(), train_w_, train_h_, seeds);
  } else {
    double sum = 0;
    for (uint8_t v : m) sum += v;
    report.selection.found = true;
    report.selection.coverage = float(sum / (255.0 * m.size()));
  }
  if (config_.pick_subject && report.selection.found && !p->depth.empty()) {
    std::vector<float> metres(p->depth.size());
    for (size_t i = 0; i < metres.size(); ++i) metres[i] = float(p->depth[i]) * config_.depth_scale;
    report.depth_removed = gate_by_depth(m.data(), train_w_, train_h_, metres.data(),
                                         p->confidence.empty() ? nullptr : p->confidence.data(), p->depth_w,
                                         p->depth_h, p->c2w, config_.fx * s, config_.fy * s, config_.cx * s,
                                         config_.cy * s, config_.anchor, config_.world_up);
    if (report.depth_removed > 0.0f) {
      // What depth cut off may leave pieces apart from the subject.
      const SubjectSelection again = select_subject(m.data(), train_w_, train_h_, seeds);
      report.selection.dropped += again.dropped;
      report.selection.removed += report.depth_removed + again.removed;
      report.selection.coverage = again.coverage;
      report.selection.found = again.found;
    }
  }
  report.select_ms = ms_since(t0);
  report.kept = report.selection.found;

  std::unique_ptr<ArchiveJob> job;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.convert_ms += report.convert_ms;
    stats_.select_ms += report.select_ms;
    stats_.regions_dropped += report.selection.dropped;
    if (report.depth_removed > 0.0f) {
      ++stats_.depth_gated;
      stats_.depth_removed_sum += report.depth_removed;
    }
    if (!report.kept) {
      ++stats_.without_subject;
      return report;
    }
    ++stats_.kept;
    stats_.coverage_sum += report.selection.coverage;
    if (!taken_) {
      DatasetFrame f;
      f.name = frame_name("images", index, "jpg");
      std::copy(p->c2w, p->c2w + 16, f.c2w);
      f.rgba = std::move(p->rgba);
      for (size_t i = 0; i < m.size(); ++i) f.rgba[i * 4 + 3] = m[i];
      if (!p->depth.empty()) {
        f.depth_width = p->depth_w;
        f.depth_height = p->depth_h;
        f.depth.resize(p->depth.size());
        for (size_t i = 0; i < p->depth.size(); ++i) f.depth[i] = float(p->depth[i]) * config_.depth_scale;
        f.depth_confidence = p->confidence;
      }
      frames_[index] = std::move(f);
    }
    if (!config_.archive_dir.empty()) {
      job = std::make_unique<ArchiveJob>();
      job->index = index;
      std::copy(p->c2w, p->c2w + 16, job->c2w);
      job->timestamp = p->timestamp;
      job->rgb = std::move(p->archive_rgb);
      job->mask = std::move(m);
      job->depth = std::move(p->depth);
      job->depth_w = p->depth_w;
      job->depth_h = p->depth_h;
      job->confidence = std::move(p->confidence);
      ++stats_.archive_queued;
    }
  }
  if (job) {
    {
      std::lock_guard<std::mutex> lock(archive_mutex_);
      archive_queue_.push_back(std::move(job));
    }
    archive_cv_.notify_all();
  }
  return report;
}

Dataset LiveCapture::take_dataset() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (taken_) throw std::logic_error("데이터셋은 한 번만 꺼낼 수 있음");
  taken_ = true;
  Dataset ds;
  ds.name = config_.name;
  ds.width = train_w_;
  ds.height = train_h_;
  const float s = float(train_scale_);
  ds.fx = config_.fx * s;
  ds.fy = config_.fy * s;
  ds.cx = config_.cx * s;
  ds.cy = config_.cy * s;
  ds.has_masks = true;
  std::copy(config_.world_up, config_.world_up + 3, ds.world_up);
  for (auto& [index, frame] : frames_) {
    frame.holdout = ds.frames.size() % 8 == 0;
    ds.has_depth = ds.has_depth || !frame.depth.empty();
    ds.frames.push_back(std::move(frame));
  }
  frames_.clear();
  compute_camera_bounds(ds);
  return ds;
}

LiveCaptureStats LiveCapture::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

void LiveCapture::archive_loop() {
  for (;;) {
    std::unique_ptr<ArchiveJob> job;
    {
      std::unique_lock<std::mutex> lock(archive_mutex_);
      archive_cv_.wait(lock, [&] { return archive_stop_ || !archive_queue_.empty(); });
      if (archive_queue_.empty()) return;  // stopping, nothing left
      job = std::move(archive_queue_.front());
      archive_queue_.pop_front();
      archive_busy_ = true;
    }
    const auto t0 = Clock::now();
    std::string error;
    try {
      write_job(*job);
    } catch (const std::exception& e) {
      error = e.what();
    }
    const double ms = ms_since(t0);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stats_.archive_ms += ms;
      if (error.empty()) {
        ++stats_.archive_written;
        auto w = std::make_unique<Written>();
        std::copy(job->c2w, job->c2w + 16, w->c2w);
        w->timestamp = job->timestamp;
        w->depth = !job->depth.empty();
        w->confidence = !job->confidence.empty();
        written_[job->index] = std::move(w);
      } else {
        ++stats_.archive_failures;
        stats_.archive_error = error;
      }
    }
    {
      std::lock_guard<std::mutex> lock(archive_mutex_);
      archive_busy_ = false;
    }
    archive_cv_.notify_all();
  }
}

void LiveCapture::write_job(const ArchiveJob& job) {
  const std::filesystem::path dir(config_.archive_dir);
  write_file((dir / frame_name("images", job.index, "jpg")).string(),
             encode_jpeg_rgb(job.rgb.data(), archive_w_, archive_h_, 90));
  write_file((dir / frame_name("masks", job.index, "png")).string(),
             encode_png_gray8(job.mask.data(), train_w_, train_h_));
  if (!job.depth.empty()) {
    write_file((dir / frame_name("depth", job.index, "png")).string(),
               encode_png_gray16(job.depth.data(), job.depth_w, job.depth_h));
    if (!job.confidence.empty()) {
      write_file((dir / frame_name("confidence", job.index, "png")).string(),
                 encode_png_gray8(job.confidence.data(), job.depth_w, job.depth_h));
    }
  }
}

void LiveCapture::complete_archive() {
  if (config_.archive_dir.empty()) return;
  {
    std::unique_lock<std::mutex> lock(archive_mutex_);
    archive_cv_.wait(lock, [&] { return archive_queue_.empty() && !archive_busy_; });
    archive_stop_ = true;
  }
  archive_cv_.notify_all();
  if (archive_thread_.joinable()) archive_thread_.join();

  std::lock_guard<std::mutex> lock(mutex_);
  if (written_.empty()) throw std::runtime_error("저장된 키프레임이 없음" + (stats_.archive_error.empty() ? std::string() : " (" + stats_.archive_error + ")"));
  const double s = archive_scale_;
  JsonWriter w;
  w.begin_object();
  w.field("camera_model", "OPENCV");
  w.field("w", archive_w_);
  w.field("h", archive_h_);
  w.key("fl_x").raw(number(config_.fx * s));
  w.key("fl_y").raw(number(config_.fy * s));
  w.key("cx").raw(number(config_.cx * s));
  w.key("cy").raw(number(config_.cy * s));
  for (const char* k : {"k1", "k2", "p1", "p2"}) w.field(k, 0);
  w.key("depth_unit_scale_factor").raw(number(config_.depth_scale));
  w.key("gsplat_lab").begin_object();
  w.field("subject", "person");
  w.field("capture", config_.source);
  w.field("app_version", config_.app_version);
  w.field("device", config_.device);
  w.key("world_up").begin_array();
  for (float v : config_.world_up) w.raw(number(v));
  w.end_array();
  w.key("subject_anchor").begin_array();
  for (float v : config_.anchor) w.raw(number(v));
  w.end_array();
  w.field("image_orientation", "sensor");
  w.field("upright_degrees", config_.upright_degrees);
  w.field("camera_width", config_.sensor_width);
  w.field("camera_height", config_.sensor_height);
  w.end_object();
  w.key("frames").begin_array();
  for (const auto& [index, f] : written_) {
    w.begin_object();
    w.field("file_path", frame_name("images", index, "jpg"));
    w.field("mask_path", frame_name("masks", index, "png"));
    if (f->depth) w.field("depth_file_path", frame_name("depth", index, "png"));
    if (f->confidence) w.field("depth_confidence_path", frame_name("confidence", index, "png"));
    w.key("transform_matrix").begin_array();
    for (int r = 0; r < 4; ++r) {
      w.begin_array();
      for (int c = 0; c < 4; ++c) w.raw(number(f->c2w[r * 4 + c]));
      w.end_array();
    }
    w.end_array();
    w.field("timestamp", f->timestamp);
    w.end_object();
  }
  w.end_array();
  w.end_object();
  const std::string& json = w.str();
  write_file((std::filesystem::path(config_.archive_dir) / "transforms.json").string(),
             std::vector<uint8_t>(json.begin(), json.end()));
}

}  // namespace gs
