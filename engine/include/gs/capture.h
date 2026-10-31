#pragma once

// Live capture: what happens to camera frames while the user walks round the
// subject, kept out of the platform code so Android (ARCore + ML Kit) and
// other platforms share it. The platform layer supplies camera images, poses,
// depth and a person mask; everything else (keyframe choice, conversion,
// picking the subject out of the mask, the training dataset and the files on
// disk) is here.

#include <array>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "gs/dataset.h"

namespace gs {

// A camera image in YUV 4:2:0 (Android YUV_420_888, iOS 420f): full-range
// BT.601 (JFIF), chroma at half resolution; U and V share a row stride and a
// pixel stride (1: planar, 2: interleaved).
struct Yuv420View {
  uint32_t width = 0;
  uint32_t height = 0;
  const uint8_t* y = nullptr;
  uint32_t y_row_stride = 0;
  const uint8_t* u = nullptr;
  const uint8_t* v = nullptr;
  uint32_t uv_row_stride = 0;
  uint32_t uv_pixel_stride = 1;
};

// RGB (3 channels) at out_w x out_h, area-averaged.
std::vector<uint8_t> yuv420_to_rgb(const Yuv420View& image, uint32_t out_w, uint32_t out_h);

// Area-average resize of an interleaved image (upscaling picks the covering pixel).
std::vector<uint8_t> resize_box(const uint8_t* src, uint32_t w, uint32_t h, uint32_t channels, uint32_t out_w,
                                uint32_t out_h);

// Rotates an interleaved image clockwise by 0, 90, 180 or 270 degrees (90 and 270 swap width and height).
std::vector<uint8_t> rotate_image(const uint8_t* src, uint32_t w, uint32_t h, uint32_t channels, int degrees);

// Variance of a Laplacian over every `step`th pixel of a luma plane: higher is sharper.
double sharpness(const uint8_t* y, uint32_t row_stride, uint32_t w, uint32_t h, uint32_t step = 4);

// Where the subject is in an image: the projections of points on its vertical
// axis, and how far round them (pixels) the subject is looked for.
struct SubjectSeeds {
  std::vector<std::array<float, 2>> points;
  float radius = 0;
};

// Pinhole projection (camera-to-world `c2w`, OpenGL axes, row-major) of a
// segment along `up` through `anchor` (the subject's middle), +-`half_height` metres.
SubjectSeeds subject_seeds(const float c2w[16], float fx, float fy, float cx, float cy, uint32_t width,
                           uint32_t height, const float anchor[3], const float up[3], float half_height = 0.3f,
                           float radius_m = 0.1f);

struct SubjectSelection {
  bool found = false;       // some part of the mask is the subject
  uint32_t regions = 0;     // separate regions in the mask (value >= 128)
  uint32_t dropped = 0;     // regions removed (other people, stray blobs)
  float coverage = 0;       // fraction of the image the kept mask covers
  float removed = 0;        // fraction of the image removed
};

// Keeps only the subject in a person mask (0..255, in place): the regions
// (value >= 128) that come within the seeds' radius, with the soft edge
// round them; everything else becomes 0. With no region near the seeds the
// nearest one within 4 radii is kept; failing that the mask is cleared.
SubjectSelection select_subject(uint8_t* mask, uint32_t w, uint32_t h, const SubjectSeeds& seeds);

// Clears mask pixels whose depth puts them further than `radius` metres from
// the subject's vertical axis (through `anchor` along `up`): someone else who
// touches the subject in the image. A depth sample (metres, 0 = none; aligned
// with the image, any resolution) counts only when it and its 8 neighbours all
// lie that far, so the subject's own edge, where depth blurs into the
// background, stays. Intrinsics are at the mask's resolution. Returns the
// fraction of the image cleared.
float gate_by_depth(uint8_t* mask, uint32_t w, uint32_t h, const float* depth, const uint8_t* confidence,
                    uint32_t depth_w, uint32_t depth_h, const float c2w[16], float fx, float fy, float cx, float cy,
                    const float anchor[3], const float up[3], float radius = 0.8f, uint8_t min_confidence = 1);

// Which frames become keyframes while recording: after the last one, the
// camera has moved (5% of its distance to the subject, at least 3 cm) or
// turned (5 degrees), at most every 200 ms, not moving too fast (60 deg/s,
// 0.8 m/s), and not blurred (sharpness under 0.6 of the median of the last 30).
class KeyframePolicy {
 public:
  static constexpr int kBins = 36;  // tenths of a turn round the subject, 10 degrees each

  struct Options {
    int64_t min_interval_ns = 200'000'000;
    float min_move_m = 0.03f;
    float move_fraction = 0.05f;
    float min_turn_deg = 5.0f;
    float max_angular_dps = 60.0f;
    float max_linear_mps = 0.8f;
    double blur_ratio = 0.6;
    size_t blur_window = 30;
    size_t blur_min_samples = 5;
    uint32_t max_keyframes = 250;
  };

  struct Counters {
    uint32_t candidates = 0;
    uint32_t rejected_fast = 0;
    uint32_t rejected_blur = 0;
    uint32_t tracking_lost = 0;
    uint32_t image_errors = 0;
  };

  KeyframePolicy() = default;
  explicit KeyframePolicy(const Options& options) : options_(options) {}

  void set_subject(const float anchor[3]);
  bool has_subject() const { return has_subject_; }
  const float* subject() const { return subject_; }
  void set_recording(bool recording) { recording_ = recording; }
  bool recording() const { return recording_; }

  // Every camera frame (rotation as a unit quaternion x, y, z, w). True when
  // this frame is a keyframe candidate: fetch its image and call accept().
  bool track(bool tracking, const float position[3], const float rotation[4], int64_t timestamp_ns);
  // The candidate's sharpness; true when it is taken as the next keyframe.
  bool accept(double sharpness);
  void image_error() { ++counters_.image_errors; }

  uint32_t keyframes() const { return keyframes_; }
  const Counters& counters() const { return counters_; }
  const std::array<bool, kBins>& coverage() const { return coverage_; }
  float angular_speed() const { return angular_speed_; }
  float linear_speed() const { return linear_speed_; }
  // Where the camera is now round the subject (-1 without a subject or right
  // above it), and its horizontal distance (metres).
  int azimuth_bin() const;
  float distance() const;

  static int azimuth_bin(const float camera[3], const float subject[3]);
  static float angle_degrees(const float a[4], const float b[4]);

 private:
  Options options_;
  bool recording_ = false;
  bool has_subject_ = false;
  float subject_[3] = {0, 0, 0};
  bool tracking_ = false;
  bool has_previous_ = false;
  float previous_position_[3] = {0, 0, 0};
  float previous_rotation_[4] = {0, 0, 0, 1};
  int64_t previous_time_ = 0;
  float angular_speed_ = 0;
  float linear_speed_ = 0;
  bool has_key_ = false;
  float key_position_[3] = {0, 0, 0};
  float key_rotation_[4] = {0, 0, 0, 1};
  int64_t key_time_ = 0;
  uint32_t keyframes_ = 0;
  std::deque<double> recent_sharpness_;
  std::array<bool, kBins> coverage_{};
  Counters counters_;
};

struct LiveCaptureConfig {
  std::string name;         // the dataset's name (its folder once saved)
  std::string archive_dir;  // where the files are written; empty keeps the capture in memory only
  uint32_t sensor_width = 0;  // camera image, in the sensor's orientation
  uint32_t sensor_height = 0;
  float fx = 0, fy = 0, cx = 0, cy = 0;  // camera image pixels
  uint32_t train_short_side = 540;     // the in-memory dataset's resolution
  uint32_t archive_short_side = 720;   // the saved images' resolution
  int upright_degrees = 90;            // clockwise rotation that stands the image upright
  float anchor[3] = {0, 0, 0};         // the subject's middle (world)
  float world_up[3] = {0, 1, 0};
  float depth_scale = 0.001f;          // depth units (millimetres) to metres
  bool pick_subject = true;            // false keeps the segmenter's mask as it is (for comparison)
  std::string source;  // what tracked the camera (e.g. "arcore"), for transforms.json
  std::string app_version;
  std::string device;
};

// What happened to one keyframe.
struct LiveFrameReport {
  bool kept = false;  // false: no subject in its mask, left out
  SubjectSelection selection;
  float depth_removed = 0;  // fraction of the image cleared by gate_by_depth
  double convert_ms = 0;  // camera image -> training and upright images
  double select_ms = 0;   // picking out the subject
};

struct LiveCaptureStats {
  uint32_t submitted = 0;
  uint32_t kept = 0;
  uint32_t without_subject = 0;
  uint32_t without_depth = 0;
  uint32_t regions_dropped = 0;  // other people / blobs removed, over all frames
  uint32_t depth_gated = 0;      // frames where depth removed part of the mask
  double depth_removed_sum = 0;
  double coverage_sum = 0;       // of kept frames
  double convert_ms = 0;         // totals
  double select_ms = 0;
  uint32_t archive_queued = 0;
  uint32_t archive_written = 0;
  uint32_t archive_failures = 0;
  double archive_ms = 0;  // encoding and writing, total
  std::string archive_error;
};

// One capture while it is recorded. submit() takes a keyframe's camera
// image, depth and pose (copied); prepare() converts it and returns the
// upright image for the person segmenter; add_mask() picks the subject out of
// the segmenter's mask and adds the frame to the in-memory dataset, and
// queues its files for a background thread. The dataset is ready the moment
// the last mask is in; the files follow.
class LiveCapture {
 public:
  explicit LiveCapture(const LiveCaptureConfig& config);
  ~LiveCapture();  // stops writing; files still queued are dropped
  LiveCapture(const LiveCapture&) = delete;
  LiveCapture& operator=(const LiveCapture&) = delete;

  const LiveCaptureConfig& config() const { return config_; }
  // Training resolution (sensor orientation) and the upright image's size.
  uint32_t width() const { return train_w_; }
  uint32_t height() const { return train_h_; }
  uint32_t upright_width() const;
  uint32_t upright_height() const;

  // Copies a keyframe; `c2w` is camera-to-world (OpenGL axes) row-major, depth
  // in `depth_scale` units with its row stride in samples. Returns its index.
  uint32_t submit(const Yuv420View& image, const float c2w[16], int64_t timestamp_ns, const uint16_t* depth,
                  uint32_t depth_w, uint32_t depth_h, uint32_t depth_row_stride, const uint8_t* confidence,
                  uint32_t confidence_row_stride);
  // RGBA, upright, upright_width() x upright_height(): what the segmenter looks at.
  std::vector<uint8_t> prepare(uint32_t index);
  // The segmenter's mask for keyframe `index` (upright, values 0..1, any size).
  LiveFrameReport add_mask(uint32_t index, const float* mask, uint32_t mask_w, uint32_t mask_h);
  // A keyframe the segmenter could not handle: left out.
  void drop(uint32_t index);

  // The frames added so far, in capture order (every 8th held out), at the
  // training resolution. Moves them out: call once, after the last add_mask().
  Dataset take_dataset();
  LiveCaptureStats stats() const;
  // Waits for the files and writes transforms.json. Throws on failure.
  void complete_archive();

 private:
  struct Pending;
  struct ArchiveJob;
  void archive_loop();
  void write_job(const ArchiveJob& job);

  LiveCaptureConfig config_;
  uint32_t train_w_ = 0, train_h_ = 0;
  uint32_t archive_w_ = 0, archive_h_ = 0;
  double train_scale_ = 1, archive_scale_ = 1;  // from the camera image
  mutable std::mutex mutex_;
  uint32_t next_index_ = 0;
  std::map<uint32_t, std::unique_ptr<Pending>> pending_;
  std::map<uint32_t, DatasetFrame> frames_;
  struct Written;
  std::map<uint32_t, std::unique_ptr<Written>> written_;  // frames whose files are on disk
  LiveCaptureStats stats_;
  bool taken_ = false;

  std::mutex archive_mutex_;
  std::condition_variable archive_cv_;
  std::deque<std::unique_ptr<ArchiveJob>> archive_queue_;
  bool archive_stop_ = false;
  bool archive_busy_ = false;
  std::thread archive_thread_;
};

}  // namespace gs
