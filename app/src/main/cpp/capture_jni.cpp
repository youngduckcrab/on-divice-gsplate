// JNI entry points for com.ondevice.gsplat.NativeCapture: the live capture
// (engine gs/capture.h) driven from ARCore frames. Keyframe choice runs on the
// GL thread; a capture's frames are converted and masked on the app's
// pipeline thread; files are written by the capture's own thread.
#include <jni.h>

#include <algorithm>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "capture_handoff.h"
#include "gs/capture.h"
#include "gs/json.h"
#include "jni_util.h"

namespace {

std::mutex g_policy_mutex;
gs::KeyframePolicy g_policy;

std::mutex g_captures_mutex;
std::map<jlong, std::shared_ptr<gs::LiveCapture>> g_captures;
jlong g_next_handle = 1;

std::mutex g_handoff_mutex;
std::unique_ptr<CapturedDataset> g_handoff;

std::shared_ptr<gs::LiveCapture> capture_of(jlong handle) {
  std::lock_guard<std::mutex> lock(g_captures_mutex);
  auto it = g_captures.find(handle);
  return it == g_captures.end() ? nullptr : it->second;
}

std::string error_json(const std::string& message) {
  gs::JsonWriter w;
  w.begin_object().field("ok", false).field("error", message).end_object();
  return w.str();
}

const uint8_t* buffer_of(JNIEnv* env, jobject buffer) {
  return buffer == nullptr ? nullptr : static_cast<const uint8_t*>(env->GetDirectBufferAddress(buffer));
}

// Column-major 4x4 (ARCore Pose.toMatrix) -> row-major.
void rows_of(JNIEnv* env, jfloatArray column_major, float out[16]) {
  float m[16];
  env->GetFloatArrayRegion(column_major, 0, 16, m);
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) out[r * 4 + c] = m[c * 4 + r];
  }
}

// RGBA bytes -> ARGB ints (android.graphics.Bitmap's int[] layout).
jintArray argb_of(JNIEnv* env, const std::vector<uint8_t>& rgba) {
  const size_t n = rgba.size() / 4;
  std::vector<jint> argb(n);
  for (size_t i = 0; i < n; ++i) {
    argb[i] = jint((uint32_t(rgba[i * 4 + 3]) << 24) | (uint32_t(rgba[i * 4]) << 16) |
                   (uint32_t(rgba[i * 4 + 1]) << 8) | uint32_t(rgba[i * 4 + 2]));
  }
  jintArray out = env->NewIntArray(jsize(n));
  env->SetIntArrayRegion(out, 0, jsize(n), argb.data());
  return out;
}

gs::Yuv420View yuv_of(JNIEnv* env, jobject y, jobject u, jobject v, jint y_row_stride, jint uv_row_stride,
                      jint uv_pixel_stride, jint width, jint height) {
  gs::Yuv420View view;
  view.width = uint32_t(width);
  view.height = uint32_t(height);
  view.y = buffer_of(env, y);
  view.u = buffer_of(env, u);
  view.v = buffer_of(env, v);
  view.y_row_stride = uint32_t(y_row_stride);
  view.uv_row_stride = uint32_t(uv_row_stride);
  view.uv_pixel_stride = uint32_t(uv_pixel_stride);
  if (view.y == nullptr || view.u == nullptr || view.v == nullptr) throw std::invalid_argument("카메라 버퍼가 direct가 아님");
  return view;
}

void write_stats(gs::JsonWriter& w, const gs::LiveCaptureStats& s) {
  w.field("submitted", s.submitted);
  w.field("kept", s.kept);
  w.field("without_subject", s.without_subject);
  w.field("without_depth", s.without_depth);
  w.field("regions_dropped", s.regions_dropped);
  w.field("depth_gated", s.depth_gated);
  w.field("depth_removed", s.depth_gated ? s.depth_removed_sum / s.depth_gated : 0.0);
  w.field("coverage", s.kept ? s.coverage_sum / s.kept : 0.0);
  w.field("convert_ms", s.submitted ? s.convert_ms / s.submitted : 0.0);
  w.field("select_ms", s.submitted ? s.select_ms / s.submitted : 0.0);
  w.field("archive_queued", s.archive_queued);
  w.field("archive_written", s.archive_written);
  w.field("archive_failures", s.archive_failures);
  w.field("archive_ms", s.archive_written ? s.archive_ms / s.archive_written : 0.0);
  w.field("archive_error", s.archive_error);
}

}  // namespace

void hand_over_capture(std::unique_ptr<CapturedDataset> capture) {
  std::lock_guard<std::mutex> lock(g_handoff_mutex);
  g_handoff = std::move(capture);
}

std::unique_ptr<CapturedDataset> take_capture(const std::string& name, uint32_t short_side) {
  std::lock_guard<std::mutex> lock(g_handoff_mutex);
  if (!g_handoff || g_handoff->name != name || g_handoff->short_side != short_side) return nullptr;
  return std::move(g_handoff);
}

void discard_capture() {
  std::lock_guard<std::mutex> lock(g_handoff_mutex);
  g_handoff.reset();
}

// --- Keyframe choice (GL thread) ----------------------------------------------

extern "C" JNIEXPORT void JNICALL Java_com_ondevice_gsplat_NativeCapture_reset(JNIEnv*, jobject) {
  std::lock_guard<std::mutex> lock(g_policy_mutex);
  g_policy = gs::KeyframePolicy();
}

extern "C" JNIEXPORT void JNICALL Java_com_ondevice_gsplat_NativeCapture_setSubject(JNIEnv*, jobject, jfloat x,
                                                                                    jfloat y, jfloat z) {
  std::lock_guard<std::mutex> lock(g_policy_mutex);
  const float anchor[3] = {x, y, z};
  g_policy.set_subject(anchor);
}

extern "C" JNIEXPORT void JNICALL Java_com_ondevice_gsplat_NativeCapture_setRecording(JNIEnv*, jobject,
                                                                                      jboolean recording) {
  std::lock_guard<std::mutex> lock(g_policy_mutex);
  g_policy.set_recording(recording == JNI_TRUE);
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_ondevice_gsplat_NativeCapture_track(JNIEnv* env, jobject,
                                                                                   jboolean tracking,
                                                                                   jfloatArray jposition,
                                                                                   jfloatArray jrotation, jlong t) {
  float position[3], rotation[4];
  env->GetFloatArrayRegion(jposition, 0, 3, position);
  env->GetFloatArrayRegion(jrotation, 0, 4, rotation);
  std::lock_guard<std::mutex> lock(g_policy_mutex);
  return g_policy.track(tracking == JNI_TRUE, position, rotation, int64_t(t)) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_ondevice_gsplat_NativeCapture_accept(JNIEnv* env, jobject, jobject y,
                                                                                    jint row_stride, jint width,
                                                                                    jint height) {
  const uint8_t* luma = buffer_of(env, y);
  if (luma == nullptr) return JNI_FALSE;
  const double sharp = gs::sharpness(luma, uint32_t(row_stride), uint32_t(width), uint32_t(height));
  std::lock_guard<std::mutex> lock(g_policy_mutex);
  return g_policy.accept(sharp) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL Java_com_ondevice_gsplat_NativeCapture_imageError(JNIEnv*, jobject) {
  std::lock_guard<std::mutex> lock(g_policy_mutex);
  g_policy.image_error();
}

extern "C" JNIEXPORT jfloatArray JNICALL Java_com_ondevice_gsplat_NativeCapture_status(JNIEnv* env, jobject) {
  constexpr int kHead = 10;
  std::vector<float> s(kHead + gs::KeyframePolicy::kBins);
  {
    std::lock_guard<std::mutex> lock(g_policy_mutex);
    const gs::KeyframePolicy::Counters& c = g_policy.counters();
    s[0] = float(g_policy.keyframes());
    s[1] = g_policy.angular_speed();
    s[2] = g_policy.linear_speed();
    s[3] = float(g_policy.azimuth_bin());
    s[4] = g_policy.distance();
    s[5] = float(c.candidates);
    s[6] = float(c.rejected_fast);
    s[7] = float(c.rejected_blur);
    s[8] = float(c.tracking_lost);
    s[9] = float(c.image_errors);
    for (int b = 0; b < gs::KeyframePolicy::kBins; ++b) s[kHead + b] = g_policy.coverage()[size_t(b)] ? 1.0f : 0.0f;
  }
  jfloatArray out = env->NewFloatArray(jsize(s.size()));
  env->SetFloatArrayRegion(out, 0, jsize(s.size()), s.data());
  return out;
}

// --- A capture ---------------------------------------------------------------------

extern "C" JNIEXPORT jstring JNICALL Java_com_ondevice_gsplat_NativeCapture_begin(
    JNIEnv* env, jobject, jstring jname, jstring jdir, jint sensor_w, jint sensor_h, jfloatArray jintrinsics,
    jint train_short, jint archive_short, jint upright, jfloatArray janchor, jstring japp, jstring jdevice) {
  try {
    gs::LiveCaptureConfig config;
    config.name = from_jstring(env, jname);
    config.archive_dir = from_jstring(env, jdir);
    config.sensor_width = uint32_t(sensor_w);
    config.sensor_height = uint32_t(sensor_h);
    float k[4];
    env->GetFloatArrayRegion(jintrinsics, 0, 4, k);
    config.fx = k[0];
    config.fy = k[1];
    config.cx = k[2];
    config.cy = k[3];
    config.train_short_side = uint32_t(train_short);
    config.archive_short_side = uint32_t(archive_short);
    config.upright_degrees = upright;
    env->GetFloatArrayRegion(janchor, 0, 3, config.anchor);
    config.source = "arcore";
    config.app_version = from_jstring(env, japp);
    config.device = from_jstring(env, jdevice);
    // A capture still in memory is replaced by this one.
    discard_capture();
    auto capture = std::make_shared<gs::LiveCapture>(config);
    jlong handle = 0;
    {
      std::lock_guard<std::mutex> lock(g_captures_mutex);
      handle = g_next_handle++;
      g_captures[handle] = capture;
    }
    gs::JsonWriter w;
    w.begin_object();
    w.field("ok", true);
    w.field("handle", int64_t(handle));
    w.field("width", capture->width());
    w.field("height", capture->height());
    w.field("upright_width", capture->upright_width());
    w.field("upright_height", capture->upright_height());
    w.end_object();
    return to_jstring(env, w.str());
  } catch (const std::exception& e) {
    return to_jstring(env, error_json(e.what()));
  }
}

extern "C" JNIEXPORT jint JNICALL Java_com_ondevice_gsplat_NativeCapture_submit(
    JNIEnv* env, jobject, jlong handle, jobject y, jobject u, jobject v, jint y_row_stride, jint uv_row_stride,
    jint uv_pixel_stride, jint width, jint height, jfloatArray jpose, jlong timestamp, jobject jdepth,
    jint depth_row_stride, jint depth_pixel_stride, jint depth_w, jint depth_h, jobject jconfidence,
    jint confidence_row_stride, jint confidence_pixel_stride) {
  std::shared_ptr<gs::LiveCapture> capture = capture_of(handle);
  if (!capture) return -1;
  try {
    const gs::Yuv420View view = yuv_of(env, y, u, v, y_row_stride, uv_row_stride, uv_pixel_stride, width, height);
    float c2w[16];
    rows_of(env, jpose, c2w);
    // Depth planes (bytes, with strides) -> packed samples.
    std::vector<uint16_t> depth;
    std::vector<uint8_t> confidence;
    const uint8_t* d = buffer_of(env, jdepth);
    if (d != nullptr && depth_w > 0 && depth_h > 0) {
      depth.resize(size_t(depth_w) * depth_h);
      for (jint r = 0; r < depth_h; ++r) {
        for (jint c = 0; c < depth_w; ++c) {
          const uint8_t* p = d + size_t(r) * depth_row_stride + size_t(c) * depth_pixel_stride;
          depth[size_t(r) * depth_w + c] = uint16_t(p[0] | (p[1] << 8));  // little-endian
        }
      }
      const uint8_t* cf = buffer_of(env, jconfidence);
      if (cf != nullptr) {
        confidence.resize(depth.size());
        for (jint r = 0; r < depth_h; ++r) {
          for (jint c = 0; c < depth_w; ++c) {
            confidence[size_t(r) * depth_w + c] = cf[size_t(r) * confidence_row_stride + size_t(c) * confidence_pixel_stride];
          }
        }
      }
    }
    return jint(capture->submit(view, c2w, int64_t(timestamp), depth.empty() ? nullptr : depth.data(),
                                uint32_t(depth_w), uint32_t(depth_h), uint32_t(depth_w),
                                confidence.empty() ? nullptr : confidence.data(), uint32_t(depth_w)));
  } catch (const std::exception&) {
    return -1;
  }
}

extern "C" JNIEXPORT jintArray JNICALL Java_com_ondevice_gsplat_NativeCapture_prepare(JNIEnv* env, jobject,
                                                                                      jlong handle, jint index) {
  std::shared_ptr<gs::LiveCapture> capture = capture_of(handle);
  if (!capture) return nullptr;
  try {
    return argb_of(env, capture->prepare(uint32_t(index)));
  } catch (const std::exception&) {
    return nullptr;
  }
}

extern "C" JNIEXPORT jstring JNICALL Java_com_ondevice_gsplat_NativeCapture_addMask(JNIEnv* env, jobject,
                                                                                    jlong handle, jint index,
                                                                                    jfloatArray jmask, jint w, jint h) {
  std::shared_ptr<gs::LiveCapture> capture = capture_of(handle);
  if (!capture) return to_jstring(env, error_json("촬영이 없음"));
  try {
    if (w <= 0 || h <= 0 || env->GetArrayLength(jmask) < w * h) throw std::invalid_argument("마스크 크기가 맞지 않음");
    std::vector<float> mask(size_t(w) * h);
    env->GetFloatArrayRegion(jmask, 0, w * h, mask.data());
    const gs::LiveFrameReport r = capture->add_mask(uint32_t(index), mask.data(), uint32_t(w), uint32_t(h));
    gs::JsonWriter out;
    out.begin_object();
    out.field("ok", true);
    out.field("kept", r.kept);
    out.field("coverage", r.selection.coverage);
    out.field("removed", r.selection.removed);
    out.field("dropped", r.selection.dropped);
    out.field("depth_removed", r.depth_removed);
    out.field("convert_ms", r.convert_ms);
    out.field("select_ms", r.select_ms);
    out.end_object();
    return to_jstring(env, out.str());
  } catch (const std::exception& e) {
    capture->drop(uint32_t(index));
    return to_jstring(env, error_json(e.what()));
  }
}

extern "C" JNIEXPORT void JNICALL Java_com_ondevice_gsplat_NativeCapture_drop(JNIEnv*, jobject, jlong handle,
                                                                              jint index) {
  if (std::shared_ptr<gs::LiveCapture> capture = capture_of(handle)) capture->drop(uint32_t(index));
}

extern "C" JNIEXPORT jstring JNICALL Java_com_ondevice_gsplat_NativeCapture_stats(JNIEnv* env, jobject,
                                                                                  jlong handle) {
  std::shared_ptr<gs::LiveCapture> capture = capture_of(handle);
  if (!capture) return to_jstring(env, error_json("촬영이 없음"));
  gs::JsonWriter w;
  w.begin_object();
  w.field("ok", true);
  write_stats(w, capture->stats());
  w.end_object();
  return to_jstring(env, w.str());
}

extern "C" JNIEXPORT jstring JNICALL Java_com_ondevice_gsplat_NativeCapture_finish(JNIEnv* env, jobject,
                                                                                   jlong handle) {
  std::shared_ptr<gs::LiveCapture> capture = capture_of(handle);
  if (!capture) return to_jstring(env, error_json("촬영이 없음"));
  try {
    const auto t0 = std::chrono::steady_clock::now();
    auto ready = std::make_unique<CapturedDataset>();
    ready->name = capture->config().name;
    ready->short_side = capture->config().train_short_side;
    ready->dataset = capture->take_dataset();
    const size_t frames = ready->dataset.frames.size();
    const size_t holdout = ready->dataset.holdout_count();
    const bool depth = ready->dataset.has_depth;
    hand_over_capture(std::move(ready));
    gs::JsonWriter w;
    w.begin_object();
    w.field("ok", true);
    w.field("frames", frames);
    w.field("holdout", holdout);
    w.field("depth", depth);
    w.field("width", capture->width());
    w.field("height", capture->height());
    w.field("short_side", capture->config().train_short_side);
    w.field("finish_ms", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    write_stats(w, capture->stats());
    w.end_object();
    return to_jstring(env, w.str());
  } catch (const std::exception& e) {
    return to_jstring(env, error_json(e.what()));
  }
}

extern "C" JNIEXPORT jstring JNICALL Java_com_ondevice_gsplat_NativeCapture_complete(JNIEnv* env, jobject,
                                                                                     jlong handle) {
  std::shared_ptr<gs::LiveCapture> capture = capture_of(handle);
  if (!capture) return to_jstring(env, error_json("촬영이 없음"));
  try {
    capture->complete_archive();
    gs::JsonWriter w;
    w.begin_object();
    w.field("ok", true);
    write_stats(w, capture->stats());
    w.end_object();
    {
      std::lock_guard<std::mutex> lock(g_captures_mutex);
      g_captures.erase(handle);
    }
    return to_jstring(env, w.str());
  } catch (const std::exception& e) {
    return to_jstring(env, error_json(e.what()));
  }
}

extern "C" JNIEXPORT void JNICALL Java_com_ondevice_gsplat_NativeCapture_release(JNIEnv*, jobject, jlong handle) {
  std::shared_ptr<gs::LiveCapture> capture;
  {
    std::lock_guard<std::mutex> lock(g_captures_mutex);
    auto it = g_captures.find(handle);
    if (it == g_captures.end()) return;
    capture = std::move(it->second);
    g_captures.erase(it);
  }
  // Dropped here, outside the lock: stops its file thread.
}

extern "C" JNIEXPORT void JNICALL Java_com_ondevice_gsplat_NativeCapture_discardMemory(JNIEnv*, jobject) {
  discard_capture();
}

// --- Live view ----------------------------------------------------------------------

extern "C" JNIEXPORT jintArray JNICALL Java_com_ondevice_gsplat_NativeCapture_liveImage(
    JNIEnv* env, jobject, jobject y, jobject u, jobject v, jint y_row_stride, jint uv_row_stride, jint uv_pixel_stride,
    jint width, jint height, jint out_w, jint out_h, jint upright) {
  try {
    const gs::Yuv420View view = yuv_of(env, y, u, v, y_row_stride, uv_row_stride, uv_pixel_stride, width, height);
    const std::vector<uint8_t> rgb = gs::yuv420_to_rgb(view, uint32_t(out_w), uint32_t(out_h));
    std::vector<uint8_t> rgba(size_t(out_w) * out_h * 4);
    for (size_t i = 0; i < size_t(out_w) * out_h; ++i) {
      rgba[i * 4] = rgb[i * 3];
      rgba[i * 4 + 1] = rgb[i * 3 + 1];
      rgba[i * 4 + 2] = rgb[i * 3 + 2];
      rgba[i * 4 + 3] = 255;
    }
    return argb_of(env, gs::rotate_image(rgba.data(), uint32_t(out_w), uint32_t(out_h), 4, upright));
  } catch (const std::exception&) {
    return nullptr;
  }
}

extern "C" JNIEXPORT jbyteArray JNICALL Java_com_ondevice_gsplat_NativeCapture_liveMask(
    JNIEnv* env, jobject, jfloatArray jmask, jint w, jint h, jint upright, jint out_w, jint out_h, jfloatArray jpose,
    jfloatArray jintrinsics, jfloatArray janchor) {
  try {
    if (w <= 0 || h <= 0 || env->GetArrayLength(jmask) < w * h) return nullptr;
    std::vector<float> mask(size_t(w) * h);
    env->GetFloatArrayRegion(jmask, 0, w * h, mask.data());
    std::vector<uint8_t> m(mask.size());
    for (size_t i = 0; i < m.size(); ++i) m[i] = uint8_t(std::clamp(mask[i], 0.0f, 1.0f) * 255.0f + 0.5f);
    const int back = (360 - upright) % 360;
    m = gs::rotate_image(m.data(), uint32_t(w), uint32_t(h), 1, back);
    const uint32_t sw = back % 180 == 0 ? uint32_t(w) : uint32_t(h), sh = back % 180 == 0 ? uint32_t(h) : uint32_t(w);
    if (sw != uint32_t(out_w) || sh != uint32_t(out_h)) m = gs::resize_box(m.data(), sw, sh, 1, uint32_t(out_w), uint32_t(out_h));
    if (janchor != nullptr) {
      // Intrinsics already at the mask's resolution.
      float c2w[16], k[4], anchor[3];
      rows_of(env, jpose, c2w);
      env->GetFloatArrayRegion(jintrinsics, 0, 4, k);
      env->GetFloatArrayRegion(janchor, 0, 3, anchor);
      const float up[3] = {0, 1, 0};
      gs::select_subject(m.data(), uint32_t(out_w), uint32_t(out_h),
                         gs::subject_seeds(c2w, k[0], k[1], k[2], k[3], uint32_t(out_w), uint32_t(out_h), anchor, up));
    }
    jbyteArray out = env->NewByteArray(jsize(m.size()));
    env->SetByteArrayRegion(out, 0, jsize(m.size()), reinterpret_cast<const jbyte*>(m.data()));
    return out;
  } catch (const std::exception&) {
    return nullptr;
  }
}
