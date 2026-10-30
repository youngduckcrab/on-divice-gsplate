// JNI entry points for com.ondevice.gsplat.NativeBridge (diagnostics),
// com.ondevice.gsplat.NativeViewer and com.ondevice.gsplat.NativeTrainer. Calls
// that can fail return JSON with {"ok":false,"error":...} instead of throwing
// into Java.
#include <android/native_window_jni.h>
#include <jni.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <fstream>
#include <optional>

#include "gs/benchmarks.h"
#include "gs/dataset.h"
#include "gs/device_probe.h"
#include "gs/json.h"
#include "gs/splat_renderer.h"
#include "gs/splat_scene.h"
#include "gs/train_selftest.h"
#include "gs/trainer.h"
#include "gs/vk_context.h"
#include "viewer.h"

namespace {

std::mutex g_mutex;
std::shared_ptr<gs::VkContext> g_context;
std::unique_ptr<Viewer> g_viewer;

std::string error_json(const std::string& message) {
  gs::JsonWriter w;
  w.begin_object().field("ok", false).field("error", message).end_object();
  return w.str();
}

// Shared context, created on first use. Caller holds g_mutex.
std::shared_ptr<gs::VkContext> context_locked() {
  if (!g_context) g_context = std::make_shared<gs::VkContext>();
  return g_context;
}

// The viewer lives for the rest of the process once created.
Viewer& viewer() {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_viewer) g_viewer = std::make_unique<Viewer>(context_locked());
  return *g_viewer;
}

// Runs `f` on the viewer; drops the call if the viewer cannot be created
// (no Vulkan device), since gesture callbacks have nothing to report to.
template <typename F>
void with_viewer(F&& f) {
  try {
    f(viewer());
  } catch (const std::exception&) {
  }
}

template <typename F>
std::string guarded(F&& f) {
  std::lock_guard<std::mutex> lock(g_mutex);
  try {
    return f();
  } catch (const gs::VulkanError& e) {
    // A lost device cannot be reused; the next call creates a fresh context.
    if (e.result == VK_ERROR_DEVICE_LOST) g_context.reset();
    return error_json(e.what());
  } catch (const std::exception& e) {
    return error_json(e.what());
  }
}

jstring to_jstring(JNIEnv* env, std::string s) {
  // NewStringUTF expects modified UTF-8. Our messages contain Korean text, so
  // build the string from UTF-8 bytes through java.lang.String instead.
  jbyteArray bytes = env->NewByteArray(static_cast<jsize>(s.size()));
  env->SetByteArrayRegion(bytes, 0, static_cast<jsize>(s.size()), reinterpret_cast<const jbyte*>(s.data()));
  jclass string_class = env->FindClass("java/lang/String");
  jmethodID ctor = env->GetMethodID(string_class, "<init>", "([BLjava/lang/String;)V");
  jstring charset = env->NewStringUTF("UTF-8");
  auto result = static_cast<jstring>(env->NewObject(string_class, ctor, bytes, charset));
  env->DeleteLocalRef(charset);
  env->DeleteLocalRef(bytes);
  env->DeleteLocalRef(string_class);
  return result;
}

std::string from_jstring(JNIEnv* env, jstring s) {
  const char* chars = env->GetStringUTFChars(s, nullptr);
  std::string out = chars != nullptr ? chars : "";
  if (chars != nullptr) env->ReleaseStringUTFChars(s, chars);
  return out;
}

std::vector<uint8_t> read_fd(int fd) {
  std::vector<uint8_t> data;
  struct stat st {};
  if (fstat(fd, &st) == 0 && st.st_size > 0) data.reserve(static_cast<size_t>(st.st_size));
  std::vector<uint8_t> chunk(1 << 20);
  ssize_t n = 0;
  while ((n = read(fd, chunk.data(), chunk.size())) > 0) data.insert(data.end(), chunk.data(), chunk.data() + n);
  close(fd);
  if (n < 0) throw std::runtime_error("파일을 읽을 수 없음");
  return data;
}

// Parses (via `make_scene`), uploads and hands a scene to the viewer.
template <typename F>
std::string load_scene(F&& make_scene) {
  using Clock = std::chrono::steady_clock;
  try {
    Viewer& v = viewer();
    std::shared_ptr<gs::VkContext> ctx;
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      ctx = context_locked();
    }
    const auto t0 = Clock::now();
    gs::SplatScene scene = make_scene();
    const auto t1 = Clock::now();
    const gs::SceneBounds bounds = gs::compute_bounds(scene);
    std::unique_ptr<gs::GpuScene> gpu = gs::upload_scene(*ctx, scene);
    const auto t2 = Clock::now();
    const bool flip = scene.convention == gs::AxisConvention::RDF;
    const int gpu_degree = gpu->sh_degree;
    v.set_scene(std::move(gpu), bounds, flip);

    gs::JsonWriter w;
    w.begin_object();
    w.field("ok", true);
    w.field("count", scene.count);
    w.field("sh_degree", scene.sh_degree);
    w.field("gpu_sh_degree", gpu_degree);
    w.field("convention", scene.convention == gs::AxisConvention::RDF ? "RDF" : "RUB");
    w.field("antialiased", scene.antialiased);
    w.field("flip_y", flip);
    w.field("load_ms", std::chrono::duration<double, std::milli>(t1 - t0).count());
    w.field("upload_ms", std::chrono::duration<double, std::milli>(t2 - t1).count());
    w.end_object();
    return w.str();
  } catch (const std::exception& e) {
    return error_json(e.what());
  }
}

}  // namespace

// --- NativeBridge (diagnostics) --------------------------------------------------

extern "C" JNIEXPORT jstring JNICALL
Java_com_ondevice_gsplat_NativeBridge_probeDevice(JNIEnv* env, jobject /*this*/) {
  return to_jstring(env, guarded([] { return gs::probe_json(*context_locked()); }));
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_ondevice_gsplat_NativeBridge_benchmarkNames(JNIEnv* env, jobject /*this*/) {
  gs::JsonWriter w;
  w.begin_array();
  for (const std::string& name : gs::benchmark_names()) w.value(name);
  w.end_array();
  return to_jstring(env, w.str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_ondevice_gsplat_NativeBridge_runBenchmark(JNIEnv* env, jobject /*this*/, jstring jname) {
  const char* chars = env->GetStringUTFChars(jname, nullptr);
  std::string name = chars != nullptr ? chars : "";
  if (chars != nullptr) env->ReleaseStringUTFChars(jname, chars);
  return to_jstring(env, guarded([&] {
    return gs::run_benchmark(*context_locked(), name, gs::BenchOptions{});
  }));
}

// --- NativeViewer ------------------------------------------------------------------

extern "C" JNIEXPORT void JNICALL
Java_com_ondevice_gsplat_NativeViewer_attach(JNIEnv* env, jobject /*this*/, jobject surface) {
  ANativeWindow* window = ANativeWindow_fromSurface(env, surface);
  if (window == nullptr) return;
  try {
    viewer().attach_window(window);
  } catch (const std::exception&) {
    ANativeWindow_release(window);
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_ondevice_gsplat_NativeViewer_detach(JNIEnv*, jobject) {
  with_viewer([](Viewer& v) { v.detach_window(); });
}

extern "C" JNIEXPORT void JNICALL
Java_com_ondevice_gsplat_NativeViewer_resized(JNIEnv*, jobject) {
  with_viewer([](Viewer& v) { v.window_resized(); });
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_ondevice_gsplat_NativeViewer_loadSynthetic(JNIEnv* env, jobject /*this*/, jint count) {
  return to_jstring(env, load_scene([&] { return gs::make_synthetic_scene(static_cast<uint32_t>(count)); }));
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_ondevice_gsplat_NativeViewer_loadFile(JNIEnv* env, jobject /*this*/, jint fd) {
  return to_jstring(env, load_scene([&] {
    std::vector<uint8_t> bytes = read_fd(fd);
    return gs::load_splat_file(bytes.data(), bytes.size());
  }));
}

extern "C" JNIEXPORT jint JNICALL
Java_com_ondevice_gsplat_NativeViewer_sceneCount(JNIEnv*, jobject) {
  try {
    return static_cast<jint>(viewer().scene_count());
  } catch (const std::exception&) {
    return 0;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_ondevice_gsplat_NativeViewer_orbit(JNIEnv*, jobject, jfloat dx, jfloat dy) {
  with_viewer([&](Viewer& v) { v.orbit(dx, dy); });
}

extern "C" JNIEXPORT void JNICALL
Java_com_ondevice_gsplat_NativeViewer_pan(JNIEnv*, jobject, jfloat dx, jfloat dy) {
  with_viewer([&](Viewer& v) { v.pan(dx, dy); });
}

extern "C" JNIEXPORT void JNICALL
Java_com_ondevice_gsplat_NativeViewer_zoom(JNIEnv*, jobject, jfloat factor) {
  with_viewer([&](Viewer& v) { v.zoom(factor); });
}

extern "C" JNIEXPORT void JNICALL
Java_com_ondevice_gsplat_NativeViewer_resetView(JNIEnv*, jobject) {
  with_viewer([](Viewer& v) { v.reset_view(); });
}

extern "C" JNIEXPORT void JNICALL
Java_com_ondevice_gsplat_NativeViewer_setFlipY(JNIEnv*, jobject, jboolean flip) {
  with_viewer([&](Viewer& v) { v.set_flip_y(flip == JNI_TRUE); });
}

extern "C" JNIEXPORT void JNICALL
Java_com_ondevice_gsplat_NativeViewer_setContinuous(JNIEnv*, jobject, jboolean continuous) {
  with_viewer([&](Viewer& v) { v.set_continuous(continuous == JNI_TRUE); });
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_ondevice_gsplat_NativeViewer_stats(JNIEnv* env, jobject /*this*/) {
  try {
    return to_jstring(env, viewer().stats_json());
  } catch (const std::exception& e) {
    return to_jstring(env, error_json(e.what()));
  }
}

// --- NativeTrainer -----------------------------------------------------------------
// Called from one background thread at a time; the mutex only guards release()
// racing a late call.

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct TrainSession {
  std::shared_ptr<gs::VkContext> ctx;
  std::unique_ptr<gs::Trainer> trainer;
  std::vector<size_t> preview_frames;  // held-out frames, else all
};

std::mutex g_train_mutex;
std::unique_ptr<TrainSession> g_train;
// Backward variant whose gradients matched the CPU reference on this GPU,
// chosen on the first prepare(). Guarded by g_train_mutex.
std::optional<gs::ReductionChoice> g_reduction;
double g_reduction_ms = 0.0;

template <typename F>
std::string with_trainer(F&& f) {
  std::lock_guard<std::mutex> lock(g_train_mutex);
  try {
    if (!g_train) return error_json("학습이 준비되지 않음");
    return f(*g_train);
  } catch (const std::exception& e) {
    return error_json(e.what());
  }
}

void write_stats(gs::JsonWriter& w, const gs::Trainer& t) {
  const gs::TrainStats& s = t.stats();
  w.field("step", s.step);
  w.field("iterations", t.options().iterations);
  w.field("gaussians", s.gaussians);
  w.field("slots", s.slots);
  w.field("pairs", s.pairs);
  w.field("l1", s.l1);
  w.field("psnr", s.psnr);
  w.field("step_ms", s.step_ms);
  w.field("sh_degree", s.sh_degree);
  w.field("densify_events", s.densify_events);
  w.field("spawned", s.spawned);
  w.field("pruned", s.pruned);
  w.field("finished", t.finished());
  w.key("timing").begin_object();
  w.field("project_wall", s.timing.project_wall_ms);
  w.field("project_gpu", s.timing.project_gpu_ms);
  w.field("main_wall", s.timing.main_wall_ms);
  w.field("main_gpu", s.timing.main_gpu_ms);
  w.end_object();
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_ondevice_gsplat_NativeTrainer_prepare(JNIEnv* env, jobject /*this*/, jstring jdir, jint width,
                                               jint iterations, jint max_gaussians, jint sh_degree, jboolean masked) {
  const std::string dir = from_jstring(env, jdir);
  std::lock_guard<std::mutex> lock(g_train_mutex);
  try {
    g_train.reset();
    auto session = std::make_unique<TrainSession>();
    {
      std::lock_guard<std::mutex> ctx_lock(g_mutex);
      session->ctx = context_locked();
    }
    const auto t0 = Clock::now();
    gs::Dataset ds = gs::load_dataset(dir, static_cast<uint32_t>(width));
    const double load_ms = ms_since(t0);

    if (!g_reduction) {
      const auto tc = Clock::now();
      g_reduction = gs::choose_reduction(*session->ctx);
      g_reduction_ms = ms_since(tc);
    }
    if (!g_reduction->found) {
      throw std::runtime_error("이 GPU에서 그래디언트가 맞는 역전파 방식을 찾지 못함 (" + g_reduction->log + ")");
    }

    gs::TrainOptions options = gs::scaled_options(static_cast<uint32_t>(iterations));
    options.reduction = g_reduction->reduction;
    options.max_gaussians = static_cast<uint32_t>(max_gaussians);
    options.sh_degree = sh_degree;
    options.masked = masked == JNI_TRUE;
    const auto t1 = Clock::now();
    gs::InitOptions init;
    init.max_gaussians = std::min<uint32_t>(options.max_gaussians / 2, 150000);
    init.subject_only = options.masked;
    gs::InitStats init_stats;
    gs::SplatScene scene = gs::initialize_gaussians(ds, init, &init_stats);
    const double init_ms = ms_since(t1);

    for (size_t i = 0; i < ds.frames.size(); ++i) {
      if (ds.frames[i].holdout) session->preview_frames.push_back(i);
    }
    if (session->preview_frames.empty()) session->preview_frames.push_back(0);

    const auto t2 = Clock::now();
    session->trainer = std::make_unique<gs::Trainer>(*session->ctx, ds, scene, options);
    const double setup_ms = ms_since(t2);
    const gs::Dataset& d = session->trainer->dataset();

    gs::JsonWriter w;
    w.begin_object();
    w.field("ok", true);
    w.field("name", d.name);
    w.field("frames", d.frames.size());
    w.field("skipped", d.skipped.size());
    w.field("skipped_first", d.skipped.empty() ? std::string() : d.skipped.front());
    w.field("holdout", d.holdout_count());
    w.field("width", d.width);
    w.field("height", d.height);
    w.field("masks", d.has_masks);
    w.field("depth", d.has_depth);
    w.field("points", d.points.size() / 3);
    w.field("masked", session->trainer->options().masked);
    w.field("init_source", init_stats.source);
    w.field("init_points", init_stats.raw_points);
    w.field("init_gaussians", scene.count);
    w.field("load_ms", load_ms);
    w.field("init_ms", init_ms);
    w.field("setup_ms", setup_ms);
    w.field("reduction", session->trainer->reduction());
    w.field("reduction_note", session->trainer->reduction_note());
    w.field("reduction_check", g_reduction->log);
    w.field("reduction_check_ms", g_reduction_ms);
    write_stats(w, *session->trainer);
    w.end_object();
    g_train = std::move(session);
    return to_jstring(env, w.str());
  } catch (const std::exception& e) {
    g_train.reset();
    return to_jstring(env, error_json(e.what()));
  }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_ondevice_gsplat_NativeTrainer_step(JNIEnv* env, jobject /*this*/, jint count, jboolean profile) {
  return to_jstring(env, with_trainer([&](TrainSession& s) {
    const auto t0 = Clock::now();
    int done = 0;
    for (; done < count && !s.trainer->finished(); ++done) {
      // Profile the first step of the batch only.
      s.trainer->step(profile == JNI_TRUE && done == 0);
    }
    gs::JsonWriter w;
    w.begin_object();
    w.field("ok", true);
    w.field("steps", done);
    w.field("wall_ms", ms_since(t0));
    write_stats(w, *s.trainer);
    const gs::PassTimes& p = s.trainer->last_profile();
    if (profile == JNI_TRUE && p.valid) {
      w.key("profile").begin_object();
      w.field("project", p.project);
      w.field("sort", p.sort);
      w.field("rasterize", p.rasterize);
      w.field("backward", p.backward);
      w.field("optimize", p.optimize);
      w.end_object();
    }
    w.end_object();
    return w.str();
  }));
}

// Renders held-out view `index` (modulo their count) and returns ARGB pixels:
// the render followed by the target, both over the same dark background.
extern "C" JNIEXPORT jintArray JNICALL
Java_com_ondevice_gsplat_NativeTrainer_preview(JNIEnv* env, jobject /*this*/, jint index) {
  std::vector<uint8_t> render, target;
  uint32_t width = 0, height = 0;
  {
    std::lock_guard<std::mutex> lock(g_train_mutex);
    if (!g_train) return nullptr;
    try {
      gs::Trainer& t = *g_train->trainer;
      const gs::Dataset& ds = t.dataset();
      const size_t frame = g_train->preview_frames[size_t(index) % g_train->preview_frames.size()];
      const float bg[3] = {0.08f, 0.09f, 0.11f};
      render = t.render(gs::dataset_camera(ds, ds.frames[frame]), bg);
      target = ds.frames[frame].rgba;
      const bool masked = t.options().masked;
      for (size_t p = 0; p < target.size() / 4; ++p) {
        const float a = masked ? target[p * 4 + 3] / 255.0f : 1.0f;
        for (int c = 0; c < 3; ++c) {
          target[p * 4 + c] = static_cast<uint8_t>(target[p * 4 + c] * a + bg[c] * 255.0f * (1.0f - a) + 0.5f);
        }
      }
      width = ds.width;
      height = ds.height;
    } catch (const std::exception&) {
      return nullptr;
    }
  }
  const size_t pixels = size_t(width) * height;
  std::vector<jint> argb(pixels * 2);
  auto pack = [](const uint8_t* p) {
    return static_cast<jint>(0xff000000u | (uint32_t(p[0]) << 16) | (uint32_t(p[1]) << 8) | uint32_t(p[2]));
  };
  for (size_t p = 0; p < pixels; ++p) {
    argb[p] = pack(&render[p * 4]);
    argb[pixels + p] = pack(&target[p * 4]);
  }
  jintArray out = env->NewIntArray(static_cast<jsize>(argb.size()));
  if (out != nullptr) env->SetIntArrayRegion(out, 0, static_cast<jsize>(argb.size()), argb.data());
  return out;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_ondevice_gsplat_NativeTrainer_evaluate(JNIEnv* env, jobject /*this*/) {
  return to_jstring(env, with_trainer([](TrainSession& s) {
    const auto t0 = Clock::now();
    const gs::EvalResult e = s.trainer->evaluate();
    gs::JsonWriter w;
    w.begin_object();
    w.field("ok", true);
    w.field("psnr", e.psnr);
    w.field("l1", e.l1);
    w.field("views", e.views);
    w.field("ms", ms_since(t0));
    w.end_object();
    return w.str();
  }));
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_ondevice_gsplat_NativeTrainer_exportPly(JNIEnv* env, jobject /*this*/, jstring jpath) {
  const std::string path = from_jstring(env, jpath);
  return to_jstring(env, with_trainer([&](TrainSession& s) {
    const auto t0 = Clock::now();
    gs::SplatScene scene = s.trainer->export_scene();
    gs::convert_convention(scene, gs::AxisConvention::RDF);
    const std::vector<uint8_t> ply = gs::save_ply(scene);
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(ply.data()), static_cast<std::streamsize>(ply.size()));
    file.close();
    if (!file) throw std::runtime_error("PLY 파일을 쓸 수 없음");
    gs::JsonWriter w;
    w.begin_object();
    w.field("ok", true);
    w.field("count", scene.count);
    w.field("bytes", ply.size());
    w.field("ms", ms_since(t0));
    w.end_object();
    return w.str();
  }));
}

// Gradient checks against the CPU reference and a small fit, on this GPU.
extern "C" JNIEXPORT jstring JNICALL
Java_com_ondevice_gsplat_NativeTrainer_selfTest(JNIEnv* env, jobject /*this*/) {
  std::lock_guard<std::mutex> lock(g_train_mutex);
  try {
    std::shared_ptr<gs::VkContext> ctx;
    {
      std::lock_guard<std::mutex> ctx_lock(g_mutex);
      ctx = context_locked();
    }
    const auto t0 = Clock::now();
    std::string json = gs::trainer_self_test_json(*ctx);
    // Append the duration before the closing brace.
    json.pop_back();
    json += ",\"ms\":" + std::to_string(ms_since(t0)) + "}";
    return to_jstring(env, json);
  } catch (const std::exception& e) {
    return to_jstring(env, error_json(e.what()));
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_ondevice_gsplat_NativeTrainer_release(JNIEnv*, jobject) {
  std::lock_guard<std::mutex> lock(g_train_mutex);
  g_train.reset();
}
