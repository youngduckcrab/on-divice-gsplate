// JNI entry points for com.ondevice.gsplat.NativeBridge (diagnostics) and
// com.ondevice.gsplat.NativeViewer. Calls that can fail return JSON with
// {"ok":false,"error":...} instead of throwing into Java.
#include <android/native_window_jni.h>
#include <jni.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "gs/benchmarks.h"
#include "gs/device_probe.h"
#include "gs/json.h"
#include "gs/splat_renderer.h"
#include "gs/splat_scene.h"
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
