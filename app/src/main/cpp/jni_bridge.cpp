// JNI entry points for com.ondevice.gsplat.NativeBridge. Every call returns a
// JSON string; failures come back as {"ok":false,"error":...} instead of throwing.
#include <jni.h>

#include <memory>
#include <mutex>
#include <string>

#include "gs/benchmarks.h"
#include "gs/device_probe.h"
#include "gs/json.h"
#include "gs/vk_context.h"

namespace {

std::mutex g_mutex;
std::unique_ptr<gs::VkContext> g_context;

std::string error_json(const std::string& message) {
  gs::JsonWriter w;
  w.begin_object().field("ok", false).field("error", message).end_object();
  return w.str();
}

// Shared context, created on first use. Caller holds g_mutex.
gs::VkContext& context() {
  if (!g_context) g_context = std::make_unique<gs::VkContext>();
  return *g_context;
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
  // NewStringUTF expects modified UTF-8. Driver strings are ASCII in practice,
  // so replace anything else rather than risk a JNI abort.
  for (char& c : s) {
    if (static_cast<unsigned char>(c) >= 0x80) c = '?';
  }
  return env->NewStringUTF(s.c_str());
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_ondevice_gsplat_NativeBridge_probeDevice(JNIEnv* env, jobject /*this*/) {
  return to_jstring(env, guarded([] { return gs::probe_json(context()); }));
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
    return gs::run_benchmark(context(), name, gs::BenchOptions{});
  }));
}
