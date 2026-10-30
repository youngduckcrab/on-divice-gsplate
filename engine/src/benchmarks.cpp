#include "gs/benchmarks.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <random>
#include <utility>

#include "gs/json.h"
#include "gs/log.h"
#include "gs/radix_sort.h"
#include "shaders/atomic_f32_cas.h"
#include "shaders/atomic_f32_native.h"
#include "shaders/atomic_u32.h"
#include "shaders/bandwidth_copy.h"
#include "shaders/fma_fp16.h"
#include "shaders/fma_fp32.h"

namespace gs {
namespace {

struct Timing {
  double median_ms = 0.0;
  double min_ms = 0.0;
  bool gpu_timestamps = false;
};

// Runs `prepare` (untimed) then `record` (timed) for warmup + repeats rounds.
Timing time_runs(VkContext& ctx, int repeats, const std::function<void(VkCommandBuffer)>& record,
                 const std::function<void(VkCommandBuffer)>& prepare = nullptr) {
  std::vector<double> samples;
  bool gpu = false;
  for (int i = 0; i < repeats + 1; ++i) {
    if (prepare) ctx.run(prepare);
    RunTiming t = ctx.run(record);
    if (i == 0) continue;  // warmup (pipeline/cache first use)
    samples.push_back(t.best_ms());
    gpu = t.gpu_ms >= 0.0;
  }
  std::sort(samples.begin(), samples.end());
  return {samples[samples.size() / 2], samples.front(), gpu};
}

void write_timing(JsonWriter& w, const Timing& t) {
  w.field("median_ms", t.median_ms);
  w.field("min_ms", t.min_ms);
  w.field("timer", t.gpu_timestamps ? "gpu" : "cpu");
}

std::string skipped(std::string_view name, std::string_view reason) {
  JsonWriter w;
  w.begin_object().field("name", name).field("ok", true).field("skipped", reason).end_object();
  return w.str();
}

std::string bench_bandwidth(VkContext& ctx, const BenchOptions& opt) {
  const auto& limits = ctx.info().properties.limits;
  uint64_t bytes = static_cast<uint64_t>(64.0 * 1024 * 1024 * opt.scale);
  bytes = std::min<uint64_t>(bytes, limits.maxStorageBufferRange);
  bytes = std::min<uint64_t>(bytes, uint64_t(limits.maxComputeWorkGroupCount[0]) * 256 * 16);
  bytes = std::max<uint64_t>(bytes, 1 << 20) & ~uint64_t(4095);
  const uint32_t count = static_cast<uint32_t>(bytes / 16);

  ComputePipeline pipe = ctx.create_compute_pipeline(shaders::bandwidth_copy, 2, sizeof(uint32_t));
  Buffer src = ctx.create_buffer(bytes, MemoryKind::Device);
  Buffer dst = ctx.create_buffer(bytes, MemoryKind::Device);
  ctx.run([&](VkCommandBuffer cmd) {
    vkCmdFillBuffer(cmd, src.handle(), 0, VK_WHOLE_SIZE, 0x3f800000u);
  });

  Timing t = time_runs(ctx, opt.repeats, [&](VkCommandBuffer cmd) {
    ctx.dispatch(cmd, pipe, {&src, &dst}, count, div_up(count, 256));
  });

  JsonWriter w;
  w.begin_object().field("name", "bandwidth").field("ok", true);
  w.field("bytes", bytes);
  write_timing(w, t);
  w.field("gb_per_s", 2.0 * double(bytes) / (t.median_ms * 1e-3) / 1e9);
  w.end_object();
  return w.str();
}

struct FmaParams {
  uint32_t iterations;
  float seed;
};

std::string bench_fma(VkContext& ctx, const BenchOptions& opt, bool fp16) {
  const char* name = fp16 ? "fp16" : "fp32";
  if (fp16 && !ctx.enabled().float16) return skipped(name, "shaderFloat16 not supported");

  const uint32_t threads = std::max<uint32_t>(256 * 64, uint32_t(1048576 * opt.scale) & ~255u);
  const uint32_t groups = std::min(threads / 256, ctx.info().properties.limits.maxComputeWorkGroupCount[0]);
  const FmaParams params{256, 0.5f};
  const double flop = double(groups) * 256 * params.iterations * 128;

  ComputePipeline pipe = ctx.create_compute_pipeline(
      fp16 ? std::span<const uint32_t>(shaders::fma_fp16) : std::span<const uint32_t>(shaders::fma_fp32),
      1, sizeof(FmaParams));
  Buffer out = ctx.create_buffer(uint64_t(groups) * 256 * 4, MemoryKind::Device);

  Timing t = time_runs(ctx, opt.repeats, [&](VkCommandBuffer cmd) {
    ctx.dispatch(cmd, pipe, {&out}, params, groups);
  });

  JsonWriter w;
  w.begin_object().field("name", name).field("ok", true);
  w.field("gflop", flop / 1e9);
  write_timing(w, t);
  w.field("gflops", flop / (t.median_ms * 1e-3) / 1e9);
  w.end_object();
  return w.str();
}

struct AtomicParams {
  uint32_t threads;
  uint32_t slot_mask;
  uint32_t per_thread;
};

std::string bench_atomics(VkContext& ctx, const BenchOptions& opt) {
  const uint32_t per_thread = 4;
  const uint32_t threads = std::max<uint32_t>(256 * 16, uint32_t(1048576 * opt.scale) & ~255u);
  const uint64_t adds = uint64_t(threads) * per_thread;

  struct Variant {
    const char* name;
    std::span<const uint32_t> spirv;
    bool is_float;
    bool supported;
  };
  const Variant variants[] = {
      {"u32_atomic", shaders::atomic_u32, false, true},
      {"f32_native_atomic", shaders::atomic_f32_native, true, ctx.enabled().atomic_float_add},
      {"f32_cas_loop", shaders::atomic_f32_cas, true, true},
  };
  // High contention (few hot slots, like gradients of large splats) vs. scattered.
  const uint32_t slot_counts[] = {256, 1u << 20};

  JsonWriter w;
  w.begin_object().field("name", "atomics").field("ok", true);
  w.field("adds", adds);
  w.key("results").begin_array();
  for (const Variant& v : variants) {
    if (!v.supported) {
      w.begin_object().field("variant", v.name).field("skipped", "not supported").end_object();
      continue;
    }
    ComputePipeline pipe = ctx.create_compute_pipeline(v.spirv, 1, sizeof(AtomicParams));
    for (uint32_t slots : slot_counts) {
      Buffer buf = ctx.create_buffer(uint64_t(slots) * 4, MemoryKind::Device);
      AtomicParams params{threads, slots - 1, per_thread};
      Timing t = time_runs(
          ctx, opt.repeats,
          [&](VkCommandBuffer cmd) { ctx.dispatch(cmd, pipe, {&buf}, params, div_up(threads, 256)); },
          [&](VkCommandBuffer cmd) { vkCmdFillBuffer(cmd, buf.handle(), 0, VK_WHOLE_SIZE, 0); });

      std::vector<uint32_t> host(slots);
      ctx.download(buf, host.data(), uint64_t(slots) * 4);
      double sum = 0.0;
      for (uint32_t bits : host) {
        if (v.is_float) {
          float f;
          std::memcpy(&f, &bits, 4);
          sum += f;
        } else {
          sum += bits;
        }
      }

      w.begin_object();
      w.field("variant", v.name);
      w.field("slots", slots);
      write_timing(w, t);
      w.field("gadds_per_s", double(adds) / (t.median_ms * 1e-3) / 1e9);
      w.field("correct", sum == double(adds));
      w.end_object();
    }
  }
  w.end_array();
  w.end_object();
  return w.str();
}

std::string bench_radix_sort(VkContext& ctx, const BenchOptions& opt) {
  const uint32_t n = std::max<uint32_t>(4096, uint32_t(1048576 * opt.scale));
  RadixSort sorter(ctx);
  sorter.reserve(n);
  Buffer src_keys = ctx.create_buffer(uint64_t(n) * 4, MemoryKind::Device);
  Buffer src_values = ctx.create_buffer(uint64_t(n) * 4, MemoryKind::Device);
  Buffer work_keys = ctx.create_buffer(uint64_t(n) * 4, MemoryKind::Device);
  Buffer work_values = ctx.create_buffer(uint64_t(n) * 4, MemoryKind::Device);

  JsonWriter w;
  w.begin_object().field("name", "radix_sort").field("ok", true).field("n", n);
  w.key("results").begin_array();
  // 32-bit keys (generic), 16-bit keys (the viewer's depth keys: 4 passes instead of 8).
  for (uint32_t key_bits : {32u, 16u}) {
    const uint32_t mask = key_bits >= 32 ? ~0u : ((1u << key_bits) - 1);
    std::vector<uint32_t> keys(n);
    std::vector<uint32_t> values(n);
    std::mt19937 rng(1234 + key_bits);
    for (uint32_t i = 0; i < n; ++i) {
      keys[i] = rng() & mask;
      values[i] = i;
    }
    ctx.upload(src_keys, keys.data(), uint64_t(n) * 4);
    ctx.upload(src_values, values.data(), uint64_t(n) * 4);

    Timing t = time_runs(
        ctx, opt.repeats,
        [&](VkCommandBuffer cmd) { sorter.record(cmd, work_keys, work_values, n, key_bits); },
        [&](VkCommandBuffer cmd) {
          VkBufferCopy region{0, 0, uint64_t(n) * 4};
          vkCmdCopyBuffer(cmd, src_keys.handle(), work_keys.handle(), 1, &region);
          vkCmdCopyBuffer(cmd, src_values.handle(), work_values.handle(), 1, &region);
        });

    std::vector<uint32_t> out_keys(n);
    std::vector<uint32_t> out_values(n);
    ctx.download(work_keys, out_keys.data(), uint64_t(n) * 4);
    ctx.download(work_values, out_values.data(), uint64_t(n) * 4);

    std::vector<std::pair<uint32_t, uint32_t>> expected(n);
    for (uint32_t i = 0; i < n; ++i) expected[i] = {keys[i], values[i]};
    std::stable_sort(expected.begin(), expected.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });
    int64_t first_mismatch = -1;
    for (uint32_t i = 0; i < n; ++i) {
      if (out_keys[i] != expected[i].first || out_values[i] != expected[i].second) {
        first_mismatch = i;
        break;
      }
    }

    w.begin_object();
    w.field("key_bits", key_bits);
    write_timing(w, t);
    w.field("mkeys_per_s", double(n) / (t.median_ms * 1e-3) / 1e6);
    w.field("correct", first_mismatch < 0);
    if (first_mismatch >= 0) w.field("first_mismatch", first_mismatch);
    w.end_object();
  }
  w.end_array();
  w.end_object();
  return w.str();
}

}  // namespace

std::vector<std::string> benchmark_names() {
  return {"bandwidth", "fp32", "fp16", "atomics", "radix_sort"};
}

std::string run_benchmark(VkContext& ctx, std::string_view name, const BenchOptions& options) {
  try {
    if (name == "bandwidth") return bench_bandwidth(ctx, options);
    if (name == "fp32") return bench_fma(ctx, options, false);
    if (name == "fp16") return bench_fma(ctx, options, true);
    if (name == "atomics") return bench_atomics(ctx, options);
    if (name == "radix_sort") return bench_radix_sort(ctx, options);
    throw std::invalid_argument("unknown benchmark");
  } catch (const VulkanError& e) {
    log_error("benchmark %.*s failed: %s", int(name.size()), name.data(), e.what());
    if (e.result == VK_ERROR_DEVICE_LOST) throw;
    JsonWriter w;
    w.begin_object().field("name", name).field("ok", false).field("error", e.what()).end_object();
    return w.str();
  } catch (const std::exception& e) {
    log_error("benchmark %.*s failed: %s", int(name.size()), name.data(), e.what());
    JsonWriter w;
    w.begin_object().field("name", name).field("ok", false).field("error", e.what()).end_object();
    return w.str();
  }
}

}  // namespace gs
