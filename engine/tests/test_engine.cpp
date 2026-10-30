// Engine tests. Run on any Vulkan device; CI uses Mesa lavapipe (software).
#include <algorithm>
#include <cstdio>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "gs/benchmarks.h"
#include "gs/device_probe.h"
#include "gs/radix_sort.h"
#include "gs/vk_context.h"

namespace {

int g_failures = 0;

#define EXPECT(cond, ...)                                     \
  do {                                                        \
    if (!(cond)) {                                            \
      ++g_failures;                                           \
      std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
      std::fprintf(stderr, __VA_ARGS__);                      \
      std::fputc('\n', stderr);                               \
    }                                                         \
  } while (0)

bool contains(const std::string& s, const char* needle) { return s.find(needle) != std::string::npos; }

void test_probe(const gs::VkContext& ctx) {
  std::string json = gs::probe_json(ctx);
  EXPECT(json.front() == '{' && json.back() == '}', "probe is not a JSON object");
  EXPECT(contains(json, "\"subgroup\""), "probe lacks subgroup info");
  EXPECT(contains(json, "\"atomic_float\""), "probe lacks atomic_float info");
  EXPECT(contains(json, "\"extensions\""), "probe lacks extensions");
}

void test_radix_sort(gs::VkContext& ctx) {
  gs::RadixSort sorter(ctx);
  std::mt19937 rng(42);
  struct Case {
    uint32_t n;
    uint32_t key_mask;  // limits key range; small masks create many duplicates
    uint32_t key_bits;
  };
  const Case cases[] = {
      {1, ~0u, 32},       {2, ~0u, 32},         {7, 0xffu, 32},     {1000, ~0u, 32},
      {1024, 0xfu, 32},   {1025, 0xffu, 8},     {4097, 0xfffu, 12}, {100000, ~0u, 32},
      {65536, 0xffffu, 16}, {300001, 0x3ffu, 32}, {131072, 0xffffffu, 24},
  };
  for (const Case& c : cases) {
    std::vector<uint32_t> keys(c.n), values(c.n);
    for (uint32_t i = 0; i < c.n; ++i) {
      keys[i] = rng() & c.key_mask;
      values[i] = i;
    }
    gs::Buffer kbuf = ctx.create_buffer(uint64_t(c.n) * 4, gs::MemoryKind::Device);
    gs::Buffer vbuf = ctx.create_buffer(uint64_t(c.n) * 4, gs::MemoryKind::Device);
    ctx.upload(kbuf, keys.data(), uint64_t(c.n) * 4);
    ctx.upload(vbuf, values.data(), uint64_t(c.n) * 4);
    sorter.reserve(c.n);
    ctx.run([&](VkCommandBuffer cmd) { sorter.record(cmd, kbuf, vbuf, c.n, c.key_bits); });

    std::vector<uint32_t> out_k(c.n), out_v(c.n);
    ctx.download(kbuf, out_k.data(), uint64_t(c.n) * 4);
    ctx.download(vbuf, out_v.data(), uint64_t(c.n) * 4);

    std::vector<std::pair<uint32_t, uint32_t>> expected(c.n);
    for (uint32_t i = 0; i < c.n; ++i) expected[i] = {keys[i], values[i]};
    std::stable_sort(expected.begin(), expected.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });
    uint32_t mismatches = 0;
    for (uint32_t i = 0; i < c.n; ++i) {
      if (out_k[i] != expected[i].first || out_v[i] != expected[i].second) ++mismatches;
    }
    EXPECT(mismatches == 0, "radix sort n=%u mask=0x%x bits=%u: %u mismatches", c.n, c.key_mask,
           c.key_bits, mismatches);
  }
}

void test_benchmarks(gs::VkContext& ctx) {
  gs::BenchOptions opt;
  opt.scale = 1.0 / 32;
  opt.repeats = 2;
  for (const std::string& name : gs::benchmark_names()) {
    std::string json = gs::run_benchmark(ctx, name, opt);
    std::printf("%s\n", json.c_str());
    EXPECT(contains(json, "\"ok\":true"), "benchmark %s failed: %s", name.c_str(), json.c_str());
    EXPECT(!contains(json, "\"correct\":false"), "benchmark %s produced wrong results: %s",
           name.c_str(), json.c_str());
  }
}

}  // namespace

int main() {
  try {
    gs::VkContext ctx;
    std::printf("device: %s\n", ctx.info().properties.deviceName);
    test_probe(ctx);
    test_radix_sort(ctx);
    test_benchmarks(ctx);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: exception: %s\n", e.what());
    return 1;
  }
  if (g_failures > 0) {
    std::fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("all tests passed\n");
  return 0;
}
