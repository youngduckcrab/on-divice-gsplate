#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "gs/vk_context.h"

namespace gs {

struct BenchOptions {
  double scale = 1.0;  // problem-size multiplier; tests use small values
  int repeats = 7;     // timed repetitions (median is reported)
};

std::vector<std::string> benchmark_names();

// Runs one benchmark and returns a JSON object. Failures are reported in the
// JSON ("ok": false) rather than thrown, except for VK_ERROR_DEVICE_LOST,
// which is rethrown so the caller can recreate the context.
std::string run_benchmark(VkContext& ctx, std::string_view name, const BenchOptions& options);

}  // namespace gs
