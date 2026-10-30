// Prints the device probe and benchmark results as JSON lines.
// Usage: gs_probe [scale]   (default scale 1.0 = same sizes as the Android app)
#include <cstdio>
#include <cstdlib>
#include <exception>

#include "gs/benchmarks.h"
#include "gs/device_probe.h"
#include "gs/vk_context.h"

int main(int argc, char** argv) {
  try {
    gs::VkContext ctx;
    std::printf("%s\n", gs::probe_json(ctx).c_str());
    gs::BenchOptions opt;
    if (argc > 1) opt.scale = std::atof(argv[1]);
    for (const std::string& name : gs::benchmark_names()) {
      std::printf("%s\n", gs::run_benchmark(ctx, name, opt).c_str());
      std::fflush(stdout);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
