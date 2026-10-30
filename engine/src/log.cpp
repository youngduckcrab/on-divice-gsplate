#include "gs/log.h"

#include <cstdarg>
#include <cstdio>

#if defined(__ANDROID__)
#include <android/log.h>
#endif

namespace gs {
namespace {

void vlog(bool error, const char* fmt, va_list args) {
#if defined(__ANDROID__)
  __android_log_vprint(error ? ANDROID_LOG_ERROR : ANDROID_LOG_INFO, "gsengine", fmt, args);
#else
  std::fprintf(stderr, "[gsengine] %s", error ? "error: " : "");
  std::vfprintf(stderr, fmt, args);
  std::fputc('\n', stderr);
#endif
}

}  // namespace

void log_info(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vlog(false, fmt, args);
  va_end(args);
}

void log_error(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vlog(true, fmt, args);
  va_end(args);
}

}  // namespace gs
