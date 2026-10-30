#pragma once

namespace gs {

// Logs to logcat on Android (tag "gsengine") and to stderr elsewhere.
void log_info(const char* fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;

void log_error(const char* fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;

}  // namespace gs
