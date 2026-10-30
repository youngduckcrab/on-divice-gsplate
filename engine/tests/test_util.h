#pragma once

#include <cstdio>
#include <string>

namespace test {

inline int failures = 0;

inline bool contains(const std::string& s, const char* needle) { return s.find(needle) != std::string::npos; }

}  // namespace test

#define EXPECT(cond, ...)                                       \
  do {                                                          \
    if (!(cond)) {                                              \
      ++test::failures;                                         \
      std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
      std::fprintf(stderr, __VA_ARGS__);                        \
      std::fputc('\n', stderr);                                 \
    }                                                           \
  } while (0)
