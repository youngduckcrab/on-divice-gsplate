#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace gs {

// Minimal streaming JSON writer. Callers are responsible for balanced
// begin/end calls; there is no validation beyond comma placement.
class JsonWriter {
 public:
  JsonWriter& begin_object() {
    prefix();
    out_ += '{';
    first_.push_back(true);
    return *this;
  }
  JsonWriter& end_object() {
    out_ += '}';
    first_.pop_back();
    return *this;
  }
  JsonWriter& begin_array() {
    prefix();
    out_ += '[';
    first_.push_back(true);
    return *this;
  }
  JsonWriter& end_array() {
    out_ += ']';
    first_.pop_back();
    return *this;
  }
  JsonWriter& key(std::string_view k) {
    prefix();
    write_string(k);
    out_ += ':';
    after_key_ = true;
    return *this;
  }

  JsonWriter& value(std::string_view v) {
    prefix();
    write_string(v);
    return *this;
  }
  JsonWriter& value(const char* v) { return value(std::string_view(v)); }
  JsonWriter& value(const std::string& v) { return value(std::string_view(v)); }
  JsonWriter& value(bool v) {
    prefix();
    out_ += v ? "true" : "false";
    return *this;
  }
  template <typename T>
    requires(std::is_integral_v<T> && !std::is_same_v<T, bool>)
  JsonWriter& value(T v) {
    prefix();
    out_ += std::to_string(v);
    return *this;
  }
  JsonWriter& value(double v) {
    prefix();
    if (!std::isfinite(v)) {
      out_ += "null";
    } else {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%.6g", v);
      out_ += buf;
    }
    return *this;
  }
  // Inserts an already-serialized JSON value verbatim.
  JsonWriter& raw(std::string_view json) {
    prefix();
    out_ += json;
    return *this;
  }

  template <typename T>
  JsonWriter& field(std::string_view k, const T& v) {
    key(k);
    return value(v);
  }

  const std::string& str() const { return out_; }

 private:
  void prefix() {
    if (after_key_) {
      after_key_ = false;
      return;
    }
    if (!first_.empty()) {
      if (!first_.back()) out_ += ',';
      first_.back() = false;
    }
  }

  void write_string(std::string_view s) {
    out_ += '"';
    for (char c : s) {
      switch (c) {
        case '"': out_ += "\\\""; break;
        case '\\': out_ += "\\\\"; break;
        case '\n': out_ += "\\n"; break;
        case '\r': out_ += "\\r"; break;
        case '\t': out_ += "\\t"; break;
        default:
          if (static_cast<unsigned char>(c) < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
            out_ += buf;
          } else {
            out_ += c;
          }
      }
    }
    out_ += '"';
  }

  std::string out_;
  std::vector<bool> first_;
  bool after_key_ = false;
};

}  // namespace gs
