#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gs {

// Small DOM for reading JSON files such as transforms.json.
struct JsonValue {
  enum class Type { Null, Bool, Number, String, Array, Object };
  Type type = Type::Null;
  bool boolean = false;
  double number = 0.0;
  std::string string;
  std::vector<JsonValue> array;
  std::vector<std::pair<std::string, JsonValue>> object;

  // Member lookup; null when absent or when this is not an object.
  const JsonValue* find(std::string_view key) const;
  double number_or(std::string_view key, double fallback) const;
  std::string string_or(std::string_view key, const std::string& fallback) const;
};

// Throws std::runtime_error with the byte offset on malformed input.
JsonValue parse_json(std::string_view text);

}  // namespace gs
