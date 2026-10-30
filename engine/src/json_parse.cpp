#include "gs/json_parse.h"

#include <cctype>
#include <cstdlib>
#include <stdexcept>

namespace gs {
namespace {

class Parser {
 public:
  explicit Parser(std::string_view text) : s_(text) {}

  JsonValue parse_document() {
    JsonValue v = parse_value();
    skip_ws();
    if (pos_ != s_.size()) fail("trailing characters");
    return v;
  }

 private:
  [[noreturn]] void fail(const char* what) const {
    throw std::runtime_error(std::string("JSON 파싱 오류 (") + what + ", 위치 " + std::to_string(pos_) + ")");
  }

  void skip_ws() {
    while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\n' || s_[pos_] == '\r' || s_[pos_] == '\t')) ++pos_;
  }

  bool consume(char c) {
    skip_ws();
    if (pos_ < s_.size() && s_[pos_] == c) {
      ++pos_;
      return true;
    }
    return false;
  }

  void expect(char c) {
    if (!consume(c)) fail("unexpected character");
  }

  bool match_word(std::string_view word) {
    if (s_.substr(pos_, word.size()) == word) {
      pos_ += word.size();
      return true;
    }
    return false;
  }

  JsonValue parse_value() {
    if (++depth_ > 64) fail("nesting too deep");
    skip_ws();
    if (pos_ >= s_.size()) fail("unexpected end");
    JsonValue v;
    const char c = s_[pos_];
    if (c == '{') {
      ++pos_;
      v.type = JsonValue::Type::Object;
      if (!consume('}')) {
        do {
          skip_ws();
          std::string key = parse_string();
          expect(':');
          v.object.emplace_back(std::move(key), parse_value());
        } while (consume(','));
        expect('}');
      }
    } else if (c == '[') {
      ++pos_;
      v.type = JsonValue::Type::Array;
      if (!consume(']')) {
        do {
          v.array.push_back(parse_value());
        } while (consume(','));
        expect(']');
      }
    } else if (c == '"') {
      v.type = JsonValue::Type::String;
      v.string = parse_string();
    } else if (match_word("true")) {
      v.type = JsonValue::Type::Bool;
      v.boolean = true;
    } else if (match_word("false")) {
      v.type = JsonValue::Type::Bool;
    } else if (match_word("null")) {
      v.type = JsonValue::Type::Null;
    } else {
      v.type = JsonValue::Type::Number;
      v.number = parse_number();
    }
    --depth_;
    return v;
  }

  double parse_number() {
    const size_t start = pos_;
    while (pos_ < s_.size() && (std::isdigit(static_cast<unsigned char>(s_[pos_])) || s_[pos_] == '-' ||
                                s_[pos_] == '+' || s_[pos_] == '.' || s_[pos_] == 'e' || s_[pos_] == 'E')) {
      ++pos_;
    }
    if (pos_ == start) fail("invalid value");
    const std::string token(s_.substr(start, pos_ - start));
    char* end = nullptr;
    const double value = std::strtod(token.c_str(), &end);
    if (end != token.c_str() + token.size()) fail("invalid number");
    return value;
  }

  std::string parse_string() {
    if (pos_ >= s_.size() || s_[pos_] != '"') fail("expected string");
    ++pos_;
    std::string out;
    while (pos_ < s_.size() && s_[pos_] != '"') {
      char c = s_[pos_++];
      if (c != '\\') {
        out += c;
        continue;
      }
      if (pos_ >= s_.size()) fail("bad escape");
      c = s_[pos_++];
      switch (c) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          if (pos_ + 4 > s_.size()) fail("bad unicode escape");
          const unsigned code = std::stoul(std::string(s_.substr(pos_, 4)), nullptr, 16);
          pos_ += 4;
          // Basic Multilingual Plane only, encoded as UTF-8.
          if (code < 0x80) {
            out += char(code);
          } else if (code < 0x800) {
            out += char(0xc0 | (code >> 6));
            out += char(0x80 | (code & 0x3f));
          } else {
            out += char(0xe0 | (code >> 12));
            out += char(0x80 | ((code >> 6) & 0x3f));
            out += char(0x80 | (code & 0x3f));
          }
          break;
        }
        default: fail("bad escape");
      }
    }
    if (pos_ >= s_.size()) fail("unterminated string");
    ++pos_;
    return out;
  }

  std::string_view s_;
  size_t pos_ = 0;
  int depth_ = 0;
};

}  // namespace

const JsonValue* JsonValue::find(std::string_view key) const {
  if (type != Type::Object) return nullptr;
  for (const auto& [k, v] : object) {
    if (k == key) return &v;
  }
  return nullptr;
}

double JsonValue::number_or(std::string_view key, double fallback) const {
  const JsonValue* v = find(key);
  return v != nullptr && v->type == Type::Number ? v->number : fallback;
}

std::string JsonValue::string_or(std::string_view key, const std::string& fallback) const {
  const JsonValue* v = find(key);
  return v != nullptr && v->type == Type::String ? v->string : fallback;
}

JsonValue parse_json(std::string_view text) { return Parser(text).parse_document(); }

}  // namespace gs
