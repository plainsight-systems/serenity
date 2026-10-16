#include "core/denoise/json.h"

#include <charconv>
#include <cstddef>
#include <string>
#include <system_error>

namespace serenity::denoise {

namespace {

// A recursive-descent reader over `text`, `at` the next byte (F.3: one
// function a production of the grammar).
class Reader {
public:
    explicit Reader(std::string_view text) : text_(text) {}

    Json document() {
        Json value = this->value(0);
        skip();
        if (at_ != text_.size()) {
            fail("more after the value");
        }
        return value;
    }

private:
    // Deeper than any layout, shallow enough that no file can exhaust the stack.
    static constexpr int max_depth = 64;

    [[noreturn]] void fail(const std::string& what) const {
        throw JsonError("JSON at byte " + std::to_string(at_) + ": " + what);
    }

    void skip() {
        while (at_ < text_.size() &&
               (text_[at_] == ' ' || text_[at_] == '\t' || text_[at_] == '\n' || text_[at_] == '\r')) {
            ++at_;
        }
    }

    char peek() {
        skip();
        if (at_ == text_.size()) {
            fail("ends early");
        }
        return text_[at_];
    }

    void expect(char c) {
        if (peek() != c) {
            fail(std::string("expected '") + c + "'");
        }
        ++at_;
    }

    void literal(std::string_view word) {
        if (text_.substr(at_, word.size()) != word) {
            fail("expected " + std::string(word));
        }
        at_ += word.size();
    }

    Json value(int depth) {
        if (depth > max_depth) {
            fail("nested deeper than " + std::to_string(max_depth));
        }
        switch (peek()) {
        case '{':
            return object(depth);
        case '[':
            return array(depth);
        case '"':
            return Json{string()};
        case 't':
            literal("true");
            return Json{true};
        case 'f':
            literal("false");
            return Json{false};
        case 'n':
            literal("null");
            return Json{nullptr};
        default:
            return Json{number()};
        }
    }

    Json object(int depth) {
        expect('{');
        JsonObject members;
        if (peek() == '}') {
            ++at_;
            return Json{std::move(members)};
        }
        for (;;) {
            if (peek() != '"') {
                fail("expected a member's name");
            }
            const std::size_t name_at = at_;
            std::string key = string();
            // A name twice is refused, at the second: at() would read the
            // first and the file would be half-read (json.h).
            for (const auto& [name, member] : members) {
                if (name == key) {
                    at_ = name_at;
                    fail("the member name \"" + key + "\" a second time in one object");
                }
            }
            expect(':');
            members.emplace_back(std::move(key), value(depth + 1));
            if (peek() == '}') {
                ++at_;
                return Json{std::move(members)};
            }
            expect(',');
        }
    }

    Json array(int depth) {
        expect('[');
        JsonArray items;
        if (peek() == ']') {
            ++at_;
            return Json{std::move(items)};
        }
        for (;;) {
            items.push_back(value(depth + 1));
            if (peek() == ']') {
                ++at_;
                return Json{std::move(items)};
            }
            expect(',');
        }
    }

    std::string string() {
        expect('"');
        std::string out;
        for (;;) {
            if (at_ == text_.size()) {
                fail("a string ends early");
            }
            const char c = text_[at_++];
            if (c == '"') {
                return out;
            }
            if (static_cast<unsigned char>(c) < 0x20) {
                fail("a control character in a string");
            }
            if (c != '\\') {
                out += c;
                continue;
            }
            if (at_ == text_.size()) {
                fail("a string ends early");
            }
            const char e = text_[at_++];
            switch (e) {
            case '"':
            case '\\':
            case '/':
                out += e;
                break;
            case 'b':
                out += '\b';
                break;
            case 'f':
                out += '\f';
                break;
            case 'n':
                out += '\n';
                break;
            case 'r':
                out += '\r';
                break;
            case 't':
                out += '\t';
                break;
            case 'u': {
                unsigned code = 0;
                const std::string_view hex = text_.substr(at_, 4);
                const auto [end, error] = std::from_chars(hex.data(), hex.data() + hex.size(), code, 16);
                if (hex.size() != 4 || error != std::errc{} || end != hex.data() + 4) {
                    fail("a \\u escape without four hex digits");
                }
                if (code > 0x7f) {
                    fail("a \\u escape past ASCII, which no layout writes");
                }
                at_ += 4;
                out += static_cast<char>(code);
                break;
            }
            default:
                fail("an unknown escape");
            }
        }
    }

    double number() {
        // RFC 8259's number: -? int frac? exp?, checked before conversion.
        const std::size_t start = at_;
        const auto digits = [&] {
            const std::size_t first = at_;
            while (at_ < text_.size() && text_[at_] >= '0' && text_[at_] <= '9') {
                ++at_;
            }
            return at_ - first;
        };
        if (at_ < text_.size() && text_[at_] == '-') {
            ++at_;
        }
        const std::size_t int_start = at_;
        const std::size_t whole = digits();
        if (whole == 0 || (whole > 1 && text_[int_start] == '0')) {
            at_ = start;
            fail("expected a value");
        }
        if (at_ < text_.size() && text_[at_] == '.') {
            ++at_;
            if (digits() == 0) {
                fail("a number's fraction has no digits");
            }
        }
        if (at_ < text_.size() && (text_[at_] == 'e' || text_[at_] == 'E')) {
            ++at_;
            if (at_ < text_.size() && (text_[at_] == '+' || text_[at_] == '-')) {
                ++at_;
            }
            if (digits() == 0) {
                fail("a number's exponent has no digits");
            }
        }
        double value = 0.0;
        const auto [end, error] = std::from_chars(text_.data() + start, text_.data() + at_, value);
        if (error != std::errc{} || end != text_.data() + at_) {
            fail("a number out of range");
        }
        return value;
    }

    std::string_view text_;
    std::size_t at_ = 0;
};

[[noreturn]] void wrong(std::string_view what, std::string_view kind) {
    throw JsonError(std::string(what) + " is not " + std::string(kind));
}

}  // namespace

const Json& Json::at(std::string_view key) const {
    const JsonObject* object = std::get_if<JsonObject>(&value);
    if (object == nullptr) {
        throw JsonError("looked for '" + std::string(key) + "' in something that is not an object");
    }
    for (const auto& [name, member] : *object) {
        if (name == key) {
            return member;
        }
    }
    throw JsonError("no '" + std::string(key) + "'");
}

double Json::number(std::string_view what) const {
    const double* n = std::get_if<double>(&value);
    if (n == nullptr) {
        wrong(what, "a number");
    }
    return *n;
}

const std::string& Json::string(std::string_view what) const {
    const std::string* s = std::get_if<std::string>(&value);
    if (s == nullptr) {
        wrong(what, "a string");
    }
    return *s;
}

const JsonArray& Json::array(std::string_view what) const {
    const JsonArray* a = std::get_if<JsonArray>(&value);
    if (a == nullptr) {
        wrong(what, "an array");
    }
    return *a;
}

Json parse_json(std::string_view text) {
    return Reader(text).document();
}

}  // namespace serenity::denoise
