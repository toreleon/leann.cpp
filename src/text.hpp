#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

// Text predicates shared by the library and the CLI.
//
// These live in one header rather than in app/main.cpp because both sides now
// depend on the same answer: Index::build and Index::load validate descriptor
// strings, and the CLI's JSON writer refuses to emit anything that fails
// is_valid_utf8. Two copies would let an index be written that `stats
// --format json` could never print.
//
// Header-only and inline, so no build-system wiring changes.

namespace leann::detail {

// Length of the UTF-8 sequence introduced by `lead`, or 0 if it is not a legal
// lead byte.
[[nodiscard]] inline std::size_t utf8_sequence_length(unsigned char lead) {
    if (lead < 0x80U) {
        return 1;
    }
    if ((lead & 0xE0U) == 0xC0U) {
        return 2;
    }
    if ((lead & 0xF0U) == 0xE0U) {
        return 3;
    }
    if ((lead & 0xF8U) == 0xF0U) {
        return 4;
    }
    return 0;
}

// Rejects overlong encodings, surrogate halves, and anything above U+10FFFF,
// so a "valid" string here is valid to every conforming JSON reader.
[[nodiscard]] inline bool is_valid_utf8(std::string_view text) {
    std::size_t index = 0;
    while (index < text.size()) {
        const auto lead = static_cast<unsigned char>(text[index]);
        const std::size_t length = utf8_sequence_length(lead);
        if (length == 0 || index + length > text.size()) {
            return false;
        }
        std::uint32_t code_point = 0;
        switch (length) {
        case 1:
            code_point = lead;
            break;
        case 2:
            code_point = lead & 0x1FU;
            break;
        case 3:
            code_point = lead & 0x0FU;
            break;
        default:
            code_point = lead & 0x07U;
            break;
        }
        for (std::size_t offset = 1; offset < length; ++offset) {
            const auto continuation =
                static_cast<unsigned char>(text[index + offset]);
            if ((continuation & 0xC0U) != 0x80U) {
                return false;
            }
            code_point = (code_point << 6U) | (continuation & 0x3FU);
        }
        static constexpr std::array<std::uint32_t, 5> minimum{
            0U, 0U, 0x80U, 0x800U, 0x10000U};
        if (code_point < minimum[length] || code_point > 0x10FFFFU ||
            (code_point >= 0xD800U && code_point <= 0xDFFFU)) {
            return false;
        }
        index += length;
    }
    return true;
}

// C0 controls and DEL. Descriptor strings are reproduced in three
// line-oriented or delimiter-sensitive places — `stats` text output, which is
// split on the first '='; the tab-separated LEANNMF1 manifest; and a shell
// command line printed by `pull`. A newline or tab in any of them forges a
// record rather than corrupting one, which is the failure that reads as
// success.
[[nodiscard]] inline bool has_control_characters(std::string_view text) {
    for (const char raw : text) {
        const auto byte = static_cast<unsigned char>(raw);
        if (byte < 0x20U || byte == 0x7fU) {
            return true;
        }
    }
    return false;
}

} // namespace leann::detail
