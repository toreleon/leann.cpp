#pragma once

#include <cstdint>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace leann::detail {

template <typename T>
void write_le(std::ostream & output, T value) {
    static_assert(std::is_unsigned_v<T>);
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        output.put(static_cast<char>((value >> (i * 8U)) & T{0xff}));
    }
    if (!output) {
        throw std::runtime_error("failed to write index data");
    }
}

template <typename T>
T read_le(std::istream & input) {
    static_assert(std::is_unsigned_v<T>);
    T value = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        const int byte = input.get();
        if (byte == std::char_traits<char>::eof()) {
            throw std::runtime_error("truncated index data");
        }
        value |= static_cast<T>(static_cast<unsigned char>(byte)) << (i * 8U);
    }
    return value;
}

inline void write_bytes(std::ostream & output, const char * data, std::size_t size) {
    output.write(data, static_cast<std::streamsize>(size));
    if (!output) {
        throw std::runtime_error("failed to write index data");
    }
}

inline void read_bytes(std::istream & input, char * data, std::size_t size) {
    input.read(data, static_cast<std::streamsize>(size));
    if (!input) {
        throw std::runtime_error("truncated index data");
    }
}

inline std::size_t checked_size(std::uint64_t value, const char * field) {
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(std::string(field) + " does not fit in memory");
    }
    return static_cast<std::size_t>(value);
}

} // namespace leann::detail
