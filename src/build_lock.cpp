#include "build_lock.hpp"

#include <charconv>
#include <chrono>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <unistd.h>
#endif

namespace leann::detail {
namespace {

// Descriptor lines are "key=value"; an unknown key is ignored so a newer
// writer can add fields without breaking an older reader.
bool parse_unsigned(std::string_view text, std::uint64_t & destination) {
    const auto * const begin = text.data();
    const auto * const end = text.data() + text.size();
    std::uint64_t parsed = 0;
    const auto [stop, error] = std::from_chars(begin, end, parsed);
    if (error != std::errc{} || stop != end) {
        return false;
    }
    destination = parsed;
    return true;
}

} // namespace

std::filesystem::path lock_path_for(const std::filesystem::path & target) {
    auto result = target;
    result += ".lock";
    return result;
}

std::uint64_t current_process_id() noexcept {
#ifdef _WIN32
    return static_cast<std::uint64_t>(GetCurrentProcessId());
#else
    return static_cast<std::uint64_t>(::getpid());
#endif
}

std::string current_host_name() {
#ifdef _WIN32
    char buffer[MAX_COMPUTERNAME_LENGTH + 1] = {};
    DWORD size = MAX_COMPUTERNAME_LENGTH + 1;
    if (GetComputerNameA(buffer, &size) == 0) {
        return {};
    }
    return std::string(buffer, size);
#else
    char buffer[256] = {};
    if (::gethostname(buffer, sizeof(buffer) - 1) != 0) {
        return {};
    }
    return std::string(buffer);
#endif
}

std::uint64_t current_unix_time() noexcept {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto seconds =
        std::chrono::duration_cast<std::chrono::seconds>(now).count();
    return seconds < 0 ? 0U : static_cast<std::uint64_t>(seconds);
}

bool process_exists(std::uint64_t pid) noexcept {
    if (pid == 0) {
        return false;
    }
#ifdef _WIN32
    HANDLE handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE,
                                static_cast<DWORD>(pid));
    if (handle == nullptr) {
        return GetLastError() == ERROR_ACCESS_DENIED;
    }
    CloseHandle(handle);
    return true;
#else
    if (pid > static_cast<std::uint64_t>(
                  std::numeric_limits<pid_t>::max())) {
        return false;
    }
    errno = 0;
    if (::kill(static_cast<pid_t>(pid), 0) == 0) {
        return true;
    }
    // EPERM means the process exists but belongs to another user.
    return errno == EPERM;
#endif
}

void write_lock_owner(const std::filesystem::path & lock) {
    std::ofstream owner(lock / lock_owner_filename,
                        std::ios::binary | std::ios::trunc);
    if (!owner) {
        return;
    }
    owner << "pid=" << current_process_id() << '\n'
          << "host=" << current_host_name() << '\n'
          << "started_unix=" << current_unix_time() << '\n';
}

void remove_lock_owner(const std::filesystem::path & lock) noexcept {
    std::error_code ignored;
    std::filesystem::remove(lock / lock_owner_filename, ignored);
}

std::optional<LockOwner> read_lock_owner(const std::filesystem::path & lock) {
    std::ifstream input(lock / lock_owner_filename, std::ios::binary);
    if (!input) {
        return std::nullopt;
    }
    LockOwner owner;
    bool saw_pid = false;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const std::size_t separator = line.find('=');
        if (separator == std::string::npos) {
            continue;
        }
        const std::string key = line.substr(0, separator);
        const std::string value = line.substr(separator + 1);
        if (key == "pid") {
            saw_pid = parse_unsigned(value, owner.pid);
        } else if (key == "host") {
            owner.host = value;
        } else if (key == "started_unix") {
            (void)parse_unsigned(value, owner.started_unix);
        }
    }
    if (!saw_pid) {
        return std::nullopt;
    }
    return owner;
}

OwnerLiveness owner_liveness(const std::optional<LockOwner> & owner) {
    if (!owner.has_value()) {
        return OwnerLiveness::Unknown;
    }
    const std::string host = current_host_name();
    if (host.empty() || owner->host.empty() || owner->host != host) {
        return OwnerLiveness::Unknown;
    }
    return process_exists(owner->pid) ? OwnerLiveness::Running
                                      : OwnerLiveness::Absent;
}

} // namespace leann::detail
