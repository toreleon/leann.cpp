#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace leann::detail {

// A build lock is an adjacent directory named "<artifact>.lock". Acquisition
// is the atomic create_directory itself; the descriptor file written inside is
// advisory metadata only, so a binary that ignores it still interoperates with
// one that writes it.
inline constexpr std::string_view lock_owner_filename = "owner";

struct LockOwner {
    std::uint64_t pid = 0;
    std::string host;
    std::uint64_t started_unix = 0;
};

// Liveness of the process that created a lock. `Unknown` is the honest answer
// whenever the lock carries no descriptor, the descriptor is unreadable, or it
// was written on a different host — this is a probe, never a proof, because a
// pid can be recycled and a network filesystem can be shared.
enum class OwnerLiveness {
    Unknown,
    Running,
    Absent,
};

[[nodiscard]] std::filesystem::path
lock_path_for(const std::filesystem::path & target);

[[nodiscard]] std::uint64_t current_process_id() noexcept;
[[nodiscard]] std::string current_host_name();
[[nodiscard]] std::uint64_t current_unix_time() noexcept;

// True when a process with this id exists on this host. A pid is recycled, so
// a true result does not prove it is the same process that took the lock.
[[nodiscard]] bool process_exists(std::uint64_t pid) noexcept;

void write_lock_owner(const std::filesystem::path & lock);
void remove_lock_owner(const std::filesystem::path & lock) noexcept;
[[nodiscard]] std::optional<LockOwner>
read_lock_owner(const std::filesystem::path & lock);

[[nodiscard]] OwnerLiveness owner_liveness(const std::optional<LockOwner> &
                                               owner);

} // namespace leann::detail
