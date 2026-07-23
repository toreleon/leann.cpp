#pragma once

#include <filesystem>
#include <string_view>
#include <system_error>

namespace leann::detail {

class ArtifactFileOperations {
  public:
    virtual ~ArtifactFileOperations() = default;

    [[nodiscard]] virtual bool
    exists(const std::filesystem::path & path) = 0;
    virtual void rename(const std::filesystem::path & from,
                        const std::filesystem::path & to,
                        std::string_view operation) = 0;
    [[nodiscard]] virtual std::error_code
    remove(const std::filesystem::path & path) = 0;
};

void publish_artifact_pair(
    const std::filesystem::path & temporary_index,
    const std::filesystem::path & temporary_documents,
    const std::filesystem::path & index_path,
    const std::filesystem::path & documents_path,
    const std::filesystem::path & index_backup,
    const std::filesystem::path & documents_backup,
    ArtifactFileOperations & operations);

void publish_artifact_pair(
    const std::filesystem::path & temporary_index,
    const std::filesystem::path & temporary_documents,
    const std::filesystem::path & index_path,
    const std::filesystem::path & documents_path,
    const std::filesystem::path & index_backup,
    const std::filesystem::path & documents_backup);

} // namespace leann::detail
