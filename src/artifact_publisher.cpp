#include "artifact_publisher.hpp"

#include <stdexcept>
#include <string>

namespace leann::detail {
namespace {

class NativeArtifactFileOperations final : public ArtifactFileOperations {
  public:
    bool exists(const std::filesystem::path & path) override {
        std::error_code error;
        const bool result = std::filesystem::exists(path, error);
        if (error) {
            throw std::runtime_error("cannot inspect artifact '" +
                                     path.string() + "': " +
                                     error.message());
        }
        return result;
    }

    void rename(const std::filesystem::path & from,
                const std::filesystem::path & to,
                std::string_view operation) override {
        std::error_code error;
        std::filesystem::rename(from, to, error);
        if (error) {
            throw std::runtime_error(std::string(operation) + ": " +
                                     error.message());
        }
    }

    std::error_code remove(const std::filesystem::path & path) override {
        std::error_code error;
        std::filesystem::remove(path, error);
        return error;
    }
};

void append_error(std::string & destination, std::string_view action,
                  const std::filesystem::path & path,
                  std::string_view reason) {
    if (!destination.empty()) {
        destination += ' ';
    }
    destination += std::string(action) + " '" + path.string() + "': " +
                   std::string(reason) + ";";
}

} // namespace

void publish_artifact_pair(
    const std::filesystem::path & temporary_index,
    const std::filesystem::path & temporary_documents,
    const std::filesystem::path & index_path,
    const std::filesystem::path & documents_path,
    const std::filesystem::path & index_backup,
    const std::filesystem::path & documents_backup,
    ArtifactFileOperations & operations) {
    const bool had_index = operations.exists(index_path);
    const bool had_documents = operations.exists(documents_path);
    bool index_backed_up = false;
    bool documents_backed_up = false;
    bool new_documents_installed = false;

    try {
        if (had_index) {
            operations.rename(index_path, index_backup,
                              "failed to back up previous index");
            index_backed_up = true;
        }
        if (had_documents) {
            operations.rename(documents_path, documents_backup,
                              "failed to back up previous document store");
            documents_backed_up = true;
        }
        operations.rename(temporary_documents, documents_path,
                          "failed to publish document store");
        new_documents_installed = true;
        operations.rename(temporary_index, index_path,
                          "failed to publish index");
    } catch (const std::exception & publish_error) {
        std::string rollback_error;
        if (new_documents_installed) {
            try {
                operations.rename(documents_path, temporary_documents,
                                  "failed to withdraw uncommitted documents");
            } catch (const std::exception & withdraw_error) {
                const std::error_code remove_error =
                    operations.remove(documents_path);
                if (remove_error) {
                    append_error(rollback_error,
                                 "cannot remove uncommitted document store",
                                 documents_path, remove_error.message());
                } else {
                    (void)withdraw_error;
                }
            }
        }
        if (documents_backed_up) {
            try {
                operations.rename(documents_backup, documents_path,
                                  "failed to restore document backup");
            } catch (const std::exception & error) {
                append_error(rollback_error, "cannot restore document backup",
                             documents_backup, error.what());
            }
        }
        if (index_backed_up && rollback_error.empty()) {
            try {
                operations.rename(index_backup, index_path,
                                  "failed to restore index backup");
            } catch (const std::exception & error) {
                append_error(rollback_error, "cannot restore index backup",
                             index_backup, error.what());
            }
        }
        throw std::runtime_error(
            std::string(publish_error.what()) +
            (rollback_error.empty()
                 ? "; previous artifact pair restored"
                 : "; rollback incomplete: " + rollback_error));
    }

    std::string cleanup_error;
    if (index_backed_up) {
        const std::error_code error = operations.remove(index_backup);
        if (error) {
            append_error(cleanup_error, "cannot remove index backup",
                         index_backup, error.message());
        }
    }
    if (documents_backed_up) {
        const std::error_code error = operations.remove(documents_backup);
        if (error) {
            append_error(cleanup_error, "cannot remove document backup",
                         documents_backup, error.message());
        }
    }
    if (!cleanup_error.empty()) {
        throw std::runtime_error(
            "new artifact pair committed; backup cleanup required: " +
            cleanup_error);
    }
}

void publish_artifact_pair(
    const std::filesystem::path & temporary_index,
    const std::filesystem::path & temporary_documents,
    const std::filesystem::path & index_path,
    const std::filesystem::path & documents_path,
    const std::filesystem::path & index_backup,
    const std::filesystem::path & documents_backup) {
    NativeArtifactFileOperations operations;
    publish_artifact_pair(temporary_index, temporary_documents, index_path,
                          documents_path, index_backup, documents_backup,
                          operations);
}

} // namespace leann::detail
