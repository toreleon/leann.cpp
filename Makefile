CXX ?= c++
CC ?= cc
HNSWLIB_DIR ?= third_party/hnswlib
BUILD_DIR ?= build-make

CPPFLAGS := -Iinclude -Isrc -isystem $(HNSWLIB_DIR)
CXXFLAGS ?= -O2 -g
CXXFLAGS += -std=c++20 -Wall -Wextra -Wpedantic
CFLAGS ?= -O2 -g
CFLAGS += -std=c11 -Wall -Wextra -Wpedantic
LDLIBS := -pthread

CORE_SOURCES := src/artifact_publisher.cpp src/build_lock.cpp src/c_api.cpp \
	src/checksum.cpp \
	src/document_store.cpp src/embedder.cpp src/index.cpp src/product_quantizer.cpp
CORE_OBJECTS := $(CORE_SOURCES:%.cpp=$(BUILD_DIR)/%.o)

.PHONY: all test persistence-test core-safety-test c-api-test cli-cache-test \
	artifact-fuzz-test check clean check-hnsw

all: check-hnsw $(BUILD_DIR)/leann

check-hnsw:
	@test -f "$(HNSWLIB_DIR)/hnswlib/hnswlib.h" || \
	  (echo "hnswlib not found; set HNSWLIB_DIR or use CMake FetchContent" && false)

$(BUILD_DIR)/leann: $(CORE_OBJECTS) $(BUILD_DIR)/app/main.o
	$(CXX) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/leann_tests: $(CORE_OBJECTS) $(BUILD_DIR)/tests/test_index.o
	$(CXX) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

test: check-hnsw $(BUILD_DIR)/leann_tests
	$(BUILD_DIR)/leann_tests

persistence-test: check-hnsw $(BUILD_DIR)/leann_persistence_tests
	$(BUILD_DIR)/leann_persistence_tests

$(BUILD_DIR)/leann_persistence_tests: $(CORE_OBJECTS) \
	$(BUILD_DIR)/tests/test_persistence.o
	$(CXX) $^ $(LDLIBS) -o $@

core-safety-test: check-hnsw $(BUILD_DIR)/leann_core_safety_tests
	$(BUILD_DIR)/leann_core_safety_tests

$(BUILD_DIR)/leann_core_safety_tests: $(CORE_OBJECTS) \
	$(BUILD_DIR)/tests/test_core_safety.o
	$(CXX) $^ $(LDLIBS) -o $@

c-api-test: check-hnsw $(BUILD_DIR)/leann_c_api_tests \
	$(BUILD_DIR)/leann_c_header_tests
	$(BUILD_DIR)/leann_c_api_tests
	$(BUILD_DIR)/leann_c_header_tests

$(BUILD_DIR)/leann_c_api_tests: $(CORE_OBJECTS) \
	$(BUILD_DIR)/tests/test_c_api.o
	$(CXX) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/leann_c_header_tests: $(CORE_OBJECTS) \
	$(BUILD_DIR)/tests/test_c_header.o
	$(CXX) $^ $(LDLIBS) -o $@

# tests/test_cli_cache.cpp #includes app/main.cpp with main renamed, so this
# target must not also link $(BUILD_DIR)/app/main.o.
cli-cache-test: check-hnsw $(BUILD_DIR)/leann_cli_cache_tests
	$(BUILD_DIR)/leann_cli_cache_tests

$(BUILD_DIR)/leann_cli_cache_tests: $(CORE_OBJECTS) \
	$(BUILD_DIR)/tests/test_cli_cache.o
	$(CXX) $^ $(LDLIBS) -o $@

artifact-fuzz-test: check-hnsw $(BUILD_DIR)/leann_artifact_fuzz_tests
	$(BUILD_DIR)/leann_artifact_fuzz_tests

$(BUILD_DIR)/leann_artifact_fuzz_tests: $(CORE_OBJECTS) \
	$(BUILD_DIR)/tests/test_artifact_fuzz.o
	$(CXX) $^ $(LDLIBS) -o $@

# `all` is included so `make check` never leaves a stale $(BUILD_DIR)/leann
# behind for manual testing: the test targets alone do not build the binary.
check: all test persistence-test core-safety-test c-api-test cli-cache-test \
	artifact-fuzz-test

clean:
	rm -rf "$(BUILD_DIR)"

-include $(CORE_OBJECTS:.o=.d)
-include $(BUILD_DIR)/app/main.d
-include $(BUILD_DIR)/tests/test_index.d
-include $(BUILD_DIR)/tests/test_persistence.d
-include $(BUILD_DIR)/tests/test_core_safety.d
-include $(BUILD_DIR)/tests/test_c_api.d
-include $(BUILD_DIR)/tests/test_c_header.d
-include $(BUILD_DIR)/tests/test_cli_cache.d
