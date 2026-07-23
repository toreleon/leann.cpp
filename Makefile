CXX ?= c++
HNSWLIB_DIR ?= third_party/hnswlib
BUILD_DIR ?= build-make

CPPFLAGS := -Iinclude -Isrc -isystem $(HNSWLIB_DIR)
CXXFLAGS ?= -O2 -g
CXXFLAGS += -std=c++20 -Wall -Wextra -Wpedantic
LDLIBS := -pthread

CORE_SOURCES := src/artifact_publisher.cpp src/checksum.cpp \
	src/document_store.cpp src/embedder.cpp src/index.cpp src/product_quantizer.cpp
CORE_OBJECTS := $(CORE_SOURCES:%.cpp=$(BUILD_DIR)/%.o)

.PHONY: all test persistence-test core-safety-test clean check-hnsw

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

clean:
	rm -rf "$(BUILD_DIR)"

-include $(CORE_OBJECTS:.o=.d)
-include $(BUILD_DIR)/app/main.d
-include $(BUILD_DIR)/tests/test_index.d
-include $(BUILD_DIR)/tests/test_persistence.d
-include $(BUILD_DIR)/tests/test_core_safety.d
