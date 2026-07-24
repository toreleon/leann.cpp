cmake_minimum_required(VERSION 3.20)
project(leann_cpp VERSION 0.3.0 LANGUAGES C CXX)

include(FetchContent)
include(GNUInstallDirs)

option(LEANN_ENABLE_LLAMA "Build the native llama.cpp GGUF embedder" OFF)
option(LEANN_BUILD_TESTS "Build leann.cpp tests" ON)
set(LEANN_HNSWLIB_SOURCE_DIR "" CACHE PATH "Existing hnswlib checkout")
set(LEANN_LLAMA_CPP_SOURCE_DIR "" CACHE PATH "Existing llama.cpp checkout")

if(LEANN_HNSWLIB_SOURCE_DIR)
    add_subdirectory("${LEANN_HNSWLIB_SOURCE_DIR}"
                     "${CMAKE_BINARY_DIR}/_deps/hnswlib-build"
                     EXCLUDE_FROM_ALL)
else()
    FetchContent_Declare(
        hnswlib
        GIT_REPOSITORY https://github.com/nmslib/hnswlib.git
        GIT_TAG d9b3608c83d83b46c96e25088cb1d729b29dcfe9
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(hnswlib)
endif()

add_library(leann_cpp
    src/artifact_publisher.cpp
    src/c_api.cpp
    src/checksum.cpp
    src/document_store.cpp
    src/embedder.cpp
    src/index.cpp
    src/product_quantizer.cpp)
add_library(leann::leann ALIAS leann_cpp)
target_compile_features(leann_cpp PUBLIC cxx_std_20)
target_include_directories(
    leann_cpp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(leann_cpp PRIVATE hnswlib::hnswlib)
target_compile_definitions(leann_cpp PRIVATE LEANN_C_API_EXPORTS=1)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(leann_cpp PUBLIC LEANN_C_API_SHARED=1)
endif()
get_target_property(LEANN_HNSWLIB_INCLUDES hnswlib INTERFACE_INCLUDE_DIRECTORIES)
if(LEANN_HNSWLIB_INCLUDES)
    target_include_directories(
        leann_cpp SYSTEM PRIVATE ${LEANN_HNSWLIB_INCLUDES})
endif()

if(MSVC)
    target_compile_options(leann_cpp PRIVATE /W4 /permissive-)
else()
    target_compile_options(leann_cpp PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(LEANN_ENABLE_LLAMA)
    set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE)

    if(LEANN_LLAMA_CPP_SOURCE_DIR)
        add_subdirectory("${LEANN_LLAMA_CPP_SOURCE_DIR}"
                         "${CMAKE_BINARY_DIR}/_deps/llama-cpp-build"
                         EXCLUDE_FROM_ALL)
    else()
        FetchContent_Declare(
            llama_cpp
            GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
            GIT_TAG c588c4f47683e73ad2d69f50480bec6cc85fd0f7
            GIT_SHALLOW TRUE)
        FetchContent_MakeAvailable(llama_cpp)
    endif()

    target_sources(leann_cpp PRIVATE src/llama_embedder.cpp)
    target_compile_definitions(leann_cpp PUBLIC LEANN_WITH_LLAMA=1)
    target_link_libraries(leann_cpp PRIVATE llama)
endif()

add_executable(leann app/main.cpp)
target_link_libraries(leann PRIVATE leann::leann hnswlib::hnswlib)
if(LEANN_HNSWLIB_INCLUDES)
    target_include_directories(leann SYSTEM PRIVATE ${LEANN_HNSWLIB_INCLUDES})
endif()

if(LEANN_BUILD_TESTS)
    enable_testing()
    add_executable(leann_tests tests/test_index.cpp)
    target_link_libraries(leann_tests PRIVATE leann::leann)
    add_test(NAME leann_tests COMMAND leann_tests)

    add_executable(leann_persistence_tests tests/test_persistence.cpp)
    target_include_directories(
        leann_persistence_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(leann_persistence_tests PRIVATE leann::leann)
    add_test(NAME leann_persistence_tests COMMAND leann_persistence_tests)

    add_executable(leann_core_safety_tests tests/test_core_safety.cpp)
    target_link_libraries(leann_core_safety_tests PRIVATE leann::leann)
    add_test(NAME leann_core_safety_tests COMMAND leann_core_safety_tests)

    add_executable(leann_c_api_tests tests/test_c_api.cpp)
    target_link_libraries(leann_c_api_tests PRIVATE leann::leann)
    add_test(NAME leann_c_api_tests COMMAND leann_c_api_tests)

    # This source is compiled as strict C11 and linked with the C++ driver only
    # after compilation, proving that the installed public header is C-safe.
    add_executable(leann_c_header_tests tests/test_c_header.c)
    set_target_properties(
        leann_c_header_tests
        PROPERTIES
            C_STANDARD 11
            C_STANDARD_REQUIRED YES
            C_EXTENSIONS NO
            LINKER_LANGUAGE CXX)
    target_link_libraries(leann_c_header_tests PRIVATE leann::leann)
    add_test(NAME leann_c_header_tests COMMAND leann_c_header_tests)
endif()

if(LEANN_ENABLE_LLAMA)
    # llama.cpp does not export an installable CMake target when embedded as a
    # subproject, so a standalone leannTargets export cannot name that
    # dependency. The CLI and archive are still installable.
    install(TARGETS leann_cpp leann
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
else()
    install(TARGETS leann_cpp leann
            EXPORT leannTargets
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(EXPORT leannTargets
            FILE leannConfig.cmake
            NAMESPACE leann::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/leann)
endif()
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
