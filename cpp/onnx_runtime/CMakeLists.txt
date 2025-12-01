cmake_minimum_required(VERSION 3.15)
project(onnx_runtime_server)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find packages
find_package(CURL REQUIRED)

# ONNX Runtime
# Users should set ONNXRUNTIME_ROOT or provide via -DONNXRUNTIME_ROOT=/path/to/onnxruntime
if(NOT DEFINED ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_ROOT "/usr/local" CACHE PATH "ONNX Runtime installation directory")
endif()

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")

# Check if ONNX Runtime is installed
if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime not found at ${ONNXRUNTIME_ROOT}. Please install ONNX Runtime or set ONNXRUNTIME_ROOT.")
endif()

# yaml-cpp
find_package(yaml-cpp REQUIRED)

# Include directories
include_directories(
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cpp-httplib
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/json/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tokenizers-cpp/include
    ${YAML_CPP_INCLUDE_DIR}
)

# Link directories
link_directories(
    ${ONNXRUNTIME_LIB_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tokenizers-cpp/lib
)

# Source files
add_executable(onnx_runtime_server
    main.cpp
)

# Link libraries
target_link_libraries(onnx_runtime_server
    onnxruntime
    tokenizers_cpp
    yaml-cpp
    pthread
    ${CURL_LIBRARIES}
    ssl
    crypto
)

# Set RPATH for finding shared libraries at runtime
set_target_properties(onnx_runtime_server PROPERTIES
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}:${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tokenizers-cpp/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Install target
install(TARGETS onnx_runtime_server DESTINATION bin)

# Print configuration
message(STATUS "ONNX Runtime root: ${ONNXRUNTIME_ROOT}")
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime lib: ${ONNXRUNTIME_LIB_DIR}")
