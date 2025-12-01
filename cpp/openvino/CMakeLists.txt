cmake_minimum_required(VERSION 3.15)
project(openvino_server)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find packages
find_package(CURL REQUIRED)
find_package(OpenVINO REQUIRED)

# yaml-cpp
find_package(yaml-cpp REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cpp-httplib
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/json/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tokenizers-cpp/include
    ${YAML_CPP_INCLUDE_DIR}
)

# Link directories
link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tokenizers-cpp/lib
)

# Source files
add_executable(openvino_server
    main.cpp
)

# Link libraries
target_link_libraries(openvino_server
    openvino::runtime
    tokenizers_cpp
    yaml-cpp
    pthread
    ${CURL_LIBRARIES}
    ssl
    crypto
)

# Set RPATH for finding shared libraries at runtime
set_target_properties(openvino_server PROPERTIES
    INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tokenizers-cpp/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Install target
install(TARGETS openvino_server DESTINATION bin)

# Print configuration
message(STATUS "OpenVINO version: ${OpenVINO_VERSION}")
message(STATUS "OpenVINO include: ${OpenVINO_INCLUDE_DIRS}")
