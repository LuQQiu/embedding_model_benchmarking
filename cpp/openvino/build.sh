#!/bin/bash
set -e

echo "========================================================================"
echo "Building OpenVINO C++ Server"
echo "========================================================================"

# Source OpenVINO environment if on Linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [ -f "/opt/intel/openvino_2024/setupvars.sh" ]; then
        source /opt/intel/openvino_2024/setupvars.sh
    fi
fi

# Clean previous build
if [ -d "build" ]; then
    rm -rf build
fi

mkdir -p build
cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

cd ..

echo ""
echo "========================================================================"
echo "Build complete!"
echo "Executable: build/openvino_server"
echo ""
echo "To run:"
echo "  export MODEL_NAME=embeddinggemma-300m"
echo "  ./build/openvino_server"
echo "========================================================================"
