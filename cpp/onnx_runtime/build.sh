#!/bin/bash
set -e

echo "========================================================================"
echo "Building ONNX Runtime C++ Server"
echo "========================================================================"

# Clean previous build
if [ -d "build" ]; then
    rm -rf build
fi

mkdir -p build
cd build

# Configure
ONNXRUNTIME_ROOT="$(pwd)/../../third_party/onnxruntime"

echo "ONNX Runtime root: $ONNXRUNTIME_ROOT"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT"

# Build
cmake --build . --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

cd ..

echo ""
echo "========================================================================"
echo "Build complete!"
echo "Executable: build/onnx_runtime_server"
echo ""
echo "To run:"
echo "  export MODEL_NAME=embeddinggemma-300m"
echo "  ./build/onnx_runtime_server"
echo "========================================================================"
