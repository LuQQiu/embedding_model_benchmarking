#!/bin/bash
set -e

# Setup script for C++ embedding servers
# This script installs dependencies and builds third-party libraries

echo "========================================================================"
echo "C++ Embedding Server - Setup"
echo "========================================================================"

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

echo "Detected OS: $OS"
echo ""

# Install system dependencies
echo "Installing system dependencies..."
if [ "$OS" == "linux" ]; then
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        libssl-dev \
        pkg-config \
        libyaml-cpp-dev
elif [ "$OS" == "macos" ]; then
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Please install Homebrew first."
        exit 1
    fi
    brew install cmake git wget curl openssl yaml-cpp
fi

echo "✓ System dependencies installed"
echo ""

# Create third_party directory
mkdir -p third_party
cd third_party

# Install cpp-httplib (header-only)
echo "Installing cpp-httplib..."
if [ ! -d "cpp-httplib" ]; then
    git clone https://github.com/yhirose/cpp-httplib.git
    cd cpp-httplib
    git checkout v0.14.0
    cd ..
fi
echo "✓ cpp-httplib installed"

# Install nlohmann/json (header-only)
echo "Installing nlohmann/json..."
if [ ! -d "json" ]; then
    git clone https://github.com/nlohmann/json.git
    cd json
    git checkout v3.11.3
    cd ..
fi
echo "✓ nlohmann/json installed"

# Install tokenizers-cpp
echo "Installing tokenizers-cpp..."
if [ ! -d "tokenizers-cpp" ]; then
    git clone https://github.com/mlc-ai/tokenizers-cpp.git
    cd tokenizers-cpp
    mkdir -p build
    cd build
    cmake ..
    cmake --build . --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
    cd ..
    mkdir -p lib include
    cp build/*.so lib/ 2>/dev/null || cp build/*.dylib lib/ 2>/dev/null || true
    cp -r include/* include/ 2>/dev/null || true
    cd ..
fi
echo "✓ tokenizers-cpp installed"

cd ..

echo ""
echo "========================================================================"
echo "Third-party libraries installed successfully!"
echo ""
echo "Next steps:"
echo "1. Install ONNX Runtime: ./install_onnxruntime.sh"
echo "2. Install OpenVINO: ./install_openvino.sh"
echo "3. Build servers:"
echo "   - ONNX Runtime: cd onnx_runtime && ./build.sh"
echo "   - OpenVINO: cd openvino && ./build.sh"
echo "========================================================================"
