#!/bin/bash
set -e

# Install ONNX Runtime C++ library

echo "========================================================================"
echo "Installing ONNX Runtime"
echo "========================================================================"

# Detect OS and architecture
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    if [ "$(uname -m)" == "x86_64" ]; then
        ARCH="x64"
    elif [ "$(uname -m)" == "aarch64" ]; then
        ARCH="aarch64"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="osx"
    if [ "$(uname -m)" == "x86_64" ]; then
        ARCH="x86_64"
    elif [ "$(uname -m)" == "arm64" ]; then
        ARCH="arm64"
    fi
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

echo "Detected: $OS $ARCH"

# ONNX Runtime version
VERSION="1.17.1"

# Build download URL
if [ "$OS" == "linux" ]; then
    PACKAGE="onnxruntime-linux-${ARCH}-${VERSION}.tgz"
elif [ "$OS" == "osx" ]; then
    PACKAGE="onnxruntime-osx-${ARCH}-${VERSION}.tgz"
fi

URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${PACKAGE}"

echo "Downloading ONNX Runtime ${VERSION}..."
echo "URL: $URL"

# Download
mkdir -p third_party
cd third_party

if [ ! -f "$PACKAGE" ]; then
    wget "$URL"
fi

# Extract
echo "Extracting..."
tar -xzf "$PACKAGE"

# Rename
EXTRACTED_DIR="onnxruntime-${OS}-${ARCH}-${VERSION}"
if [ -d "onnxruntime" ]; then
    rm -rf onnxruntime
fi
mv "$EXTRACTED_DIR" onnxruntime

cd ..

echo ""
echo "========================================================================"
echo "ONNX Runtime installed successfully!"
echo "Location: $(pwd)/third_party/onnxruntime"
echo ""
echo "To build the ONNX Runtime server:"
echo "  cd onnx_runtime"
echo "  ./build.sh"
echo "========================================================================"
