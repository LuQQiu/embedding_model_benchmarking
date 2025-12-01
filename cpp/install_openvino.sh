#!/bin/bash
set -e

# Install OpenVINO C++ library

echo "========================================================================"
echo "Installing OpenVINO"
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

if [ "$OS" == "linux" ]; then
    # Ubuntu/Debian
    echo "Installing OpenVINO via APT repository..."

    # Add Intel GPG key
    wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | sudo apt-key add -

    # Add repository
    echo "deb https://apt.repos.intel.com/openvino/2024 ubuntu20 main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2024.list

    # Update and install
    sudo apt-get update
    sudo apt-get install -y openvino-2024.0.0

    # Setup environment
    echo "source /opt/intel/openvino_2024/setupvars.sh" >> ~/.bashrc

    echo ""
    echo "OpenVINO installed to /opt/intel/openvino_2024"
    echo "Run: source /opt/intel/openvino_2024/setupvars.sh"

elif [ "$OS" == "macos" ]; then
    echo "Installing OpenVINO via Homebrew..."

    # Install via Homebrew
    brew install openvino

    echo ""
    echo "OpenVINO installed via Homebrew"
fi

echo ""
echo "========================================================================"
echo "OpenVINO installed successfully!"
echo ""
echo "To build the OpenVINO server:"
echo "  cd openvino"
echo "  ./build.sh"
echo "========================================================================"
