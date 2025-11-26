#!/bin/bash
set -e

echo "========================================="
echo "Embedding Model Benchmark Setup"
echo "========================================="
echo ""

# Check if running on correct instance type
if [ -f /sys/devices/virtual/dmi/id/product_name ]; then
    INSTANCE_TYPE=$(cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null || echo "unknown")
    echo "Instance type: $INSTANCE_TYPE"
fi

# Create required directories
echo "Creating directories..."
mkdir -p models results config docker orchestrator scripts analysis

# Upgrade pip and install build tools
echo ""
echo "Upgrading pip and installing build tools..."
pip3 install --upgrade pip setuptools wheel

# Install PyTorch (CPU version for model downloads)
echo ""
echo "Installing PyTorch (CPU version)..."
pip3 install torch --index-url https://download.pytorch.org/whl/cpu

# Install ML and model dependencies
echo ""
echo "Installing ML and model dependencies..."
pip3 install \
    sentence-transformers \
    transformers \
    tokenizers \
    huggingface-hub \
    onnx \
    onnxruntime \
    pyyaml \
    requests \
    psutil \
    numpy \
    tqdm \
    docker-compose

# Add ~/.local/bin to PATH in .bashrc (for huggingface-cli)
echo ""
echo "Configuring PATH for HuggingFace CLI..."
if ! grep -q 'export PATH=$PATH:~/.local/bin' ~/.bashrc; then
    echo 'export PATH=$PATH:~/.local/bin' >> ~/.bashrc
    echo "✓ Added ~/.local/bin to PATH in ~/.bashrc"
else
    echo "✓ PATH already configured"
fi

# Make PATH available in current session
export PATH=$PATH:~/.local/bin

# Make scripts executable
chmod +x scripts/*.sh scripts/*.py orchestrator/*.py 2>/dev/null || true

# Check Docker installation
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    echo "Please run the Terraform provisioning or install Docker manually"
    exit 1
fi

# Check Docker Compose installation
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed"
    exit 1
fi

echo "✓ Docker version: $(docker --version)"
echo "✓ Docker Compose version: $(docker-compose --version)"

# Test Docker access
if ! docker ps &> /dev/null; then
    echo "Error: Cannot access Docker. You may need to:"
    echo "  1. Add your user to docker group: sudo usermod -aG docker $USER"
    echo "  2. Log out and log back in"
    exit 1
fi

# Pull base images to save time later
echo ""
echo "Pulling base Docker images (this may take a while)..."
docker pull python:3.11-slim
docker pull rust:1.75-slim

# Install Claude Code CLI
echo ""
echo "Installing Claude Code CLI..."
if ! command -v claude &> /dev/null; then
    curl -fsSL https://raw.githubusercontent.com/anthropics/claude-code/main/install.sh | bash
    echo "✓ Claude Code installed"
else
    echo "✓ Claude Code already installed ($(claude --version))"
fi

# Verify installations
echo ""
echo "Verifying installations..."
echo "  Python: $(python3 --version)"
echo "  pip: $(pip3 --version | head -n1)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "  HuggingFace CLI: $(huggingface-cli --version 2>&1 | head -n1 || echo 'installed')"
echo "  Claude Code: $(claude --version 2>&1 || echo 'installed')"

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "  1. Authenticate with HuggingFace (for gated models):"
echo "     bash scripts/huggingface_login.sh"
echo "     Or manually: huggingface-cli login"
echo ""
echo "  2. Download a model:"
echo "     python3 scripts/download_model.py --model embeddinggemma-300m"
echo ""
echo "  3. Build Docker containers:"
echo "     docker-compose build"
echo ""
echo "  4. Run a single benchmark:"
echo "     docker-compose run --rm pytorch"
echo ""
echo "  5. Run all benchmarks with CSV export:"
echo "     python3 orchestrator/runner.py --model embeddinggemma-300m --csv"
echo ""
echo "Note: If you see command not found errors, log out and log back in"
echo "      to refresh your PATH, or run: source ~/.bashrc"
echo ""
