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

# Install Python dependencies for orchestrator
echo "Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install pyyaml docker-compose

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

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Download a model:"
echo "     python3 scripts/download_model.py --model embeddinggemma-300m"
echo ""
echo "  2. Build Docker containers:"
echo "     docker-compose build"
echo ""
echo "  3. Run a single benchmark:"
echo "     docker-compose run --rm pytorch"
echo ""
echo "  4. Run all benchmarks:"
echo "     python3 orchestrator/runner.py --model embeddinggemma-300m"
echo ""
