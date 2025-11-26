#!/bin/bash
set -e

# User data script for Rust benchmark instance
# Runs on first boot to configure the instance

echo "==================================="
echo "Rust Benchmark Instance Setup"
echo "==================================="

# Update system
apt-get update
apt-get upgrade -y

# Install Docker
apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker GPG key
mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Add Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add ubuntu user to docker group
usermod -aG docker ubuntu

# Install Rust (for potential local builds)
su - ubuntu -c 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'

# Install additional tools
apt-get install -y \
    git \
    htop \
    vim \
    tmux \
    jq \
    python3-pip

# Install Python packages (for benchmark client)
pip3 install --upgrade pip
pip3 install \
    pyyaml \
    requests \
    numpy \
    psutil \
    tqdm

# Configure performance governor (for consistent benchmarking)
apt-get install -y linux-tools-generic
echo "performance" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable swap for consistent performance
swapoff -a
sed -i '/swap/d' /etc/fstab

# Create directories
mkdir -p /home/ubuntu/{models,results,config}
chown -R ubuntu:ubuntu /home/ubuntu/{models,results,config}

echo "==================================="
echo "Setup complete!"
echo "==================================="
