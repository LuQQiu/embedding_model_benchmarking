#!/bin/bash
set -e

# Log everything to /var/log/user-data.log
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "Starting benchmark instance setup..."

# Update system
apt-get update
apt-get upgrade -y

# Install essential tools
apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    vim \
    htop \
    unzip \
    jq \
    ca-certificates \
    gnupg \
    lsb-release

# Install Docker
echo "Installing Docker..."
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add ubuntu user to docker group
usermod -aG docker ubuntu

# Enable Docker service
systemctl enable docker
systemctl start docker

# Install Docker Compose (standalone)
DOCKER_COMPOSE_VERSION="2.24.5"
curl -SL "https://github.com/docker/compose/releases/download/v${DOCKER_COMPOSE_VERSION}/docker-compose-linux-x86_64" \
    -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose

# Install Python 3.11
apt-get install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Set Python 3.11 as default
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
update-alternatives --set python3 /usr/bin/python3.11

# Install Rust for ubuntu user
echo "Installing Rust..."
su - ubuntu -c 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'

# Install CMake (for ONNX Native C++)
apt-get install -y cmake

# Create workspace directory
mkdir -p /home/ubuntu/benchmark
chown -R ubuntu:ubuntu /home/ubuntu/benchmark

# System optimizations for benchmarking
echo "Applying system optimizations..."
# Disable CPU frequency scaling
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable swap to avoid performance variance
swapoff -a

# Increase file descriptor limits
cat >> /etc/security/limits.conf <<EOF
ubuntu soft nofile 65536
ubuntu hard nofile 65536
EOF

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install
rm -rf awscliv2.zip aws

# Create completion marker
touch /var/log/user-data-complete

echo "Benchmark instance setup complete!"
echo "Instance is ready for benchmarking."
