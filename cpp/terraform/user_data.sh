#!/bin/bash
set -e

# Log everything to /var/log/user-data.log
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "========================================================================"
echo "Starting C++ Benchmark Instance Setup"
echo "========================================================================"
echo "Instance type: $(ec2-metadata --instance-type | cut -d ' ' -f 2)"
echo "CPU info: $(lscpu | grep 'Model name' | cut -d ':' -f 2 | xargs)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "Start time: $(date)"
echo "========================================================================"

# Update system
echo "Updating system..."
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get upgrade -y

# Install essential tools
echo "Installing essential tools..."
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
    lsb-release \
    software-properties-common \
    pkg-config \
    libssl-dev \
    libcurl4-openssl-dev

# Install CMake (latest version for C++ builds)
echo "Installing CMake..."
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null
apt-get update
apt-get install -y cmake

# Install yaml-cpp
echo "Installing yaml-cpp..."
apt-get install -y libyaml-cpp-dev

# Install Python 3.11 (for model conversion scripts)
echo "Installing Python 3.11..."
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Set Python 3.11 as default
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
update-alternatives --set python3 /usr/bin/python3.11

# Upgrade pip
python3 -m pip install --upgrade pip

# Install Python dependencies for model conversion
echo "Installing Python dependencies..."
python3 -m pip install --no-cache-dir \
    torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu \
    transformers==4.38.0 \
    onnx==1.15.0 \
    onnxruntime==1.17.1 \
    optimum[exporters]==1.17.1 \
    pyyaml==6.0.1 \
    requests==2.31.0

# Install OpenVINO
echo "Installing OpenVINO..."
wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | apt-key add -
echo "deb https://apt.repos.intel.com/openvino/2024 ubuntu20 main" | tee /etc/apt/sources.list.d/intel-openvino-2024.list
apt-get update
apt-get install -y openvino-2024.0.0

# Setup OpenVINO environment for all users
echo "source /opt/intel/openvino_2024/setupvars.sh" >> /etc/profile.d/openvino.sh
chmod +x /etc/profile.d/openvino.sh

# Source for current session
source /opt/intel/openvino_2024/setupvars.sh

# Install OpenVINO Python tools
python3 -m pip install openvino-dev==2024.0.0

# Install AWS CLI
echo "Installing AWS CLI..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install
rm -rf awscliv2.zip aws

# Create workspace directory
echo "Creating workspace..."
mkdir -p /home/ubuntu/benchmark
chown -R ubuntu:ubuntu /home/ubuntu/benchmark

# Install ONNX Runtime C++ (system-wide)
echo "Installing ONNX Runtime C++..."
ONNXRUNTIME_VERSION="1.17.1"
ONNXRUNTIME_DIR="/usr/local/onnxruntime"
mkdir -p "$ONNXRUNTIME_DIR"
cd /tmp
wget "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz"
tar -xzf "onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz"
mv "onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}"/* "$ONNXRUNTIME_DIR/"
rm -rf "onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz" "onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}"

# Add ONNX Runtime to library path
echo "$ONNXRUNTIME_DIR/lib" > /etc/ld.so.conf.d/onnxruntime.conf
ldconfig

# Install third-party C++ libraries for ubuntu user
echo "Installing third-party C++ libraries..."
su - ubuntu -c 'bash -s' << 'EOF_USER'
set -e

cd /home/ubuntu/benchmark
mkdir -p third_party
cd third_party

# cpp-httplib (header-only HTTP server)
if [ ! -d "cpp-httplib" ]; then
    git clone https://github.com/yhirose/cpp-httplib.git
    cd cpp-httplib
    git checkout v0.14.0
    cd ..
fi

# nlohmann/json (header-only JSON parser)
if [ ! -d "json" ]; then
    git clone https://github.com/nlohmann/json.git
    cd json
    git checkout v3.11.3
    cd ..
fi

# tokenizers-cpp (fast tokenizer)
if [ ! -d "tokenizers-cpp" ]; then
    git clone https://github.com/mlc-ai/tokenizers-cpp.git
    cd tokenizers-cpp
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    cd ..
    mkdir -p lib include
    cp build/*.so lib/ 2>/dev/null || true
    cd ..
fi

echo "Third-party libraries installed successfully!"
EOF_USER

# System optimizations for benchmarking
echo "Applying system optimizations..."

# Set CPU governor to performance mode
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Make CPU governor persist on reboot
cat > /etc/systemd/system/cpu-performance.service <<EOF
[Unit]
Description=Set CPU governor to performance mode
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'

[Install]
WantedBy=multi-user.target
EOF

systemctl enable cpu-performance.service

# Disable swap to avoid performance variance
swapoff -a
sed -i '/ swap / s/^/#/' /etc/fstab

# Increase file descriptor limits
cat >> /etc/security/limits.conf <<EOF
ubuntu soft nofile 65536
ubuntu hard nofile 65536
EOF

# Disable transparent huge pages (can cause latency spikes)
echo never > /sys/kernel/mm/transparent_hugepage/enabled
echo never > /sys/kernel/mm/transparent_hugepage/defrag

# Make THP settings persist
cat > /etc/systemd/system/disable-thp.service <<EOF
[Unit]
Description=Disable Transparent Huge Pages
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'echo never > /sys/kernel/mm/transparent_hugepage/enabled && echo never > /sys/kernel/mm/transparent_hugepage/defrag'

[Install]
WantedBy=multi-user.target
EOF

systemctl enable disable-thp.service

# Set kernel parameters for performance
cat >> /etc/sysctl.conf <<EOF
# Performance tuning for benchmarking
vm.swappiness=0
vm.dirty_ratio=80
vm.dirty_background_ratio=5
net.core.somaxconn=4096
net.ipv4.tcp_max_syn_backlog=4096
EOF

sysctl -p

# Install systemd service templates for C++ servers
echo "Creating systemd service templates..."

# ONNX Runtime C++ service
cat > /etc/systemd/system/onnx-cpp-server.service <<'EOF'
[Unit]
Description=ONNX Runtime C++ Embedding Server
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/benchmark/cpp/onnx_runtime
Environment="MODEL_NAME=embeddinggemma-300m"
Environment="LD_LIBRARY_PATH=/usr/local/onnxruntime/lib:/home/ubuntu/benchmark/cpp/third_party/tokenizers-cpp/lib"
ExecStart=/home/ubuntu/benchmark/cpp/onnx_runtime/build/onnx_runtime_server
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# OpenVINO C++ service
cat > /etc/systemd/system/openvino-cpp-server.service <<'EOF'
[Unit]
Description=OpenVINO C++ Embedding Server
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/benchmark/cpp/openvino
Environment="MODEL_NAME=embeddinggemma-300m"
Environment="LD_LIBRARY_PATH=/home/ubuntu/benchmark/cpp/third_party/tokenizers-cpp/lib"
EnvironmentFile=/opt/intel/openvino_2024/setupvars.sh
ExecStartPre=/bin/bash -c 'source /opt/intel/openvino_2024/setupvars.sh'
ExecStart=/home/ubuntu/benchmark/cpp/openvino/build/openvino_server
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload

# Print system info
echo ""
echo "========================================================================"
echo "System Information:"
echo "========================================================================"
echo "OS: $(lsb_release -d | cut -f2)"
echo "Kernel: $(uname -r)"
echo "CPU: $(lscpu | grep 'Model name' | cut -d ':' -f 2 | xargs)"
echo "CPU cores: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "CMake version: $(cmake --version | head -n1)"
echo "GCC version: $(gcc --version | head -n1)"
echo "Python version: $(python3 --version)"
echo "ONNX Runtime: $ONNXRUNTIME_DIR"
echo "OpenVINO: /opt/intel/openvino_2024"
echo "========================================================================"

# Create completion marker
touch /var/log/user-data-complete
echo "Setup complete!" > /var/log/user-data-complete
echo "$(date)" >> /var/log/user-data-complete

echo ""
echo "========================================================================"
echo "C++ Benchmark Instance Setup Complete!"
echo "========================================================================"
echo "Instance is ready for C++ benchmarking."
echo "Next steps:"
echo "1. SSH into the instance"
echo "2. Clone your repository to /home/ubuntu/benchmark"
echo "3. Build the C++ servers (see cpp/README.md)"
echo "4. Run benchmarks with cpp/benchmark_comparison.py"
echo "========================================================================"
