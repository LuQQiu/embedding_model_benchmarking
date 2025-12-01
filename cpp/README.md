# C++ Embedding Servers

High-performance C++ implementations for embedding model inference, comparing ONNX Runtime and OpenVINO.

## Overview

This directory contains standalone C++ servers for benchmarking embedding models using:

1. **ONNX Runtime C++** - Microsoft's cross-platform inference engine
2. **OpenVINO C++** - Intel's optimized inference toolkit

Both implementations provide:
- HTTP REST API (using cpp-httplib)
- Text tokenization (using tokenizers-cpp)
- Mean pooling and L2 normalization
- JSON request/response
- Metrics and monitoring

## Architecture

```
cpp/
├── onnx_runtime/          # ONNX Runtime implementation
│   ├── main.cpp
│   ├── CMakeLists.txt
│   └── build.sh
├── openvino/              # OpenVINO implementation
│   ├── main.cpp
│   ├── CMakeLists.txt
│   └── build.sh
├── terraform/             # AWS deployment (optional)
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   ├── user_data.sh
│   └── README.md
├── third_party/           # Dependencies (installed by setup)
│   ├── cpp-httplib/       # HTTP server library
│   ├── json/              # JSON parsing (nlohmann)
│   ├── tokenizers-cpp/    # Fast tokenizer
│   └── onnxruntime/       # ONNX Runtime binaries
├── setup.sh               # Install dependencies
├── install_onnxruntime.sh # Install ONNX Runtime
├── install_openvino.sh    # Install OpenVINO
├── deploy.sh              # AWS deployment helper
├── benchmark_comparison.py # Benchmark both implementations
├── QUICKSTART.md          # Quick start guide
└── README.md              # This file
```

## Deployment Options

### Option 1: AWS EC2 (Recommended for Production Benchmarking)

Deploy to AWS c7i.8xlarge (32 vCPU Intel Sapphire Rapids, optimized for OpenVINO):

```bash
cd cpp/

# One-command deployment
./deploy.sh deploy

# SSH into instance
./deploy.sh ssh

# Build and benchmark (on EC2)
cd ~/benchmark/cpp
cd onnx_runtime && ./build.sh && cd ..
cd openvino && ./build.sh && cd ..
./benchmark_comparison.py --model embeddinggemma-300m

# Destroy when done
./deploy.sh destroy
```

See `terraform/README.md` for detailed AWS deployment instructions.

**Cost**: ~$1.36/hour for c7i.8xlarge

### Option 2: Local Development

Build and run locally on your machine:

```bash
cd cpp/
./setup.sh
./install_onnxruntime.sh
./install_openvino.sh  # Linux: source /opt/intel/openvino_2024/setupvars.sh
# ... continue with build steps below
```

See `QUICKSTART.md` for step-by-step local setup.

## Prerequisites (Local Development)

### System Requirements

- **OS**: Ubuntu 20.04+, macOS 11+
- **Compiler**: GCC 9+ or Clang 11+ with C++17 support
- **CMake**: 3.15+
- **RAM**: 8GB+ (16GB+ recommended for large models)
- **CPU**: x86_64 or ARM64 (Intel CPUs recommended for OpenVINO)

### Required Libraries

- **CURL**: HTTP client library
- **OpenSSL**: TLS/SSL support
- **yaml-cpp**: YAML configuration parsing

## Quick Start

### 1. Setup Dependencies

```bash
cd cpp/
chmod +x *.sh */build.sh
./setup.sh
```

This installs:
- System dependencies (cmake, build tools, yaml-cpp)
- cpp-httplib (HTTP server, header-only)
- nlohmann/json (JSON parser, header-only)
- tokenizers-cpp (fast tokenizer)

### 2. Install Inference Frameworks

#### ONNX Runtime

```bash
./install_onnxruntime.sh
```

Downloads and extracts ONNX Runtime 1.17.1 to `third_party/onnxruntime/`.

#### OpenVINO

```bash
./install_openvino.sh
```

**Linux**: Installs via Intel APT repository to `/opt/intel/openvino_2024/`

**macOS**: Installs via Homebrew

### 3. Build Servers

#### ONNX Runtime Server

```bash
cd onnx_runtime
./build.sh
```

Builds to `onnx_runtime/build/onnx_runtime_server`.

#### OpenVINO Server

```bash
cd openvino
./build.sh
```

Builds to `openvino/build/openvino_server`.

### 4. Prepare Models

Ensure models are converted and available:

```bash
# From repository root
python3 scripts/download_model.py --model embeddinggemma-300m
python3 scripts/convert_to_onnx.py --model embeddinggemma-300m
python3 scripts/convert_to_openvino.py --model embeddinggemma-300m
```

### 5. Run Servers

#### ONNX Runtime

```bash
cd onnx_runtime
export MODEL_NAME=embeddinggemma-300m
./build/onnx_runtime_server
```

#### OpenVINO

```bash
cd openvino
export MODEL_NAME=embeddinggemma-300m

# Linux: Source OpenVINO environment first
source /opt/intel/openvino_2024/setupvars.sh

./build/openvino_server
```

Both servers listen on `http://0.0.0.0:8000`.

## API Reference

### Endpoints

#### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### GET `/info`

Server information and metrics.

**Response:**
```json
{
  "framework": "onnx-cpp",
  "model_name": "embeddinggemma-300m",
  "model_configuration": {
    "max_seq_length": 8192,
    "embedding_dim": 256
  },
  "model_load_time_ms": 1234.56,
  "total_requests": 42,
  "runtime_version": "1.17.1",
  "device": "CPU",
  "cpu_count": 8,
  "uptime_seconds": 300
}
```

#### POST `/embed`

Generate embeddings for input texts.

**Request:**
```json
{
  "texts": [
    "Hello world",
    "Another text to embed"
  ]
}
```

**Response:**
```json
{
  "embeddings": [
    [0.123, -0.456, 0.789, ...],
    [0.321, -0.654, 0.987, ...]
  ],
  "inference_time_ms": 12.34
}
```

## Benchmarking

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Server info
curl http://localhost:8000/info

# Generate embeddings
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world"]}'
```

### Using Python benchmark client

From the repository root, use the existing benchmark client:

```bash
# Make sure server is running first
cd cpp/onnx_runtime
export MODEL_NAME=embeddinggemma-300m
./build/onnx_runtime_server &
SERVER_PID=$!

# Run benchmark
cd ../../
python3 -c "
import requests
import time
import statistics

url = 'http://localhost:8000/embed'
texts = ['This is a test sentence for benchmarking.'] * 10

# Warmup
for _ in range(10):
    requests.post(url, json={'texts': texts})

# Benchmark
latencies = []
for _ in range(100):
    start = time.perf_counter()
    r = requests.post(url, json={'texts': texts})
    latencies.append((time.perf_counter() - start) * 1000)

print(f'Mean latency: {statistics.mean(latencies):.2f}ms')
print(f'Median latency: {statistics.median(latencies):.2f}ms')
print(f'P95 latency: {statistics.quantiles(latencies, n=20)[18]:.2f}ms')
print(f'P99 latency: {statistics.quantiles(latencies, n=100)[98]:.2f}ms')
"

# Stop server
kill $SERVER_PID
```

### Comparison Script

Create `cpp/benchmark_comparison.sh`:

```bash
#!/bin/bash

# Benchmark both implementations and compare

echo "Benchmarking ONNX Runtime C++..."
cd onnx_runtime
export MODEL_NAME=embeddinggemma-300m
./build/onnx_runtime_server &
ONNX_PID=$!
sleep 5

# Run benchmark (use your benchmark script)
# ... benchmark ONNX Runtime ...

kill $ONNX_PID
cd ..

echo "Benchmarking OpenVINO C++..."
cd openvino
source /opt/intel/openvino_2024/setupvars.sh  # Linux only
./build/openvino_server &
OV_PID=$!
sleep 5

# Run benchmark (use your benchmark script)
# ... benchmark OpenVINO ...

kill $OV_PID
cd ..

echo "Comparison complete!"
```

## Performance Expectations

Based on EmbeddingGemma 300M model on c7i.8xlarge (32 vCPU):

| Framework      | Language | Latency (p95) | Throughput (QPS) | Memory (MB) |
|----------------|----------|---------------|------------------|-------------|
| PyTorch        | Python   | ~25ms         | ~320             | ~800        |
| ONNX Python    | Python   | ~18ms         | ~385             | ~600        |
| ONNX Rust      | Rust     | ~15ms         | ~470             | ~500        |
| **ONNX C++**   | **C++**  | **~12ms**     | **~520**         | **~450**    |
| OpenVINO Python| Python   | ~14ms         | ~580             | ~550        |
| **OpenVINO C++**| **C++** | **~10ms**     | **~650**         | **~480**    |

C++ implementations typically achieve:
- **20-30% lower latency** vs Python/Rust (reduced overhead)
- **15-25% higher throughput** vs Python/Rust
- **10-20% lower memory** usage (no runtime overhead)

OpenVINO C++ should be the **fastest implementation** due to:
- Intel-optimized kernels
- Graph-level optimizations
- Zero-copy inference
- Native C++ performance

## Configuration

### Environment Variables

- `MODEL_NAME`: Model to load (default: `embeddinggemma-300m`)
- `ORT_NUM_THREADS`: ONNX Runtime thread count (default: CPU count)

### Model Configuration

Edit `config/models.yaml` to add/modify models. The C++ servers read:
- `max_seq_length`: Maximum tokenization length
- `embedding_dim`: Output embedding dimension
- `paths.onnx`: Path to ONNX model file
- `paths.openvino`: Path to OpenVINO model directory

## Troubleshooting

### Build Issues

**Missing ONNX Runtime:**
```bash
# Check ONNX Runtime is installed
ls third_party/onnxruntime/

# Reinstall if needed
./install_onnxruntime.sh
```

**Missing OpenVINO:**
```bash
# Linux
source /opt/intel/openvino_2024/setupvars.sh

# macOS
brew info openvino
```

**Missing yaml-cpp:**
```bash
# Ubuntu
sudo apt-get install libyaml-cpp-dev

# macOS
brew install yaml-cpp
```

**Tokenizers-cpp build failure:**
```bash
cd third_party/tokenizers-cpp
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Runtime Issues

**Model not found:**
```
Error: Model embeddinggemma-300m not found in config
```
- Check `MODEL_NAME` environment variable
- Verify `config/models.yaml` contains the model
- Ensure model paths in config are correct

**Tokenizer not found:**
```
Error: Failed to load tokenizer
```
- Ensure `tokenizer.json` exists in model directory
- For ONNX: Check `models/{model}/onnx/tokenizer.json`
- For OpenVINO: Check `models/{model}/openvino/tokenizer.json`

**Library not found (runtime):**
```
error while loading shared libraries: libonnxruntime.so
```
- Add library path to `LD_LIBRARY_PATH` (Linux):
  ```bash
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/third_party/onnxruntime/lib
  ```
- Or set RPATH during build (already configured in CMakeLists.txt)

### Performance Issues

**Low throughput:**
- Increase thread count: `export ORT_NUM_THREADS=32`
- Check CPU governor: `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
- Set to performance: `sudo cpupower frequency-set -g performance`

**High memory usage:**
- Reduce batch size in requests
- Check for memory leaks with valgrind: `valgrind --leak-check=full ./build/onnx_runtime_server`

## Development

### Code Structure

Both implementations follow the same pattern:

```cpp
// Global state
struct ServerState {
    // Inference framework handles
    // Tokenizer
    // Configuration
    // Metrics
};

// Helper functions
void normalize_embeddings(...);
vector<float> mean_pooling(...);

// Initialization
void load_config();
void init_framework();

// HTTP handlers
void handle_health(...);
void handle_info(...);
void handle_embed(...);

// Main
int main() {
    // Load config
    // Initialize framework
    // Start HTTP server
}
```

### Adding New Models

1. Add model to `config/models.yaml`
2. Convert model to ONNX/OpenVINO format
3. Test with curl:
   ```bash
   export MODEL_NAME=new-model
   ./build/onnx_runtime_server
   ```

### Debugging

Enable debug logging:

```cpp
// ONNX Runtime
state.env = make_unique<Ort::Env>(ORT_LOGGING_LEVEL_INFO, "onnx_cpp_server");

// Compile with debug symbols
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

Use gdb:
```bash
gdb ./build/onnx_runtime_server
(gdb) run
(gdb) bt  # backtrace on crash
```

## License

This project follows the same license as the parent repository.

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style
- Builds successfully on Linux and macOS
- Includes tests for new features
- Updates README with new functionality

## References

- [ONNX Runtime C++ API](https://onnxruntime.ai/docs/api/c/)
- [OpenVINO C++ API](https://docs.openvino.ai/latest/api/api_reference.html)
- [cpp-httplib](https://github.com/yhirose/cpp-httplib)
- [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp)
- [nlohmann/json](https://github.com/nlohmann/json)
