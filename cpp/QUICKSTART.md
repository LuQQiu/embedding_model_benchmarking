# C++ Embedding Servers - Quick Start Guide

Get the C++ implementations up and running in 5 minutes!

## Prerequisites

- Ubuntu 20.04+ or macOS 11+
- GCC 9+ or Clang 11+
- CMake 3.15+
- 8GB+ RAM

## Step 1: Setup Dependencies (5 minutes)

```bash
cd cpp/
./setup.sh
```

This installs system dependencies and third-party libraries (cpp-httplib, nlohmann/json, tokenizers-cpp).

## Step 2: Install Inference Frameworks (2-5 minutes each)

### ONNX Runtime

```bash
./install_onnxruntime.sh
```

Downloads ONNX Runtime 1.17.1 binaries (~200MB).

### OpenVINO

**Ubuntu:**
```bash
./install_openvino.sh
source /opt/intel/openvino_2024/setupvars.sh
```

**macOS:**
```bash
./install_openvino.sh  # Uses Homebrew
```

## Step 3: Build Servers (1-2 minutes each)

### ONNX Runtime

```bash
cd onnx_runtime
./build.sh
cd ..
```

### OpenVINO

```bash
cd openvino
./build.sh
cd ..
```

## Step 4: Prepare Models (one-time setup)

```bash
# From repository root
cd ..
python3 scripts/download_model.py --model embeddinggemma-300m
python3 scripts/convert_to_onnx.py --model embeddinggemma-300m
python3 scripts/convert_to_openvino.py --model embeddinggemma-300m
cd cpp/
```

## Step 5: Run Benchmark Comparison

```bash
./benchmark_comparison.py --model embeddinggemma-300m --iterations 100
```

This will:
1. Start ONNX Runtime C++ server
2. Run 100 benchmark iterations
3. Stop ONNX Runtime server
4. Start OpenVINO C++ server
5. Run 100 benchmark iterations
6. Stop OpenVINO server
7. Compare results and show winner

## Manual Testing

### ONNX Runtime

Terminal 1:
```bash
cd onnx_runtime
export MODEL_NAME=embeddinggemma-300m
./build/onnx_runtime_server
```

Terminal 2:
```bash
# Health check
curl http://localhost:8000/health

# Generate embeddings
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "Test sentence"]}'
```

### OpenVINO

Terminal 1:
```bash
cd openvino

# Linux only - source OpenVINO environment
source /opt/intel/openvino_2024/setupvars.sh

export MODEL_NAME=embeddinggemma-300m
./build/openvino_server
```

Terminal 2:
```bash
# Health check
curl http://localhost:8000/health

# Generate embeddings
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "Test sentence"]}'
```

## Expected Performance

On c7i.8xlarge (32 vCPU, 64GB RAM) with EmbeddingGemma 300M:

| Metric          | ONNX Runtime C++ | OpenVINO C++ | Winner        |
|-----------------|------------------|--------------|---------------|
| P95 Latency     | ~12ms            | ~10ms        | OpenVINO -17% |
| Throughput      | ~520 QPS         | ~650 QPS     | OpenVINO +25% |
| Memory          | ~450MB           | ~480MB       | ONNX Runtime  |
| Model Load Time | ~1200ms          | ~1500ms      | ONNX Runtime  |

**Summary:** OpenVINO C++ is typically **15-25% faster** than ONNX Runtime C++ due to Intel-optimized kernels.

## Troubleshooting

### Build fails with "ONNX Runtime not found"

```bash
ls third_party/onnxruntime/
# If empty:
./install_onnxruntime.sh
```

### Build fails with "OpenVINO not found"

**Linux:**
```bash
source /opt/intel/openvino_2024/setupvars.sh
cd openvino && ./build.sh
```

**macOS:**
```bash
brew info openvino
# If not installed:
./install_openvino.sh
```

### Runtime error: "Model not found"

```bash
# Check model exists
ls ../models/embeddinggemma-300m/onnx/model.onnx
ls ../models/embeddinggemma-300m/openvino/model.xml

# If missing, convert models:
cd ..
python3 scripts/convert_to_onnx.py --model embeddinggemma-300m
python3 scripts/convert_to_openvino.py --model embeddinggemma-300m
```

### Runtime error: "Library not found"

**Linux:**
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/third_party/onnxruntime/lib
```

**macOS:**
```bash
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$(pwd)/third_party/onnxruntime/lib
```

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Customize models in `../config/models.yaml`
- Adjust benchmark parameters: `./benchmark_comparison.py --help`
- Integrate with your application using the REST API

## Support

For issues or questions:
- Check [README.md](README.md) troubleshooting section
- Review build logs in `onnx_runtime/build/` or `openvino/build/`
- File an issue in the repository

Happy benchmarking! 🚀
