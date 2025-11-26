# Benchmark Architecture: Multi-Language Framework Orchestration

## Problem
How to benchmark the same model across 6 different frameworks in different languages:
- **Python**: PyTorch, ONNX Runtime Python
- **Rust**: Candle, ONNX Runtime Rust (ort crate)
- **C++**: ONNX Runtime Native
- **C++**: OpenVINO

## Solution: Docker + YAML Configuration

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Orchestrator (Python)                      │
│  - Reads YAML configs                                        │
│  - Spawns Docker containers                                  │
│  - Collects results                                          │
│  - Generates reports                                         │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴──────────────┐
                │                            │
        ┌───────▼────────┐          ┌───────▼────────┐
        │ models.yaml    │          │ benchmark.yaml │
        │ - Model configs│          │ - Test scenarios│
        │ - HF IDs       │          │ - Concurrency   │
        └────────────────┘          └─────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐   ┌────────▼────────┐   ┌──────▼─────────┐
│ PyTorch        │   │ ONNX Runtime    │   │ Candle         │
│ Container      │   │ Python Container│   │ Container      │
│                │   │                 │   │                │
│ Python 3.11    │   │ Python 3.11     │   │ Rust 1.75      │
│ transformers   │   │ onnxruntime     │   │ candle-core    │
│ sentence-tr... │   │ numpy           │   │ tokenizers     │
└────────────────┘   └─────────────────┘   └────────────────┘

┌────────────────┐   ┌─────────────────┐   ┌────────────────┐
│ ONNX Runtime   │   │ ONNX Runtime    │   │ OpenVINO       │
│ Rust Container │   │ C++ Container   │   │ Container      │
│                │   │                 │   │                │
│ Rust 1.75      │   │ Ubuntu + g++    │   │ Ubuntu 22.04   │
│ ort crate      │   │ onnxruntime lib │   │ openvino       │
│ tokenizers     │   │ HTTP server     │   │ Python API     │
└────────────────┘   └─────────────────┘   └────────────────┘
```

## Directory Structure

```
embedding_model_benchmarking/
├── config/
│   ├── models.yaml              # Model definitions
│   └── benchmark.yaml           # Benchmark scenarios
│
├── docker/
│   ├── pytorch/
│   │   ├── Dockerfile
│   │   └── benchmark.py         # Standardized interface
│   ├── onnx_python/
│   │   ├── Dockerfile
│   │   └── benchmark.py
│   ├── onnx_rust/
│   │   ├── Dockerfile
│   │   ├── Cargo.toml
│   │   └── src/main.rs
│   ├── onnx_native/
│   │   ├── Dockerfile
│   │   ├── CMakeLists.txt
│   │   └── src/benchmark.cpp
│   ├── candle/
│   │   ├── Dockerfile
│   │   ├── Cargo.toml
│   │   └── src/main.rs
│   └── openvino/
│       ├── Dockerfile
│       └── benchmark.py
│
├── docker-compose.yml           # Orchestrate all containers
│
├── orchestrator/
│   ├── runner.py                # Main orchestrator
│   ├── docker_manager.py        # Container management
│   └── results_collector.py    # Aggregate results
│
├── models/                      # Shared volume (mounted in containers)
│   └── embeddinggemma-300m/
│       ├── pytorch/
│       ├── onnx/
│       └── openvino/
│
├── results/                     # Shared volume (results written here)
│   └── embeddinggemma-300m/
│       ├── pytorch.json
│       ├── onnx_python.json
│       ├── onnx_rust.json
│       ├── onnx_native.json
│       ├── candle.json
│       └── openvino.json
│
└── scripts/
    ├── setup.sh                 # One-time setup
    └── convert_models.sh        # Model conversion
```

## Configuration Files

### config/models.yaml
```yaml
models:
  embeddinggemma-300m:
    name: "EmbeddingGemma 300M"
    huggingface_id: "google/embeddinggemma-300m"
    type: "text_embedding"
    max_seq_length: 8192
    embedding_dim: 256

    # Framework-specific paths (relative to /models in container)
    paths:
      pytorch: "/models/embeddinggemma-300m/pytorch"
      onnx: "/models/embeddinggemma-300m/onnx/model.onnx"
      openvino: "/models/embeddinggemma-300m/openvino"
```

### config/benchmark.yaml
```yaml
test_config:
  warmup_requests: 100
  dataset: "ms_marco"
  sequence_lengths: [32, 128, 512]

scenarios:
  - name: "single_query"
    concurrency: 1
    num_requests: 1000
    batch_size: 1

  - name: "low_concurrency"
    concurrency: 4
    num_requests: 5000
    batch_size: 1

  - name: "medium_concurrency"
    concurrency: 16
    num_requests: 10000
    batch_size: 1

  - name: "high_concurrency"
    concurrency: 64
    num_requests: 20000
    batch_size: 1
```

### docker-compose.yml
```yaml
version: '3.8'

services:
  pytorch:
    build: ./docker/pytorch
    volumes:
      - ./models:/models:ro
      - ./results:/results
      - ./config:/config:ro
    environment:
      - MODEL_NAME=embeddinggemma-300m
    command: python benchmark.py --config /config/benchmark.yaml --model /config/models.yaml

  onnx-python:
    build: ./docker/onnx_python
    volumes:
      - ./models:/models:ro
      - ./results:/results
      - ./config:/config:ro
    environment:
      - MODEL_NAME=embeddinggemma-300m
    command: python benchmark.py --config /config/benchmark.yaml --model /config/models.yaml

  onnx-rust:
    build: ./docker/onnx_rust
    volumes:
      - ./models:/models:ro
      - ./results:/results
      - ./config:/config:ro
    environment:
      - MODEL_NAME=embeddinggemma-300m
    command: /app/benchmark --config /config/benchmark.yaml --model /config/models.yaml

  onnx-native:
    build: ./docker/onnx_native
    volumes:
      - ./models:/models:ro
      - ./results:/results
      - ./config:/config:ro
    environment:
      - MODEL_NAME=embeddinggemma-300m
    command: /app/benchmark --config /config/benchmark.yaml --model /config/models.yaml

  candle:
    build: ./docker/candle
    volumes:
      - ./models:/models:ro
      - ./results:/results
      - ./config:/config:ro
    environment:
      - MODEL_NAME=embeddinggemma-300m
    command: /app/benchmark --config /config/benchmark.yaml --model /config/models.yaml

  openvino:
    build: ./docker/openvino
    volumes:
      - ./models:/models:ro
      - ./results:/results
      - ./config:/config:ro
    environment:
      - MODEL_NAME=embeddinggemma-300m
    command: python benchmark.py --config /config/benchmark.yaml --model /config/models.yaml
```

## Standardized Output Format

All benchmarks output to `/results/{model_name}/{framework}.json`:

```json
{
  "framework": "pytorch",
  "model": "embeddinggemma-300m",
  "timestamp": "2025-11-25T10:00:00Z",
  "system_info": {
    "cpu": "Intel Xeon 8488C",
    "cores": 16,
    "memory_gb": 32
  },
  "model_load_time_ms": 1234.5,
  "peak_memory_mb": 512,
  "scenarios": {
    "single_query": {
      "concurrency": 1,
      "num_requests": 1000,
      "latency_ms": {
        "mean": 12.3,
        "p50": 11.5,
        "p95": 15.2,
        "p99": 18.7,
        "p999": 23.4
      },
      "throughput_qps": 81.3,
      "cpu_utilization_pct": 45.2
    },
    "low_concurrency": {
      // ...
    }
  }
}
```

## Orchestrator Implementation

### orchestrator/runner.py
```python
#!/usr/bin/env python3
import subprocess
import yaml
import json
from pathlib import Path

class BenchmarkRunner:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.frameworks = [
            "pytorch",
            "onnx-python",
            "onnx-rust",
            "onnx-native",
            "candle",
            "openvino"
        ]

    def run_all_benchmarks(self):
        """Run all framework benchmarks sequentially"""
        for framework in self.frameworks:
            print(f"Running {framework} benchmark...")
            self.run_single_benchmark(framework)

    def run_single_benchmark(self, framework: str):
        """Run single framework benchmark using docker-compose"""
        cmd = [
            "docker-compose",
            "run",
            "--rm",
            framework
        ]
        subprocess.run(cmd, check=True)

    def collect_results(self):
        """Aggregate all results"""
        results = {}
        results_dir = Path("results") / self.model_name

        for framework in self.frameworks:
            result_file = results_dir / f"{framework}.json"
            if result_file.exists():
                with open(result_file) as f:
                    results[framework] = json.load(f)

        return results

if __name__ == "__main__":
    runner = BenchmarkRunner("embeddinggemma-300m")
    runner.run_all_benchmarks()
    results = runner.collect_results()

    # Generate comparison report
    print(json.dumps(results, indent=2))
```

## Benefits of This Architecture

### ✅ Clean Separation
- Each framework in its own container
- No dependency conflicts
- Easy to debug (run one container at a time)

### ✅ Easy Configuration
- Change models: Edit `models.yaml`
- Change test scenarios: Edit `benchmark.yaml`
- No code changes needed

### ✅ Language Agnostic
- Python, Rust, C++ all work the same way
- Standardized input (YAML) and output (JSON)
- Framework-specific optimizations possible

### ✅ Reproducible
- Docker ensures same environment every time
- Version-pinned dependencies
- Can share exact setup with others

### ✅ Scalable
- Easy to add new frameworks (new Dockerfile + docker-compose entry)
- Easy to add new models (update YAML)
- Can run containers in parallel if needed

### ✅ Production-Ready
- Same approach works on EC2, locally, or CI/CD
- Easy to automate
- Results in standard JSON format

## Alternative Approaches

### ❌ Option 1: Single Environment
Install all frameworks on bare metal:
- **Cons**: Dependency hell, conflicts, hard to reproduce
- **When to use**: Never for this use case

### ❌ Option 2: Individual VMs
Separate VM per framework:
- **Cons**: Expensive, slow to provision, hard to manage
- **When to use**: If strict isolation required

### ⚠️ Option 3: Conda/Virtual Envs
Use Python venvs for isolation:
- **Cons**: Doesn't help with Rust/C++, still dependency issues
- **When to use**: Python-only benchmarks

## Execution Workflow

```bash
# 1. One-time setup
./scripts/setup.sh

# 2. Convert models to all formats
./scripts/convert_models.sh embeddinggemma-300m

# 3. Run ALL benchmarks (sequential)
python orchestrator/runner.py --model embeddinggemma-300m

# OR run single framework for debugging
docker-compose run --rm pytorch

# 4. Generate report
python analysis/report.py --results results/embeddinggemma-300m/

# 5. To test a different model
# - Update config/models.yaml
# - Run: ./scripts/convert_models.sh qwen3-embedding-600m
# - Run: python orchestrator/runner.py --model qwen3-embedding-600m
```

## Summary

**Best Approach**: Docker + YAML

- **YAML** for all configuration (models, benchmarks, scenarios)
- **Docker** for framework isolation (one container per framework)
- **Docker Compose** for orchestration
- **Python** orchestrator to run everything and collect results
- **Shared volumes** for models and results

This gives you:
- ✅ Easy to modify (just edit YAML)
- ✅ Clean environments (Docker isolation)
- ✅ Multi-language support (Python/Rust/C++)
- ✅ Reproducible results
- ✅ Simple to debug (run one container at a time)
- ✅ Scalable (add models/frameworks easily)
