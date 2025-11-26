# BGE-M3 Multi-Framework Benchmark Design

## Overview
Benchmark BGE-M3 embedding model across 6 different inference frameworks on Intel CPU to compare:
- **Latency**: Time per inference request (p50, p95, p99)
- **QPS (Throughput)**: Queries handled per second
- **Concurrency**: Performance under concurrent load

## Frameworks to Benchmark
1. **PyTorch** (baseline)
2. **Candle** (Rust ML framework)
3. **ONNX Runtime Native** (C++ binary)
4. **ONNX Runtime Rust** (ort crate)
5. **ONNX Runtime Python** (onnxruntime package)
6. **OpenVINO** (Intel optimized)

## Infrastructure Design

### AWS EC2 Instance
- **Instance Type**: `c7i.4xlarge` (Intel Xeon 4th gen, 16 vCPUs, 32 GB RAM)
  - Optimized for compute workloads
  - Consistent Intel CPU architecture
  - Network optimized (up to 12.5 Gbps)
- **OS**: Ubuntu 22.04 LTS
- **Storage**: 100 GB gp3 SSD
- **Terraform**: Automated provisioning with consistent configuration

### Directory Structure
```
embedding_model_benchmarking/
├── terraform/                    # Infrastructure as code
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   └── user_data.sh             # Instance initialization script
├── models/                       # Model files (gitignored)
│   ├── embeddinggemma-300m/     # Start here
│   │   ├── pytorch/
│   │   ├── onnx/
│   │   └── openvino/
│   ├── qwen3-embedding-600m/
│   │   ├── pytorch/
│   │   ├── onnx/
│   │   └── openvino/
│   ├── bge-m3/
│   │   ├── pytorch/
│   │   ├── onnx/
│   │   └── openvino/
│   ├── siglip/
│   │   ├── pytorch/
│   │   ├── onnx/
│   │   └── openvino/
│   └── clip/
│       ├── pytorch/
│       ├── onnx/
│       └── openvino/
├── benchmarks/
│   ├── pytorch_bench/           # PyTorch implementation
│   ├── candle_bench/            # Candle (Rust) implementation
│   ├── onnx_native_bench/       # ONNX C++ implementation
│   ├── onnx_rust_bench/         # ONNX Rust implementation
│   ├── onnx_python_bench/       # ONNX Python implementation
│   └── openvino_bench/          # OpenVINO implementation
├── harness/                      # Benchmark orchestration
│   ├── runner.py                # Main benchmark runner
│   ├── config.yaml              # Benchmark configuration
│   └── dataset.py               # Test dataset preparation
├── results/                      # Benchmark results
│   └── .gitkeep
├── scripts/
│   ├── setup_environment.sh     # Environment setup on EC2
│   ├── convert_models.sh        # Model format conversion
│   └── run_all_benchmarks.sh    # Run all benchmarks
└── analysis/
    ├── visualize.py             # Generate charts
    └── report.py                # Generate markdown report
```

## Benchmark Metrics

### Primary Metrics
1. **Latency**
   - Mean, Median (p50), p95, p99, p99.9
   - Per-request inference time
   - Measured in milliseconds

2. **Throughput (QPS)**
   - Queries processed per second
   - Measured under different concurrency levels

3. **Concurrency Performance**
   - Test with: 1, 2, 4, 8, 16, 32, 64, 128 concurrent requests
   - Measure latency degradation
   - Identify optimal concurrency level

### Secondary Metrics
- **Model Load Time**: Time to load model into memory
- **Memory Usage**: RSS and peak memory consumption
- **CPU Utilization**: Average CPU usage during inference
- **First Request Latency**: Cold start performance

## Benchmark Harness Design

### Test Configuration
```yaml
# models.yaml - Model configurations
models:
  # Text Embedding Models
  embeddinggemma-300m:
    name: "EmbeddingGemma 300M"
    type: "text_embedding"
    huggingface_id: "google/embeddinggemma-300m"
    max_seq_length: 8192
    embedding_dim: 256  # Supports MRL (can truncate)
    params: "308M"
    multilingual: true

  qwen3-embedding-600m:
    name: "Qwen3 Embedding 0.6B"
    type: "text_embedding"
    huggingface_id: "Qwen/Qwen3-Embedding-0.6B"
    max_seq_length: 8192
    embedding_dim: 768
    params: "600M"
    multilingual: true

  bge-m3:
    name: "BGE-M3"
    type: "text_embedding"
    huggingface_id: "BAAI/bge-m3"
    max_seq_length: 8192
    embedding_dim: 1024
    params: "600M"
    multilingual: true

  # Multimodal Embedding Models
  siglip:
    name: "SigLIP Base"
    type: "multimodal"
    huggingface_id: "google/siglip-base-patch16-224"
    max_seq_length: 64
    embedding_dim: 768
    image_size: 224
    params: "~400M"
    modalities: ["text", "image"]

  clip:
    name: "OpenAI CLIP ViT-B/16"
    type: "multimodal"
    huggingface_id: "openai/clip-vit-base-patch16"
    max_seq_length: 77
    embedding_dim: 512
    image_size: 224
    params: "~150M"
    modalities: ["text", "image"]

# benchmark_config.yaml - Benchmark settings
active_models:
  - "embeddinggemma-300m"
  - "qwen3-embedding-600m"
  - "bge-m3"
  - "siglip"
  - "clip"

test_scenarios:
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

  - name: "batch_processing"
    concurrency: 1
    num_requests: 500
    batch_size: 8

input_config:
  sequence_lengths: [32, 128, 512]  # Test different input sizes
  warmup_requests: 100
```

### Test Dataset
- Use a fixed set of sentences from a standard corpus (e.g., MS MARCO)
- Different sequence lengths to test performance across input sizes
- Consistent across all frameworks for fair comparison

### Benchmark Protocol
1. **Setup Phase**
   - Load model
   - Measure load time and memory

2. **Warmup Phase**
   - Run 100 inference requests
   - Discard results (JIT compilation, cache warming)

3. **Measurement Phase**
   - Execute configured number of requests
   - Collect timing data for each request
   - Monitor system resources

4. **Teardown Phase**
   - Clean up resources
   - Save results to JSON

## Implementation Architecture

### Standard API Interface
Each framework implementation exposes the same interface:

```python
# Common interface for all benchmarks
class EmbeddingModel:
    def load_model(self) -> float:
        """Load model, return load time in seconds"""
        pass

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for input texts"""
        pass

    def get_memory_usage(self) -> int:
        """Return current memory usage in bytes"""
        pass
```

### HTTP Server Option (for Candle/Rust/C++)
For native implementations, wrap in a simple HTTP server:
- Accepts POST requests with JSON payload
- Returns embeddings as JSON
- Easier to benchmark with standard load testing tools

### Load Testing Tools
- **wrk2** or **oha**: For HTTP-based benchmarks (C++ latency-focused)
- **Python asyncio**: For direct library benchmarks
- **Custom harness**: Orchestrates all tests with consistent methodology

## Model Conversion Pipeline

```bash
# Generic model conversion script
# Usage: ./scripts/convert_models.sh <model_config_key>

# 1. Download PyTorch model (from HuggingFace)
python scripts/download_model.py \
  --config models.yaml \
  --model bge-m3

# 2. Export to ONNX
python scripts/export_to_onnx.py \
  --config models.yaml \
  --model bge-m3 \
  --output models/bge-m3/onnx/model.onnx \
  --opset 14

# 3. Convert to OpenVINO IR
mo --input_model models/bge-m3/onnx/model.onnx \
   --output_dir models/bge-m3/openvino/ \
   --compress_to_fp16

# 4. Optimize ONNX model (optional)
python -m onnxruntime.quantization.preprocess \
  --input models/bge-m3/onnx/model.onnx \
  --output models/bge-m3/onnx/model_optimized.onnx

# Convert all models at once
./scripts/convert_all_models.sh
```

## Execution Plan

### Phase 1: Infrastructure Setup (1-2 hours)
1. Write Terraform configuration
2. Provision EC2 instance
3. Install dependencies (Docker, build tools, Python, Rust, etc.)
4. Download and convert models

### Phase 2: Implementation (1-2 days)
1. Implement PyTorch benchmark (baseline)
2. Implement ONNX Python benchmark
3. Implement ONNX Rust benchmark
4. Implement Candle benchmark
5. Implement OpenVINO benchmark
6. Implement ONNX Native C++ benchmark

### Phase 3: Benchmark Execution (2-4 hours)
1. Run warmup tests
2. Execute full benchmark suite
3. Collect results

### Phase 4: Analysis (1-2 hours)
1. Generate visualization charts
2. Create comparison report
3. Identify best performer for different scenarios

## Expected Outcomes

### Performance Hypotheses
1. **OpenVINO**: Likely fastest on Intel CPU (optimized for Intel hardware)
2. **ONNX Runtime**: Good balance of speed and compatibility
3. **PyTorch**: Slowest but most flexible (baseline)
4. **Candle**: Competitive with ONNX, better than PyTorch
5. **Rust implementations**: Better resource efficiency

### Deliverables
1. Automated infrastructure provisioning
2. Benchmark implementations for all 6 frameworks
3. Comprehensive performance report with:
   - Latency comparison charts
   - QPS vs Concurrency graphs
   - Memory usage comparison
   - Resource utilization heatmaps
4. Recommendations for production deployment

## Cost Estimation
- **c7i.4xlarge**: ~$0.68/hour
- **Storage**: ~$10/month for 100 GB
- **Data transfer**: Minimal (< $1)
- **Total for 8-hour benchmark session**: ~$6-10

## Quick Start Commands

```bash
# 1. Provision infrastructure
cd terraform && terraform apply

# 2. SSH into instance
ssh -i ~/.ssh/benchmark_key.pem ubuntu@<instance-ip>

# 3. Run setup
./scripts/setup_environment.sh

# 4. Convert models
./scripts/convert_models.sh

# 5. Run benchmarks
./scripts/run_all_benchmarks.sh

# 6. Generate report
python analysis/report.py --results results/
```

## Adding New Models

To benchmark a new model, simply:

1. **Add model configuration** to `models.yaml`:
   ```yaml
   my-new-model:
     name: "vendor/model-name"
     type: "embedding"
     max_seq_length: 512
     embedding_dim: 768
     huggingface_id: "vendor/model-name"
   ```

2. **Update active model** in `benchmark_config.yaml`:
   ```yaml
   active_model: "my-new-model"
   ```

3. **Run conversion and benchmarks**:
   ```bash
   ./scripts/convert_models.sh my-new-model
   ./scripts/run_all_benchmarks.sh
   ```

All framework implementations automatically adapt to the configured model!

## Next Steps

1. Review this design and provide feedback
2. Implement Terraform configuration
3. Create benchmark harness
4. Implement framework-specific benchmarks
5. Execute and analyze results
