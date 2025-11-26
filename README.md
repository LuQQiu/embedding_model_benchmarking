# Embedding Model Benchmarking

Comprehensive benchmarking suite for embedding models across multiple inference frameworks on Intel CPU.

## Overview

Benchmark embedding models across 6 different frameworks:
- **PyTorch** (baseline)
- **ONNX Runtime Python**
- **ONNX Runtime Rust**
- **ONNX Runtime Native C++**
- **Candle** (Rust ML framework)
- **OpenVINO** (Intel optimized)

### Metrics

- **Latency**: p50, p95, p99, p99.9
- **Throughput**: QPS (queries per second)
- **Concurrency**: Performance at 1, 4, 16, 64, 128 concurrent requests
- **Resources**: CPU usage, memory consumption

### Models

Starting with **EmbeddingGemma 300M**, expandable to:
1. EmbeddingGemma 300M (google/embeddinggemma-300m)
2. Qwen3-Embedding 0.6B (Qwen/Qwen3-Embedding-0.6B)
3. BGE-M3 (BAAI/bge-m3)
4. CLIP (openai/clip-vit-base-patch16)
5. SigLIP (google/siglip-base-patch16-224)

## Architecture

```
┌─────────────────────────────────────────┐
│       EC2 c7i.4xlarge Instance          │
│  (Intel Xeon 8488C, 16 vCPUs, 32 GB)   │
│                                          │
│  ┌────────────────────────────────┐    │
│  │   Docker Compose Orchestrator   │    │
│  └────────────────────────────────┘    │
│              │                           │
│    ┌─────────┼─────────┐                │
│    │         │         │                │
│  ┌─▼──┐   ┌─▼──┐   ┌─▼──┐             │
│  │PyT │   │ONNX│   │Can │ (6 total)   │
│  │orch│   │Rust│   │dle │             │
│  └────┘   └────┘   └────┘             │
│                                          │
│  Shared Volumes:                        │
│    - /models (read-only)                │
│    - /results (write)                   │
│    - /config (read-only)                │
└─────────────────────────────────────────┘
```

## Quick Start

### 1. Provision Infrastructure

```bash
cd terraform

# Update variables.tf with your settings
# - AWS region
# - SSH key path
# - Allowed IP for SSH

terraform init
terraform plan
terraform apply

# Save the output
export INSTANCE_IP=$(terraform output -raw instance_public_ip)
echo "Instance IP: $INSTANCE_IP"
```

### 2. Connect to Instance

```bash
# SSH into the instance
ssh -i ~/.ssh/id_rsa ubuntu@$INSTANCE_IP

# Clone your repository (or copy files)
git clone https://github.com/yourusername/embedding_model_benchmarking.git
cd embedding_model_benchmarking
```

### 3. Setup Environment

```bash
# Run setup script
bash scripts/setup.sh

# This installs Docker, Docker Compose, and dependencies
```

### 4. Download Model

```bash
# Download EmbeddingGemma 300M
python3 scripts/download_model.py --model embeddinggemma-300m

# Model is saved to models/embeddinggemma-300m/pytorch/
```

### 5. Build Docker Containers

```bash
# Build all framework containers
docker-compose build

# Or build a specific one
docker-compose build pytorch
```

### 6. Run Benchmarks

```bash
# Run a single framework (for testing)
docker-compose run --rm pytorch

# Run all frameworks sequentially
python3 orchestrator/runner.py --model embeddinggemma-300m

# Results are saved to results/embeddinggemma-300m/
```

### 7. View Results

```bash
# Print summary
python3 orchestrator/runner.py --model embeddinggemma-300m --summary-only

# Export to CSV with timestamp (auto-generated filename)
python3 orchestrator/runner.py --model embeddinggemma-300m --summary-only --csv

# Export to CSV with custom filename
python3 orchestrator/runner.py --model embeddinggemma-300m --summary-only --csv results/my_results.csv

# View detailed JSON results
cat results/embeddinggemma-300m/pytorch.json | jq .
```

**CSV Output Format:**
- Columns: `timestamp, model, runtime, language, concurrency, qps, p50_latency_ms, p90_latency_ms, p95_latency_ms, p99_latency_ms`
- One row per framework per concurrency level
- Example: `results/embeddinggemma-300m/benchmark_results_20241126_153045.csv`

## Configuration

### Models (config/models.yaml)

Add or modify models:

```yaml
models:
  my-model:
    name: "My Custom Model"
    type: "text_embedding"
    huggingface_id: "username/model-name"
    max_seq_length: 512
    embedding_dim: 768
    # ...
```

### Benchmarks (config/benchmark.yaml)

Modify test scenarios:

```yaml
scenarios:
  - name: "my_scenario"
    concurrency: 8
    num_requests: 5000
    batch_size: 1
```

## Directory Structure

```
embedding_model_benchmarking/
├── config/
│   ├── models.yaml              # Model definitions
│   └── benchmark.yaml           # Benchmark scenarios
├── docker/
│   ├── pytorch/                 # PyTorch implementation
│   │   ├── Dockerfile
│   │   └── benchmark.py
│   ├── onnx_python/            # ONNX Python
│   ├── onnx_rust/              # ONNX Rust
│   ├── onnx_native/            # ONNX C++
│   ├── candle/                 # Candle
│   └── openvino/               # OpenVINO
├── terraform/                   # Infrastructure as code
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   └── user_data.sh
├── orchestrator/
│   └── runner.py               # Main orchestrator
├── scripts/
│   ├── setup.sh                # Environment setup
│   └── download_model.py       # Model downloader
├── models/                     # Downloaded models (gitignored)
├── results/                    # Benchmark results (gitignored)
└── docker-compose.yml          # Container orchestration
```

## Usage Examples

### Run Specific Frameworks

```bash
# PyTorch only
python3 orchestrator/runner.py --model embeddinggemma-300m --frameworks pytorch

# PyTorch and ONNX Python
python3 orchestrator/runner.py --model embeddinggemma-300m --frameworks pytorch onnx-python
```

### Continue on Error

```bash
# Don't stop if one framework fails
python3 orchestrator/runner.py --model embeddinggemma-300m --skip-on-error
```

### Export Results to CSV

```bash
# Run benchmarks and export to CSV automatically
python3 orchestrator/runner.py --model embeddinggemma-300m --csv

# Run specific frameworks and export
python3 orchestrator/runner.py --model embeddinggemma-300m --frameworks pytorch onnx-python --csv

# Export existing results to CSV (no new benchmark run)
python3 orchestrator/runner.py --model embeddinggemma-300m --summary-only --csv

# Specify custom CSV filename
python3 orchestrator/runner.py --model embeddinggemma-300m --summary-only --csv results/benchmark_$(date +%Y%m%d).csv
```

### Test Different Model

```bash
# Download model
python3 scripts/download_model.py --model qwen3-embedding-600m

# Update config/benchmark.yaml:
#   active_model: "qwen3-embedding-600m"

# Run benchmarks
python3 orchestrator/runner.py --model qwen3-embedding-600m
```

## Benchmark Results

Results are saved as JSON in `results/{model_name}/{framework}.json`:

```json
{
  "framework": "pytorch",
  "model": "EmbeddingGemma 300M",
  "model_load_time_ms": 1234.5,
  "first_inference_ms": 45.2,
  "scenarios": {
    "single_query": {
      "latency_ms": {
        "mean": 12.3,
        "median": 11.5,
        "p95": 15.2,
        "p99": 18.7
      },
      "throughput_qps": 81.3,
      "cpu_percent": 45.2,
      "memory_rss_mb": 512.3
    }
  }
}
```

## Cost Estimation

- **Instance**: c7i.4xlarge @ ~$0.68/hour
- **Storage**: 100 GB gp3 @ ~$10/month
- **Typical benchmark run**: 2-4 hours
- **Total cost per run**: ~$2-5

## Performance Tips

1. **Warmup**: Always enabled by default (100 requests)
2. **CPU Isolation**: Instance uses performance governor
3. **No Swap**: Swap disabled for consistent performance
4. **Sequential**: Frameworks run one at a time for fair comparison
5. **Resource Limits**: Docker containers limited to full instance resources

## Troubleshooting

### Docker permission denied

```bash
sudo usermod -aG docker $USER
# Then logout and login again
```

### Out of memory

```bash
# Check available memory
free -h

# Reduce num_requests in config/benchmark.yaml
# Or use smaller batch_size
```

### Model download fails

```bash
# Set HuggingFace token if needed
export HF_TOKEN=your_token_here

# Try manual download
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('google/embeddinggemma-300m')"
```

## Next Steps

1. **Implement remaining frameworks**: ONNX Rust, ONNX Native, Candle, OpenVINO
2. **Add model conversion scripts**: PyTorch → ONNX, ONNX → OpenVINO
3. **Analysis tools**: Generate comparison charts, HTML reports
4. **Multimodal support**: Extend for CLIP and SigLIP with image inputs
5. **Automation**: CI/CD pipeline for automated benchmarking

## Contributing

See [DESIGN.md](DESIGN.md) for architecture details and [ARCHITECTURE.md](ARCHITECTURE.md) for implementation approach.

## License

MIT
