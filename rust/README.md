# Rust Embedding Model Benchmarking

Rust implementation of embedding model benchmarking infrastructure with ONNX Runtime.

## Overview

This directory contains a complete Rust-based benchmarking system that runs on a separate EC2 instance from the Python benchmarks.

### Components

1. **ONNX Runtime Rust Server** - High-performance inference server using `ort` crate
2. **Common Library** - Shared types, configuration, and statistics
3. **Benchmark Client** - Uses existing Python client (Rust client TBD)
4. **Infrastructure** - Separate Terraform configuration for dedicated Rust instance

### Architecture

```
rust/
├── common/              # Shared library (types, config, stats)
├── onnx_server/        # ONNX Runtime server (ort crate)
├── benchmark_client/   # Rust client (placeholder - uses Python for now)
├── docker/             # Dockerfiles
├── terraform/          # Infrastructure as code
├── docker-compose.yml  # Container orchestration
└── scripts/            # Deployment and utility scripts
```

## Quick Start

### Prerequisites

- AWS account with credentials configured
- Terraform >= 1.0
- SSH key pair created in AWS
- Models and config directories (copy from parent project)

### 1. Provision Infrastructure

```bash
cd terraform

# Update variables.tf with your settings:
# - AWS region
# - SSH key name
# - Allowed IP for SSH

terraform init
terraform plan
terraform apply

# Save the output
export RUST_INSTANCE_IP=$(terraform output -raw instance_public_ip)
echo "Rust Instance IP: $RUST_INSTANCE_IP"
```

### 2. Connect to Instance

```bash
# SSH into the instance
ssh -i ~/.ssh/id_rsa ubuntu@$RUST_INSTANCE_IP

# Clone repository (or copy files)
git clone https://github.com/yourusername/embedding_model_benchmarking.git
cd embedding_model_benchmarking/rust
```

### 3. Copy Models and Config

From your local machine:

```bash
# Copy models (from parent Python benchmarks)
scp -i ~/.ssh/id_rsa -r ../models ubuntu@$RUST_INSTANCE_IP:~/embedding_model_benchmarking/
scp -i ~/.ssh/id_rsa -r ../config ubuntu@$RUST_INSTANCE_IP:~/embedding_model_benchmarking/
```

Or download directly on the instance:

```bash
# On the instance
cd ~/embedding_model_benchmarking
python3 ../scripts/download_model.py --model embeddinggemma-300m
python3 ../scripts/convert_to_onnx.py --model embeddinggemma-300m
```

### 4. Build Docker Containers

```bash
cd ~/embedding_model_benchmarking/rust
docker compose build
```

This will:
- Build the ONNX Rust server (multi-stage build)
- Compile Rust code in release mode
- Include ONNX Runtime libraries
- Take ~10-15 minutes on first build

### 5. Run Benchmarks

```bash
# Test server startup
docker compose up onnx-rust-server

# In another terminal, check health
curl http://localhost:8000/health

# Run full benchmark
docker compose run --rm -e MODEL_NAME=embeddinggemma-300m onnx-rust-client

# Results saved to ../results/embeddinggemma-300m/onnx-rust.json
```

## Development

### Local Rust Development

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build workspace
cd rust
cargo build --release

# Run tests
cargo test

# Run server locally (needs ONNX Runtime libraries)
export ORT_DYLIB_PATH=/path/to/onnxruntime/lib
cargo run -p onnx_server
```

### Docker Development

```bash
# Build only the server
docker compose build onnx-rust-server

# Run server with logs
docker compose up onnx-rust-server

# Execute client
docker compose run --rm onnx-rust-client
```

## Configuration

### Models

Models are configured in `../config/models.yaml` (shared with Python implementation):

```yaml
models:
  embeddinggemma-300m:
    name: "EmbeddingGemma 300M"
    paths:
      pytorch: "/models/embeddinggemma-300m/pytorch"
      onnx: "/models/embeddinggemma-300m/onnx/model.onnx"
```

### Benchmarks

Benchmark scenarios in `../config/benchmark.yaml`:

```yaml
scenarios:
  - name: "concurrency_1"
    concurrency: 1
    num_requests: 1000
  - name: "concurrency_16"
    concurrency: 16
    num_requests: 12000
```

## Performance Expectations

### vs Python ONNX Runtime

- **Latency**: ~5-15% lower (less overhead)
- **Throughput**: ~10-20% higher QPS
- **Memory**: ~50-100MB less (no Python interpreter)
- **Startup**: ~2x faster model loading

### Instance Resources

- **CPU**: c7i.4xlarge (16 vCPUs, 32 GB RAM)
- **Server**: 16 vCPUs, 32GB limit
- **Client**: 14 vCPUs, 16GB limit

## Troubleshooting

### Build fails with ONNX Runtime errors

```bash
# Check ONNX Runtime download
docker compose build --no-cache onnx-rust-server
```

### Server won't start

```bash
# Check logs
docker compose logs onnx-rust-server

# Verify model path
docker compose run --rm onnx-rust-server ls -la /models/
```

### Connection refused

```bash
# Check server is running
docker compose ps

# Check health endpoint
curl http://localhost:8000/health
```

## Comparison with Python Implementation

| Feature | Python | Rust |
|---------|--------|------|
| Runtime | ONNX Runtime Python | ONNX Runtime (ort crate) |
| Language | Python 3.11 | Rust 1.82 |
| Web Framework | FastAPI | Axum |
| Async Runtime | uvicorn | Tokio |
| Memory Usage | ~2-3 GB | ~1.5-2 GB |
| Startup Time | ~5-10s | ~2-5s |
| Latency | Baseline | 5-15% lower |
| Throughput | Baseline | 10-20% higher |

## Cost Estimation

- **Instance**: c7i.4xlarge @ ~$0.68/hour (US West 2)
- **Storage**: 100 GB gp3 @ ~$10/month
- **Typical benchmark run**: 1-2 hours
- **Total cost per run**: ~$1-2

## Next Steps

1. **Implement Rust benchmark client** - Replace Python client with native Rust
2. **Add Candle server** - Pure Rust ML framework implementation
3. **Optimize ONNX Runtime** - Tune session options, graph optimizations
4. **Automated deployment** - CI/CD pipeline for benchmarking
5. **Comparison analysis** - Automated Python vs Rust comparison reports

## Contributing

See parent project's ARCHITECTURE.md for overall design.

Rust-specific design decisions:
- **ort crate**: Official ONNX Runtime bindings
- **Axum**: Fast, ergonomic web framework
- **Tokio**: De facto async runtime
- **Multi-stage builds**: Smaller final images
- **Shared configs**: Reuse Python config files

## License

MIT
