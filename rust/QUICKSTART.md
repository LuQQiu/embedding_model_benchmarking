# Quick Start Guide - Rust Benchmarks

Get the Rust ONNX Runtime benchmark running in under 30 minutes.

## Prerequisites

- AWS account
- AWS CLI configured (`aws configure`)
- SSH key in AWS (note the key name)
- Terraform installed
- Your IP address (for SSH access)

## Step-by-Step

### 1. Deploy Infrastructure (5 minutes)

```bash
cd rust/terraform

# Edit variables.tf:
# - Set your AWS region (default: us-west-2)
# - Set your SSH key name (default: id_rsa)
# - Set your IP for SSH access (default: 0.0.0.0/0 - CHANGE THIS!)

# Deploy
terraform init
terraform apply

# Save the IP
export RUST_IP=$(terraform output -raw instance_public_ip)
echo "Instance: $RUST_IP"
```

### 2. Wait for Instance Setup (5-10 minutes)

The instance will auto-install Docker, Rust, and dependencies.

```bash
# Wait for instance to be ready
# Try connecting (will fail until setup completes):
ssh -i ~/.ssh/id_rsa ubuntu@$RUST_IP

# Once connected, check setup:
docker --version
rustc --version
```

### 3. Copy Project Files (2 minutes)

From your local machine:

```bash
# Option A: Clone from Git
ssh -i ~/.ssh/id_rsa ubuntu@$RUST_IP
git clone https://github.com/yourusername/embedding_model_benchmarking.git
cd embedding_model_benchmarking/rust

# Option B: Copy from local
# From your local repo root:
scp -i ~/.ssh/id_rsa -r rust ubuntu@$RUST_IP:~/embedding_model_benchmarking/
```

### 4. Copy Models and Config (Varies)

You need the model files and config from the Python benchmarks:

```bash
# Option A: Copy from Python instance
# If you already have models on the Python instance:
export PYTHON_IP=<your-python-instance-ip>

ssh -i ~/.ssh/id_rsa ubuntu@$RUST_IP
rsync -av ubuntu@$PYTHON_IP:~/embedding_model_benchmarking/models/ \
    ~/embedding_model_benchmarking/models/
rsync -av ubuntu@$PYTHON_IP:~/embedding_model_benchmarking/config/ \
    ~/embedding_model_benchmarking/config/

# Option B: Download directly
cd ~/embedding_model_benchmarking
python3 scripts/download_model.py --model embeddinggemma-300m
python3 scripts/convert_to_onnx.py --model embeddinggemma-300m
```

### 5. Build Docker Image (10-15 minutes)

```bash
cd ~/embedding_model_benchmarking/rust
docker compose build
```

### 6. Run Benchmark (5-10 minutes)

```bash
# Quick test
./scripts/run_benchmark.sh embeddinggemma-300m

# Or manually:
docker compose up -d onnx-rust-server
docker compose run --rm -e MODEL_NAME=embeddinggemma-300m onnx-rust-client
docker compose down
```

### 7. View Results

```bash
# Summary
cat ../results/embeddinggemma-300m/onnx-rust.json | jq .scenarios.concurrency_1

# Full results
cat ../results/embeddinggemma-300m/onnx-rust.json | jq .
```

## Expected Output

```json
{
  "framework": "onnx-rust",
  "model_name": "embeddinggemma-300m",
  "model_load_time_ms": 2340.5,
  "first_inference_ms": 15.2,
  "scenarios": {
    "concurrency_1": {
      "latency_ms": {
        "mean": 22.5,
        "median": 21.8,
        "p95": 25.3,
        "p99": 28.1
      },
      "throughput_qps": 44.4,
      "total_duration_sec": 22.5,
      "errors": 0,
      "error_rate": 0.0
    }
  }
}
```

## Troubleshooting

### Instance won't SSH
- Check security group allows your IP
- Wait 5 minutes for user_data script to complete

### Docker permission denied
```bash
# Re-login (user_data adds ubuntu to docker group)
exit
ssh -i ~/.ssh/id_rsa ubuntu@$RUST_IP
```

### Model not found
```bash
# Check path
ls -la ~/embedding_model_benchmarking/models/embeddinggemma-300m/onnx/
# Should see: model.onnx, tokenizer.json, ...
```

### Build fails
```bash
# Try clean build
docker compose build --no-cache onnx-rust-server
```

## Cleanup

```bash
# On your local machine:
cd rust/terraform
terraform destroy

# Confirm with: yes
```

Cost: ~$0.70/hour for c7i.4xlarge. Total for quickstart: ~$2-3.

## Next Steps

- Compare with Python benchmarks
- Add Candle implementation
- Implement Rust benchmark client
- Automate comparison analysis

See README.md for full details.
