# Quick Start Guide

Fast setup guide for AWS EC2 instances.

## One-Command Setup

SSH into your EC2 instance and run:

```bash
git clone https://github.com/LuQQiu/embedding_model_benchmarking.git
cd embedding_model_benchmarking
bash scripts/setup.sh
```

This single script will:
- ✅ Install all Python dependencies (PyTorch, transformers, sentence-transformers, etc.)
- ✅ Install HuggingFace CLI
- ✅ Configure PATH for command-line tools
- ✅ Pull Docker base images
- ✅ Install Claude Code CLI
- ✅ Verify all installations

**Time**: ~5-10 minutes

## After Setup

### 1. Authenticate with HuggingFace (for gated models)

```bash
bash scripts/huggingface_login.sh
```

Or manually:
```bash
huggingface-cli login
# Paste your token when prompted
```

Get your token from: https://huggingface.co/settings/tokens

### 2. Download Model

```bash
python3 scripts/download_model.py --model embeddinggemma-300m
```

**Available models:**
- `embeddinggemma-300m` - Google's efficient 300M model (requires auth)
- `qwen3-embedding-600m` - Qwen's 600M multilingual model
- `bge-m3` - BAAI's multilingual 600M model
- `bge-small-en` - BAAI's small 33M English model (public, no auth)

### 3. Build Docker Containers

```bash
docker-compose build
```

**Available frameworks:**
- `pytorch` - PyTorch baseline
- `onnx-python` - ONNX Runtime (Python)
- `openvino` - Intel OpenVINO (optimized for Intel CPUs)

### 4. Run Benchmarks

**Single framework:**
```bash
docker-compose run --rm pytorch
```

**All frameworks with CSV export:**
```bash
python3 orchestrator/runner.py --model embeddinggemma-300m --csv
```

**Specific frameworks:**
```bash
python3 orchestrator/runner.py --model embeddinggemma-300m --frameworks pytorch onnx-python --csv
```

### 5. View Results

**Terminal summary:**
```bash
python3 orchestrator/runner.py --model embeddinggemma-300m --summary-only
```

**CSV export:**
```bash
# Auto-generated filename with timestamp
python3 orchestrator/runner.py --model embeddinggemma-300m --summary-only --csv

# Custom filename
python3 orchestrator/runner.py --model embeddinggemma-300m --summary-only --csv results/my_results.csv
```

**View CSV:**
```bash
cat results/embeddinggemma-300m/benchmark_results_*.csv | column -t -s,
```

**JSON results:**
```bash
cat results/embeddinggemma-300m/pytorch.json | jq .
```

## Troubleshooting

### "command not found" errors

After setup, refresh your shell:
```bash
source ~/.bashrc
# Or logout and login again
```

### Docker permission denied

```bash
sudo usermod -aG docker $USER
# Logout and login again
```

### Out of memory

Reduce requests in `config/benchmark.yaml`:
```yaml
scenarios:
  - name: "concurrency_1"
    num_requests: 500  # Reduce from 1000
```

### Model download fails

1. Check authentication:
   ```bash
   huggingface-cli whoami
   ```

2. For gated models, accept terms first:
   - Visit model page on HuggingFace
   - Click "Agree and access repository"

## Configuration Files

- `config/models.yaml` - Model definitions
- `config/benchmark.yaml` - Test scenarios and concurrency levels
- `docker-compose.yml` - Container configurations

## Full Documentation

- **Deployment**: See [DEPLOYMENT.md](DEPLOYMENT.md) for AWS infrastructure setup
- **Architecture**: See [DESIGN.md](DESIGN.md) and [ARCHITECTURE.md](ARCHITECTURE.md)
- **README**: See [README.md](README.md) for comprehensive guide

## Cost Management

**Stop instance when not in use:**
```bash
# From your local machine (not on EC2)
aws ec2 stop-instances --instance-ids <your-instance-id>
```

**Start again:**
```bash
aws ec2 start-instances --instance-ids <your-instance-id>
```

**Destroy everything:**
```bash
cd terraform
terraform destroy
```

## Typical Workflow

```bash
# 1. SSH in
ssh -i ~/.ssh/id_rsa ubuntu@<instance-ip>

# 2. Setup (first time only)
cd embedding_model_benchmarking
bash scripts/setup.sh
source ~/.bashrc

# 3. Authenticate (first time only)
bash scripts/huggingface_login.sh

# 4. Download model
python3 scripts/download_model.py --model embeddinggemma-300m

# 5. Build containers
docker-compose build

# 6. Run benchmarks
python3 orchestrator/runner.py --model embeddinggemma-300m --csv

# 7. View results
python3 orchestrator/runner.py --model embeddinggemma-300m --summary-only --csv

# 8. Copy results to local machine (from your laptop)
scp -i ~/.ssh/id_rsa ubuntu@<instance-ip>:~/embedding_model_benchmarking/results/*.csv .
```

## Expected Runtime

- Setup script: ~5-10 minutes
- Model download: ~2-5 minutes
- Docker build: ~5-10 minutes per framework
- Benchmark run (all 3 frameworks): ~30-60 minutes

## Instance Costs

**c7i.8xlarge (32 vCPU, 64 GB):**
- us-east-1: ~$1.36/hour
- us-west-2: ~$1.44/hour

**Complete benchmark run:**
- Duration: ~1-2 hours (including setup)
- Total cost: ~$2-3

**Remember to stop or terminate when done!**
