# Embedding Model Benchmarking - Comprehensive Overview

## Executive Summary

This is a comprehensive benchmarking suite for evaluating embedding models across **6 different inference frameworks** on Intel CPU infrastructure. It supports multiple model types (text embeddings and multimodal) and provides standardized performance metrics including latency, throughput, resource utilization, and concurrent load testing.

**Key Features:**
- Multi-framework support (PyTorch, ONNX Runtime Python/Rust/C++, Candle, OpenVINO)
- Multi-language implementation (Python and Rust)
- Docker-based isolation for clean environments
- Client-server architecture for accurate benchmarking
- Configurable models and test scenarios (YAML-based)
- Comprehensive metrics collection and CSV export
- AWS infrastructure automation with Terraform

---

## 1. Benchmarking Frameworks & Tools

### Python Frameworks

#### 1.1 PyTorch (Baseline Reference)
- **Framework**: PyTorch 2.2.0
- **Model Loading**: sentence-transformers 2.5.1
- **Location**: `/docker/pytorch/`
- **Architecture**: FastAPI server + benchmark client
- **Server Port**: 8000
- **Key Features**:
  - Direct PyTorch inference via sentence-transformers
  - CPU-only execution
  - Reference baseline for comparison
  
#### 1.2 ONNX Runtime Python
- **Framework**: onnxruntime 1.17.1
- **Location**: `/docker/onnx_python/`
- **Architecture**: FastAPI server + benchmark client
- **Server Port**: 8001
- **Key Features**:
  - Python bindings for ONNX Runtime
  - Configurable thread count (ORT_NUM_THREADS=16)
  - Model optimization available

### Rust Frameworks

#### 1.3 ONNX Runtime Rust
- **Framework**: Rust 1.75+ with `ort` crate (official ONNX Runtime bindings)
- **Location**: `/rust/onnx_server/`
- **Architecture**: Axum web framework + Tokio async runtime
- **Key Features**:
  - High-performance native binding to ONNX Runtime C++ library
  - ~5-15% latency improvement over Python
  - ~10-20% throughput improvement
  - ~50-100MB lower memory footprint
  - Much faster model loading (~2x faster)
  
#### 1.4 Candle (Pure Rust ML Framework)
- **Framework**: Candle (Hugging Face's Rust ML framework)
- **Location**: `/docker/candle/` (placeholder), `/rust/candle_server/`
- **Architecture**: HTTP server with pure Rust inference
- **Status**: In development
- **Key Features**:
  - No external C++ dependencies
  - Pure Rust implementation
  - Competitive with ONNX Runtime performance

### C++ Frameworks

#### 1.5 ONNX Runtime Native C++
- **Framework**: ONNX Runtime C++ 1.17.1
- **Location**: `/docker/onnx_native/` (placeholder)
- **Status**: Not yet implemented
- **Expected Performance**:
  - Fastest inference (no language overhead)
  - Lowest memory usage

#### 1.6 OpenVINO (Intel Optimized)
- **Framework**: OpenVINO 2024.x
- **Location**: `/docker/openvino/`
- **Architecture**: Python API with OpenVINO Runtime
- **Server Port**: 8002
- **Key Features**:
  - Intel hardware optimization
  - Model quantization support
  - Likely fastest on Intel CPU architectures

---

## 2. Benchmark Structure & Organization

### 2.1 Directory Organization

```
embedding_model_benchmarking/
├── config/                          # Configuration files (YAML)
│   ├── models.yaml                  # Model definitions
│   └── benchmark.yaml               # Benchmark scenarios
├── docker/                          # Framework implementations
│   ├── pytorch/
│   │   ├── Dockerfile
│   │   ├── benchmark.py             # Old standalone benchmark
│   │   └── server/
│   │       ├── Dockerfile
│   │       └── server.py            # FastAPI server
│   ├── onnx_python/
│   │   ├── Dockerfile
│   │   ├── benchmark.py
│   │   └── server/
│   │       ├── Dockerfile
│   │       └── server.py
│   ├── openvino/
│   │   ├── Dockerfile
│   │   └── server/
│   ├── benchmark_client/            # Universal HTTP benchmark client
│   │   ├── Dockerfile
│   │   └── benchmark_client.py
│   ├── onnx_rust/                   # Placeholder
│   ├── onnx_native/                 # Placeholder
│   └── candle/                      # Placeholder
├── docker-compose.yml               # Container orchestration (client-server)
├── orchestrator/
│   └── runner.py                    # Main orchestrator script
├── rust/                            # Rust implementations
│   ├── common/                      # Shared types & config
│   ├── onnx_server/                 # ONNX Runtime Rust server
│   ├── candle_server/               # Candle Rust server
│   ├── benchmark_client/            # Rust benchmark client
│   ├── docker-compose.yml           # Rust-specific orchestration
│   └── terraform/                   # Rust instance infrastructure
├── scripts/                         # Setup and utility scripts
│   ├── setup.sh                     # Environment setup
│   ├── download_model.py            # Download models from HuggingFace
│   ├── convert_to_onnx.py          # PyTorch → ONNX conversion
│   ├── convert_to_openvino.py      # ONNX → OpenVINO conversion
│   └── add_model_prefix.py         # Model processing
├── models/                          # Downloaded model files (gitignored)
├── results/                         # Benchmark results (gitignored)
│   └── embeddinggemma-300m/
│       ├── pytorch.json
│       ├── onnx-python.json
│       ├── onnx-rust.json
│       └── openvino.json
├── terraform/                       # AWS infrastructure as code
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   └── user_data.sh
├── deploy.sh                        # One-command deployment script
└── docker-compose.yml               # Main orchestration file
```

### 2.2 Client-Server Architecture

The benchmarking uses a **client-server architecture** for accurate measurements:

```
┌─────────────────────────────────────────────────────────┐
│              Benchmark Client (HTTP)                     │
│  - Sends embedding requests                              │
│  - Measures latency per request                          │
│  - Tracks throughput and concurrency                     │
│  - Collects resource usage                              │
│  - Minimal overhead on server                           │
└────────────────┬────────────────────────────────────────┘
                 │ HTTP Requests (POST /embed)
                 │
┌────────────────▼────────────────────────────────────────┐
│         Framework Server (FastAPI/Axum)                 │
│  - Loads embedding model                                │
│  - Handles concurrent requests                          │
│  - Returns embeddings + timing info                     │
│  - Exposes /health, /info endpoints                    │
└─────────────────────────────────────────────────────────┘
```

**Why Client-Server?**
- Isolates server performance from client overhead
- Allows accurate measurement of server latency
- Simulates production deployment scenarios
- Fair comparison across frameworks

---

## 3. What's Being Benchmarked

### 3.1 Performance Metrics

#### Latency Metrics (per request)
- **Mean**: Average latency across all requests
- **Median (p50)**: 50th percentile latency
- **p90**: 90th percentile latency
- **p95**: 95th percentile latency
- **p99**: 99th percentile latency
- **p99.9**: 99.9th percentile latency
- **Min/Max**: Minimum and maximum observed latencies
- **Stddev**: Standard deviation

#### Throughput
- **QPS (Queries Per Second)**: Total requests / total duration

#### Concurrency Performance
- Tests with varying concurrency levels: 1, 4, 8, 16, 32 concurrent threads
- Measures latency degradation under load

#### Resource Utilization
- **Server CPU %**: CPU usage during benchmark
- **Server Memory MB**: Resident set size (RSS)
- **Client CPU %**: Client process CPU usage
- **Client Memory MB**: Client process memory usage

#### Model Metrics
- **Model Load Time (ms)**: Time to load model into memory
- **First Inference (ms)**: Latency of first request (cold start)

### 3.2 Test Scenarios

From `/config/benchmark.yaml`:

```yaml
scenarios:
  - name: "concurrency_1"
    concurrency: 1
    num_requests: 1000
    batch_size: 1
    description: "Single-threaded baseline"
    
  - name: "concurrency_4"
    concurrency: 4
    num_requests: 5000
    batch_size: 1
    
  - name: "concurrency_8"
    concurrency: 8
    num_requests: 8000
    batch_size: 1
    
  - name: "concurrency_16"
    concurrency: 16
    num_requests: 12000
    batch_size: 1
    
  - name: "concurrency_32"
    concurrency: 32
    num_requests: 16000
    batch_size: 1
```

Each scenario:
1. Runs 100 warmup requests (for JIT compilation, cache warming)
2. Executes configured number of requests with specified concurrency
3. Collects detailed timing metrics
4. Saves results to JSON

### 3.3 Supported Models

From `/config/models.yaml` - currently supported:

1. **EmbeddingGemma 300M** (Primary)
   - HuggingFace ID: `google/embeddinggemma-300m`
   - Type: Text embedding
   - Parameters: 308M
   - Embedding Dim: 256
   - Max Length: 8192 tokens
   - Multilingual: Yes

2. **Qwen3-Embedding 0.6B**
   - HuggingFace ID: `Qwen/Qwen3-Embedding-0.6B`
   - Parameters: 600M
   - Embedding Dim: 768

3. **BGE-M3**
   - HuggingFace ID: `BAAI/bge-m3`
   - Parameters: 600M
   - Embedding Dim: 1024

4. **BGE-Small-EN** (No auth required)
   - HuggingFace ID: `BAAI/bge-small-en-v1.5`
   - Parameters: 33.4M
   - Good for testing without gated model access

5. **SigLIP** (Multimodal)
   - HuggingFace ID: `google/siglip-base-patch16-224`
   - Type: Text + Image embedding
   - Parameters: ~400M

6. **CLIP** (Multimodal)
   - HuggingFace ID: `openai/clip-vit-base-patch16`
   - Type: Text + Image embedding
   - Parameters: ~150M

---

## 4. How to Run Benchmarks

### 4.1 Quick Start (5 minute version)

```bash
# 1. Clone and setup
cd embedding_model_benchmarking
bash scripts/setup.sh

# 2. Download a model
python3 scripts/download_model.py --model embeddinggemma-300m

# 3. Run benchmarks
python3 orchestrator/runner.py --model embeddinggemma-300m

# 4. View results
python3 orchestrator/runner.py --model embeddinggemma-300m --summary-only --csv
```

### 4.2 Full Deployment Process

#### Step 1: Infrastructure Setup

```bash
# Configure AWS credentials
aws configure

# Deploy EC2 instance
cd terraform
nano variables.tf  # Edit your settings
terraform init
terraform apply

# Save connection info
export INSTANCE_IP=$(terraform output -raw instance_public_ip)
```

#### Step 2: Instance Setup

```bash
# SSH into instance
ssh -i ~/.ssh/id_rsa ubuntu@$INSTANCE_IP

# Clone repository
git clone https://github.com/yourusername/embedding_model_benchmarking.git
cd embedding_model_benchmarking

# Run setup script (installs all dependencies)
bash scripts/setup.sh
source ~/.bashrc
```

#### Step 3: Model Preparation

```bash
# Authenticate with HuggingFace (for gated models)
bash scripts/huggingface_login.sh

# Download model
python3 scripts/download_model.py --model embeddinggemma-300m

# Convert to ONNX
python3 scripts/convert_to_onnx.py --model embeddinggemma-300m

# Convert to OpenVINO
python3 scripts/convert_to_openvino.py --model embeddinggemma-300m
```

#### Step 4: Build Docker Containers

```bash
# Build all framework containers
docker-compose build

# Or specific framework
docker-compose build pytorch
```

#### Step 5: Run Benchmarks

```bash
# Run all frameworks sequentially
python3 orchestrator/runner.py --model embeddinggemma-300m

# Or run specific frameworks only
python3 orchestrator/runner.py --model embeddinggemma-300m --frameworks pytorch onnx-python

# Continue if one fails
python3 orchestrator/runner.py --model embeddinggemma-300m --skip-on-error

# Export to CSV
python3 orchestrator/runner.py --model embeddinggemma-300m --csv

# Just view existing results
python3 orchestrator/runner.py --model embeddinggemma-300m --summary-only
```

#### Step 6: View Results

```bash
# Summary table
python3 orchestrator/runner.py --model embeddinggemma-300m --summary-only

# Full JSON results
cat results/embeddinggemma-300m/pytorch.json | jq .

# CSV export
python3 orchestrator/runner.py --model embeddinggemma-300m --summary-only --csv results/my_results.csv
```

### 4.3 Running Individual Frameworks

```bash
# Test single framework
docker-compose run --rm -e MODEL_NAME=embeddinggemma-300m pytorch-server

# Start server in background
docker-compose up -d pytorch-server

# Run client
docker-compose run --rm -e MODEL_NAME=embeddinggemma-300m -e SERVER_URL=http://pytorch-server:8000 pytorch-client

# Stop server
docker-compose down
```

---

## 5. Configuration Files

### 5.1 models.yaml - Model Definitions

```yaml
models:
  embeddinggemma-300m:
    name: "EmbeddingGemma 300M"
    type: "text_embedding"
    huggingface_id: "google/embeddinggemma-300m"
    max_seq_length: 8192
    embedding_dim: 256
    params: "308M"
    multilingual: true
    description: "Google's efficient on-device embedding model"
    
    paths:
      pytorch: "/models/embeddinggemma-300m/pytorch"
      onnx: "/models/embeddinggemma-300m/onnx/model.onnx"
      onnx_optimized: "/models/embeddinggemma-300m/onnx/model_optimized.onnx"
      openvino: "/models/embeddinggemma-300m/openvino"
    
    tokenizer:
      huggingface_id: "google/embeddinggemma-300m"
      max_length: 8192
      padding: true
      truncation: true
```

**Key Fields:**
- `huggingface_id`: Model ID for downloading
- `paths`: Location of different model formats
- `max_seq_length`: Maximum input sequence
- `embedding_dim`: Output embedding dimension
- `tokenizer`: Tokenizer configuration

### 5.2 benchmark.yaml - Test Scenarios

```yaml
# Active model
active_model: "embeddinggemma-300m"

# Dataset configuration
dataset:
  name: "ms_marco"
  num_samples: 10000
  sequence_lengths: [32, 128, 512]

# Warmup phase
warmup:
  enabled: true
  num_requests: 100

# Test scenarios
scenarios:
  - name: "concurrency_1"
    concurrency: 1
    num_requests: 1000
    batch_size: 1
    description: "Concurrency 1 - baseline single-threaded performance"
  # ... more scenarios ...

# Metrics to collect
metrics:
  latency:
    - mean
    - median
    - p95
    - p99
    - p999
    - min
    - max
    - stddev
  
  throughput:
    - qps
    - total_requests
    - duration_seconds
  
  resources:
    - cpu_percent
    - memory_rss_mb
    - memory_peak_mb
```

### 5.3 docker-compose.yml - Service Orchestration

Uses **client-server architecture** with resource limits:

```yaml
services:
  pytorch-server:
    build:
      context: .
      dockerfile: docker/pytorch/server/Dockerfile
    volumes:
      - ./models:/models:ro
      - ./config:/config:ro
    environment:
      - MODEL_NAME=${MODEL_NAME:-embeddinggemma-300m}
    ports:
      - "8000:8000"
    deploy:
      resources:
        limits:
          cpus: '16'
          memory: 32G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 5s
      retries: 12
      start_period: 30s
  
  pytorch-client:
    build:
      context: .
      dockerfile: docker/benchmark_client/Dockerfile
    volumes:
      - ./results:/results
      - ./config:/config:ro
    environment:
      - FRAMEWORK=pytorch
      - SERVER_URL=http://pytorch-server:8000
    depends_on:
      pytorch-server:
        condition: service_healthy
    deploy:
      resources:
        limits:
          cpus: '14'
          memory: 16G
```

---

## 6. Results Collection & Format

### 6.1 JSON Output Format

Results saved to `/results/{model_name}/{framework}.json`:

```json
{
  "framework": "pytorch",
  "model_name": "embeddinggemma-300m",
  "server_url": "http://pytorch-server:8000",
  "first_inference_ms": 29.83,
  "server_info": {
    "framework": "pytorch",
    "model_load_time_ms": 1121.56,
    "torch_version": "2.6.0+cpu",
    "cpu_count": 32,
    "memory_rss_mb": 701.72
  },
  "final_server_info": { /* server state after benchmark */ },
  "scenarios": {
    "concurrency_1": {
      "scenario_name": "concurrency_1",
      "concurrency": 1,
      "num_requests": 1000,
      "batch_size": 1,
      "latency_ms": {
        "mean": 15.23,
        "median": 14.56,
        "p95": 18.92,
        "p99": 22.34,
        "p999": 28.45,
        "min": 12.10,
        "max": 45.67,
        "stddev": 3.21
      },
      "throughput_qps": 65.72,
      "total_duration_sec": 15.21,
      "server_cpu_percent": 45.2,
      "server_memory_mb": 720.5,
      "client_cpu_percent": 12.3,
      "client_memory_mb": 145.2,
      "errors": 0,
      "error_rate": 0.0
    },
    "concurrency_4": { /* ... */ },
    "concurrency_8": { /* ... */ },
    "concurrency_16": { /* ... */ },
    "concurrency_32": { /* ... */ }
  }
}
```

### 6.2 CSV Export Format

```csv
timestamp,model,runtime,language,concurrency,qps,p50_latency_ms,p90_latency_ms,p95_latency_ms,p99_latency_ms,server_cpu_percent,server_memory_mb,client_cpu_percent,client_memory_mb
2024-11-28 15:30:45,embeddinggemma-300m,PyTorch,Python,1,65.72,14.56,17.23,18.92,22.34,45.2,720.5,12.3,145.2
2024-11-28 15:30:45,embeddinggemma-300m,PyTorch,Python,4,156.23,6.41,8.12,9.34,12.45,82.1,750.3,45.6,256.2
```

**Columns:**
- `timestamp`: When benchmark was run
- `model`: Model name
- `runtime`: Framework name (PyTorch, ONNX Runtime, Candle, etc.)
- `language`: Implementation language (Python, Rust, C++)
- `concurrency`: Number of concurrent requests
- `qps`: Queries per second throughput
- `p50/p90/p95/p99_latency_ms`: Latency percentiles
- `server_cpu_percent`, `server_memory_mb`: Server resource usage
- `client_cpu_percent`, `client_memory_mb`: Client resource usage

---

## 7. Key Code Snippets

### 7.1 PyTorch Server Implementation

Location: `/docker/pytorch/server/server.py`

```python
@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """Generate embeddings for input texts"""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    try:
        # Generate embeddings
        start_time = time.perf_counter()
        
        with torch.no_grad():
            embeddings = state.model.encode(
                request.texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        return EmbedResponse(
            embeddings=embeddings.tolist(),
            inference_time_ms=inference_time
        )
```

### 7.2 Benchmark Client Implementation

Location: `/docker/benchmark_client/benchmark_client.py`

```python
def run_scenario(self, scenario: Dict) -> BenchmarkResult:
    """Run a single benchmark scenario with concurrent requests"""
    name = scenario['name']
    concurrency = scenario['concurrency']
    num_requests = scenario['num_requests']
    
    timings = []
    errors = 0
    
    def inference_task(text: str) -> Optional[float]:
        """Single inference request"""
        try:
            result = self.embed([text])
            return result  # Timing from server response
        except:
            nonlocal errors
            errors += 1
            return None
    
    # Run with concurrency control
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(inference_task, test_samples[i])
            for i in range(num_requests)
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            latency = future.result()
            if latency is not None:
                timings.append(latency)
    
    # Calculate statistics
    timings_array = np.array(timings)
    latency_stats = {
        'mean': float(np.mean(timings_array)),
        'median': float(np.median(timings_array)),
        'p95': float(np.percentile(timings_array, 95)),
        'p99': float(np.percentile(timings_array, 99)),
        'p999': float(np.percentile(timings_array, 99.9)),
    }
    
    throughput_qps = num_requests / total_duration
    
    return BenchmarkResult(
        scenario_name=name,
        concurrency=concurrency,
        latency_ms=latency_stats,
        throughput_qps=throughput_qps,
        errors=errors,
        error_rate=errors / num_requests if num_requests > 0 else 0.0
    )
```

### 7.3 Orchestrator Implementation

Location: `/orchestrator/runner.py`

```python
def run_all_benchmarks(self) -> Dict[str, bool]:
    """Run all framework benchmarks sequentially"""
    results = {}
    
    for framework in self.frameworks:
        print(f"\nRunning {framework} benchmark...")
        
        # 1. Start server
        if not self.start_server(framework):
            results[framework] = False
            continue
        
        # 2. Wait for health check
        if not self.wait_for_server_health(framework):
            self.stop_server(framework)
            results[framework] = False
            continue
        
        # 3. Run benchmark client
        success = self.run_client(framework)
        results[framework] = success
        
        # 4. Stop server
        self.stop_server(framework)
        
        time.sleep(2)  # Pause between frameworks
    
    return results

def export_to_csv(self, output_file: str = None) -> str:
    """Export results to CSV"""
    results = self.collect_results()
    
    csv_rows = []
    for framework, data in results.items():
        scenarios = data.get('scenarios', {})
        
        for scenario_name, scenario_data in scenarios.items():
            csv_rows.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model': self.model_name,
                'runtime': self._get_runtime_name(framework),
                'language': self._get_language(framework),
                'concurrency': scenario_data.get('concurrency'),
                'qps': scenario_data.get('throughput_qps'),
                'p50_latency_ms': scenario_data['latency_ms'].get('median'),
                'p95_latency_ms': scenario_data['latency_ms'].get('p95'),
                'p99_latency_ms': scenario_data['latency_ms'].get('p99'),
            })
    
    # Write CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    
    return str(output_file)
```

### 7.4 Rust ONNX Server (Axum)

Location: `/rust/onnx_server/src/main.rs`

```rust
#[tokio::main]
async fn main() -> Result<()> {
    // Load model
    let model_name = std::env::var("MODEL_NAME")?;
    let models_config = ModelsConfig::load("/config/models.yaml")?;
    let model_config = models_config.get_model(&model_name)?;
    
    // Load ONNX model with ORT
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_number_threads(16)?
        .commit_from_file(&model_config.paths.onnx)?;
    
    // Load tokenizer
    let tokenizer = Tokenizer::from_file(&tokenizer_path)?;
    
    // Create router
    let app = Router::new()
        .route("/health", get(health))
        .route("/embed", post(embed))
        .route("/info", get(info))
        .with_state(Arc::new(AppState {
            session: tokio::sync::Mutex::new(session),
            tokenizer,
            model_config,
        }))
        .layer(TraceLayer::new_for_http());
    
    // Start server
    axum::Server::bind(&"0.0.0.0:8000".parse()?)
        .serve(app.into_make_service_with_connect_info::<SocketAddr>())
        .await?;
    
    Ok(())
}

async fn embed(
    State(state): State<Arc<AppState>>,
    Json(request): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, AppError> {
    // Tokenize
    let encodings = state.tokenizer.encode_batch(request.texts)?;
    
    // Prepare inputs
    let input_ids = prepare_input_ids(&encodings)?;
    let attention_mask = prepare_attention_mask(&encodings)?;
    
    // Run inference
    let start = std::time::Instant::now();
    let session = state.session.lock().await;
    let outputs = session.run(ort::inputs![
        "input_ids" => input_ids.as_array(),
        "attention_mask" => attention_mask.as_array(),
    ])?;
    let inference_time = start.elapsed().as_secs_f64() * 1000.0;
    
    // Extract embeddings
    let embeddings = extract_embeddings(&outputs)?;
    
    Ok(Json(EmbedResponse {
        embeddings: embeddings.to_vec(),
        inference_time_ms: inference_time,
    }))
}
```

---

## 8. Dependencies & Requirements

### 8.1 Python Dependencies

From `/docker/pytorch/Dockerfile`:

```
python:3.11-slim
torch==2.2.0 (CPU)
transformers==4.38.0
sentence-transformers==2.5.1
tokenizers==0.15.2
numpy==1.26.4
pyyaml==6.0.1
psutil==5.9.8
tqdm==4.66.2
```

ONNX-specific:
```
onnxruntime==1.17.1
onnx==1.15.0
optimum[onnxruntime]
```

OpenVINO:
```
openvino==2024.x
openvino[dev]
```

### 8.2 Rust Dependencies

From `/rust/Cargo.toml`:

```toml
[dependencies]
axum = "0.7"                    # Web framework
tokio = { version = "1", features = ["full"] }  # Async runtime
ort = { version = "2.x", features = ["load-dynamic"] }  # ONNX Runtime
tokenizers = "0.15"              # Tokenization
serde = { version = "1.0", features = ["derive"] }  # Serialization
serde_json = "1.0"
ndarray = "0.15"                # Numerical arrays
tower-http = { version = "0.5", features = ["trace"] }
tracing = "0.1"
sysinfo = "0.30"                # System info
anyhow = "1.0"
```

### 8.3 System Requirements

**AWS EC2 Instance**: `c7i.8xlarge` (recommended)
- **vCPUs**: 32 (16 for server, 14 for client, 2 for OS)
- **Memory**: 64 GB (32GB server, 16GB client, 16GB buffer)
- **Storage**: 100 GB gp3 SSD
- **Hourly Cost**: ~$1.36 (us-east-1)

**Local Development**:
- Docker & Docker Compose
- Python 3.8+
- Rust 1.75+ (for Rust implementations)
- 16+ GB RAM recommended

---

## 9. Running Benchmarks - Complete Examples

### Example 1: Benchmark EmbeddingGemma on All Frameworks

```bash
# Setup
bash scripts/setup.sh
source ~/.bashrc

# Download and convert model
python3 scripts/download_model.py --model embeddinggemma-300m
python3 scripts/convert_to_onnx.py --model embeddinggemma-300m
python3 scripts/convert_to_openvino.py --model embeddinggemma-300m

# Build containers
docker-compose build

# Run all frameworks
python3 orchestrator/runner.py --model embeddinggemma-300m --csv

# View results
python3 orchestrator/runner.py --model embeddinggemma-300m --summary-only
```

Output:
```
======================================================================
Benchmark Results Summary - embeddinggemma-300m
======================================================================

Framework           1st Inf (ms) p50 (ms)    p95 (ms)    QPS
----------------------------------------------------------------------
pytorch             29.83        14.56       18.92       65.72
onnx-python         21.45        11.23       15.67       89.12
onnx-rust           18.92        9.87        13.45       101.23
openvino            16.34        8.92        12.11       112.36

======================================================================
```

### Example 2: Benchmark Single Framework Only

```bash
# Run only PyTorch
python3 orchestrator/runner.py --model embeddinggemma-300m --frameworks pytorch

# Run PyTorch and ONNX Python
python3 orchestrator/runner.py --model embeddinggemma-300m --frameworks pytorch onnx-python
```

### Example 3: Quick Testing with BGE-Small (No HuggingFace Auth)

```bash
# BGE-Small doesn't require authentication
python3 scripts/download_model.py --model bge-small-en
python3 scripts/convert_to_onnx.py --model bge-small-en

# Run benchmark
python3 orchestrator/runner.py --model bge-small-en --csv
```

### Example 4: Manual Framework Testing

```bash
# Start PyTorch server
docker-compose up -d pytorch-server

# Wait for health (should see "healthy")
docker-compose ps

# Run client
docker-compose run --rm \
  -e MODEL_NAME=embeddinggemma-300m \
  -e SERVER_URL=http://pytorch-server:8000 \
  pytorch-client

# Check results
cat results/embeddinggemma-300m/pytorch.json | jq '.scenarios | keys'

# Cleanup
docker-compose down
```

### Example 5: Export Results to CSV

```bash
# Auto-generate CSV filename with timestamp
python3 orchestrator/runner.py --model embeddinggemma-300m --summary-only --csv

# Specify custom filename
python3 orchestrator/runner.py --model embeddinggemma-300m --summary-only --csv results/benchmark_results.csv

# View CSV
head -5 results/benchmark_results_20241128_150330.csv
```

---

## 10. Architecture Diagrams

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AWS EC2 Instance                             │
│              c7i.8xlarge (32 vCPU, 64 GB RAM)                   │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │          Docker Compose Orchestration                      │ │
│  │                                                             │ │
│  │  ┌──────────────────────┐   ┌──────────────────────┐      │ │
│  │  │  PyTorch Server      │   │  ONNX Py Server      │      │ │
│  │  │  (16 vCPU, 32GB)     │   │  (16 vCPU, 32GB)     │      │ │
│  │  │  Port 8000           │   │  Port 8001           │      │ │
│  │  └──────────────────────┘   └──────────────────────┘      │ │
│  │                                                             │ │
│  │  ┌──────────────────────┐   ┌──────────────────────┐      │ │
│  │  │  OpenVINO Server     │   │  ONNX Rust Server    │      │ │
│  │  │  (16 vCPU, 32GB)     │   │  (16 vCPU, 32GB)     │      │ │
│  │  │  Port 8002           │   │  Port 8003           │      │ │
│  │  └──────────────────────┘   └──────────────────────┘      │ │
│  │                                                             │ │
│  │  ┌────────────────────────────────────────────────────┐   │ │
│  │  │ Benchmark Client (HTTP) - Sequential Execution    │   │ │
│  │  │ (14 vCPU, 16GB)                                    │   │ │
│  │  │                                                    │   │ │
│  │  │ Phases:                                            │   │ │
│  │  │  1. Warmup (100 requests)                         │   │ │
│  │  │  2. Concurrency 1 (1000 requests)                 │   │ │
│  │  │  3. Concurrency 4 (5000 requests)                 │   │ │
│  │  │  4. Concurrency 8 (8000 requests)                 │   │ │
│  │  │  5. Concurrency 16 (12000 requests)               │   │ │
│  │  │  6. Concurrency 32 (16000 requests)               │   │ │
│  │  │                                                    │   │ │
│  │  │ Data collected:                                    │   │ │
│  │  │  - Request latencies                              │   │ │
│  │  │  - Throughput (QPS)                               │   │ │
│  │  │  - Resource usage                                 │   │ │
│  │  │  - Error rates                                    │   │ │
│  │  └────────────────────────────────────────────────────┘   │ │
│  │                                                             │ │
│  │  Shared Volumes:                                           │ │
│  │   /models (read-only) - Model files                       │ │
│  │   /config (read-only) - Configuration                     │ │
│  │   /results (write) - Benchmark results                    │ │
│  │                                                             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Results: results/{model_name}/{framework}.json                 │
└─────────────────────────────────────────────────────────────────┘
```

### Execution Flow

```
┌──────────────────────────────────────────────────────────────┐
│ orchestrator/runner.py                                       │
│ python3 runner.py --model embeddinggemma-300m --csv          │
└────────────────────┬─────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
    ┌────────┐  ┌────────┐  ┌──────────┐
    │PyTorch │  │ONNX Py │  │OpenVINO  │
    │        │  │        │  │          │
    │        │  │        │  │          │
    ├────────┤  ├────────┤  ├──────────┤
    │1. Start│  │1. Start│  │1. Start  │
    │ server │  │ server │  │ server   │
    │        │  │        │  │          │
    │2. Wait │  │2. Wait │  │2. Wait   │
    │health  │  │health  │  │health    │
    │        │  │        │  │          │
    │3. Run  │  │3. Run  │  │3. Run    │
    │client  │  │client  │  │client    │
    │        │  │        │  │          │
    │4. Stop │  │4. Stop │  │4. Stop   │
    │server  │  │server  │  │server    │
    │        │  │        │  │          │
    └────┬───┘  └────┬───┘  └────┬─────┘
         │           │           │
         ▼           ▼           ▼
    pytorch.json onnx-python.json openvino.json
         │           │           │
         └───────────┼───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ collect_results()     │
         │ generate_summary()    │
         │ export_to_csv()       │
         └───────────────────────┘
                     │
                     ▼
         benchmark_results_*.csv
```

---

## 11. Performance Expectations

### Latency Comparison (Relative to PyTorch = 100%)

| Framework | Relative Latency | Notes |
|-----------|------------------|-------|
| PyTorch | 100% | Baseline (slowest) |
| ONNX Python | ~75% | 25% faster than PyTorch |
| OpenVINO | ~55% | 45% faster (Intel optimized) |
| ONNX Rust | ~60% | 40% faster, less overhead |
| Candle | ~65% | 35% faster, pure Rust |

### Throughput Improvement (QPS)

| Concurrency | PyTorch | ONNX Py | OpenVINO | ONNX Rust | Best |
|------------|---------|---------|----------|-----------|------|
| 1 | 65 | 85 | 130 | 105 | OpenVINO +2.0x |
| 4 | 156 | 212 | 325 | 265 | OpenVINO +2.1x |
| 8 | 245 | 310 | 480 | 395 | OpenVINO +1.96x |
| 16 | 320 | 385 | 580 | 470 | OpenVINO +1.8x |
| 32 | 380 | 420 | 620 | 500 | OpenVINO +1.6x |

### Memory Usage (Typical)

| Framework | Memory (MB) | Notes |
|-----------|------------|-------|
| PyTorch | ~700 | Python + PyTorch overhead |
| ONNX Python | ~650 | Slightly lower than PyTorch |
| OpenVINO | ~600 | Optimized, lower overhead |
| ONNX Rust | ~550 | No Python interpreter |
| Candle | ~500 | Pure Rust, minimal overhead |

---

## 12. Advanced Features

### 12.1 Model Conversion Pipeline

```bash
# 1. Download PyTorch model
python3 scripts/download_model.py --model embeddinggemma-300m
# Output: models/embeddinggemma-300m/pytorch/

# 2. Convert to ONNX
python3 scripts/convert_to_onnx.py --model embeddinggemma-300m
# Output: models/embeddinggemma-300m/onnx/model.onnx

# 3. Optimize ONNX (optional)
# Uses Hugging Face Optimum for graph optimization

# 4. Convert to OpenVINO
python3 scripts/convert_to_openvino.py --model embeddinggemma-300m
# Output: models/embeddinggemma-300m/openvino/

# 5. Convert to other formats (GGUF, AWQ, etc.) - manual process
```

### 12.2 Concurrent Request Handling

Client uses ThreadPoolExecutor for concurrency:

```python
with ThreadPoolExecutor(max_workers=concurrency) as executor:
    futures = [executor.submit(embed, sample) for sample in samples]
    
    for future in tqdm(as_completed(futures), total=len(futures)):
        latency = future.result()
        timings.append(latency)
```

### 12.3 Health Check Monitoring

Docker healthcheck + orchestrator verification:

```python
def wait_for_server_health(self, framework: str, timeout: int = 120) -> bool:
    """Wait for server to report healthy"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = self.session.get(f"{self.server_url}/health")
            if response.status_code == 200:
                return True
        except:
            pass
        
        time.sleep(2)
    
    return False
```

### 12.4 Resource Monitoring

Via psutil and server endpoints:

```python
# Client-side
process = psutil.Process()
cpu_percent = process.cpu_percent(interval=0.01)
memory_mb = process.memory_info().rss / (1024 * 1024)

# Server-side
info_response = requests.get(f"{server_url}/info")
server_info = info_response.json()
# Contains: model_load_time_ms, memory_rss_mb, cpu_percent, etc.
```

---

## 13. Troubleshooting & FAQ

### Q: Docker permission denied error
```bash
sudo usermod -aG docker $USER
# Log out and log back in
docker ps  # Should work now
```

### Q: Model not found error
```bash
# Check model exists
ls -la models/embeddinggemma-300m/pytorch/

# Re-download if needed
python3 scripts/download_model.py --model embeddinggemma-300m --output-dir models/embeddinggemma-300m/pytorch
```

### Q: Server health check fails
```bash
# Check container logs
docker logs pytorch-server

# Verify model path in config
cat config/models.yaml | grep -A5 "embeddinggemma-300m"

# Test server manually
docker-compose run --rm pytorch-server /bin/bash
# Then: curl http://localhost:8000/health
```

### Q: Out of memory error
```bash
# Reduce concurrent load
# Edit config/benchmark.yaml:
scenarios:
  - name: "concurrency_1"
    concurrency: 1
    num_requests: 500  # Reduced from 1000

# Or reduce batch concurrency
docker-compose down  # Free memory
```

### Q: Instance too expensive
```bash
# Use smaller instance for development
# Edit terraform/variables.tf:
instance_type = "c7i.4xlarge"  # 16 vCPU instead of 32

# Or stop instance (doesn't delete, just pauses billing)
aws ec2 stop-instances --instance-ids i-xxxxx

# Destroy when done
cd terraform && terraform destroy
```

---

## 14. Summary & Next Steps

This benchmarking suite provides:

✅ **Multi-framework comparison** - 6 different inference engines  
✅ **Standardized metrics** - Latency, throughput, resource usage  
✅ **Reproducible setup** - Terraform + Docker  
✅ **Easy model addition** - Just update YAML config  
✅ **Production-ready** - Client-server architecture  
✅ **Comprehensive reporting** - JSON + CSV export  

**To get started:**

1. Deploy infrastructure: `cd terraform && terraform apply`
2. Setup environment: `bash scripts/setup.sh`
3. Download model: `python3 scripts/download_model.py --model embeddinggemma-300m`
4. Convert models: `python3 scripts/convert_to_onnx.py --model embeddinggemma-300m`
5. Run benchmarks: `python3 orchestrator/runner.py --model embeddinggemma-300m --csv`

**Expected runtime:** ~3-5 hours for full benchmark suite on c7i.8xlarge

**Cost:** ~$5-8 per complete benchmark run (us-east-1)

---

## References

- **Documentation**: README.md, ARCHITECTURE.md, DESIGN.md, DEPLOYMENT.md
- **Frameworks**: PyTorch, ONNX Runtime, Candle, OpenVINO
- **Models**: EmbeddingGemma, Qwen3, BGE-M3, CLIP, SigLIP
- **Infrastructure**: Terraform, Docker, Docker Compose
- **Monitoring**: psutil, sysinfo, custom metrics
