#!/bin/bash
#
# Run Candle Rust benchmark end-to-end
#
# Usage:
#   ./scripts/run_candle_benchmark.sh [MODEL_NAME]
#

set -e

MODEL_NAME=${1:-embeddinggemma-300m}

echo "========================================================================"
echo "Candle Rust Benchmark Runner"
echo "========================================================================"
echo "Model: $MODEL_NAME"
echo ""

# Check if we're in the rust directory
if [ ! -f "docker-compose.yml" ]; then
    echo "Error: Must run from rust/ directory"
    exit 1
fi

# Check if models exist
if [ ! -d "../models/$MODEL_NAME" ]; then
    echo "Error: Model not found at ../models/$MODEL_NAME"
    echo "Please download the model first:"
    echo "  python3 ../scripts/download_model.py --model $MODEL_NAME"
    exit 1
fi

echo "Starting benchmark..."
echo ""

# Start server
echo "1. Starting Candle Rust server..."
docker compose up -d candle-rust-server

# Wait for health check
echo "2. Waiting for server to be healthy..."
timeout 60 bash -c 'until curl -sf http://localhost:8001/health > /dev/null; do sleep 2; done' || {
    echo "Error: Server failed to start"
    docker compose logs candle-rust-server
    docker compose down
    exit 1
}

echo "   ✓ Server is healthy"
echo ""

# Get server info
echo "3. Server info:"
curl -s http://localhost:8001/info | jq .
echo ""

# Run benchmark
echo "4. Running benchmark..."
docker compose run --rm \
    -e MODEL_NAME=$MODEL_NAME \
    candle-rust-client

# Stop server
echo ""
echo "5. Stopping server..."
docker compose down

echo ""
echo "========================================================================"
echo "Benchmark complete!"
echo "Results: ../results/$MODEL_NAME/candle-rust.json"
echo "========================================================================"

# Show quick summary
if [ -f "../results/$MODEL_NAME/candle-rust.json" ]; then
    echo ""
    echo "Quick Summary:"
    jq '.scenarios.concurrency_1 | {
        mean_latency: .latency_ms.mean,
        p95_latency: .latency_ms.p95,
        throughput_qps: .throughput_qps
    }' "../results/$MODEL_NAME/candle-rust.json"
fi
