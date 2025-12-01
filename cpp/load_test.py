#!/usr/bin/env python3
"""
Comprehensive Load Testing Script for C++ Embedding Servers

Tests both ONNX Runtime C++ and OpenVINO C++ servers across multiple
concurrency levels, measuring:
- Queries Per Second (QPS)
- Latency percentiles (p50, p95, p99)
- CPU usage during load

Usage:
    ./load_test.py --url http://localhost:8000 --duration 30
"""

import argparse
import json
import requests
import time
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import sys

# Test texts for embedding
TEST_TEXTS = [
    "This is a test sentence for benchmarking embedding models.",
    "Machine learning models require careful performance evaluation."
]

def get_cpu_usage_ssh(host):
    """Get CPU usage from remote host via SSH"""
    try:
        # Run top command for 2 seconds and extract CPU usage
        cmd = f"ssh -i ~/.ssh/id_rsa ubuntu@{host} \"top -bn2 -d 1 | grep 'Cpu(s)' | tail -n1 | awk '{{print 100 - \\$8}}'\""
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception as e:
        print(f"Warning: Could not get CPU usage: {e}", file=sys.stderr)
    return None

def make_request(url, texts):
    """Make a single embedding request and return latency in ms"""
    start = time.perf_counter()
    try:
        response = requests.post(
            f"{url}/embed",
            json={"texts": texts},
            timeout=30
        )
        latency_ms = (time.perf_counter() - start) * 1000

        if response.status_code == 200:
            return {"success": True, "latency_ms": latency_ms}
        else:
            return {"success": False, "latency_ms": latency_ms, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return {"success": False, "latency_ms": latency_ms, "error": str(e)}

def run_load_test(url, concurrency, duration_seconds, texts, host=None):
    """
    Run load test with specified concurrency level for given duration

    Returns dict with:
        - qps: queries per second
        - p50, p95, p99: latency percentiles in ms
        - cpu_usage: average CPU usage percentage (if host provided)
        - error_rate: percentage of failed requests
    """
    print(f"  Testing concurrency={concurrency} for {duration_seconds}s...", end="", flush=True)

    latencies = []
    errors = 0
    start_time = time.time()
    requests_made = 0

    # CPU sampling
    cpu_samples = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []

        while time.time() - start_time < duration_seconds:
            # Submit requests up to concurrency level
            while len(futures) < concurrency and time.time() - start_time < duration_seconds:
                future = executor.submit(make_request, url, texts)
                futures.append(future)
                requests_made += 1

            # Check completed requests
            done_futures = [f for f in futures if f.done()]
            for future in done_futures:
                result = future.result()
                if result["success"]:
                    latencies.append(result["latency_ms"])
                else:
                    errors += 1
                futures.remove(future)

            # Sample CPU periodically (every 0.5s)
            if host and requests_made % (concurrency * 5) == 0:
                cpu = get_cpu_usage_ssh(host)
                if cpu is not None:
                    cpu_samples.append(cpu)

            time.sleep(0.01)  # Small sleep to prevent busy-waiting

        # Wait for remaining requests
        for future in as_completed(futures):
            result = future.result()
            if result["success"]:
                latencies.append(result["latency_ms"])
            else:
                errors += 1

    # Final CPU sample
    if host:
        cpu = get_cpu_usage_ssh(host)
        if cpu is not None:
            cpu_samples.append(cpu)

    elapsed = time.time() - start_time

    if not latencies:
        print(" FAILED (no successful requests)")
        return None

    # Calculate metrics
    latencies.sort()
    qps = len(latencies) / elapsed
    p50 = latencies[int(len(latencies) * 0.50)]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    error_rate = (errors / requests_made) * 100 if requests_made > 0 else 0
    avg_cpu = statistics.mean(cpu_samples) if cpu_samples else None

    print(f" ✓ ({len(latencies)} successful requests)")

    return {
        "concurrency": concurrency,
        "qps": qps,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "cpu_usage": avg_cpu,
        "error_rate": error_rate,
        "total_requests": requests_made,
        "successful_requests": len(latencies)
    }

def print_results_table(results, server_name):
    """Print results in a formatted table"""
    print(f"\n{'='*80}")
    print(f"{server_name} - Load Test Results")
    print(f"{'='*80}")
    print(f"{'Concurrency':<12} {'QPS':<10} {'p50 (ms)':<12} {'p95 (ms)':<12} {'p99 (ms)':<12} {'CPU %':<10}")
    print(f"{'-'*80}")

    for r in results:
        cpu_str = f"{r['cpu_usage']:.1f}" if r['cpu_usage'] is not None else "N/A"
        print(f"{r['concurrency']:<12} {r['qps']:<10.2f} {r['p50_ms']:<12.2f} "
              f"{r['p95_ms']:<12.2f} {r['p99_ms']:<12.2f} {cpu_str:<10}")

    print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(description="Load test C++ embedding servers")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--host", help="SSH host for CPU monitoring (e.g., 44.212.155.111)")
    parser.add_argument("--duration", type=int, default=30, help="Test duration per concurrency level (seconds)")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16, 32],
                       help="Concurrency levels to test (default: 1 4 8 16 32)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for requests")
    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    # Prepare test texts
    texts = TEST_TEXTS[:args.batch_size]

    # Verify server is reachable
    print(f"Checking server at {args.url}...")
    try:
        response = requests.get(f"{args.url}/health", timeout=5)
        if response.status_code != 200:
            print(f"Error: Server returned status {response.status_code}")
            sys.exit(1)
        print("✓ Server is healthy\n")

        # Get server info
        info = requests.get(f"{args.url}/info", timeout=5).json()
        server_name = f"{info['framework']} - {info['model_name']}"
        print(f"Server: {server_name}")
        print(f"CPU cores: {info.get('cpu_count', 'unknown')}")
        print(f"Model: {info['model_name']}")
        print()
    except Exception as e:
        print(f"Error: Could not connect to server: {e}")
        sys.exit(1)

    # Run warmup
    print("Warming up server (10 requests)...")
    for _ in range(10):
        make_request(args.url, texts)
    print("✓ Warmup complete\n")

    # Run load tests
    print(f"Running load tests (duration={args.duration}s per level):")
    results = []

    for concurrency in sorted(args.concurrency):
        result = run_load_test(args.url, concurrency, args.duration, texts, args.host)
        if result:
            results.append(result)

    # Print results
    print_results_table(results, server_name)

    # Save to file if requested
    if args.output:
        output_data = {
            "server": server_name,
            "test_config": {
                "duration_per_level": args.duration,
                "batch_size": args.batch_size,
                "concurrency_levels": args.concurrency
            },
            "results": results
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
