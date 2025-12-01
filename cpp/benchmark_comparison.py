#!/usr/bin/env python3
"""
C++ Framework Comparison Benchmark

Benchmarks ONNX Runtime C++ vs OpenVINO C++ implementations
and provides detailed performance comparison.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from typing import Dict, List, Tuple
import statistics

import requests


def start_server(name: str, executable_path: str, model_name: str) -> Tuple[subprocess.Popen, int]:
    """Start a C++ server and return process and PID"""
    print(f"\n{'='*70}")
    print(f"Starting {name} server...")
    print(f"Executable: {executable_path}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")

    env = os.environ.copy()
    env['MODEL_NAME'] = model_name

    process = subprocess.Popen(
        [executable_path],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )

    # Wait for server to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get('http://localhost:8000/health', timeout=1)
            if response.status_code == 200:
                print(f"✓ Server ready after {i+1} attempts")
                return process, process.pid
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)

    raise RuntimeError(f"Failed to start {name} server after {max_retries} seconds")


def stop_server(process: subprocess.Popen, name: str):
    """Stop a server process gracefully"""
    print(f"\nStopping {name} server...")
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=5)
    except:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    print(f"✓ {name} server stopped")


def benchmark_server(
    name: str,
    url: str = 'http://localhost:8000/embed',
    texts: List[str] = None,
    warmup: int = 10,
    iterations: int = 100
) -> Dict:
    """Run benchmark on a server and return metrics"""
    if texts is None:
        texts = [
            "This is a test sentence for benchmarking the embedding model.",
            "Another sentence to include in the batch for more realistic testing."
        ]

    print(f"\n{'='*70}")
    print(f"Benchmarking {name}")
    print(f"{'='*70}")
    print(f"Warmup iterations: {warmup}")
    print(f"Benchmark iterations: {iterations}")
    print(f"Batch size: {len(texts)}")
    print()

    # Get server info
    info = requests.get('http://localhost:8000/info').json()

    # Warmup
    print("Running warmup...")
    for i in range(warmup):
        requests.post(url, json={'texts': texts})
    print(f"✓ Warmup complete")

    # Benchmark
    print(f"Running benchmark ({iterations} iterations)...")
    latencies = []
    server_times = []

    for i in range(iterations):
        start = time.perf_counter()
        response = requests.post(url, json={'texts': texts})
        end = time.perf_counter()

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            continue

        data = response.json()
        latencies.append((end - start) * 1000)
        server_times.append(data['inference_time_ms'])

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{iterations}")

    print(f"✓ Benchmark complete")

    # Calculate statistics
    def percentile(data, p):
        return statistics.quantiles(data, n=100)[p-1] if len(data) >= 100 else sorted(data)[int(len(data) * p / 100)]

    results = {
        'framework': name,
        'info': info,
        'batch_size': len(texts),
        'iterations': iterations,
        'total_latency_ms': {
            'mean': statistics.mean(latencies),
            'median': statistics.median(latencies),
            'stddev': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            'min': min(latencies),
            'max': max(latencies),
            'p50': percentile(latencies, 50),
            'p95': percentile(latencies, 95),
            'p99': percentile(latencies, 99),
        },
        'server_inference_ms': {
            'mean': statistics.mean(server_times),
            'median': statistics.median(server_times),
            'stddev': statistics.stdev(server_times) if len(server_times) > 1 else 0,
            'min': min(server_times),
            'max': max(server_times),
            'p50': percentile(server_times, 50),
            'p95': percentile(server_times, 95),
            'p99': percentile(server_times, 99),
        },
        'throughput_qps': 1000 / statistics.mean(latencies) * len(texts),
    }

    return results


def print_results(results: Dict):
    """Print benchmark results in a readable format"""
    print(f"\n{'='*70}")
    print(f"Results: {results['framework']}")
    print(f"{'='*70}")
    print(f"Framework: {results['info']['framework']}")
    print(f"Model: {results['info']['model_name']}")
    print(f"Runtime version: {results['info']['runtime_version']}")
    print(f"Model load time: {results['info']['model_load_time_ms']:.2f}ms")
    print()
    print(f"Batch size: {results['batch_size']}")
    print(f"Iterations: {results['iterations']}")
    print()
    print("Total Latency (including network + client overhead):")
    for key, value in results['total_latency_ms'].items():
        print(f"  {key:8s}: {value:8.2f}ms")
    print()
    print("Server Inference Time (pure model inference):")
    for key, value in results['server_inference_ms'].items():
        print(f"  {key:8s}: {value:8.2f}ms")
    print()
    print(f"Throughput: {results['throughput_qps']:.2f} QPS")
    print(f"{'='*70}")


def compare_results(results1: Dict, results2: Dict):
    """Compare two benchmark results"""
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")

    name1 = results1['framework']
    name2 = results2['framework']

    print(f"\n{name1} vs {name2}:")
    print()

    # Compare latencies
    print("Latency (P95):")
    lat1 = results1['total_latency_ms']['p95']
    lat2 = results2['total_latency_ms']['p95']
    diff = ((lat1 - lat2) / lat2) * 100
    faster = name2 if lat1 > lat2 else name1
    print(f"  {name1}: {lat1:.2f}ms")
    print(f"  {name2}: {lat2:.2f}ms")
    print(f"  {faster} is {abs(diff):.1f}% faster")
    print()

    # Compare throughput
    print("Throughput:")
    qps1 = results1['throughput_qps']
    qps2 = results2['throughput_qps']
    diff = ((qps2 - qps1) / qps1) * 100
    faster = name2 if qps2 > qps1 else name1
    print(f"  {name1}: {qps1:.2f} QPS")
    print(f"  {name2}: {qps2:.2f} QPS")
    print(f"  {faster} is {abs(diff):.1f}% faster")
    print()

    # Compare model load time
    print("Model Load Time:")
    load1 = results1['info']['model_load_time_ms']
    load2 = results2['info']['model_load_time_ms']
    diff = ((load1 - load2) / load2) * 100
    faster = name2 if load1 > load2 else name1
    print(f"  {name1}: {load1:.2f}ms")
    print(f"  {name2}: {load2:.2f}ms")
    print(f"  {faster} is {abs(diff):.1f}% faster")

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark C++ embedding servers')
    parser.add_argument('--model', default='embeddinggemma-300m', help='Model name')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--iterations', type=int, default=100, help='Benchmark iterations')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--onnx-only', action='store_true', help='Benchmark ONNX Runtime only')
    parser.add_argument('--openvino-only', action='store_true', help='Benchmark OpenVINO only')
    parser.add_argument('--output', help='Output JSON file for results')

    args = parser.parse_args()

    # Generate test texts
    texts = [
        f"This is test sentence number {i} for benchmarking the embedding model."
        for i in range(args.batch_size)
    ]

    results = {}

    # Benchmark ONNX Runtime C++
    if not args.openvino_only:
        onnx_executable = 'onnx_runtime/build/onnx_runtime_server'
        if not os.path.exists(onnx_executable):
            print(f"Error: {onnx_executable} not found. Please build it first.")
            print("Run: cd onnx_runtime && ./build.sh")
            sys.exit(1)

        try:
            process, pid = start_server('ONNX Runtime C++', onnx_executable, args.model)
            time.sleep(2)  # Extra stabilization time
            results['onnx-cpp'] = benchmark_server(
                'ONNX Runtime C++',
                texts=texts,
                warmup=args.warmup,
                iterations=args.iterations
            )
            print_results(results['onnx-cpp'])
        finally:
            stop_server(process, 'ONNX Runtime C++')
            time.sleep(2)

    # Benchmark OpenVINO C++
    if not args.onnx_only:
        openvino_executable = 'openvino/build/openvino_server'
        if not os.path.exists(openvino_executable):
            print(f"Error: {openvino_executable} not found. Please build it first.")
            print("Run: cd openvino && ./build.sh")
            sys.exit(1)

        try:
            process, pid = start_server('OpenVINO C++', openvino_executable, args.model)
            time.sleep(2)  # Extra stabilization time
            results['openvino-cpp'] = benchmark_server(
                'OpenVINO C++',
                texts=texts,
                warmup=args.warmup,
                iterations=args.iterations
            )
            print_results(results['openvino-cpp'])
        finally:
            stop_server(process, 'OpenVINO C++')

    # Compare results
    if 'onnx-cpp' in results and 'openvino-cpp' in results:
        compare_results(results['onnx-cpp'], results['openvino-cpp'])

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {args.output}")

    print("\n✓ Benchmark complete!")


if __name__ == '__main__':
    main()
