#!/usr/bin/env python3
"""
Universal Benchmark Client for Embedding Model Servers

This client sends HTTP requests to any framework server and measures performance.
It ensures that benchmarking overhead doesn't affect server inference performance.
"""

import os
import sys
import time
import json
import yaml
import requests
import psutil
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


@dataclass
class BenchmarkResult:
    """Benchmark results for a single scenario"""
    scenario_name: str
    concurrency: int
    num_requests: int
    batch_size: int
    latency_ms: Dict[str, float]
    throughput_qps: float
    total_duration_sec: float
    cpu_percent: float
    memory_rss_mb: float
    errors: int
    error_rate: float


class BenchmarkClient:
    """Universal HTTP client for benchmarking embedding servers"""

    def __init__(self, framework: str, model_name: str, server_url: str):
        self.framework = framework
        self.model_name = model_name
        self.server_url = server_url
        self.session = requests.Session()
        self.process = psutil.Process()

        # Load benchmark config
        self._load_config()

    def _load_config(self):
        """Load benchmark configuration"""
        benchmark_config_path = Path("/config/benchmark.yaml")
        with open(benchmark_config_path) as f:
            self.benchmark_config = yaml.safe_load(f)

    def wait_for_server(self, timeout: int = 120, interval: int = 2) -> bool:
        """
        Wait for server to be ready

        Args:
            timeout: Maximum time to wait in seconds
            interval: Check interval in seconds

        Returns:
            True if server is ready, False if timeout
        """
        health_url = f"{self.server_url}/health"
        start_time = time.time()

        print(f"Waiting for server at {self.server_url}...")

        while time.time() - start_time < timeout:
            try:
                response = self.session.get(health_url, timeout=5)
                if response.status_code == 200:
                    print(f"✓ Server is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass

            time.sleep(interval)

        print(f"✗ Server failed to start within {timeout}s")
        return False

    def get_server_info(self) -> Dict[str, Any]:
        """Get server information and verify it's running"""
        try:
            response = self.session.get(f"{self.server_url}/info", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting server info: {e}")
            return {}

    def embed(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Send embedding request to server

        Args:
            texts: List of texts to embed

        Returns:
            Embeddings as numpy array, or None if error
        """
        try:
            response = self.session.post(
                f"{self.server_url}/embed",
                json={"texts": texts},
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            embeddings = np.array(result["embeddings"])
            return embeddings

        except requests.exceptions.RequestException as e:
            print(f"Error during embedding request: {e}")
            return None

    def warmup(self, num_iterations: int = 100):
        """
        Warmup the server

        Args:
            num_iterations: Number of warmup iterations
        """
        print(f"\nWarming up server ({num_iterations} iterations)...")

        warmup_text = "This is a warmup sentence to initialize the model."
        errors = 0

        for _ in tqdm(range(num_iterations), desc="Warmup"):
            result = self.embed([warmup_text])
            if result is None:
                errors += 1

        if errors > 0:
            print(f"⚠ Warning: {errors}/{num_iterations} warmup requests failed")
        else:
            print("✓ Warmup complete")

    def calculate_stats(self, timings: List[float]) -> Dict[str, float]:
        """Calculate statistics from timing measurements"""
        if not timings:
            return {
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "p999": 0.0,
                "min": 0.0,
                "max": 0.0,
                "stddev": 0.0
            }

        timings_array = np.array(timings)

        return {
            "mean": float(np.mean(timings_array)),
            "median": float(np.median(timings_array)),
            "p95": float(np.percentile(timings_array, 95)),
            "p99": float(np.percentile(timings_array, 99)),
            "p999": float(np.percentile(timings_array, 99.9)),
            "min": float(np.min(timings_array)),
            "max": float(np.max(timings_array)),
            "stddev": float(np.std(timings_array))
        }

    def run_scenario(self, scenario: Dict, test_data: List[str]) -> BenchmarkResult:
        """
        Run a single benchmark scenario

        Args:
            scenario: Scenario configuration
            test_data: Test dataset

        Returns:
            BenchmarkResult with metrics
        """
        scenario_name = scenario['name']
        concurrency = scenario['concurrency']
        num_requests = scenario['num_requests']
        batch_size = scenario.get('batch_size', 1)

        print(f"\n{'='*70}")
        print(f"Scenario: {scenario_name}")
        print(f"  Concurrency: {concurrency}")
        print(f"  Requests: {num_requests}")
        print(f"  Batch size: {batch_size}")
        print(f"{'='*70}")

        # Prepare samples
        samples = []
        for i in range(num_requests):
            text_idx = i % len(test_data)
            if batch_size == 1:
                samples.append([test_data[text_idx]])
            else:
                batch = [test_data[(text_idx + j) % len(test_data)] for j in range(batch_size)]
                samples.append(batch)

        # Track client resource usage
        self.process.cpu_percent()  # Initialize

        # Benchmark function
        error_count = 0

        def inference_task(texts: List[str]) -> Optional[float]:
            """Single inference task"""
            nonlocal error_count

            start = time.perf_counter()
            result = self.embed(texts)
            latency = (time.perf_counter() - start) * 1000

            if result is None:
                error_count += 1
                return None

            return latency

        # Run benchmark
        timings = []
        start_time = time.perf_counter()

        if concurrency == 1:
            # Single-threaded execution
            for sample in tqdm(samples, desc=f"  Progress"):
                latency = inference_task(sample)
                if latency is not None:
                    timings.append(latency)
        else:
            # Multi-threaded execution
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(inference_task, sample) for sample in samples]

                for future in tqdm(as_completed(futures), total=len(futures), desc=f"  Progress"):
                    latency = future.result()
                    if latency is not None:
                        timings.append(latency)

        total_duration = time.perf_counter() - start_time

        # Calculate metrics
        latency_stats = self.calculate_stats(timings)
        throughput_qps = len(timings) / total_duration if total_duration > 0 else 0
        cpu_percent = self.process.cpu_percent()
        memory_rss = self.process.memory_info().rss / 1024 / 1024
        error_rate = error_count / num_requests if num_requests > 0 else 0

        # Print results
        print(f"\n  Results:")
        print(f"    Latency (ms):")
        print(f"      Mean:   {latency_stats['mean']:8.2f}")
        print(f"      Median: {latency_stats['median']:8.2f}")
        print(f"      P95:    {latency_stats['p95']:8.2f}")
        print(f"      P99:    {latency_stats['p99']:8.2f}")
        print(f"      P99.9:  {latency_stats['p999']:8.2f}")
        print(f"    Throughput: {throughput_qps:.2f} QPS")
        print(f"    Client CPU: {cpu_percent:.1f}%")
        print(f"    Client Memory: {memory_rss:.1f} MB")
        if error_count > 0:
            print(f"    ⚠ Errors: {error_count}/{num_requests} ({error_rate*100:.2f}%)")

        return BenchmarkResult(
            scenario_name=scenario_name,
            concurrency=concurrency,
            num_requests=num_requests,
            batch_size=batch_size,
            latency_ms=latency_stats,
            throughput_qps=throughput_qps,
            total_duration_sec=total_duration,
            cpu_percent=cpu_percent,
            memory_rss_mb=memory_rss,
            errors=error_count,
            error_rate=error_rate
        )

    def run_all_scenarios(self) -> Dict[str, Any]:
        """
        Run all benchmark scenarios

        Returns:
            Complete benchmark results
        """
        print("\n" + "="*70)
        print(f"Benchmarking {self.framework.upper()}")
        print(f"Model: {self.model_name}")
        print(f"Server: {self.server_url}")
        print("="*70)

        # Wait for server
        if not self.wait_for_server():
            raise RuntimeError("Server failed to start")

        # Get server info
        server_info = self.get_server_info()
        print(f"\nServer info:")
        for key, value in server_info.items():
            print(f"  {key}: {value}")

        # Warmup
        warmup_iterations = self.benchmark_config.get('warmup_iterations', 100)
        self.warmup(warmup_iterations)

        # First inference timing
        test_text = "This is a test sentence for benchmarking."
        start = time.perf_counter()
        _ = self.embed([test_text])
        first_inference_time = (time.perf_counter() - start) * 1000

        print(f"\nFirst inference: {first_inference_time:.2f}ms")

        # Load test data
        test_data = self.benchmark_config.get('test_data', [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models are trained on large datasets.",
            "Natural language processing enables computers to understand text.",
            "Deep learning has revolutionized artificial intelligence.",
            "Embedding models convert text into numerical vectors."
        ])

        # Run scenarios
        scenario_results = {}
        for scenario in self.benchmark_config['scenarios']:
            result = self.run_scenario(scenario, test_data)
            scenario_results[scenario['name']] = asdict(result)

        # Get final server info (includes server-side metrics)
        final_server_info = self.get_server_info()

        # Compile results
        results = {
            "framework": self.framework,
            "model_name": self.model_name,
            "server_url": self.server_url,
            "first_inference_ms": first_inference_time,
            "server_info": server_info,
            "final_server_info": final_server_info,
            "scenarios": scenario_results,
            "client_info": {
                "cpu_count": os.cpu_count(),
                "python_version": sys.version.split()[0]
            }
        }

        return results

    def save_results(self, results: Dict[str, Any]):
        """Save results to JSON file"""
        output_dir = Path("/results") / self.model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{self.framework}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")


def main():
    """Main entry point"""
    framework = os.environ.get('FRAMEWORK', 'pytorch')
    model_name = os.environ.get('MODEL_NAME', 'embeddinggemma-300m')
    server_url = os.environ.get('SERVER_URL', f'http://{framework}-server:8000')

    print("="*70)
    print("Embedding Model Benchmark Client")
    print(f"Framework: {framework}")
    print(f"Model: {model_name}")
    print(f"Server: {server_url}")
    print("="*70)

    try:
        client = BenchmarkClient(framework, model_name, server_url)
        results = client.run_all_scenarios()
        client.save_results(results)

        print("\n" + "="*70)
        print("Benchmark completed successfully!")
        print("="*70)

    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
