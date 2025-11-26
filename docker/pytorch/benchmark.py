#!/usr/bin/env python3
"""
PyTorch Benchmark for Embedding Models
Supports all transformer-based embedding models via sentence-transformers
"""

import os
import time
import json
import yaml
import psutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


@dataclass
class BenchmarkResult:
    """Results from a single benchmark scenario"""
    scenario_name: str
    concurrency: int
    num_requests: int
    batch_size: int
    latency_ms: Dict[str, float]
    throughput_qps: float
    total_duration_sec: float
    cpu_percent: float
    memory_rss_mb: float
    memory_peak_mb: float


class PyTorchBenchmark:
    """PyTorch embedding model benchmark"""

    def __init__(self, model_config: Dict, benchmark_config: Dict):
        self.model_config = model_config
        self.benchmark_config = benchmark_config
        self.model = None
        self.model_load_time = 0.0
        self.first_inference_time = 0.0
        self.process = psutil.Process()

    def load_model(self) -> float:
        """Load the model and return load time in seconds"""
        print(f"Loading model: {self.model_config['huggingface_id']}")
        start_time = time.time()

        # Load using sentence-transformers for better performance
        self.model = SentenceTransformer(
            self.model_config['huggingface_id'],
            device='cpu'
        )

        # Set to eval mode
        self.model.eval()

        load_time = time.time() - start_time
        self.model_load_time = load_time * 1000  # Convert to ms

        print(f"Model loaded in {self.model_load_time:.2f} ms")
        return load_time

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for input texts"""
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=len(texts)
            )
        return embeddings

    def get_memory_usage(self) -> int:
        """Return current memory usage in bytes"""
        return self.process.memory_info().rss

    def load_test_data(self) -> List[str]:
        """Load or generate test sentences"""
        dataset_config = self.benchmark_config.get('dataset', {})
        num_samples = dataset_config.get('num_samples', 10000)

        # For now, generate synthetic data
        # TODO: Load from actual dataset
        print(f"Generating {num_samples} test sentences...")

        sentences = []
        templates = [
            "This is a test sentence number {i} for benchmarking embedding models.",
            "The quick brown fox jumps over the lazy dog. Iteration {i}.",
            "Machine learning models are used for various natural language processing tasks. Sample {i}.",
            "Embedding models convert text into numerical vectors for semantic search. Example {i}.",
        ]

        for i in range(num_samples):
            template = templates[i % len(templates)]
            sentences.append(template.format(i=i))

        return sentences

    def warmup(self, test_data: List[str]):
        """Run warmup inferences"""
        warmup_config = self.benchmark_config.get('warmup', {})
        if not warmup_config.get('enabled', True):
            return

        num_warmup = warmup_config.get('num_requests', 100)
        print(f"Running {num_warmup} warmup requests...")

        # Take a subset for warmup
        warmup_data = test_data[:num_warmup]

        # First inference timing
        start_time = time.time()
        self.embed([warmup_data[0]])
        self.first_inference_time = (time.time() - start_time) * 1000

        # Rest of warmup
        for i in range(1, len(warmup_data), 10):
            batch = warmup_data[i:i+10]
            self.embed(batch)

        print(f"Warmup complete. First inference: {self.first_inference_time:.2f} ms")

    def run_scenario(self, scenario: Dict, test_data: List[str]) -> BenchmarkResult:
        """Run a single benchmark scenario"""
        name = scenario['name']
        concurrency = scenario['concurrency']
        num_requests = scenario['num_requests']
        batch_size = scenario['batch_size']

        print(f"\nRunning scenario: {name}")
        print(f"  Concurrency: {concurrency}, Requests: {num_requests}, Batch: {batch_size}")

        # Prepare test samples
        samples = []
        for i in range(num_requests):
            start_idx = (i * batch_size) % len(test_data)
            end_idx = start_idx + batch_size
            batch = test_data[start_idx:end_idx]
            if len(batch) < batch_size:
                batch = test_data[:batch_size]
            samples.append(batch)

        # Track timings
        timings = []
        memory_before = self.get_memory_usage()
        peak_memory = memory_before

        # Monitor CPU and memory
        cpu_samples = []

        def inference_task(texts: List[str]) -> float:
            """Single inference task"""
            nonlocal peak_memory

            start = time.perf_counter()
            self.embed(texts)
            duration = time.perf_counter() - start

            # Update peak memory
            current_memory = self.get_memory_usage()
            peak_memory = max(peak_memory, current_memory)

            return duration * 1000  # Convert to ms

        # Run benchmark
        start_time = time.perf_counter()

        if concurrency == 1:
            # Single-threaded
            for sample in tqdm(samples, desc=f"  Progress"):
                latency = inference_task(sample)
                timings.append(latency)
                cpu_samples.append(self.process.cpu_percent(interval=0.01))
        else:
            # Multi-threaded
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(inference_task, sample) for sample in samples]

                for future in tqdm(as_completed(futures), total=len(futures), desc=f"  Progress"):
                    latency = future.result()
                    timings.append(latency)
                    cpu_samples.append(self.process.cpu_percent(interval=0.01))

        total_duration = time.perf_counter() - start_time

        # Calculate metrics
        timings_array = np.array(timings)
        latency_stats = {
            'mean': float(np.mean(timings_array)),
            'median': float(np.median(timings_array)),
            'p95': float(np.percentile(timings_array, 95)),
            'p99': float(np.percentile(timings_array, 99)),
            'p999': float(np.percentile(timings_array, 99.9)),
            'min': float(np.min(timings_array)),
            'max': float(np.max(timings_array)),
            'stddev': float(np.std(timings_array))
        }

        throughput_qps = num_requests / total_duration
        memory_rss_mb = self.get_memory_usage() / (1024 * 1024)
        memory_peak_mb = peak_memory / (1024 * 1024)
        avg_cpu = np.mean(cpu_samples) if cpu_samples else 0.0

        print(f"  Latency (p50/p95/p99): {latency_stats['median']:.2f}/{latency_stats['p95']:.2f}/{latency_stats['p99']:.2f} ms")
        print(f"  Throughput: {throughput_qps:.2f} QPS")
        print(f"  CPU: {avg_cpu:.1f}%, Memory: {memory_rss_mb:.1f} MB")

        return BenchmarkResult(
            scenario_name=name,
            concurrency=concurrency,
            num_requests=num_requests,
            batch_size=batch_size,
            latency_ms=latency_stats,
            throughput_qps=throughput_qps,
            total_duration_sec=total_duration,
            cpu_percent=avg_cpu,
            memory_rss_mb=memory_rss_mb,
            memory_peak_mb=memory_peak_mb
        )

    def run_all_scenarios(self, test_data: List[str]) -> List[BenchmarkResult]:
        """Run all benchmark scenarios"""
        scenarios = self.benchmark_config.get('scenarios', [])
        results = []

        for scenario in scenarios:
            result = self.run_scenario(scenario, test_data)
            results.append(result)

        return results

    def save_results(self, results: List[BenchmarkResult], output_path: Path):
        """Save benchmark results to JSON"""
        output_data = {
            'framework': 'pytorch',
            'model': self.model_config['name'],
            'model_id': self.model_config['huggingface_id'],
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'system_info': {
                'cpu': 'Intel Xeon 8488C',
                'cores': psutil.cpu_count(logical=True),
                'memory_gb': psutil.virtual_memory().total / (1024**3)
            },
            'model_load_time_ms': self.model_load_time,
            'first_inference_ms': self.first_inference_time,
            'scenarios': {r.scenario_name: asdict(r) for r in results}
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {output_path}")


def main():
    """Main benchmark entry point"""
    # Load configurations
    model_config_path = Path(os.getenv('MODEL_CONFIG', '/config/models.yaml'))
    benchmark_config_path = Path(os.getenv('BENCHMARK_CONFIG', '/config/benchmark.yaml'))
    model_name = os.getenv('MODEL_NAME', 'embeddinggemma-300m')

    print(f"Loading configurations...")
    print(f"  Model config: {model_config_path}")
    print(f"  Benchmark config: {benchmark_config_path}")
    print(f"  Model: {model_name}")

    with open(model_config_path) as f:
        models_config = yaml.safe_load(f)

    with open(benchmark_config_path) as f:
        benchmark_config = yaml.safe_load(f)

    model_config = models_config['models'][model_name]

    print(f"\n{'='*60}")
    print(f"PyTorch Benchmark - {model_config['name']}")
    print(f"{'='*60}\n")

    # Run benchmark
    benchmark = PyTorchBenchmark(model_config, benchmark_config)

    # Load model
    benchmark.load_model()

    # Load test data
    test_data = benchmark.load_test_data()

    # Warmup
    benchmark.warmup(test_data)

    # Run all scenarios
    results = benchmark.run_all_scenarios(test_data)

    # Save results
    output_path = Path(f"/results/{model_name}/pytorch.json")
    benchmark.save_results(results, output_path)

    print(f"\n{'='*60}")
    print(f"Benchmark complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
