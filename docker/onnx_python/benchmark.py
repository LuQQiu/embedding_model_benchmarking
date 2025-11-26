#!/usr/bin/env python3
"""
ONNX Runtime Python Benchmark for Embedding Models
"""

import os
import sys
import time
import json
import yaml
import psutil
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import onnxruntime as ort
from transformers import AutoTokenizer


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
    memory_peak_mb: float


class ONNXPythonBenchmark:
    """ONNX Runtime Python benchmark implementation"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_config = None
        self.session = None
        self.tokenizer = None
        self.process = psutil.Process()

        # Load configs
        self._load_configs()

    def _load_configs(self):
        """Load model and benchmark configurations"""
        # Load models.yaml
        models_config_path = Path("/config/models.yaml")
        with open(models_config_path) as f:
            models_config = yaml.safe_load(f)

        if self.model_name not in models_config['models']:
            raise ValueError(f"Model {self.model_name} not found in config")

        self.model_config = models_config['models'][self.model_name]

        # Load benchmark.yaml
        benchmark_config_path = Path("/config/benchmark.yaml")
        with open(benchmark_config_path) as f:
            self.benchmark_config = yaml.safe_load(f)

    def load_model(self) -> float:
        """
        Load ONNX model and tokenizer

        Returns:
            Model load time in milliseconds
        """
        print("Loading ONNX model...")

        start_time = time.perf_counter()

        # Get ONNX model path
        onnx_path = self.model_config['paths']['onnx']
        if not Path(onnx_path).exists():
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_path}\n"
                f"Please run: python3 scripts/convert_to_onnx.py --model {self.model_name}"
            )

        # Configure ONNX Runtime session options
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = os.cpu_count()
        session_options.inter_op_num_threads = os.cpu_count()

        # Create inference session
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=session_options,
            providers=providers
        )

        # Load tokenizer from ONNX directory
        tokenizer_path = Path(onnx_path).parent
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

        load_time = (time.perf_counter() - start_time) * 1000

        print(f"✓ Model loaded in {load_time:.2f}ms")
        print(f"  ONNX Runtime version: {ort.__version__}")
        print(f"  Providers: {self.session.get_providers()}")
        print(f"  Intra-op threads: {session_options.intra_op_num_threads}")

        return load_time

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using ONNX Runtime

        Args:
            texts: List of input texts

        Returns:
            Embeddings as numpy array
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.model_config['max_seq_length'],
            return_tensors="np"
        )

        # Prepare inputs for ONNX Runtime
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }

        # Run inference
        outputs = self.session.run(None, ort_inputs)

        # Get last hidden state (first output)
        last_hidden_state = outputs[0]

        # Mean pooling
        attention_mask = inputs["attention_mask"]
        attention_mask_expanded = np.expand_dims(attention_mask, -1)

        sum_embeddings = np.sum(last_hidden_state * attention_mask_expanded, axis=1)
        sum_mask = np.clip(attention_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)

        embeddings = sum_embeddings / sum_mask

        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings

    def warmup(self, num_iterations: int = 100):
        """
        Warmup the model

        Args:
            num_iterations: Number of warmup iterations
        """
        print(f"\nWarming up model ({num_iterations} iterations)...")

        warmup_text = "This is a warmup sentence to initialize the model."

        for _ in tqdm(range(num_iterations), desc="Warmup"):
            _ = self.embed([warmup_text])

        print("✓ Warmup complete")

    def calculate_stats(self, timings: List[float]) -> Dict[str, float]:
        """Calculate statistics from timing measurements"""
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

        # Track resource usage
        self.process.cpu_percent()  # Initialize
        memory_before = self.process.memory_info().rss / 1024 / 1024
        peak_memory = memory_before

        # Benchmark function
        def inference_task(texts: List[str]) -> float:
            """Single inference task"""
            nonlocal peak_memory

            start = time.perf_counter()
            _ = self.embed(texts)
            latency = (time.perf_counter() - start) * 1000

            # Track peak memory
            current_memory = self.process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)

            return latency

        # Run benchmark
        timings = []
        start_time = time.perf_counter()

        if concurrency == 1:
            # Single-threaded execution
            for sample in tqdm(samples, desc=f"  Progress"):
                latency = inference_task(sample)
                timings.append(latency)
        else:
            # Multi-threaded execution
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(inference_task, sample) for sample in samples]

                for future in tqdm(as_completed(futures), total=len(futures), desc=f"  Progress"):
                    latency = future.result()
                    timings.append(latency)

        total_duration = time.perf_counter() - start_time

        # Calculate metrics
        latency_stats = self.calculate_stats(timings)
        throughput_qps = num_requests / total_duration
        cpu_percent = self.process.cpu_percent()
        memory_rss = self.process.memory_info().rss / 1024 / 1024

        # Print results
        print(f"\n  Results:")
        print(f"    Latency (ms):")
        print(f"      Mean:   {latency_stats['mean']:8.2f}")
        print(f"      Median: {latency_stats['median']:8.2f}")
        print(f"      P95:    {latency_stats['p95']:8.2f}")
        print(f"      P99:    {latency_stats['p99']:8.2f}")
        print(f"      P99.9:  {latency_stats['p999']:8.2f}")
        print(f"    Throughput: {throughput_qps:.2f} QPS")
        print(f"    CPU: {cpu_percent:.1f}%")
        print(f"    Memory: {memory_rss:.1f} MB (peak: {peak_memory:.1f} MB)")

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
            memory_peak_mb=peak_memory
        )

    def run_all_scenarios(self) -> Dict[str, Any]:
        """
        Run all benchmark scenarios

        Returns:
            Complete benchmark results
        """
        print("\n" + "="*70)
        print("ONNX Runtime Python Benchmark")
        print(f"Model: {self.model_config['name']}")
        print("="*70)

        # Load model
        model_load_time = self.load_model()

        # Warmup
        warmup_iterations = self.benchmark_config.get('warmup_iterations', 100)
        self.warmup(warmup_iterations)

        # First inference (cold start)
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

        # Compile results
        results = {
            "framework": "onnx-python",
            "onnx_runtime_version": ort.__version__,
            "model": self.model_config['name'],
            "model_name": self.model_name,
            "model_load_time_ms": model_load_time,
            "first_inference_ms": first_inference_time,
            "scenarios": scenario_results,
            "system_info": {
                "cpu_count": os.cpu_count(),
                "python_version": sys.version.split()[0]
            }
        }

        return results

    def save_results(self, results: Dict[str, Any]):
        """Save results to JSON file"""
        output_dir = Path("/results") / self.model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "onnx-python.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")


def main():
    """Main entry point"""
    model_name = os.environ.get('MODEL_NAME', 'embeddinggemma-300m')

    print("="*70)
    print("ONNX Runtime Python Embedding Benchmark")
    print(f"Model: {model_name}")
    print("="*70)

    try:
        benchmark = ONNXPythonBenchmark(model_name)
        results = benchmark.run_all_scenarios()
        benchmark.save_results(results)

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
