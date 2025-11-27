#!/usr/bin/env python3
"""
Benchmark Orchestrator - Client-Server Architecture

Manages lifecycle of server-client pairs for each framework:
1. Start server
2. Wait for health check
3. Run client benchmark
4. Stop server
5. Clean up
"""

import argparse
import subprocess
import json
import yaml
import time
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


class BenchmarkOrchestrator:
    """Orchestrates multi-framework benchmarks with client-server architecture"""

    def __init__(self, model_name: str, frameworks: List[str] = None):
        self.model_name = model_name
        self.frameworks = frameworks or [
            "pytorch",
            "onnx-python",
            "openvino"
            # "onnx-rust",  # Not implemented yet
            # "onnx-native",  # Not implemented yet
            # "candle",  # Not implemented yet
        ]
        self.results_dir = Path("results") / model_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def start_server(self, framework: str) -> bool:
        """
        Start framework server using docker-compose

        Args:
            framework: Framework name (e.g., 'pytorch', 'onnx-python')

        Returns:
            True if server started successfully
        """
        service_name = f"{framework}-server"

        print(f"  Starting {framework} server...")

        try:
            cmd = [
                "docker-compose",
                "up",
                "-d",
                service_name
            ]

            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            print(f"  ✓ Server started: {service_name}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to start server: {e.stderr}")
            return False

    def wait_for_server_health(self, framework: str, timeout: int = 120) -> bool:
        """
        Wait for server to be healthy

        Args:
            framework: Framework name
            timeout: Maximum time to wait in seconds

        Returns:
            True if server is healthy
        """
        service_name = f"{framework}-server"

        print(f"  Waiting for {framework} server to be healthy...")

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Check container health status
                cmd = [
                    "docker",
                    "inspect",
                    "--format={{.State.Health.Status}}",
                    service_name
                ]

                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True
                )

                health_status = result.stdout.strip()

                if health_status == "healthy":
                    print(f"  ✓ Server is healthy")
                    return True
                # Don't fail immediately on "unhealthy" - could be during start_period
                # Only fail if we timeout

            except subprocess.CalledProcessError:
                pass

            time.sleep(2)

        print(f"  ✗ Server failed to become healthy within {timeout}s")
        return False

    def run_client(self, framework: str) -> bool:
        """
        Run benchmark client

        Args:
            framework: Framework name

        Returns:
            True if benchmark completed successfully
        """
        service_name = f"{framework}-client"

        print(f"  Running {framework} benchmark client...")

        try:
            cmd = [
                "docker-compose",
                "run",
                "--rm",
                "-e", f"MODEL_NAME={self.model_name}",
                service_name
            ]

            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,  # Show output in real-time
                text=True
            )

            print(f"  ✓ Benchmark completed")
            return True

        except subprocess.CalledProcessError as e:
            print(f"  ✗ Benchmark failed with error code {e.returncode}")
            return False

    def stop_server(self, framework: str):
        """
        Stop framework server

        Args:
            framework: Framework name
        """
        service_name = f"{framework}-server"

        print(f"  Stopping {framework} server...")

        try:
            cmd = [
                "docker-compose",
                "stop",
                service_name
            ]

            subprocess.run(cmd, check=True, capture_output=True)

            # Remove container
            cmd = [
                "docker-compose",
                "rm",
                "-f",
                service_name
            ]

            subprocess.run(cmd, check=True, capture_output=True)

            print(f"  ✓ Server stopped")

        except subprocess.CalledProcessError as e:
            print(f"  ⚠ Warning: Failed to stop server: {e.stderr}")

    def run_single_benchmark(self, framework: str) -> bool:
        """
        Run complete benchmark for a single framework

        Args:
            framework: Framework name

        Returns:
            True if benchmark completed successfully
        """
        print(f"\n{'='*70}")
        print(f"Benchmarking {framework.upper()}")
        print(f"Model: {self.model_name}")
        print(f"{'='*70}\n")

        try:
            # 1. Start server
            if not self.start_server(framework):
                return False

            # 2. Wait for server to be healthy
            if not self.wait_for_server_health(framework):
                self.stop_server(framework)
                return False

            # 3. Run benchmark client
            success = self.run_client(framework)

            # 4. Stop server
            self.stop_server(framework)

            # Brief pause before next benchmark
            time.sleep(2)

            return success

        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")
            self.stop_server(framework)
            return False

    def run_all_benchmarks(self, skip_on_error: bool = False) -> Dict[str, bool]:
        """
        Run all framework benchmarks sequentially

        Args:
            skip_on_error: Continue on error

        Returns:
            Dictionary mapping framework names to success status
        """
        results = {}

        start_time = time.time()

        for framework in self.frameworks:
            success = self.run_single_benchmark(framework)
            results[framework] = success

            if not success and not skip_on_error:
                print(f"\nStopping benchmarks due to {framework} failure")
                break

        total_time = time.time() - start_time

        # Print summary
        print(f"\n{'='*70}")
        print(f"Benchmark Summary")
        print(f"{'='*70}")
        print(f"Model: {self.model_name}")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"\nResults:")

        for framework, success in results.items():
            status = "✓ Success" if success else "✗ Failed"
            print(f"  {framework:20s}: {status}")

        return results

    def collect_results(self) -> Dict[str, Any]:
        """Collect all benchmark results"""
        results = {}

        for framework in self.frameworks:
            result_file = self.results_dir / f"{framework}.json"

            if result_file.exists():
                with open(result_file) as f:
                    results[framework] = json.load(f)
            else:
                print(f"Warning: Results not found for {framework}")

        return results

    def generate_summary(self) -> str:
        """Generate a summary report"""
        results = self.collect_results()

        if not results:
            return "No results found"

        summary = []
        summary.append(f"\n{'='*70}")
        summary.append(f"Benchmark Results Summary - {self.model_name}")
        summary.append(f"{'='*70}\n")

        # Table header
        summary.append(f"{'Framework':<20} {'1st Inf (ms)':<14} {'p50 (ms)':<12} {'p95 (ms)':<12} {'QPS':<10}")
        summary.append("-" * 70)

        # Extract key metrics from each framework
        for framework, data in results.items():
            first_inference = data.get('first_inference_ms', 0)

            # Get single_query scenario (baseline)
            scenarios = data.get('scenarios', {})
            single_query = scenarios.get('single_query', {})

            if single_query:
                latency = single_query.get('latency_ms', {})
                p50 = latency.get('median', 0)
                p95 = latency.get('p95', 0)
                qps = single_query.get('throughput_qps', 0)

                summary.append(
                    f"{framework:<20} {first_inference:<14.2f} {p50:<12.2f} {p95:<12.2f} {qps:<10.2f}"
                )

        summary.append("\n" + "="*70)

        return "\n".join(summary)

    def export_to_csv(self, output_file: str = None) -> str:
        """
        Export benchmark results to CSV format

        CSV columns: timestamp, model, runtime, language, concurrency, qps, p50_latency_ms, p90_latency_ms,
                     p95_latency_ms, p99_latency_ms, server_cpu_percent, server_memory_mb,
                     client_cpu_percent, client_memory_mb

        Args:
            output_file: Path to output CSV file (default: results/{model}/benchmark_results_{timestamp}.csv)

        Returns:
            Path to generated CSV file
        """
        results = self.collect_results()

        if not results:
            print("No results to export")
            return None

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Determine output file
        if output_file is None:
            timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.results_dir / f"benchmark_results_{timestamp_file}.csv"
        else:
            output_file = Path(output_file)

        # Prepare CSV data
        csv_rows = []

        for framework, data in results.items():
            # Determine runtime and language
            runtime_map = {
                'pytorch': ('PyTorch', 'Python'),
                'onnx-python': ('ONNX Runtime', 'Python'),
                'onnx-rust': ('ONNX Runtime', 'Rust'),
                'onnx-native': ('ONNX Runtime', 'C++'),
                'candle': ('Candle', 'Rust'),
                'openvino': ('OpenVINO', 'Python')
            }

            runtime, language = runtime_map.get(framework, (framework, 'Unknown'))

            # Extract server resource info (from final_server_info)
            final_server_info = data.get('final_server_info', {})
            server_cpu_percent = final_server_info.get('cpu_percent', 0)
            server_memory_mb = final_server_info.get('memory_rss_mb', 0)

            # Extract scenarios
            scenarios = data.get('scenarios', {})

            for scenario_name, scenario_data in scenarios.items():
                concurrency = scenario_data.get('concurrency', 0)
                qps = scenario_data.get('throughput_qps', 0)

                latency = scenario_data.get('latency_ms', {})
                p50 = latency.get('median', 0)
                p90 = latency.get('p90', 0)  # Will be 0 if not present
                p95 = latency.get('p95', 0)
                p99 = latency.get('p99', 0)

                # Client resource usage (per scenario)
                client_cpu_percent = scenario_data.get('cpu_percent', 0)
                client_memory_mb = scenario_data.get('memory_rss_mb', 0)

                csv_rows.append({
                    'timestamp': timestamp,
                    'model': self.model_name,
                    'runtime': runtime,
                    'language': language,
                    'concurrency': concurrency,
                    'qps': f"{qps:.2f}",
                    'p50_latency_ms': f"{p50:.2f}",
                    'p90_latency_ms': f"{p90:.2f}",
                    'p95_latency_ms': f"{p95:.2f}",
                    'p99_latency_ms': f"{p99:.2f}",
                    'server_cpu_percent': f"{server_cpu_percent:.1f}",
                    'server_memory_mb': f"{server_memory_mb:.1f}",
                    'client_cpu_percent': f"{client_cpu_percent:.1f}",
                    'client_memory_mb': f"{client_memory_mb:.1f}"
                })

        # Write CSV
        fieldnames = [
            'timestamp', 'model', 'runtime', 'language', 'concurrency',
            'qps', 'p50_latency_ms', 'p90_latency_ms', 'p95_latency_ms', 'p99_latency_ms',
            'server_cpu_percent', 'server_memory_mb', 'client_cpu_percent', 'client_memory_mb'
        ]

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"\n✓ CSV results exported to: {output_file}")
        print(f"  Total rows: {len(csv_rows)}")

        return str(output_file)


def main():
    parser = argparse.ArgumentParser(
        description='Run embedding model benchmarks with client-server architecture'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='embeddinggemma-300m',
        help='Model to benchmark (default: embeddinggemma-300m)'
    )
    parser.add_argument(
        '--frameworks',
        nargs='+',
        help='Specific frameworks to run (default: all available)'
    )
    parser.add_argument(
        '--skip-on-error',
        action='store_true',
        help='Continue running other frameworks if one fails'
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Only print summary of existing results'
    )
    parser.add_argument(
        '--csv',
        type=str,
        nargs='?',
        const='auto',
        help='Export results to CSV file (specify path or use auto-generated name)'
    )

    args = parser.parse_args()

    orchestrator = BenchmarkOrchestrator(
        model_name=args.model,
        frameworks=args.frameworks
    )

    if args.summary_only:
        print(orchestrator.generate_summary())
        # Export to CSV if requested
        if args.csv:
            csv_path = None if args.csv == 'auto' else args.csv
            orchestrator.export_to_csv(csv_path)
    else:
        # Run benchmarks
        orchestrator.run_all_benchmarks(skip_on_error=args.skip_on_error)

        # Print summary
        print(orchestrator.generate_summary())

        # Export to CSV if requested
        if args.csv:
            csv_path = None if args.csv == 'auto' else args.csv
            orchestrator.export_to_csv(csv_path)


if __name__ == '__main__':
    main()
