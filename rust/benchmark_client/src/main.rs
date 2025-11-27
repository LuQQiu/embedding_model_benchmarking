use anyhow::{Context, Result};
use common::{BenchmarkConfig, EmbedRequest, Statistics};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use sysinfo::{ProcessRefreshKind, RefreshKind, System};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkResult {
    scenario_name: String,
    concurrency: usize,
    num_requests: usize,
    batch_size: usize,
    latency_ms: Statistics,
    throughput_qps: f64,
    total_duration_sec: f64,
    cpu_percent: f32,
    memory_rss_mb: f64,
    errors: usize,
    error_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FullBenchmarkResult {
    framework: String,
    model_name: String,
    server_info: serde_json::Value,
    results: HashMap<String, BenchmarkResult>,
    timestamp: String,
}

struct BenchmarkClient {
    framework: String,
    model_name: String,
    server_url: String,
    client: reqwest::blocking::Client,
}

impl BenchmarkClient {
    fn new(framework: String, model_name: String, server_url: String) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()?;

        Ok(Self {
            framework,
            model_name,
            server_url,
            client,
        })
    }

    fn wait_for_server(&self, timeout_secs: u64) -> Result<()> {
        let health_url = format!("{}/health", self.server_url);
        let start = Instant::now();

        println!("Waiting for server at {}...", self.server_url);

        while start.elapsed().as_secs() < timeout_secs {
            if let Ok(response) = self.client.get(&health_url).send() {
                if response.status().is_success() {
                    println!("✓ Server is ready!");
                    return Ok(());
                }
            }
            std::thread::sleep(Duration::from_secs(2));
        }

        anyhow::bail!("✗ Server failed to start within {}s", timeout_secs)
    }

    fn get_server_info(&self) -> Result<serde_json::Value> {
        let response = self
            .client
            .get(format!("{}/info", self.server_url))
            .send()
            .context("Failed to get server info")?;

        let info: serde_json::Value = response.json()?;
        Ok(info)
    }

    fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let request = EmbedRequest { texts };

        let response = self
            .client
            .post(format!("{}/embed", self.server_url))
            .json(&request)
            .send()
            .context("Failed to send embed request")?;

        if !response.status().is_success() {
            anyhow::bail!("Server returned error: {}", response.status());
        }

        let result: serde_json::Value = response.json()?;
        let embeddings: Vec<Vec<f32>> = serde_json::from_value(result["embeddings"].clone())?;

        Ok(embeddings)
    }

    fn warmup(&self, num_iterations: usize) -> Result<()> {
        println!("\nWarming up server ({} iterations)...", num_iterations);

        let pb = ProgressBar::new(num_iterations as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{msg} [{bar:40.cyan/blue}] {pos}/{len}")
                .unwrap()
                .progress_chars("=>-"),
        );
        pb.set_message("Warmup");

        let warmup_text = "This is a warmup sentence to initialize the model.".to_string();
        let mut errors = 0;

        for _ in 0..num_iterations {
            if self.embed(vec![warmup_text.clone()]).is_err() {
                errors += 1;
            }
            pb.inc(1);
        }

        pb.finish_with_message("✓ Warmup complete");

        if errors > 0 {
            println!(
                "⚠ Warning: {}/{} warmup requests failed",
                errors, num_iterations
            );
        }

        Ok(())
    }

    fn run_scenario(
        &self,
        scenario: &common::ScenarioConfig,
        test_data: &[String],
    ) -> Result<BenchmarkResult> {
        println!("\n{}", "=".repeat(70));
        println!("Scenario: {}", scenario.name);
        println!("  Concurrency: {}", scenario.concurrency);
        println!("  Requests: {}", scenario.num_requests);
        println!("  Batch size: {}", scenario.batch_size);
        println!("{}", "=".repeat(70));

        // Prepare samples
        let mut samples = Vec::new();
        for i in 0..scenario.num_requests {
            let batch: Vec<String> = test_data
                .iter()
                .cycle()
                .skip(i * scenario.batch_size)
                .take(scenario.batch_size)
                .cloned()
                .collect();
            samples.push(batch);
        }

        // Run benchmark
        let errors = Arc::new(AtomicUsize::new(0));
        let latencies = Arc::new(std::sync::Mutex::new(Vec::new()));

        let pb = ProgressBar::new(scenario.num_requests as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{msg} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("=>-"),
        );
        pb.set_message("Running");

        let start_time = Instant::now();

        // Use rayon to run requests in parallel
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(scenario.concurrency)
            .build()?;

        pool.install(|| {
            samples.par_iter().for_each(|batch| {
                let request_start = Instant::now();

                match self.embed(batch.clone()) {
                    Ok(_) => {
                        let latency = request_start.elapsed().as_secs_f64() * 1000.0;
                        latencies.lock().unwrap().push(latency);
                    }
                    Err(_) => {
                        errors.fetch_add(1, Ordering::SeqCst);
                    }
                }

                pb.inc(1);
            });
        });

        let total_duration = start_time.elapsed().as_secs_f64();
        pb.finish_with_message("✓ Complete");

        // Calculate statistics
        let latencies_vec = latencies.lock().unwrap().clone();
        let latency_stats = Statistics::from_samples(&latencies_vec);

        let error_count = errors.load(Ordering::SeqCst);
        let successful_requests = scenario.num_requests - error_count;
        let throughput = successful_requests as f64 / total_duration;
        let error_rate = error_count as f64 / scenario.num_requests as f64;

        // Get system metrics
        let mut sys = System::new_with_specifics(
            RefreshKind::new().with_processes(ProcessRefreshKind::everything()),
        );
        sys.refresh_all();

        let current_pid = sysinfo::get_current_pid().unwrap();
        let process = sys.process(current_pid).unwrap();

        let result = BenchmarkResult {
            scenario_name: scenario.name.clone(),
            concurrency: scenario.concurrency,
            num_requests: scenario.num_requests,
            batch_size: scenario.batch_size,
            latency_ms: latency_stats,
            throughput_qps: throughput,
            total_duration_sec: total_duration,
            cpu_percent: process.cpu_usage(),
            memory_rss_mb: process.memory() as f64 / 1024.0 / 1024.0,
            errors: error_count,
            error_rate,
        };

        // Print results
        println!("\nResults:");
        println!("  Total duration: {:.2}s", result.total_duration_sec);
        println!("  Throughput: {:.2} req/s", result.throughput_qps);
        println!("  Latency (mean): {:.2}ms", result.latency_ms.mean);
        println!("  Latency (p50): {:.2}ms", result.latency_ms.median);
        println!("  Latency (p95): {:.2}ms", result.latency_ms.p95);
        println!("  Latency (p99): {:.2}ms", result.latency_ms.p99);
        println!("  Errors: {}", result.errors);
        println!("  Error rate: {:.2}%", result.error_rate * 100.0);

        Ok(result)
    }

    fn run_benchmark(&self, config: &BenchmarkConfig) -> Result<FullBenchmarkResult> {
        // Get server info
        let server_info = self.get_server_info()?;
        println!("\nServer Info:");
        println!("{}", serde_json::to_string_pretty(&server_info)?);

        // Load test data (simple for now)
        let test_data: Vec<String> = (0..100)
            .map(|i| format!("This is test sentence number {}. It is used for benchmarking the embedding model performance.", i))
            .collect();

        // Warmup
        if config.warmup.enabled {
            self.warmup(config.warmup.num_requests)?;
        }

        // Run scenarios
        let mut results = HashMap::new();

        for scenario in &config.scenarios {
            let result = self.run_scenario(scenario, &test_data)?;
            results.insert(scenario.name.clone(), result);
        }

        Ok(FullBenchmarkResult {
            framework: self.framework.clone(),
            model_name: self.model_name.clone(),
            server_info,
            results,
            timestamp: chrono::Utc::now().to_rfc3339(),
        })
    }
}

fn main() -> Result<()> {
    println!("========================================================================");
    println!("ONNX Rust Benchmark Client");
    println!("========================================================================");

    // Get environment variables
    let framework = env::var("FRAMEWORK").unwrap_or_else(|_| "onnx-rust".to_string());
    let model_name = env::var("MODEL_NAME").context("MODEL_NAME not set")?;
    let server_url = env::var("SERVER_URL").unwrap_or_else(|_| "http://localhost:8000".to_string());

    println!("Framework: {}", framework);
    println!("Model: {}", model_name);
    println!("Server URL: {}", server_url);

    // Load benchmark config
    let config_path = Path::new("/config/benchmark.yaml");
    let config = BenchmarkConfig::load(config_path)
        .context("Failed to load benchmark configuration")?;

    // Create client
    let client = BenchmarkClient::new(framework.clone(), model_name.clone(), server_url)?;

    // Wait for server
    client.wait_for_server(120)?;

    // Run benchmarks
    let results = client.run_benchmark(&config)?;

    // Save results
    let output_dir = Path::new("/results").join(&model_name);
    fs::create_dir_all(&output_dir)?;

    let output_path = output_dir.join(format!("{}.json", framework));
    let json = serde_json::to_string_pretty(&results)?;
    fs::write(&output_path, json)?;

    println!("\n========================================================================");
    println!("Benchmark complete!");
    println!("Results saved to: {}", output_path.display());
    println!("========================================================================");

    Ok(())
}
