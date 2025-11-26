use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedRequest {
    pub texts: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub inference_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub model_loaded: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfoResponse {
    pub framework: String,
    pub model_name: String,
    pub model_configuration: HashMap<String, serde_json::Value>,
    pub model_load_time_ms: f64,
    pub total_requests: u64,
    pub runtime_version: String,
    pub device: String,
    pub cpu_count: usize,
    pub memory_rss_mb: f64,
    pub cpu_percent: f32,
}
