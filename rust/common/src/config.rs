use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    pub models: HashMap<String, ModelConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    #[serde(rename = "type")]
    pub model_type: String,
    pub huggingface_id: String,
    pub max_seq_length: usize,
    pub embedding_dim: usize,
    pub params: String,
    pub paths: ModelPaths,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPaths {
    pub pytorch: String,
    pub onnx: String,
    pub openvino: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub active_model: String,
    pub dataset: DatasetConfig,
    pub warmup: WarmupConfig,
    pub scenarios: Vec<ScenarioConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub name: String,
    pub source: String,
    pub subset: String,
    pub num_samples: usize,
    pub sequence_lengths: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupConfig {
    pub enabled: bool,
    pub num_requests: usize,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioConfig {
    pub name: String,
    pub concurrency: usize,
    pub num_requests: usize,
    pub batch_size: usize,
    pub description: String,
}

impl ModelsConfig {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: ModelsConfig = serde_yaml::from_str(&contents)?;
        Ok(config)
    }

    pub fn get_model(&self, name: &str) -> Option<&ModelConfig> {
        self.models.get(name)
    }
}

impl BenchmarkConfig {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: BenchmarkConfig = serde_yaml::from_str(&contents)?;
        Ok(config)
    }
}
