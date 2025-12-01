use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::gemma::{Config, Model};
use common::{
    config::ModelsConfig, EmbedRequest, EmbedResponse, HealthResponse, InfoResponse, ModelConfig,
};
use serde_json::json;
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};
use sysinfo::System;
use tokenizers::Tokenizer;
use tower_http::trace::TraceLayer;
use tracing::info;

// Model pool for concurrent inference
struct ModelPool {
    models: Vec<Arc<tokio::sync::Mutex<Model>>>,
    next_idx: AtomicU64,
}

impl ModelPool {
    fn new(models: Vec<Model>) -> Self {
        Self {
            models: models.into_iter().map(|m| Arc::new(tokio::sync::Mutex::new(m))).collect(),
            next_idx: AtomicU64::new(0),
        }
    }

    fn get_model(&self) -> Arc<tokio::sync::Mutex<Model>> {
        let idx = self.next_idx.fetch_add(1, Ordering::Relaxed) as usize % self.models.len();
        self.models[idx].clone()
    }
}

// Application state
struct AppState {
    model_pool: Arc<ModelPool>,
    tokenizer: Tokenizer,
    device: Device,
    model_config: ModelConfig,
    model_load_time_ms: f64,
    total_requests: AtomicU64,
    system: Arc<tokio::sync::Mutex<System>>,
    max_seq_length: usize,
}

// Error handling
enum AppError {
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AppError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        let body = Json(json!({
            "error": message,
        }));

        (status, body).into_response()
    }
}

impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        AppError::Internal(err.to_string())
    }
}

impl From<candle_core::Error> for AppError {
    fn from(err: candle_core::Error) -> Self {
        AppError::Internal(format!("Candle error: {}", err))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter("candle_server=info,tower_http=info")
        .init();

    info!("======================================================================");
    info!("Candle Rust Server - Starting");
    info!("======================================================================");

    // Load configuration
    let model_name =
        std::env::var("MODEL_NAME").unwrap_or_else(|_| "embeddinggemma-300m".to_string());
    info!("Model: {}", model_name);

    let models_config = ModelsConfig::load("/config/models.yaml")?;
    let model_config = models_config
        .get_model(&model_name)
        .ok_or_else(|| anyhow::anyhow!("Model {} not found in config", model_name))?
        .clone();

    let max_seq_length = model_config.max_seq_length;

    // Determine number of model instances (one per CPU core)
    let num_workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    info!("Creating {} model instances for concurrent inference", num_workers);

    // Load model
    info!("Loading model: {}", model_config.name);
    let start_time = Instant::now();

    // Use CPU for now (can add GPU support later)
    let device = Device::Cpu;
    info!("Using device: {:?}", device);

    // Load model from local files
    //  Candle version uses model_candle.safetensors which has "model." prefix on all tensors
    let model_dir = format!("/models/{}/pytorch", model_name);
    info!("Loading model from: {}", model_dir);

    let config_filename = format!("{}/config.json", model_dir);
    let tokenizer_filename = format!("{}/tokenizer.json", model_dir);
    let weights_filename = format!("{}/model_candle.safetensors", model_dir);

    info!("✓ Model file paths resolved");

    // Load config
    let config_content = std::fs::read_to_string(&config_filename)?;
    let config: Config = serde_json::from_str(&config_content)?;
    info!("✓ Config loaded");

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(&tokenizer_filename)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    info!("✓ Tokenizer loaded");

    // Create multiple model instances
    let mut models = Vec::new();
    for i in 0..num_workers {
        info!("  Loading model instance {}/{}...", i + 1, num_workers);

        // Load weights for this instance
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[weights_filename.clone()],
                candle_core::DType::F32,
                &device
            )?
        };

        // Create model (use_flash_attn=false for CPU inference)
        let model = Model::new(false, &config, vb)?;
        models.push(model);
    }

    let model_load_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    info!("✓ {} model instances loaded in {:.2}ms", num_workers, model_load_time_ms);
    info!("  Candle version: 0.8");
    info!("  Device: {:?}", device);
    info!("");
    info!("Server ready on http://0.0.0.0:8000");
    info!("======================================================================");

    // Create model pool
    let model_pool = Arc::new(ModelPool::new(models));

    // Create application state
    let state = Arc::new(AppState {
        model_pool,
        tokenizer,
        device,
        model_config,
        model_load_time_ms,
        total_requests: AtomicU64::new(0),
        system: Arc::new(tokio::sync::Mutex::new(System::new_all())),
        max_seq_length,
    });

    // Build router
    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/info", get(info_handler))
        .route("/embed", post(embed))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // Start server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn root() -> Json<serde_json::Value> {
    Json(json!({
        "service": "Candle Rust Embedding Server",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "embed": "/embed (POST)"
        }
    }))
}

async fn health(State(_state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        model_loaded: true,
    })
}

async fn info_handler(State(state): State<Arc<AppState>>) -> Result<Json<InfoResponse>, AppError> {
    let mut system = state.system.lock().await;
    system.refresh_all();

    let process = system
        .process(sysinfo::get_current_pid().unwrap())
        .unwrap();
    let memory_rss_mb = process.memory() as f64 / 1024.0 / 1024.0;
    let cpu_percent = process.cpu_usage();

    // Convert model config to HashMap
    let mut model_configuration = HashMap::new();
    model_configuration.insert("name".to_string(), json!(state.model_config.name));
    model_configuration.insert("type".to_string(), json!(state.model_config.model_type));
    model_configuration.insert(
        "max_seq_length".to_string(),
        json!(state.model_config.max_seq_length),
    );
    model_configuration.insert(
        "embedding_dim".to_string(),
        json!(state.model_config.embedding_dim),
    );

    let cpu_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    Ok(Json(InfoResponse {
        framework: "candle-rust".to_string(),
        model_name: std::env::var("MODEL_NAME")
            .unwrap_or_else(|_| "embeddinggemma-300m".to_string()),
        model_configuration,
        model_load_time_ms: state.model_load_time_ms,
        total_requests: state.total_requests.load(Ordering::Relaxed),
        runtime_version: "0.8".to_string(),
        device: format!("{:?}", state.device),
        cpu_count,
        memory_rss_mb,
        cpu_percent,
    }))
}

async fn embed(
    State(state): State<Arc<AppState>>,
    Json(request): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, AppError> {
    if request.texts.is_empty() {
        return Err(AppError::Internal("No texts provided".to_string()));
    }

    let start_time = Instant::now();

    // Log the request for debugging
    tracing::debug!("Embedding request for {} texts", request.texts.len());

    // Tokenize
    let encodings = state
        .tokenizer
        .encode_batch(request.texts.clone(), true)
        .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;

    // Find max length
    let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);
    let max_len = max_len.min(state.max_seq_length);

    // Prepare input tensors
    let batch_size = encodings.len();
    let mut input_ids_vec = Vec::with_capacity(batch_size * max_len);
    let mut attention_mask_vec = Vec::with_capacity(batch_size * max_len);

    for encoding in &encodings {
        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();

        for i in 0..max_len {
            if i < ids.len() {
                input_ids_vec.push(ids[i] as u32);
                attention_mask_vec.push(mask[i] as f32);
            } else {
                input_ids_vec.push(0);
                attention_mask_vec.push(0.0);
            }
        }
    }

    // Create tensors
    let input_ids = Tensor::from_vec(input_ids_vec, (batch_size, max_len), &state.device)?;
    let attention_mask =
        Tensor::from_vec(attention_mask_vec.clone(), (batch_size, max_len), &state.device)?;

    // Get a model from the pool for this request
    let model = state.model_pool.get_model();

    // Run inference (seqlen_offset=0 for fresh sequences)
    let token_embeddings = {
        let mut model = model.lock().await;
        match model.forward(&input_ids, 0) {
            Ok(embeddings) => embeddings,
            Err(e) => {
                tracing::error!("Model forward pass failed: {:?}", e);
                return Err(AppError::Internal(format!("Model forward pass failed: {:?}", e)));
            }
        }
    };

    // Mean pooling with attention mask
    // token_embeddings shape: (batch_size, seq_len, embedding_dim)
    let attention_mask_expanded = attention_mask.unsqueeze(2)?; // (batch_size, seq_len, 1)

    // Multiply embeddings by attention mask
    let masked_embeddings = token_embeddings.broadcast_mul(&attention_mask_expanded)?;

    // Sum over sequence dimension
    let sum_embeddings = masked_embeddings.sum(1)?; // (batch_size, embedding_dim)

    // Sum attention mask
    let sum_mask = attention_mask.sum(1)?; // (batch_size,)
    let sum_mask = sum_mask.clamp(1e-9, f32::MAX)?; // Avoid division by zero

    // Divide to get mean
    let sum_mask_expanded = sum_mask.unsqueeze(1)?; // (batch_size, 1)
    let mut embeddings = sum_embeddings.broadcast_div(&sum_mask_expanded)?;

    // L2 normalization
    let norms = embeddings.sqr()?.sum_keepdim(1)?.sqrt()?.clamp(1e-9, f32::MAX)?;
    embeddings = embeddings.broadcast_div(&norms)?;

    let inference_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    // Update request counter
    state.total_requests.fetch_add(1, Ordering::Relaxed);

    // Convert to Vec<Vec<f32>>
    let embeddings_data = embeddings.to_vec2::<f32>()?;

    Ok(Json(EmbedResponse {
        embeddings: embeddings_data,
        inference_time_ms,
    }))
}
