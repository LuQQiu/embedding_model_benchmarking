use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use common::{EmbedRequest, EmbedResponse, HealthResponse, InfoResponse, ModelConfig, ModelsConfig};
use ndarray::{Array2, Axis};
use ort::{GraphOptimizationLevel, Session};
use serde_json::json;
use std::{
    collections::HashMap,
    sync::{Arc, atomic::{AtomicU64, Ordering}},
    time::Instant,
};
use sysinfo::System;
use tokenizers::Tokenizer;
use tower_http::trace::TraceLayer;
use tracing::{info, error};

// Application state
struct AppState {
    session: Session,
    tokenizer: Tokenizer,
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

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter("onnx_server=info,tower_http=info")
        .init();

    info!("======================================================================");
    info!("ONNX Runtime Rust Server - Starting");
    info!("======================================================================");

    // Load configuration
    let model_name = std::env::var("MODEL_NAME").unwrap_or_else(|_| "embeddinggemma-300m".to_string());
    info!("Model: {}", model_name);

    let models_config = ModelsConfig::load("/config/models.yaml")?;
    let model_config = models_config
        .get_model(&model_name)
        .ok_or_else(|| anyhow::anyhow!("Model {} not found in config", model_name))?
        .clone();

    let max_seq_length = model_config.max_seq_length;

    // Load model
    info!("Loading model: {}", model_config.name);
    let start_time = Instant::now();

    let onnx_path = &model_config.paths.onnx;
    info!("ONNX path: {}", onnx_path);

    if !std::path::Path::new(onnx_path).exists() {
        error!("ONNX model not found at {}", onnx_path);
        error!("Please run: python3 scripts/convert_to_onnx.py --model {}", model_name);
        return Err(anyhow::anyhow!("Model file not found"));
    }

    // Create ONNX Runtime session
    let cpu_count = num_cpus::get();
    info!("CPU count: {}", cpu_count);

    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(cpu_count as i16)?
        .with_inter_threads(cpu_count as i16)?
        .commit_from_file(onnx_path)?;

    info!("✓ ONNX session created");

    // Load tokenizer
    let tokenizer_path = std::path::Path::new(onnx_path).parent().unwrap();
    let tokenizer = Tokenizer::from_file(tokenizer_path.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    info!("✓ Tokenizer loaded");

    let model_load_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    info!("✓ Model loaded in {:.2}ms", model_load_time_ms);
    info!("  ONNX Runtime version: {}", ort::version());
    info!("  CPU count: {}", cpu_count);
    info!("");
    info!("Server ready on http://0.0.0.0:8000");
    info!("======================================================================");

    // Create application state
    let state = Arc::new(AppState {
        session,
        tokenizer,
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
        .route("/info", get(info))
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
        "service": "ONNX Runtime Rust Embedding Server",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "embed": "/embed (POST)"
        }
    }))
}

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        model_loaded: true,
    })
}

async fn info(State(state): State<Arc<AppState>>) -> Result<Json<InfoResponse>, AppError> {
    let mut system = state.system.lock().await;
    system.refresh_all();

    let process = system.process(sysinfo::get_current_pid().unwrap()).unwrap();
    let memory_rss_mb = process.memory() as f64 / 1024.0 / 1024.0;
    let cpu_percent = process.cpu_usage();

    // Convert model config to HashMap
    let mut model_configuration = HashMap::new();
    model_configuration.insert("name".to_string(), json!(state.model_config.name));
    model_configuration.insert("type".to_string(), json!(state.model_config.model_type));
    model_configuration.insert("max_seq_length".to_string(), json!(state.model_config.max_seq_length));
    model_configuration.insert("embedding_dim".to_string(), json!(state.model_config.embedding_dim));

    Ok(Json(InfoResponse {
        framework: "onnx-rust".to_string(),
        model_name: std::env::var("MODEL_NAME").unwrap_or_else(|_| "embeddinggemma-300m".to_string()),
        model_configuration,
        model_load_time_ms: state.model_load_time_ms,
        total_requests: state.total_requests.load(Ordering::Relaxed),
        runtime_version: ort::version(),
        device: "CPU".to_string(),
        cpu_count: num_cpus::get(),
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
                input_ids_vec.push(ids[i] as i64);
                attention_mask_vec.push(mask[i] as i64);
            } else {
                input_ids_vec.push(0);
                attention_mask_vec.push(0);
            }
        }
    }

    let input_ids = Array2::from_shape_vec((batch_size, max_len), input_ids_vec)?;
    let attention_mask = Array2::from_shape_vec((batch_size, max_len), attention_mask_vec)?;

    // Run inference
    let outputs = state.session.run(ort::inputs![
        "input_ids" => input_ids.view(),
        "attention_mask" => attention_mask.view(),
    ]?)?;

    // Get last hidden state
    let last_hidden_state: ort::Value = outputs["last_hidden_state"].try_extract_tensor()?;
    let last_hidden_state_array = last_hidden_state.view();
    let shape = last_hidden_state_array.shape();

    // Convert to ndarray Array3
    let last_hidden_state_nd: Array2<f32> = ndarray::ArrayView3::<f32>::from_shape(
        (shape[0], shape[1], shape[2]),
        last_hidden_state_array.as_slice().unwrap(),
    )?
    .into_owned()
    .into_dimensionality()?;

    // Mean pooling
    let attention_mask_f32: Array2<f32> = attention_mask.mapv(|x| x as f32);
    let attention_mask_expanded = attention_mask_f32.insert_axis(Axis(2));

    let last_hidden_state_3d = last_hidden_state_nd.into_shape((batch_size, max_len, shape[2]))?;

    let sum_embeddings = (&last_hidden_state_3d * &attention_mask_expanded)
        .sum_axis(Axis(1));

    let sum_mask = attention_mask_expanded
        .sum_axis(Axis(1))
        .mapv(|x| x.max(1e-9));

    let mut embeddings = sum_embeddings / sum_mask;

    // L2 normalization
    for mut row in embeddings.axis_iter_mut(Axis(0)) {
        let norm = row.mapv(|x| x * x).sum().sqrt().max(1e-9);
        row /= norm;
    }

    let inference_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    // Update request counter
    state.total_requests.fetch_add(1, Ordering::Relaxed);

    // Convert to Vec<Vec<f32>>
    let embeddings_vec: Vec<Vec<f32>> = embeddings
        .axis_iter(Axis(0))
        .map(|row| row.to_vec())
        .collect();

    Ok(Json(EmbedResponse {
        embeddings: embeddings_vec,
        inference_time_ms,
    }))
}
