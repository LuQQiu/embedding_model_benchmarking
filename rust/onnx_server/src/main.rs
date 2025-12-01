use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use common::{
    config::ModelsConfig, EmbedRequest, EmbedResponse, HealthResponse, InfoResponse, ModelConfig,
};
use ndarray::{Array2, Array3, Axis};
use ort::{
    execution_providers::OneDNNExecutionProvider,
    session::builder::GraphOptimizationLevel,
    session::Session,
    value::Value,
};
use serde_json::json;
use std::{
    collections::HashMap,
    sync::{Arc, atomic::{AtomicU64, AtomicUsize, Ordering}},
    time::Instant,
};
use sysinfo::System;
use tokenizers::Tokenizer;
use tower_http::trace::TraceLayer;
use tracing::{info, error};

// Session pool for concurrent inference with improved scheduling
struct SessionPool {
    sessions: Vec<tokio::sync::Mutex<Session>>,  // Use async mutex
    pool_size: usize,
    round_robin_counter: AtomicUsize,  // For better load distribution
}

impl SessionPool {
    fn new(sessions: Vec<Session>) -> Self {
        let pool_size = sessions.len();
        let sessions = sessions.into_iter().map(tokio::sync::Mutex::new).collect();
        Self {
            sessions,
            pool_size,
            round_robin_counter: AtomicUsize::new(0),
        }
    }

    // Optimized acquisition with true round-robin and minimal contention
    async fn acquire(&self) -> tokio::sync::MutexGuard<'_, Session> {
        // Simple round-robin without try_lock overhead
        let idx = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) % self.pool_size;
        self.sessions[idx].lock().await
    }
}

// Application state
struct AppState {
    session_pool: Arc<SessionPool>,  // Pool of sessions for concurrent inference
    tokenizer: Arc<Tokenizer>,  // Tokenizer is thread-safe, only needs Arc
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

impl From<ndarray::ShapeError> for AppError {
    fn from(err: ndarray::ShapeError) -> Self {
        AppError::Internal(format!("Shape error: {}", err))
    }
}

impl From<ort::Error> for AppError {
    fn from(err: ort::Error) -> Self {
        AppError::Internal(format!("ORT error: {}", err))
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

    // Create ONNX Runtime session pool for concurrent inference
    let cpu_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    info!("CPU count: {}", cpu_count);

    // Create multiple sessions for concurrent inference
    // Use pool_size from env or default to 4
    let pool_size = std::env::var("POOL_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(4);

    let threads_per_session = cpu_count / pool_size;
    info!("Creating session pool: {} sessions with {} threads each", pool_size, threads_per_session);

    let mut sessions = Vec::with_capacity(pool_size);
    for i in 0..pool_size {
        let session = Session::builder()?
            .with_execution_providers([
                OneDNNExecutionProvider::default().build()
            ])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads_per_session)?
            .with_inter_threads(1)?  // Only 1 inter-thread since we have multiple sessions
            .commit_from_file(onnx_path)?;
        sessions.push(session);
        info!("  ✓ Session {} created with oneDNN", i + 1);
    }

    let session_pool = SessionPool::new(sessions);
    info!("✓ Session pool created with {} sessions", pool_size);

    // Load tokenizer
    let tokenizer_path = std::path::Path::new(onnx_path).parent().unwrap();
    let tokenizer = Tokenizer::from_file(tokenizer_path.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    info!("✓ Tokenizer loaded");

    let model_load_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    info!("✓ Model loaded in {:.2}ms", model_load_time_ms);
    info!("  ONNX Runtime version: 2.0.0-rc.10");
    info!("  CPU count: {}", cpu_count);
    info!("  Pool size: {} sessions", pool_size);
    info!("  Threads per session: {}", threads_per_session);
    info!("");
    info!("Server ready on http://0.0.0.0:8000");
    info!("======================================================================");

    // Create application state
    let state = Arc::new(AppState {
        session_pool: Arc::new(session_pool),
        tokenizer: Arc::new(tokenizer),
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
        "service": "ONNX Runtime Rust Embedding Server",
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

    let process = system.process(sysinfo::get_current_pid().unwrap()).unwrap();
    let memory_rss_mb = process.memory() as f64 / 1024.0 / 1024.0;
    let cpu_percent = process.cpu_usage();

    // Convert model config to HashMap
    let mut model_configuration = HashMap::new();
    model_configuration.insert("name".to_string(), json!(state.model_config.name));
    model_configuration.insert("type".to_string(), json!(state.model_config.model_type));
    model_configuration.insert("max_seq_length".to_string(), json!(state.model_config.max_seq_length));
    model_configuration.insert("embedding_dim".to_string(), json!(state.model_config.embedding_dim));

    let cpu_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    Ok(Json(InfoResponse {
        framework: "onnx-rust".to_string(),
        model_name: std::env::var("MODEL_NAME").unwrap_or_else(|_| "embeddinggemma-300m".to_string()),
        model_configuration,
        model_load_time_ms: state.model_load_time_ms,
        total_requests: state.total_requests.load(Ordering::Relaxed),
        runtime_version: "2.0.0-rc.10".to_string(),
        device: "CPU (oneDNN)".to_string(),
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
    let attention_mask = Array2::from_shape_vec((batch_size, max_len), attention_mask_vec.clone())?;

    // Run inference - ORT 2.0 RC requires creating Value objects
    let input_ids_value = Value::from_array(input_ids)?;
    let attention_mask_value = Value::from_array(attention_mask)?;

    // Acquire a session from the pool and run inference
    let last_hidden_state_3d = {
        let mut session = state.session_pool.acquire().await;

        let outputs = session.run(ort::inputs![
            "input_ids" => input_ids_value,
            "attention_mask" => attention_mask_value,
        ])?;

        // Get token embeddings - ORT 2.0 RC returns tuple (shape, data)
        let (shape, data) = outputs["token_embeddings"].try_extract_tensor::<f32>()?;

        // Convert to Array3 before dropping lock
        Array3::from_shape_vec(
            (shape.as_ref()[0] as usize, shape.as_ref()[1] as usize, shape.as_ref()[2] as usize),
            data.to_vec(),
        )?
    };

    // Recreate attention mask for pooling
    let attention_mask_pooling = Array2::from_shape_vec((batch_size, max_len), attention_mask_vec)?;
    let attention_mask_f32: Array2<f32> = attention_mask_pooling.mapv(|x| x as f32);
    let attention_mask_expanded = attention_mask_f32.insert_axis(Axis(2));

    let sum_embeddings = (&last_hidden_state_3d * &attention_mask_expanded)
        .sum_axis(Axis(1));

    let sum_mask = attention_mask_expanded
        .sum_axis(Axis(1))
        .mapv(|x: f32| x.max(1e-9));

    let mut embeddings = sum_embeddings / sum_mask;

    // L2 normalization
    for mut row in embeddings.axis_iter_mut(Axis(0)) {
        let norm: f32 = row.mapv(|x: f32| x * x).sum().sqrt().max(1e-9);
        row /= norm;
    }

    let inference_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    // Update request counter
    state.total_requests.fetch_add(1, Ordering::Relaxed);

    // Convert to Vec<Vec<f32>>
    let embeddings_vec: Vec<Vec<f32>> = embeddings
        .axis_iter(Axis(0))
        .map(|row: ndarray::ArrayView1<f32>| row.to_vec())
        .collect();

    Ok(Json(EmbedResponse {
        embeddings: embeddings_vec,
        inference_time_ms,
    }))
}
