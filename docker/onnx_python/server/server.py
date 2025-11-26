#!/usr/bin/env python3
"""
ONNX Runtime Python Embedding Model Server

FastAPI server for serving embeddings using ONNX Runtime
"""

import os
import sys
import time
import yaml
import psutil
import numpy as np
from pathlib import Path
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
from transformers import AutoTokenizer


# Global state
class ServerState:
    session: ort.InferenceSession = None
    tokenizer: AutoTokenizer = None
    model_config: dict = None
    model_load_time_ms: float = 0
    total_requests: int = 0
    process: psutil.Process = None
    max_seq_length: int = 512


state = ServerState()


# Request/Response models
class EmbedRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class InfoResponse(BaseModel):
    framework: str
    model_name: str
    model_configuration: dict
    model_load_time_ms: float
    total_requests: int
    onnxruntime_version: str
    providers: List[str]
    cpu_count: int
    memory_rss_mb: float
    cpu_percent: float


# Lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    print("="*70)
    print("ONNX Runtime Python Server - Starting")
    print("="*70)

    model_name = os.environ.get('MODEL_NAME', 'embeddinggemma-300m')

    # Load model configuration
    models_config_path = Path("/config/models.yaml")
    with open(models_config_path) as f:
        models_config = yaml.safe_load(f)

    if model_name not in models_config['models']:
        raise ValueError(f"Model {model_name} not found in config")

    state.model_config = models_config['models'][model_name]
    state.max_seq_length = state.model_config['max_seq_length']
    state.process = psutil.Process()

    # Load model
    print(f"Loading model: {state.model_config['name']}")
    start_time = time.perf_counter()

    onnx_path = state.model_config['paths']['onnx']

    if not Path(onnx_path).exists():
        raise FileNotFoundError(
            f"ONNX model not found at {onnx_path}\n"
            f"Please run: python3 scripts/convert_to_onnx.py --model {model_name}"
        )

    # Configure ONNX Runtime session options
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = os.cpu_count()
    session_options.inter_op_num_threads = os.cpu_count()

    # Create inference session
    providers = ['CPUExecutionProvider']
    state.session = ort.InferenceSession(
        onnx_path,
        sess_options=session_options,
        providers=providers
    )

    # Load tokenizer
    tokenizer_path = Path(onnx_path).parent
    state.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    state.model_load_time_ms = (time.perf_counter() - start_time) * 1000

    print(f"✓ Model loaded in {state.model_load_time_ms:.2f}ms")
    print(f"  ONNX Runtime version: {ort.__version__}")
    print(f"  Providers: {state.session.get_providers()}")
    print(f"  Intra-op threads: {session_options.intra_op_num_threads}")
    print(f"  CPU count: {os.cpu_count()}")
    print(f"\nServer ready on http://0.0.0.0:8000")
    print("="*70)

    yield

    # Shutdown
    print("\n" + "="*70)
    print("ONNX Runtime Python Server - Shutting down")
    print(f"Total requests served: {state.total_requests}")
    print("="*70)


# Create FastAPI app
app = FastAPI(
    title="ONNX Runtime Python Embedding Server",
    description="Embedding model inference using ONNX Runtime",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=state.session is not None
    )


@app.get("/info", response_model=InfoResponse)
async def info():
    """Server information endpoint"""
    if state.session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return InfoResponse(
        framework="onnx-python",
        model_name=os.environ.get('MODEL_NAME', 'embeddinggemma-300m'),
        model_configuration=state.model_config,
        model_load_time_ms=state.model_load_time_ms,
        total_requests=state.total_requests,
        onnxruntime_version=ort.__version__,
        providers=state.session.get_providers(),
        cpu_count=os.cpu_count(),
        memory_rss_mb=state.process.memory_info().rss / 1024 / 1024,
        cpu_percent=state.process.cpu_percent()
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """
    Generate embeddings for input texts

    Args:
        request: EmbedRequest with list of texts

    Returns:
        EmbedResponse with embeddings and timing
    """
    if state.session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    try:
        # Generate embeddings
        start_time = time.perf_counter()

        # Tokenize
        inputs = state.tokenizer(
            request.texts,
            padding=True,
            truncation=True,
            max_length=state.max_seq_length,
            return_tensors="np"
        )

        # Prepare inputs for ONNX Runtime
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }

        # Run inference
        outputs = state.session.run(None, ort_inputs)

        # Get last hidden state
        last_hidden_state = outputs[0]

        # Mean pooling
        attention_mask = inputs["attention_mask"]
        attention_mask_expanded = np.expand_dims(attention_mask, -1)

        sum_embeddings = np.sum(last_hidden_state * attention_mask_expanded, axis=1)
        sum_mask = np.clip(attention_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)

        embeddings = sum_embeddings / sum_mask

        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        inference_time = (time.perf_counter() - start_time) * 1000

        # Update request counter
        state.total_requests += 1

        # Convert to list for JSON serialization
        embeddings_list = embeddings.tolist()

        return EmbedResponse(
            embeddings=embeddings_list,
            inference_time_ms=inference_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "ONNX Runtime Python Embedding Server",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "embed": "/embed (POST)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
