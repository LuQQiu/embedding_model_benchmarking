# Multi-stage build for ONNX Runtime Rust server
FROM rustlang/rust:nightly-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Download ONNX Runtime binaries (v1.22.x for ort 2.0.0-rc.10 compatibility)
WORKDIR /tmp
RUN curl -LO https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz && \
    tar xzf onnxruntime-linux-x64-1.22.0.tgz && \
    mv onnxruntime-linux-x64-1.22.0 /onnxruntime

# Set ONNX Runtime environment variables for build
ENV ORT_DYLIB_PATH=/onnxruntime/lib
ENV LD_LIBRARY_PATH=/onnxruntime/lib:$LD_LIBRARY_PATH

# Build workspace
WORKDIR /build
COPY rust/Cargo.toml rust/Cargo.lock* ./
COPY rust/common ./common
COPY rust/onnx_server ./onnx_server
COPY rust/benchmark_client ./benchmark_client

# Build release binary
RUN cargo build --release -p onnx_server

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy ONNX Runtime libraries
COPY --from=builder /onnxruntime/lib/*.so* /usr/local/lib/

# Copy binary
COPY --from=builder /build/target/release/onnx_server /usr/local/bin/

# Update library cache
RUN ldconfig

# Set library path for runtime
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Create working directory
WORKDIR /app

# Expose port
EXPOSE 8000

# Run server
CMD ["onnx_server"]
