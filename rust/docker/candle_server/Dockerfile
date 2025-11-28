# Multi-stage build for Candle Rust server
FROM rustlang/rust:nightly-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Build workspace
WORKDIR /build
COPY rust/Cargo.toml rust/Cargo.lock* ./
COPY rust/common ./common
COPY rust/candle_server ./candle_server
COPY rust/benchmark_client ./benchmark_client

# Build release binary
RUN cargo build --release -p candle_server

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy binary
COPY --from=builder /build/target/release/candle_server /usr/local/bin/

# Create working directory
WORKDIR /app

# Expose port
EXPOSE 8000

# Run server
CMD ["candle_server"]
