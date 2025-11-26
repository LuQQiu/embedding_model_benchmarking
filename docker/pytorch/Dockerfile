FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu \
    transformers==4.38.0 \
    sentence-transformers==2.5.1 \
    tokenizers==0.15.2 \
    numpy==1.26.4 \
    pyyaml==6.0.1 \
    psutil==5.9.8 \
    tqdm==4.66.2

# Copy benchmark script
COPY docker/pytorch/benchmark.py /app/benchmark.py

# Run as non-root user
RUN useradd -m -u 1000 benchmark && \
    chown -R benchmark:benchmark /app
USER benchmark

# Default command
CMD ["python", "/app/benchmark.py"]
