FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    onnxruntime==1.17.1 \
    transformers==4.38.0 \
    tokenizers==0.15.2 \
    numpy==1.26.4 \
    pyyaml==6.0.1 \
    psutil==5.9.8 \
    tqdm==4.66.2

# Create working directory
WORKDIR /app

# Copy benchmark script
COPY docker/onnx_python/benchmark.py /app/

# Entry point
CMD ["python", "benchmark.py"]
