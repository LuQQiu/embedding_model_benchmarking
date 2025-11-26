FROM python:3.11-slim

# Install system dependencies including curl for health checks
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
# Install PyTorch first with CPU-only index
# Using PyTorch 2.6+ for EmbeddingGemma compatibility (requires or_mask_function/and_mask_function)
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# Install other packages from PyPI
RUN pip install --no-cache-dir \
    transformers>=4.45.0 \
    sentence-transformers>=3.0.0 \
    tokenizers>=0.19.0 \
    numpy==1.26.4 \
    pyyaml==6.0.1 \
    psutil==5.9.8 \
    fastapi==0.109.2 \
    uvicorn==0.27.1

# Create working directory
WORKDIR /app

# Copy server script
COPY docker/pytorch/server/server.py /app/

# Expose port
EXPOSE 8000

# Entry point
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
