FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    openvino==2024.0.0 \
    transformers==4.38.0 \
    tokenizers==0.15.2 \
    numpy==1.26.4 \
    pyyaml==6.0.1 \
    psutil==5.9.8 \
    fastapi==0.109.2 \
    uvicorn==0.27.1

# Create working directory
WORKDIR /app

# Copy server script
COPY docker/openvino/server/server.py /app/

# Expose port
EXPOSE 8000

# Entry point
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
