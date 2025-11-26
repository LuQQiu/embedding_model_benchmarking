FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    requests==2.31.0 \
    numpy==1.26.4 \
    pyyaml==6.0.1 \
    psutil==5.9.8 \
    tqdm==4.66.2

# Create working directory
WORKDIR /app

# Copy benchmark client script
COPY docker/benchmark_client/benchmark_client.py /app/

# Entry point
CMD ["python", "benchmark_client.py"]
