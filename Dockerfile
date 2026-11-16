FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Install curl for container healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application directories
COPY api/ ./api/
COPY src/ ./src/
COPY monitoring/ ./monitoring/
COPY data/ ./data/

# Configure Python environment
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1
