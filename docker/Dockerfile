# 🌌 Nebula Living Research Organism - Unified Docker Image
# Production container with Reality Bridge, Nebula Bridge, and Research Bridge

FROM python:3.12-slim

LABEL maintainer="VoxSigil Team"
LABEL description="Nebula Living Research Organism - Complete AI Evolution System"
LABEL version="2.0.1"

# Install system dependencies including graphviz for architecture visualization
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    graphviz \
    libgraphviz-dev \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv (much faster than pip)
# Using cpu variant for broader compatibility - GPU will be detected at runtime
RUN uv sync --extra cpu --no-dev

# Copy entire codebase
COPY . .

# Create necessary directories
RUN mkdir -p \
    /data/models \
    /data/checkpoints \
    /data/logs \
    /data/blockchain \
    /data/research \
    /data/architectures \
    architecture_storage \
    config

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV NEBULA_CHAIN=test
ENV NEBULA_DATA_DIR=/data
ENV REDIS_HOST=redis
ENV NEO4J_URI=bolt://neo4j:7687
ENV NEO4J_USER=neo4j
ENV NEO4J_PASSWORD=nebula123

# Expose ports
# 8090: Health Monitor Dashboard
# 9080: Dual Chain API
# 9081: Worker/Task Fabric API
# 8765: WebSocket
# 5001: P2P Mesh
EXPOSE 8090 9080 9081 8765 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8090/health || exit 1

# Default command - start the Living Organism
CMD ["uv", "run", "python", "start_nebula_bridge.py"]
