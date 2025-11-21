# Multi-stage Dockerfile for Docling Serve with CUDA 12.8 support
# Build with: docker build -t docling-serve:cuda128 .

# Stage 1: UV Builder
FROM ghcr.io/astral-sh/uv:0.8.19 AS uv

# Stage 2: Build stage
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04 AS builder

# Install UV from first stage
COPY --from=uv /uv /usr/local/bin/uv

# Set environment variables for build
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/app-root \
    HF_HUB_DOWNLOAD_TIMEOUT=90 \
    HF_HUB_ETAG_TIMEOUT=90

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3-pip \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    libglvnd0 \
    libgl1 \
    libglib2.0-0 \
    git \
    curl \
    ca-certificates \
    locales \
    && locale-gen en_US.UTF-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app directory and user
RUN mkdir -p /opt/app-root/src && \
    groupadd -g 1001 appuser && \
    useradd -r -u 1001 -g appuser -d /opt/app-root -s /sbin/nologin appuser && \
    chown -R 1001:1001 /opt/app-root

WORKDIR /opt/app-root/src

# Copy dependency files
COPY --chown=1001:1001 pyproject.toml uv.lock ./

# Install dependencies with CUDA 12.8 support
# Use --no-group pypi --group cu128 to install CUDA 12.8 PyTorch
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-group pypi --group cu128

# Copy application code
COPY --chown=1001:1001 docling_serve ./docling_serve

# Download model weights at build time
RUN --mount=type=cache,target=/root/.cache/huggingface \
    /opt/app-root/bin/python -c "from docling.models import download_models_hf; download_models_hf()" || true

# Stage 3: Runtime stage
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04 AS runtime

# Install runtime dependencies only
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PYTHONIOENCODING=utf-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract5 \
    libglvnd0 \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    locales \
    && locale-gen en_US.UTF-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app directory and user
RUN mkdir -p /opt/app-root/src && \
    groupadd -g 1001 appuser && \
    useradd -r -u 1001 -g appuser -d /opt/app-root -s /sbin/nologin appuser && \
    chown -R 1001:1001 /opt/app-root

# Copy application and dependencies from builder
COPY --from=builder --chown=1001:1001 /opt/app-root /opt/app-root

WORKDIR /opt/app-root/src

# Set runtime environment variables with defaults from docs/configuration.md
ENV PATH="/opt/app-root/bin:$PATH" \
    PYTHONPATH="/opt/app-root/src" \
    OMP_NUM_THREADS=4 \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/ \
    # Uvicorn Web Server Configuration
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=5001 \
    UVICORN_RELOAD=false \
    UVICORN_WORKERS=1 \
    UVICORN_ROOT_PATH="" \
    UVICORN_PROXY_HEADERS=true \
    UVICORN_TIMEOUT_KEEP_ALIVE=60 \
    # Docling Serve Application Configuration
    DOCLING_SERVE_ENABLE_UI=false \
    DOCLING_SERVE_SHOW_VERSION_INFO=true \
    DOCLING_SERVE_ENABLE_REMOTE_SERVICES=false \
    DOCLING_SERVE_ALLOW_EXTERNAL_PLUGINS=false \
    DOCLING_SERVE_SINGLE_USE_RESULTS=true \
    DOCLING_SERVE_RESULT_REMOVAL_DELAY=300 \
    DOCLING_SERVE_MAX_DOCUMENT_TIMEOUT=604800 \
    DOCLING_SERVE_SYNC_POLL_INTERVAL=2 \
    DOCLING_SERVE_MAX_SYNC_WAIT=120 \
    DOCLING_SERVE_LOAD_MODELS_AT_BOOT=True \
    DOCLING_SERVE_FREE_VRAM_ON_IDLE=False \
    DOCLING_SERVE_CLEANUP_POLL_INTERVAL=5.0 \
    DOCLING_SERVE_UNLOAD_EXTERNAL_MODEL_TIMEOUT=10.0 \
    DOCLING_SERVE_OPTIONS_CACHE_SIZE=2 \
    # SECURITY WARNING: Default CORS settings allow all origins. Override in production!
    DOCLING_SERVE_CORS_ORIGINS='["*"]' \
    DOCLING_SERVE_CORS_METHODS='["*"]' \
    DOCLING_SERVE_CORS_HEADERS='["*"]' \
    DOCLING_SERVE_ENG_KIND=local \
    # Docling Core Configuration
    DOCLING_NUM_THREADS=4 \
    DOCLING_PERF_PAGE_BATCH_SIZE=4 \
    DOCLING_PERF_ELEMENTS_BATCH_SIZE=8 \
    DOCLING_DEBUG_PROFILE_PIPELINE_TIMINGS=false \
    # Compute Engine - Local
    DOCLING_SERVE_ENG_LOC_NUM_WORKERS=2 \
    DOCLING_SERVE_ENG_LOC_SHARE_MODELS=False \
    # HuggingFace Configuration
    HF_HUB_DOWNLOAD_TIMEOUT=90 \
    HF_HUB_ETAG_TIMEOUT=90

# Switch to non-root user
USER 1001

# Expose the default port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3.12 -c "import urllib.request; urllib.request.urlopen('http://localhost:5001/health').read()" || exit 1

# Run the application
CMD ["docling-serve", "run"]
