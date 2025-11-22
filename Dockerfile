# Optimized Dockerfile for Docling Serve with CUDA 12.8 support
# Based on official Containerfile approach for minimal image size
# Build with: docker build --build-arg UV_SYNC_EXTRA_ARGS="--no-group pypi --group cu128" -t docling-serve:cuda128 .

ARG BASE_IMAGE=quay.io/sclorg/python-312-c9s:c9s
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.8.19
ARG UV_SYNC_EXTRA_ARGS=""

###################################################################################################
# UV Stage - Extract UV binary for temporary mounting                                            #
###################################################################################################

FROM ${UV_IMAGE} AS uv_stage

###################################################################################################
# Main Build Stage                                                                                #
###################################################################################################

FROM ${BASE_IMAGE} AS docling-base

# Switch to root for system package installation
USER 0

# Install system dependencies from os-packages.txt
RUN --mount=type=bind,source=os-packages.txt,target=/tmp/os-packages.txt \
    dnf -y install --best --nodocs --setopt=install_weak_deps=False dnf-plugins-core && \
    dnf config-manager --best --nodocs --setopt=install_weak_deps=False --save && \
    dnf config-manager --enable crb && \
    dnf -y update && \
    dnf install -y $(cat /tmp/os-packages.txt) && \
    dnf -y clean all && \
    rm -rf /var/cache/dnf

# Fix permissions for cache directory
RUN /usr/bin/fix-permissions /opt/app-root/src/.cache

# Set environment variables
ENV TESSDATA_PREFIX=/usr/share/tesseract/tessdata/ \
    OMP_NUM_THREADS=4 \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/app-root \
    DOCLING_SERVE_ARTIFACTS_PATH=/opt/app-root/src/.cache/docling/models \
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
    # VRAM Management: Set to True for process-based isolation (complete VRAM release)
    # False: Thread-based (faster, ~2GB VRAM accumulation) | True: Process-based (slower startup, ~50MB idle)
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
    HF_HUB_ETAG_TIMEOUT=90 \
    PATH="/opt/app-root/bin:$PATH" \
    PYTHONPATH="/opt/app-root/src"

# Switch to non-root user for application setup
USER 1001

WORKDIR /opt/app-root/src

# Build argument for CUDA group selection
ARG UV_SYNC_EXTRA_ARGS

# Install Python dependencies with two-phase approach for flash-attention
# Phase 1: Install all dependencies except flash-attn
# Phase 2: Install flash-attn with special build flags to skip CUDA compilation
RUN --mount=from=uv_stage,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/opt/app-root/src/.cache/uv,uid=1001 \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    umask 002 && \
    UV_SYNC_ARGS="--frozen --no-install-project --no-dev --all-extras" && \
    uv sync ${UV_SYNC_ARGS} ${UV_SYNC_EXTRA_ARGS} --no-extra flash-attn && \
    FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE uv sync ${UV_SYNC_ARGS} ${UV_SYNC_EXTRA_ARGS} --no-build-isolation-package=flash-attn

# Download models explicitly using docling-tools
# This ensures all required models are preloaded in the image
ARG MODELS_LIST="layout tableformer picture_classifier rapidocr easyocr"

RUN echo "Downloading models..." && \
    HF_HUB_DOWNLOAD_TIMEOUT="90" \
    HF_HUB_ETAG_TIMEOUT="90" \
    docling-tools models download -o "${DOCLING_SERVE_ARTIFACTS_PATH}" ${MODELS_LIST} && \
    chown -R 1001:0 ${DOCLING_SERVE_ARTIFACTS_PATH} && \
    chmod -R g=u ${DOCLING_SERVE_ARTIFACTS_PATH}

# Copy application code
COPY --chown=1001:0 ./docling_serve ./docling_serve

# Final sync to install the project itself
RUN --mount=from=uv_stage,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/opt/app-root/src/.cache/uv,uid=1001 \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    umask 002 && uv sync --frozen --no-dev --all-extras ${UV_SYNC_EXTRA_ARGS}

# Expose the default port
EXPOSE 5001

# Health check (enhancement over official Containerfile)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5001/health').read()" || exit 1

# Run the application
CMD ["docling-serve", "run"]
