# Docker Deployment Guide

This guide explains how to build and deploy Docling Serve using Docker with CUDA 12.8 support.

## Prerequisites

- Docker Engine 20.10+ with Compose V2
- NVIDIA GPU with CUDA 12.8 support
- NVIDIA Container Toolkit installed
- NVIDIA Driver >= 550.54.14

### Installing NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Verify GPU access:
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

## Quick Start

### 1. Build the Docker Image

```bash
# Build with CUDA 12.8 support
docker build -t docling-serve:cuda128 .
```

### 2. Run with Docker Compose

```bash
# Start the service
docker compose up -d

# View logs
docker compose logs -f docling-serve

# Stop the service
docker compose down
```

### 3. Run with Docker

```bash
docker run -d \
  --name docling-serve \
  --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -p 5001:5001 \
  -v docling-models:/opt/app-root/src/.cache/docling/models \
  docling-serve:cuda128
```

## Testing the Deployment

```bash
# Check health endpoint
curl http://localhost:5001/health

# Check version info
curl http://localhost:5001/version

# View API documentation in browser
# macOS: open http://localhost:5001/docs
# Linux: xdg-open http://localhost:5001/docs
# Windows: start http://localhost:5001/docs
# Or simply navigate to: http://localhost:5001/docs
```

## Configuration

All configuration is done via environment variables. See `docker-compose.yml` for the complete list of available options.

### Common Configurations

#### Enable UI (Gradio)

```yaml
environment:
  DOCLING_SERVE_ENABLE_UI: "true"
```

Then access the UI at `http://localhost:5001/ui`

#### Configure Workers

```yaml
environment:
  DOCLING_SERVE_ENG_LOC_NUM_WORKERS: 4
  UVICORN_WORKERS: 2
```

#### Set Processing Limits

```yaml
environment:
  DOCLING_SERVE_MAX_NUM_PAGES: 500
  DOCLING_SERVE_MAX_FILE_SIZE: 52428800  # 50MB
  DOCLING_SERVE_MAX_DOCUMENT_TIMEOUT: 3600  # 1 hour
```

#### Configure GPU Device

```yaml
environment:
  DOCLING_DEVICE: cuda
  NVIDIA_VISIBLE_DEVICES: "0,1"  # Use specific GPUs
```

#### Enable API Key Authentication

```yaml
environment:
  DOCLING_SERVE_API_KEY: your_secret_api_key_here
```

Then make requests with the API key:
```bash
curl -H "X-API-Key: your_secret_api_key_here" http://localhost:5001/version
```

#### Use Redis Queue (RQ) Compute Engine

Uncomment the `redis` and `rq-worker` services in `docker-compose.yml`, then:

```yaml
environment:
  DOCLING_SERVE_ENG_KIND: rq
  DOCLING_SERVE_ENG_RQ_REDIS_URL: redis://redis:6379/0
```

## Volume Mounts

### Model Artifacts (Recommended)

Persist downloaded model weights:

```yaml
volumes:
  - docling-models:/opt/app-root/src/.cache/docling/models
```

### Scratch Directory

For temporary processing files:

```yaml
volumes:
  - ./scratch:/opt/app-root/scratch
environment:
  DOCLING_SERVE_SCRATCH_PATH: /opt/app-root/scratch
```

### Custom Model Artifacts

Use your own pre-downloaded models:

```yaml
volumes:
  - ./my-models:/opt/app-root/artifacts:ro
environment:
  DOCLING_SERVE_ARTIFACTS_PATH: /opt/app-root/artifacts
```

### SSL Certificates

For HTTPS:

```yaml
volumes:
  - ./certs:/certs:ro
environment:
  UVICORN_SSL_CERTFILE: /certs/cert.pem
  UVICORN_SSL_KEYFILE: /certs/key.pem
```

## Multi-Container Setup with RQ Workers

For distributed processing with Redis Queue:

```bash
# Start all services (docling-serve + redis + rq-worker)
docker compose up -d docling-serve redis rq-worker

# Scale RQ workers
docker compose up -d --scale rq-worker=3

# View worker logs
docker compose logs -f rq-worker
```

## Resource Limits

Configure GPU and memory limits in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 16G
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0']  # Specific GPU
          capabilities: [gpu]
```

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi

# Check Docker daemon configuration
cat /etc/docker/daemon.json
# Should contain:
# {
#   "runtimes": {
#     "nvidia": {
#       "path": "nvidia-container-runtime",
#       "runtimeArgs": []
#     }
#   }
# }
```

### Out of Memory Errors

Enable VRAM cleanup on idle with auto-restart:

```yaml
environment:
  DOCLING_SERVE_FREE_VRAM_ON_IDLE: "True"
  DOCLING_SERVE_MAX_TASKS_BEFORE_RESTART: 50  # Restart after N tasks for complete VRAM cleanup
```

**Why auto-restart?** ONNX Runtime's CUDA allocator caches memory and doesn't release it back to the OS in-process. Restarting the container after N tasks ensures complete VRAM cleanup. The value depends on your workload:
- `1-5`: Maximum VRAM efficiency, best for large documents or limited VRAM
- `20-50`: Balanced approach (recommended for most cases)
- `100+`: Minimal restart overhead, for systems with abundant VRAM

Or integrate with Ollama/llama-swap for model swapping:

```yaml
environment:
  DOCLING_SERVE_UNLOAD_OLLAMA_BASE_URL: http://ollama:11434
  DOCLING_SERVE_UNLOAD_OLLAMA_MODEL: llama3.2-vision
  # Or for llama-swap:
  DOCLING_SERVE_UNLOAD_LLAMA_SWAP_BASE_URL: http://llama-swap:9292/v1
```

**Complete VRAM management example:**

```yaml
environment:
  # Enable VRAM cleanup
  DOCLING_SERVE_FREE_VRAM_ON_IDLE: "True"

  # Auto-restart for complete cleanup (required for ONNX Runtime memory release)
  DOCLING_SERVE_MAX_TASKS_BEFORE_RESTART: 50

  # Unload external models before loading Docling models
  DOCLING_SERVE_UNLOAD_LLAMA_SWAP_BASE_URL: http://172.17.0.1:9292/v1
```

**Note:** The `restart: unless-stopped` policy in docker-compose.yml is required for auto-restart to work.

### Model Download Timeouts

Increase HuggingFace timeouts:

```yaml
environment:
  HF_HUB_DOWNLOAD_TIMEOUT: 180
  HF_HUB_ETAG_TIMEOUT: 180
```

### Permission Denied Errors

Ensure volumes have correct permissions:

```bash
# Create directories with correct ownership
mkdir -p scratch artifacts
sudo chown -R 1001:1001 scratch artifacts
```

## Performance Tuning

### Batch Processing

```yaml
environment:
  DOCLING_SERVE_OCR_BATCH_SIZE: 16
  DOCLING_SERVE_LAYOUT_BATCH_SIZE: 8
  DOCLING_SERVE_TABLE_BATCH_SIZE: 8
  DOCLING_PERF_PAGE_BATCH_SIZE: 8
  DOCLING_PERF_ELEMENTS_BATCH_SIZE: 16
```

### Thread Configuration

```yaml
environment:
  DOCLING_NUM_THREADS: 8
  OMP_NUM_THREADS: 8
```

### Model Caching

```yaml
environment:
  DOCLING_SERVE_OPTIONS_CACHE_SIZE: 5
  DOCLING_SERVE_LOAD_MODELS_AT_BOOT: "True"
```

## Production Deployment

### Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name docling.example.com;

    location / {
        proxy_pass http://localhost:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support for async processing
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Increase timeouts for long-running conversions
        proxy_read_timeout 7200s;
        proxy_send_timeout 7200s;
    }
}
```

Configure Docling Serve for reverse proxy:

```yaml
environment:
  UVICORN_ROOT_PATH: /
  UVICORN_PROXY_HEADERS: "true"
```

### CORS Configuration

For cross-origin requests:

```yaml
environment:
  DOCLING_SERVE_CORS_ORIGINS: '["https://example.com", "https://app.example.com"]'
  DOCLING_SERVE_CORS_METHODS: '["GET", "POST", "OPTIONS"]'
  DOCLING_SERVE_CORS_HEADERS: '["Content-Type", "Authorization", "X-API-Key"]'
```

### Monitoring

```bash
# Container stats
docker stats docling-serve

# GPU utilization
docker exec docling-serve nvidia-smi

# Application logs
docker compose logs -f --tail=100 docling-serve
```

## Building for Different Platforms

### CPU-Only Build

Modify `Dockerfile` to use CPU PyTorch:

```bash
# In the build stage, replace:
uv sync --frozen --no-dev --no-group pypi --group cpu
```

### AMD ROCm Build

For AMD GPUs, use ROCm:

```bash
uv sync --frozen --no-dev --no-group pypi --group rocm
```

## Upgrading

```bash
# Pull latest code
git pull

# Rebuild image
docker compose build --no-cache

# Restart services
docker compose down
docker compose up -d

# Clean old images
docker image prune -f
```

## Reference

- [Configuration Documentation](docs/configuration.md)
- [Docling Documentation](https://github.com/docling-project/docling)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
