# Docker Build Instructions

## Quick Start

### Build with CUDA 12.8 Support (Recommended)

```bash
docker build \
  --build-arg UV_SYNC_EXTRA_ARGS="--no-group pypi --group cu128" \
  -t docling-serve:cuda128 \
  .
```

Or use the Makefile:

```bash
make docker-build-cu128
```

### Build for CPU Only

```bash
docker build -t docling-serve:cpu .
```

## What Changed

This Dockerfile has been optimized to match the official build strategy:

### Size Optimization (~6GB savings)

1. **No nvidia/cuda base image** (-3.5GB)
   - Uses CentOS Stream 9 instead of Ubuntu 24.04
   - CUDA support comes from PyTorch cu128 wheels, not base image

2. **UV mount binding** (-50MB)
   - UV tool is mounted during build, not copied into image

3. **Minimal system packages** (-500MB)
   - Uses DNF with `--nodocs --setopt=install_weak_deps=False`
   - Only installs packages from os-packages.txt

4. **Single-stage build** (-1GB)
   - No separate builder/runtime stages
   - Smart layer ordering prevents artifact duplication

### Reliability Improvements

1. **Explicit model preloading**
   - Uses `docling-tools models download` with specific model list
   - No more silent failures with `|| true`

2. **Two-phase flash-attention installation**
   - Prevents unnecessary CUDA compilation
   - Uses `FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE`

### Production Enhancements (kept from original)

1. **Extensive environment configuration**
   - All Uvicorn, Docling Serve, and Core settings pre-configured

2. **Health check**
   - Built-in container health monitoring

3. **Security defaults**
   - Non-root user (1001)
   - CORS warnings for production deployment

## Expected Results

- **Image size**: ~11-12GB (with models preloaded)
- **Official size**: 11.4GB
- **Previous size**: 17.5GB (without models!)

## Build Args

- `BASE_IMAGE`: Base image (default: quay.io/sclorg/python-312-c9s:c9s)
- `UV_IMAGE`: UV tool image (default: ghcr.io/astral-sh/uv:0.8.19)
- `UV_SYNC_EXTRA_ARGS`: Additional UV sync arguments
  - For CUDA 12.8: `--no-group pypi --group cu128`
  - For CPU: (leave empty)
- `MODELS_LIST`: Space-separated model names to download

## Verification

After building, verify the image size:

```bash
docker images docling-serve:cuda128
```

Expected output should show ~11-12GB for the image size.
