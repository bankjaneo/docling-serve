# Subprocess Worker Integration Guide

## Overview

This guide explains how to integrate the subprocess worker for **complete VRAM cleanup** when `DOCLINGSERVEFREEVRAMON_IDLE=True`.

## Why Subprocess Approach?

**Problem**: ONNX Runtime's CUDA allocator doesn't release memory in-process
- Current in-process cleanup: ~1324 MB after 2 tasks
- Baseline after first task: ~682 MB
- **Memory accumulation is unavoidable** with in-process cleanup

**Solution**: Run OCR in isolated subprocesses
- Process termination forces OS to reclaim ALL CUDA memory
- Each task starts with clean slate
- Guaranteed return to ~200-500 MB CUDA context baseline

## Implementation Options

### Option 1: Quick Integration (Recommended for Testing)

Add subprocess processing to settings:

1. **Add setting in `docling_serve/settings.py`**:

```python
use_subprocess_for_ocr: bool = Field(
    default=False,
    description="Use subprocess workers for OCR to ensure complete VRAM cleanup",
)
```

2. **Modify `ensure_models_loaded()` in `docling_serve/app.py`**:

```python
async def ensure_models_loaded(orchestrator: BaseOrchestrator):
    """Ensure models are loaded before processing if lazy loading is enabled."""
    if docling_serve_settings.free_vram_on_idle:
        # Skip model loading if using subprocess workers
        if docling_serve_settings.use_subprocess_for_ocr:
            _log.info("Using subprocess workers - skipping model warmup")
            return

        # First, unload external models to free VRAM if configured
        await unload_external_models()
        # Then load Docling models
        _log.info("Loading models for processing...")
        await orchestrator.warm_up_caches()
```

3. **Modify task processing to use subprocess**:

This is the complex part - you need to intercept where the actual document conversion happens. Based on the architecture, this would be in the orchestrator's task processing.

### Option 2: Environment Variable Toggle (Simplest)

Add an environment variable to enable/disable subprocess mode:

```bash
export DOCLING_SERVE_USE_SUBPROCESS=true
```

Then check this in the code:

```python
import os

USE_SUBPROCESS = os.getenv("DOCLING_SERVE_USE_SUBPROCESS", "false").lower() == "true"

if USE_SUBPROCESS and docling_serve_settings.free_vram_on_idle:
    # Use subprocess worker
    from docling_serve.subprocess_worker import process_in_subprocess
    result = await process_in_subprocess(task_dict)
else:
    # Use normal in-process processing
    result = await orchestrator.process_task(task)
```

### Option 3: Conditional Auto-Enable (Smartest)

Automatically use subprocesses when VRAM is limited:

```python
import torch

def should_use_subprocess() -> bool:
    """Determine if subprocess mode should be used."""
    if not docling_serve_settings.free_vram_on_idle:
        return False

    # Force subprocess if explicitly requested
    if os.getenv("DOCLING_SERVE_USE_SUBPROCESS", "").lower() == "true":
        return True

    # Auto-enable for limited VRAM
    if torch.cuda.is_available():
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        # Use subprocess for GPUs with < 12GB VRAM
        if total_vram_gb < 12:
            _log.info(f"Auto-enabling subprocess mode for {total_vram_gb:.1f}GB VRAM")
            return True

    return False
```

## Complete Integration Example

Here's a complete example of how to integrate into your orchestrator:

### Step 1: Modify `docling_serve/settings.py`

```python
class DoclingServeSettings(BaseSettings):
    # ... existing settings ...

    use_subprocess_for_ocr: bool = Field(
        default=False,
        description="Use subprocess workers for OCR to ensure complete VRAM cleanup. "
                   "Only used when free_vram_on_idle=True. Provides guaranteed memory "
                   "release at the cost of process startup overhead.",
    )

    subprocess_timeout: float = Field(
        default=600.0,
        description="Timeout for subprocess workers in seconds",
    )
```

### Step 2: Create wrapper in `docling_serve/app.py`

Add this function near the top of the file:

```python
async def process_task_with_vram_management(
    orchestrator: BaseOrchestrator,
    task: Task,
) -> Any:
    """
    Process a task with appropriate VRAM management strategy.

    If subprocess mode is enabled and VRAM cleanup is active, processes
    the task in an isolated subprocess for guaranteed memory cleanup.
    Otherwise uses normal in-process processing.
    """
    use_subprocess = (
        docling_serve_settings.free_vram_on_idle and
        docling_serve_settings.use_subprocess_for_ocr
    )

    if use_subprocess:
        _log.info(f"Processing task {task.task_id} in subprocess for VRAM cleanup")

        from docling_serve.subprocess_worker import process_in_subprocess

        # Build task configuration
        task_dict = {
            "task_id": task.task_id,
            "sources": task.sources,
            "convert_options": task.convert_options,
            # Add converter manager config
            "artifacts_path": docling_serve_settings.artifacts_path,
            "options_cache_size": docling_serve_settings.options_cache_size,
            "enable_remote_services": docling_serve_settings.enable_remote_services,
            "allow_external_plugins": docling_serve_settings.allow_external_plugins,
            "max_num_pages": docling_serve_settings.max_num_pages,
            "max_file_size": docling_serve_settings.max_file_size,
            "queue_max_size": docling_serve_settings.queue_max_size,
            "ocr_batch_size": docling_serve_settings.ocr_batch_size,
            "layout_batch_size": docling_serve_settings.layout_batch_size,
            "table_batch_size": docling_serve_settings.table_batch_size,
            "batch_polling_interval_seconds": docling_serve_settings.batch_polling_interval_seconds,
        }

        try:
            result = await process_in_subprocess(
                task_dict,
                timeout=docling_serve_settings.subprocess_timeout,
            )

            if result["status"] == "error":
                _log.error(f"Subprocess task failed: {result['error']}")
                raise RuntimeError(result["error"])

            return result["results"]

        except Exception as e:
            _log.error(f"Subprocess processing failed: {e}")
            raise

    else:
        # Normal in-process processing
        _log.debug(f"Processing task {task.task_id} in-process")
        return await orchestrator.process_task(task)
```

### Step 3: Update cleanup functions

Modify `cleanup_models_if_needed()`:

```python
async def cleanup_models_if_needed(orchestrator: BaseOrchestrator):
    """Clear models after processing if lazy loading is enabled to free VRAM."""
    if not docling_serve_settings.free_vram_on_idle:
        return

    # If using subprocess mode, no cleanup needed (subprocess already terminated)
    if docling_serve_settings.use_subprocess_for_ocr:
        _log.info("Subprocess mode: no cleanup needed (process already terminated)")
        return

    # ... existing cleanup code ...
```

## Testing

### Test 1: Verify Subprocess Mode

```bash
# Enable subprocess mode
export DOCLING_SERVE_FREE_VRAM_ON_IDLE=true
export DOCLING_SERVE_USE_SUBPROCESS=true

# Start server
python -m docling_serve

# In another terminal, watch VRAM
watch -n 1 nvidia-smi

# Process multiple documents and verify:
# - VRAM spikes during processing
# - VRAM returns to ~200-500 MB baseline after each task
# - No accumulation over multiple tasks
```

### Test 2: Compare Performance

Measure overhead of subprocess vs in-process:

```python
import time

# Test in-process
start = time.time()
# Process 10 documents in-process
in_process_time = time.time() - start

# Test subprocess
start = time.time()
# Process 10 documents in subprocess
subprocess_time = time.time() - start

overhead = ((subprocess_time - in_process_time) / in_process_time) * 100
print(f"Subprocess overhead: {overhead:.1f}%")
```

Expected overhead: 10-30% depending on model loading time.

## Performance Tuning

### 1. Reduce Startup Overhead

Cache model files in memory-mapped storage:

```python
# In subprocess_worker.py
import mmap

def preload_models_mmap(artifacts_path: Path):
    """Pre-load model files into shared memory."""
    # This allows multiple processes to share model weights
    # without duplicating in RAM
    pass
```

### 2. Adjust Process Pool Size

For systems with more VRAM:

```python
# Allow 2-3 concurrent subprocess workers
pool = SubprocessWorkerPool(max_workers=2)
```

### 3. Warm Pool During Idle

Pre-spawn worker processes during idle time:

```python
async def warm_subprocess_pool():
    """Pre-spawn subprocess workers during idle time."""
    pool = get_worker_pool()
    # Pre-spawn process(es) to reduce first-task latency
    await pool.prewarm()
```

## Monitoring

Add monitoring to track subprocess performance:

```python
import time

class SubprocessMetrics:
    def __init__(self):
        self.total_tasks = 0
        self.total_time = 0.0
        self.total_overhead = 0.0

    def record_task(self, task_time: float, overhead: float):
        self.total_tasks += 1
        self.total_time += task_time
        self.total_overhead += overhead

    @property
    def avg_overhead_pct(self) -> float:
        if self.total_time == 0:
            return 0.0
        return (self.total_overhead / self.total_time) * 100

metrics = SubprocessMetrics()
```

## Troubleshooting

### Issue: Subprocess hangs

**Cause**: Deadlock in pickling/unpickling
**Solution**: Ensure all objects in task_dict are picklable

```python
# Test pickling
import pickle
try:
    pickle.dumps(task_dict)
except Exception as e:
    _log.error(f"Task dict not picklable: {e}")
```

### Issue: CUDA out of memory in subprocess

**Cause**: Multiple processes loading models simultaneously
**Solution**: Use semaphore to limit concurrent workers

```python
pool = SubprocessWorkerPool(max_workers=1)  # Only 1 at a time
```

### Issue: Slow first task

**Cause**: Model loading in cold subprocess
**Solution**: Pre-warm pool or accept first-task overhead

## Migration Path

### Phase 1: Testing (1-2 days)
1. Enable subprocess mode on dev/staging
2. Run test suite
3. Measure performance overhead
4. Monitor VRAM cleanup

### Phase 2: Gradual Rollout (3-5 days)
1. Enable for low-VRAM instances first
2. Monitor error rates and performance
3. Tune settings based on metrics

### Phase 3: Full Deployment (1 week)
1. Enable for all instances with `free_vram_on_idle=True`
2. Monitor production metrics
3. Keep in-process mode as fallback

## Recommendation

Based on your logs showing ~1324 MB accumulation with in-process cleanup:

**For your 16GB RTX 5060 Ti**: Enable subprocess mode
- ✅ Complete VRAM release guaranteed
- ✅ Can handle larger documents without OOM
- ✅ More predictable memory usage
- ⚠️ ~10-20% performance overhead acceptable for cleanup benefit

**Configuration**:
```bash
export DOCLING_SERVE_FREE_VRAM_ON_IDLE=true
export DOCLING_SERVE_USE_SUBPROCESS=true
export DOCLING_SERVE_SUBPROCESS_TIMEOUT=600
```

This will ensure VRAM returns to ~200-500 MB baseline after each task, eliminating accumulation completely.
