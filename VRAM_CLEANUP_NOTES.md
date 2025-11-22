# VRAM Cleanup Improvements

## Problem
When `DOCLINGSERVEFREEVRAMON_IDLE=True`, VRAM is not fully released between OCR tasks, leading to accumulation:
- First run: ~682 MB
- Second run: ~1338 MB
- Third run: ~1995 MB

This is caused by:
1. CUDA context overhead (normal 200-500 MB baseline)
2. ONNX Runtime CUDA memory pool caching
3. PyTorch reserved memory not fully released
4. Model references not properly garbage collected

## Solution 1: Enhanced In-Process Cleanup (Implemented)

### Changes in `docling_serve/app.py`

The `cleanup_models_if_needed()` function has been significantly improved:

#### Key Improvements:

1. **Recursive CPU Migration** (`_move_to_cpu_recursive()`)
   - Deep traversal of all objects to find PyTorch modules/tensors
   - Moves everything to CPU before deletion
   - Sets modules to eval mode to release training buffers
   - Prevents circular reference issues with visited set

2. **Explicit Reference Deletion**
   - Deletes ONNX Runtime sessions before clearing cache
   - Explicitly removes each cache entry individually
   - Clears both converter cache and internal references

3. **Advanced PyTorch Memory Management**
   - Per-device memory cleanup
   - Aggressive memory fraction reset (0.0 → 1.0)
   - Multiple synchronization points
   - Reset of all memory stats

4. **Multi-Pass Garbage Collection**
   - 3 full GC passes (generation 2)
   - 2 additional passes for good measure
   - Targeted GC after ONNX cleanup

5. **Better Logging**
   - Tracks memory before and after
   - Reports memory freed
   - Distinguishes between normal CUDA overhead vs. leaks

### Expected Results:
- Better memory release between runs
- Reduced accumulation (should stay close to CUDA baseline ~200-500 MB)
- More informative logging to track cleanup effectiveness

### Testing:
Run multiple OCR tasks and monitor:
```bash
watch nvidia-smi
```

Check logs for:
- "VRAM allocated before cleanup"
- "VRAM freed"
- "VRAM allocated after cleanup"

## Solution 2: Child Process Approach (Future Consideration)

If in-process cleanup is still insufficient, consider moving OCR to child processes.

### Benefits:
- **Complete VRAM release**: Process termination forces OS to reclaim all GPU memory
- **Isolation**: Each task starts with clean state
- **No accumulation**: Memory leaks die with process
- **Simpler code**: No complex cleanup logic needed

### Trade-offs:
- **Startup overhead**: Process creation + model loading per task
- **IPC complexity**: Need to pass data between processes
- **Resource overhead**: Process creation cost
- **Debugging**: Harder to debug child processes

### Implementation Approach:

#### Option A: Multiprocessing with CUDA

```python
import multiprocessing as mp
from multiprocessing import Process, Queue

def ocr_worker_process(task_queue: Queue, result_queue: Queue):
    """Worker process that handles OCR tasks."""
    import torch
    import os

    # Set CUDA device in child process
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Initialize models in child process
    from docling import DocumentConverter
    converter = DocumentConverter()

    while True:
        task = task_queue.get()
        if task is None:  # Poison pill to shutdown
            break

        try:
            # Process document
            result = converter.convert(task['document'])
            result_queue.put({'status': 'success', 'result': result})
        except Exception as e:
            result_queue.put({'status': 'error', 'error': str(e)})
        finally:
            # Process will be terminated after task
            break

    # Explicit cleanup before exit
    del converter
    torch.cuda.empty_cache()


async def process_with_child_process(document):
    """Process document in isolated child process."""
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # Start worker process
    process = Process(
        target=ocr_worker_process,
        args=(task_queue, result_queue)
    )
    process.start()

    # Send task
    task_queue.put({'document': document})

    # Wait for result with timeout
    try:
        result = result_queue.get(timeout=300)  # 5 min timeout
    finally:
        # Ensure process is terminated
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join()

    return result
```

#### Option B: Process Pool with Lifecycle Management

```python
from concurrent.futures import ProcessPoolExecutor
import asyncio

class OCRProcessPool:
    """Managed pool of OCR worker processes."""

    def __init__(self, max_workers=1):
        self.max_workers = max_workers
        self.executor = None

    def start(self):
        """Start the process pool."""
        self.executor = ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=mp.get_context('spawn')  # Use spawn for CUDA
        )

    def shutdown(self):
        """Shutdown pool and wait for processes to terminate."""
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None

    async def process_document(self, document):
        """Process document in pool worker."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            ocr_task_function,
            document
        )

        # Restart pool after each task for complete cleanup
        if docling_serve_settings.free_vram_on_idle:
            self.shutdown()
            self.start()

        return result


def ocr_task_function(document):
    """Function executed in worker process."""
    import torch
    from docling import DocumentConverter

    try:
        converter = DocumentConverter()
        result = converter.convert(document)
        return result
    finally:
        # Cleanup before process returns to pool
        del converter
        import gc
        gc.collect()
        torch.cuda.empty_cache()
```

#### Option C: One-Shot Process Per Task (Simplest)

```python
async def process_in_subprocess(document_path: str):
    """Run OCR in completely isolated subprocess."""

    # Create temporary script
    script = f"""
import sys
import torch
from docling import DocumentConverter

try:
    converter = DocumentConverter()
    result = converter.convert("{document_path}")

    # Save result
    import json
    with open("/tmp/result.json", "w") as f:
        json.dump(result.export_to_dict(), f)

    sys.exit(0)
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    sys.exit(1)
"""

    # Run in subprocess
    process = await asyncio.create_subprocess_exec(
        "python", "-c", script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()

    if process.returncode == 0:
        # Load result
        with open("/tmp/result.json") as f:
            return json.load(f)
    else:
        raise RuntimeError(f"OCR failed: {stderr.decode()}")
```

### Integration Points:

Would need to modify:
- `_enque_file()` - Use child process for actual conversion
- `_enque_source()` - Use child process for actual conversion
- `ensure_models_loaded()` - Skip if using child processes
- `cleanup_models_if_needed()` - Skip if using child processes

### When to Consider:

Move to child process approach if:
1. In-process cleanup still shows >1GB accumulation after improvements
2. ONNX Runtime or other libraries have unfixable memory leaks
3. Complete isolation is required for stability
4. Task frequency allows for process startup overhead

### Recommended Next Steps:

1. **Test current improvements first**
   - Run 10+ consecutive OCR tasks
   - Monitor VRAM via nvidia-smi
   - Check if memory stays under 700MB baseline

2. **If accumulation persists > 1GB**
   - Implement Option B (Process Pool)
   - Use pool size of 1 worker
   - Restart pool after each task
   - Measure startup overhead vs. memory savings

3. **Performance tuning**
   - Cache model weights on disk
   - Use shared memory for models if possible
   - Pre-warm worker processes during idle time

## References

- [PyTorch CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [Clearing GPU Memory After Training](https://www.geeksforgeeks.org/deep-learning/clearing-gpu-memory-after-pytorch-training-without-kernel-restart/)
- [PyTorch empty_cache()](https://pytorch.org/docs/stable/generated/torch.cuda.empty_cache.html)
- [Moving from GPU to CPU](https://discuss.pytorch.org/t/moving-from-gpu-to-cpu-not-freeing-gpuram/85313)
- [How to Delete Tensor in GPU](https://discuss.pytorch.org/t/how-to-delete-a-tensor-in-gpu-to-free-up-memory/48879)
- [How to Clear CUDA Memory](https://stackoverflow.com/questions/55322434/how-to-clear-cuda-memory-in-pytorch)
