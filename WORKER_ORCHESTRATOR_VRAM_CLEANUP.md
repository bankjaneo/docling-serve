# WorkerOrchestrator for Complete VRAM Cleanup

## Problem

The original VRAM cleanup implementation in docling-serve suffered from persistent CUDA context overhead:
- Even with aggressive cleanup, ~1.7GB of VRAM remained allocated after each OCR task
- VRAM accumulated over multiple tasks (682MB → 1.3GB → 1.9GB)
- CUDA contexts persisted within the same process
- Risk of crashes and timeouts with aggressive cleanup attempts

## Solution: Worker Process Isolation

The WorkerOrchestrator solves this by running **each OCR task in a completely separate process**:

### How It Works

1. **Process Isolation**: Each OCR task spawns a new Python process
2. **Complete Cleanup**: When the task finishes, the entire process is terminated
3. **VRAM Release**: All CUDA contexts, memory allocations, and model weights are released automatically
4. **No Accumulation**: Each task starts with clean VRAM state

### Architecture

```
Main docling-serve process (minimal VRAM usage)
    ├── Task 1 → Worker Process 1 (loads models, processes OCR, terminates)
    ├── Task 2 → Worker Process 2 (loads models, processes OCR, terminates)
    ├── Task 3 → Worker Process 3 (loads models, processes OCR, terminates)
    └── ...
```

### Key Benefits

1. **Complete VRAM Release**: ~0MB VRAM between tasks (only main process overhead)
2. **No Accumulation**: Each task starts fresh, no memory buildup
3. **Crash Isolation**: Worker process crashes don't affect main service
4. **Simplicity**: No complex cleanup logic needed - process termination handles everything
5. **Safety**: No dangerous CUDA context manipulation

## Implementation Details

### WorkerOrchestrator Class

Located in `docling_serve/worker_orchestrator.py`:

- **Process Management**: Spawns subprocesses for each task
- **Communication**: Uses file-based I/O for task data and results
- **Monitoring**: Tracks worker processes and handles timeouts
- **Resource Cleanup**: Automatically terminates workers and cleans up files

### Worker Process

Each worker process:
1. **Loads Models**: Fresh docling models and CUDA contexts
2. **Processes OCR**: Handles the actual document conversion
3. **Saves Results**: Writes output to result file
4. **Terminates**: Entire process exits, releasing all VRAM

### Automatic Selection

The WorkerOrchestrator is automatically used when:
- `DOCLING_SERVE_FREE_VRAM_ON_IDLE=True` is set
- WorkerOrchestrator import succeeds

## Expected VRAM Usage

### Before (with in-process cleanup)
- **Initial**: ~6GB VRAM
- **After Task 1**: ~1.7GB persistent
- **After Task 2**: ~3.4GB persistent
- **After Task 3**: ~5.1GB persistent
- **Problem**: Continuous accumulation

### After (with WorkerOrchestrator)
- **Initial**: ~50MB (main process only)
- **During Task**: ~6GB (worker process)
- **After Task**: ~50MB (worker terminated)
- **Result**: Consistent low VRAM usage

## Configuration

### Environment Variable

```bash
export DOCLING_SERVE_FREE_VRAM_ON_IDLE=True
```

### Required Dependencies

- Redis (for task status and result storage)
- Sufficient disk space for temporary worker files

### Redis Configuration

WorkerOrchestrator uses Redis for:
- Task metadata storage
- Result storage
- Process coordination

Redis URL defaults to `redis://localhost:6379` and can be configured via docling-serve settings.

## Monitoring

### Logs

Look for these log messages:

```
INFO: Using WorkerOrchestrator for complete VRAM isolation
INFO: Task abc-123 queued for worker processing
INFO: Task abc-123 completed with status: success
DEBUG: Task abc-123 completed - WorkerOrchestrator handles VRAM cleanup automatically
```

### VRAM Monitoring

Use `nvidia-smi` to monitor VRAM:

```bash
# Watch VRAM usage in real-time
watch -n 1 nvidia-smi
```

Expected pattern:
- **Idle**: ~50MB (main process only)
- **Processing**: ~6GB (worker process active)
- **Complete**: Returns to ~50MB

## Troubleshooting

### Common Issues

1. **Worker Fails to Start**
   - Check Redis connection
   - Verify Python path and environment
   - Check disk space for temporary files

2. **Worker Timeout**
   - Increase `DOCLING_SERVE_MAX_DOCUMENT_TIMEOUT`
   - Check if tasks are too large/complex

3. **Redis Connection Issues**
   - Ensure Redis is running
   - Check Redis URL configuration
   - Verify network connectivity

### Debug Mode

Enable debug logging:
```bash
export DOCLING_SERVE_LOG_LEVEL=DEBUG
```

This will show detailed worker process information and Redis operations.

## Performance Considerations

### Process Startup Overhead

- **Tradeoff**: Slight delay for process startup (~1-2 seconds)
- **Benefit**: Complete VRAM cleanup and isolation
- **Net Result**: Better for memory-constrained environments

### Concurrency

- Multiple workers can run simultaneously
- Each worker gets its own VRAM allocation
- Monitor total VRAM usage with concurrent tasks

### Resource Limits

- Set appropriate worker timeouts
- Monitor disk space for temporary files
- Consider Redis memory usage for many concurrent tasks

## Migration

### From Standard Orchestrator

1. Set `DOCLING_SERVE_FREE_VRAM_ON_IDLE=True`
2. Restart docling-serve
3. WorkerOrchestrator will be used automatically

### Backward Compatibility

- If WorkerOrchestrator fails to import, falls back to standard orchestrator
- Existing API endpoints remain unchanged
- All configuration options are preserved

## File Cleanup

WorkerOrchestrator automatically cleans up:
- Input files (`task_*_input.json`)
- Output files (`task_*_output.json`)
- Terminated worker processes

Files are cleaned up immediately after task completion or timeout.

## Security

### Process Isolation

- Workers run in separate process groups
- Worker termination doesn't affect main process
- File I/O is isolated to scratch directory

### Input Validation

- Task inputs are validated before worker execution
- File paths are restricted to scratch directory
- Worker processes have limited permissions

## Conclusion

The WorkerOrchestrator provides a robust solution for complete VRAM cleanup by leveraging process isolation. This approach eliminates VRAM accumulation issues while maintaining system stability and crash isolation.