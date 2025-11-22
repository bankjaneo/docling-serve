# Quick Subprocess Integration

## Current Status

Your environment variables are set correctly:
```yaml
- DOCLING_SERVE_USE_SUBPROCESS=true
- DOCLING_SERVE_SUBPROCESS_TIMEOUT=600
- DOCLING_SERVE_FREE_VRAM_ON_IDLE=True
```

**But**: The subprocess worker code exists but isn't integrated into the orchestrator yet.

## Test Results (10 OCR tasks)

- Memory stayed at ~1340-1762 MB
- NOT returning to baseline ~682 MB
- Confirms ONNX Runtime CUDA allocator is the issue
- Subprocess mode is the ONLY solution

## Problem

The subprocess integration requires understanding how your orchestrator processes tasks, which depends on whether you're using:
- Local orchestrator
- RQ orchestrator
- KFP orchestrator

This is complex and requires knowledge of the docling-jobkit internals.

## Alternative: Simpler Workaround

Instead of integrating into the orchestrator, there's a simpler approach:

**Restart the container after N tasks** to force complete VRAM cleanup.

### Option A: Auto-Restart Container After N Tasks

Add a task counter and auto-restart when limit is reached.

**1. Add to `docling_serve/app.py`** (after imports):

```python
import os
import signal

# Task counter for auto-restart
_task_counter = 0
_max_tasks_before_restart = int(os.getenv("DOCLING_SERVE_MAX_TASKS_BEFORE_RESTART", "0"))
```

**2. Add counter increment in `cleanup_models_after_task()`**:

```python
async def cleanup_models_after_task(orchestrator: BaseOrchestrator, task_id: str):
    """
    Cleanup models after a task completes (for synchronous endpoints).
    Only clears models if there are no other active tasks running.
    """
    global _task_counter

    if not docling_serve_settings.free_vram_on_idle:
        return

    # Increment task counter
    _task_counter += 1
    _log.info(f"Task counter: {_task_counter}/{_max_tasks_before_restart}")

    # Check if we should restart
    if _max_tasks_before_restart > 0 and _task_counter >= _max_tasks_before_restart:
        _log.warning(f"Reached max tasks ({_max_tasks_before_restart}), initiating graceful shutdown...")
        _log.warning("Container will restart automatically if restart policy is configured")

        # Give time for response to be sent
        await asyncio.sleep(1.0)

        # Graceful shutdown
        os.kill(os.getpid(), signal.SIGTERM)
        return

    # ... rest of existing function
```

**3. Update docker-compose.yml**:

```yaml
services:
  docling-serve:
    image: bankja/docling:cuda
    container_name: docling
    ports:
      - "5001:5001"
    environment:
      - DOCLING_SERVE_ENABLE_UI=1
      - DOCLING_SERVE_MAX_SYNC_WAIT=600
      - DOCLING_SERVE_ENABLE_REMOTE_SERVICES=1
      - DOCLING_NUM_THREADS=7
      - DOCLING_SERVE_FREE_VRAM_ON_IDLE=True
      - DOCLING_SERVE_UNLOAD_LLAMA_SWAP_BASE_URL=http://172.17.0.1:9292/v1
      - DOCLING_SERVE_MAX_TASKS_BEFORE_RESTART=20  # Restart after 20 tasks
    volumes:
      - ./thai.pth:/opt/app-root/src/.cache/docling/models/EasyOcr/thai.pth:ro
    restart: unless-stopped  # This ensures auto-restart after shutdown
    logging:
      driver: "json-file"
      options:
        max-file: "1"
        max-size: "10m"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

**Benefits**:
- ✅ Simple to implement (just a counter)
- ✅ Guaranteed VRAM cleanup (container restart)
- ✅ Graceful shutdown between restarts
- ✅ Configurable restart frequency
- ✅ No complex orchestrator changes needed

**Drawbacks**:
- ⚠️ Brief downtime during restart (~5-10 seconds)
- ⚠️ Need to tune restart frequency (too frequent = overhead, too rare = memory buildup)

### Option B: Manual Restart Endpoint

Add an endpoint to manually trigger restart when needed.

**Add to `docling_serve/app.py`** (in the `create_app()` function):

```python
@app.post("/v1/admin/restart", tags=["admin"], include_in_schema=False)
async def trigger_restart(
    auth: Annotated[AuthenticationResult, Depends(require_auth)],
):
    """
    Trigger graceful server restart. Useful for clearing VRAM when
    free_vram_on_idle cannot fully clean up ONNX Runtime CUDA memory.
    """
    _log.warning("Manual restart triggered via API")

    # Return response before shutdown
    response = {"status": "restarting", "message": "Server will restart in 1 second"}

    async def delayed_shutdown():
        await asyncio.sleep(1.0)
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.create_task(delayed_shutdown())

    return response
```

**Usage**:
```bash
# Manually restart when VRAM is high
curl -X POST http://localhost:5001/v1/admin/restart
```

### Option C: Scheduled Restart (Cron-based)

Add a cron job to restart the container periodically.

**1. Create cron job on host**:

```bash
# Restart docling container every 6 hours
0 */6 * * * docker restart docling
```

**2. Or use a simple monitor script**:

```bash
#!/bin/bash
# monitor-and-restart.sh

while true; do
    # Check VRAM usage
    VRAM_MB=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | head -1)

    if [ "$VRAM_MB" -gt 1500 ]; then
        echo "VRAM usage ${VRAM_MB}MB exceeds threshold, restarting container..."
        docker restart docling
        sleep 30  # Wait for restart
    fi

    sleep 60  # Check every minute
done
```

**Run in background**:
```bash
chmod +x monitor-and-restart.sh
nohup ./monitor-and-restart.sh > /var/log/docling-monitor.log 2>&1 &
```

## Recommendation

Given your test results showing stable but elevated memory (~1340-1762 MB):

**Best approach for immediate fix**: **Option A (Auto-Restart After N Tasks)**

Why:
1. ✅ Simplest to implement (just add counter)
2. ✅ Guaranteed to work (container restart = VRAM cleared)
3. ✅ Predictable behavior
4. ✅ No external dependencies
5. ✅ ~5-10 second downtime acceptable for guaranteed cleanup

**Suggested configuration**:
```yaml
DOCLING_SERVE_MAX_TASKS_BEFORE_RESTART=50
```

This means:
- After 50 OCR tasks, container auto-restarts
- Takes ~5-10 seconds to restart
- VRAM returns to baseline ~682 MB
- Client experiences brief downtime (can implement retry logic)

**Long-term solution**: Full subprocess integration
- More complex (requires orchestrator changes)
- No downtime
- Better performance
- Requires deeper integration work

## Next Steps

1. **Immediate fix**: Implement Option A (auto-restart counter)
   - Add counter to `cleanup_models_after_task()`
   - Set `DOCLING_SERVE_MAX_TASKS_BEFORE_RESTART=50`
   - Test with 60+ tasks to verify restart happens

2. **Test and tune**:
   - Monitor how many tasks until memory becomes problematic
   - Adjust restart frequency accordingly
   - Balance between VRAM usage and restart overhead

3. **Later** (if needed): Full subprocess integration
   - Only if auto-restart approach has unacceptable downtime
   - Requires understanding orchestrator internals
   - I can help with this if you decide to proceed

The auto-restart approach is a practical, production-ready solution that solves the ONNX Runtime memory issue without requiring complex code changes.
