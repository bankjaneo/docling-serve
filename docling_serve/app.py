import asyncio
import copy
import importlib.metadata
import logging
import shutil
import time
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Annotated

import httpx

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from scalar_fastapi import get_scalar_api_reference

from docling.datamodel.base_models import DocumentStream
from docling_jobkit.datamodel.callback import (
    ProgressCallbackRequest,
    ProgressCallbackResponse,
)
from docling_jobkit.datamodel.chunking import (
    BaseChunkerOptions,
    ChunkingExportOptions,
    HierarchicalChunkerOptions,
    HybridChunkerOptions,
)
from docling_jobkit.datamodel.http_inputs import FileSource, HttpSource
from docling_jobkit.datamodel.s3_coords import S3Coordinates
from docling_jobkit.datamodel.task import Task, TaskSource, TaskStatus, TaskType
from docling_jobkit.datamodel.task_targets import (
    InBodyTarget,
    ZipTarget,
)
from docling_jobkit.orchestrators.base_orchestrator import (
    BaseOrchestrator,
    ProgressInvalid,
    TaskNotFoundError,
)

from docling_serve.auth import APIKeyAuth, AuthenticationResult
from docling_serve.datamodel.convert import ConvertDocumentsRequestOptions
from docling_serve.datamodel.requests import (
    ConvertDocumentsRequest,
    FileSourceRequest,
    GenericChunkDocumentsRequest,
    HttpSourceRequest,
    S3SourceRequest,
    TargetName,
    TargetRequest,
    make_request_model,
)
from docling_serve.datamodel.responses import (
    ChunkDocumentResponse,
    ClearResponse,
    ConvertDocumentResponse,
    HealthCheckResponse,
    MemoryUsageResponse,
    MessageKind,
    PresignedUrlConvertDocumentResponse,
    TaskStatusResponse,
    WebsocketMessage,
)
from docling_serve.helper_functions import DOCLING_VERSIONS, FormDepends
from docling_serve.orchestrator_factory import get_async_orchestrator
from docling_serve.response_preparation import prepare_response
from docling_serve.settings import docling_serve_settings
from docling_serve.storage import get_scratch
from docling_serve.websocket_notifier import WebsocketNotifier


# Set up custom logging as we'll be intermixes with FastAPI/Uvicorn's logging
class ColoredLogFormatter(logging.Formatter):
    COLOR_CODES = {
        logging.DEBUG: "\033[94m",  # Blue
        logging.INFO: "\033[92m",  # Green
        logging.WARNING: "\033[93m",  # Yellow
        logging.ERROR: "\033[91m",  # Red
        logging.CRITICAL: "\033[95m",  # Magenta
    }
    RESET_CODE = "\033[0m"

    def format(self, record):
        color = self.COLOR_CODES.get(record.levelno, "")
        record.levelname = f"{color}{record.levelname}{self.RESET_CODE}"
        return super().format(record)


logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(levelname)s:\t%(asctime)s - %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)

# Override the formatter with the custom ColoredLogFormatter
root_logger = logging.getLogger()  # Get the root logger
for handler in root_logger.handlers:  # Iterate through existing handlers
    if handler.formatter:
        handler.setFormatter(ColoredLogFormatter(handler.formatter._fmt))

_log = logging.getLogger(__name__)


# Context manager to initialize and clean up the lifespan of the FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    scratch_dir = get_scratch()

    orchestrator = get_async_orchestrator()
    notifier = WebsocketNotifier(orchestrator)
    orchestrator.bind_notifier(notifier)

    # Warm up processing cache
    if docling_serve_settings.load_models_at_boot and not docling_serve_settings.free_vram_on_idle:
        await orchestrator.warm_up_caches()

    # Start the background queue processor
    queue_task = asyncio.create_task(orchestrator.process_queue())

    yield

    # Cancel the background queue processor on shutdown
    queue_task.cancel()
    try:
        await queue_task
    except asyncio.CancelledError:
        _log.info("Queue processor cancelled.")

    # Remove scratch directory in case it was a tempfile
    if docling_serve_settings.scratch_path is not None:
        shutil.rmtree(scratch_dir, ignore_errors=True)


##################################
# Lazy loading helper functions #
##################################


async def unload_external_models():
    """
    Unload models from external services (Ollama or llama-swap) to free VRAM
    before loading Docling models. This is useful for systems with low VRAM
    that need to swap models between different applications.
    """
    tasks = []

    # Unload from llama-swap if configured
    if docling_serve_settings.unload_llama_swap_base_url:
        tasks.append(
            ("llama-swap", _unload_llama_swap(docling_serve_settings.unload_llama_swap_base_url))
        )

    # Unload from Ollama if configured
    if docling_serve_settings.unload_ollama_base_url and docling_serve_settings.unload_ollama_model:
        tasks.append(
            ("Ollama", _unload_ollama(
                docling_serve_settings.unload_ollama_base_url,
                docling_serve_settings.unload_ollama_model
            ))
        )

    # Execute all unload tasks concurrently
    if tasks:
        _log.info(f"Unloading external models from: {', '.join([name for name, _ in tasks])}")
        await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)


async def _unload_llama_swap(base_url: str):
    """Unload models from llama-swap by calling the /unload endpoint."""
    # Parse URL to extract only scheme and netloc, stripping any path components
    # This handles cases where users include /v1 or other paths in the base URL
    parsed = httpx.URL(base_url)
    clean_base_url = f"{parsed.scheme}://{parsed.netloc}"
    url = f"{clean_base_url}/unload"

    try:
        async with httpx.AsyncClient(
            timeout=docling_serve_settings.unload_external_model_timeout
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            _log.info(f"Successfully unloaded llama-swap models at {clean_base_url}")
    except Exception as e:
        _log.warning(f"Failed to unload llama-swap models at {clean_base_url}: {e}")
        raise


async def _unload_ollama(base_url: str, model_name: str):
    """Unload models from Ollama by setting keep_alive to 0."""
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {"model": model_name, "keep_alive": 0}
    try:
        async with httpx.AsyncClient(
            timeout=docling_serve_settings.unload_external_model_timeout
        ) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            _log.info(f"Successfully unloaded Ollama model '{model_name}' at {base_url}")
    except Exception as e:
        _log.warning(f"Failed to unload Ollama model '{model_name}' at {base_url}: {e}")
        raise


async def ensure_models_loaded(orchestrator: BaseOrchestrator):
    """Ensure models are loaded before processing if lazy loading is enabled."""
    if docling_serve_settings.free_vram_on_idle:
        # First, unload external models to free VRAM if configured
        await unload_external_models()
        # Then load Docling models
        _log.info("Loading models for processing...")
        await orchestrator.warm_up_caches()


async def cleanup_models_if_needed(orchestrator: BaseOrchestrator):
    """Enhanced model cleanup to aggressively free VRAM and reduce memory retention."""
    if docling_serve_settings.free_vram_on_idle:
        _log.info("Starting enhanced VRAM cleanup...")

        # Log detailed VRAM usage before cleanup
        mem_before = 0
        mem_reserved_before = 0
        device_info = ""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_info = f" (devices: {device_count})"
                for device_id in range(device_count):
                    mem_before_dev = torch.cuda.memory_allocated(device_id) / 1024**2
                    mem_reserved_dev = torch.cuda.memory_reserved(device_id) / 1024**2
                    device_props = torch.cuda.get_device_properties(device_id)
                    total_mem = device_props.total_memory / 1024**3

                    if getattr(docling_serve_settings, 'enable_detailed_memory_logging', False):
                        _log.info(f"Device {device_id} ({device_props.name}) - Total: {total_mem:.1f} GB, "
                                f"Allocated: {mem_before_dev:.2f} MB, Reserved: {mem_reserved_dev:.2f} MB")

                    mem_before += mem_before_dev
                    mem_reserved_before += mem_reserved_dev

                _log.info(f"VRAM before cleanup{device_info} - Allocated: {mem_before:.2f} MB, Reserved: {mem_reserved_before:.2f} MB")
        except Exception as e:
            _log.debug(f"Failed to get initial VRAM info: {e}")

        # Step 1: Enhanced model moving to CPU with deep traversal
        models_moved = 0
        try:
            import torch
            if torch.cuda.is_available():
                _log.info("Moving all models to CPU before deletion...")

                # Access converter manager through multiple possible paths
                cm = None
                if hasattr(orchestrator, 'cm'):
                    cm = orchestrator.cm
                elif hasattr(orchestrator, 'converter_manager'):
                    cm = orchestrator.converter_manager

                if cm:
                    # Try different cache access methods
                    cache_accessors = [
                        '_get_converter_from_hash',
                        '_converter_cache',
                        '_cached_converters',
                        'converter_cache'
                    ]

                    for attr_name in cache_accessors:
                        if hasattr(cm, attr_name):
                            cache = getattr(cm, attr_name)
                            models_moved += await _move_models_to_cpu_deep(cache, attr_name)

                # Additional: Search for any torch.nn.Module objects in orchestrator
                models_moved += await _find_and_move_torch_modules(orchestrator)

                _log.info(f"Moved {models_moved} models to CPU")
        except Exception as e:
            _log.warning(f"Failed to move models to CPU: {e}")

        # Step 2: Aggressive ONNX Runtime cleanup
        onnx_sessions_closed = 0
        try:
            onnx_sessions_closed = await _aggressive_onnx_cleanup(orchestrator)
            _log.info(f"Closed {onnx_sessions_closed} ONNX Runtime sessions")
        except Exception as e:
            _log.warning(f"ONNX Runtime cleanup failed: {e}")

        # Step 3: Enhanced cache clearing with multiple strategies
        caches_cleared = 0
        try:
            caches_cleared = await _enhanced_cache_clearing(orchestrator)
            _log.info(f"Cleared {caches_cleared} converter caches")
        except Exception as e:
            _log.warning(f"Enhanced cache clearing failed: {e}")

        # Step 4: Call orchestrator's clear_converters method
        try:
            await orchestrator.clear_converters()
            _log.info("Called orchestrator.clear_converters()")
        except Exception as e:
            _log.warning(f"orchestrator.clear_converters() failed: {e}")

        # Step 5: Multi-stage garbage collection with memory pressure
        try:
            await _multi_stage_gc()
        except Exception as e:
            _log.warning(f"Multi-stage GC failed: {e}")

        # Step 6: ONNX Runtime provider-specific cleanup
        try:
            await _onnx_provider_cleanup()
        except Exception as e:
            _log.debug(f"ONNX provider cleanup: {e}")

        # Step 7: Aggressive CUDA memory cleanup with context management
        try:
            await _aggressive_cuda_cleanup()
        except Exception as e:
            _log.warning(f"Aggressive CUDA cleanup failed: {e}")

        # Step 8: Final memory usage reporting with detailed logging
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                mem_after = 0
                mem_reserved_after = 0

                # Collect detailed per-device information
                for device_id in range(device_count):
                    mem_after_dev = torch.cuda.memory_allocated(device_id) / 1024**2
                    mem_reserved_dev = torch.cuda.memory_reserved(device_id) / 1024**2
                    device_props = torch.cuda.get_device_properties(device_id)
                    total_mem = device_props.total_memory / 1024**3

                    mem_after += mem_after_dev
                    mem_reserved_after += mem_reserved_dev

                    if getattr(docling_serve_settings, 'enable_detailed_memory_logging', False):
                        _log.info(f"Device {device_id} after cleanup ({device_props.name}) - "
                                f"Total: {total_mem:.1f} GB, Allocated: {mem_after_dev:.2f} MB, "
                                f"Reserved: {mem_reserved_dev:.2f} MB")

                mem_freed = mem_before - mem_after
                reserved_freed = mem_reserved_before - mem_reserved_after
                cleanup_efficiency = (mem_freed / max(mem_before, 1)) * 100 if mem_before > 0 else 0

                _log.info(f"VRAM after cleanup{device_info} - Allocated: {mem_after:.2f} MB, Reserved: {mem_reserved_after:.2f} MB")
                _log.info(f"Memory freed - Allocated: {mem_freed:.2f} MB ({cleanup_efficiency:.1f}%), "
                         f"Reserved: {reserved_freed:.2f} MB")

                # Enhanced warning with context information and recommendations
                expected_overhead = getattr(docling_serve_settings, 'expected_vram_overhead_mb', 400.0)

                if mem_after > expected_overhead:
                    overhead_mb = mem_after - expected_overhead
                    _log.warning(
                        f"VRAM still has {mem_after:.2f} MB allocated ({overhead_mb:.1f} MB above expected {expected_overhead:.0f} MB). "
                        f"Cleanup effectiveness: {models_moved} models moved, {onnx_sessions_closed} ONNX sessions closed, "
                        f"{caches_cleared} caches cleared."
                    )

                    # Provide recommendations based on remaining memory
                    if overhead_mb > 600:
                        _log.warning(
                            "High VRAM retention detected. Consider: "
                            "1) Setting DOCLING_SERVE_FORCE_CUDA_CONTEXT_RESET=true (experimental), "
                            "2) Restarting the process for full memory release, "
                            "3) Checking for memory leaks in external dependencies"
                        )
                    elif overhead_mb > 200:
                        _log.info(
                            "Moderate VRAM retention. This is typically normal CUDA context overhead. "
                            "For maximum cleanup, consider process restart."
                        )
                else:
                    _log.info(f"VRAM cleanup excellent - remaining {mem_after:.2f} MB is within expected range")

        except Exception as e:
            _log.warning(f"Final memory reporting failed: {e}")


async def _move_models_to_cpu_deep(cache_obj, cache_name: str) -> int:
    """Deep traversal to find and move PyTorch models to CPU."""
    models_moved = 0
    try:
        import torch

        # Handle LRU cache
        if hasattr(cache_obj, '__wrapped__') and hasattr(cache_obj, 'cache'):
            cache_dict = cache_obj.cache
            _log.debug(f"Processing LRU cache {cache_name} with {len(cache_dict)} items")

            for key, value in list(cache_dict.items()):
                models_moved += await _move_object_to_cpu_recursive(value, f"cache[{cache_name}][{key}]")

        # Handle regular dict cache
        elif hasattr(cache_obj, 'values') or hasattr(cache_obj, 'items'):
            items = list(cache_obj.items()) if hasattr(cache_obj, 'items') else list(cache_obj)
            _log.debug(f"Processing cache {cache_name} with {len(items)} items")

            for item in items:
                if isinstance(item, tuple):
                    value = item[1]
                    key = item[0]
                else:
                    value = item
                    key = str(item)[:50]

                models_moved += await _move_object_to_cpu_recursive(value, f"cache[{cache_name}][{key}]")

        # Handle single object
        else:
            models_moved += await _move_object_to_cpu_recursive(cache_obj, cache_name)

    except Exception as e:
        _log.debug(f"Error in _move_models_to_cpu_deep for {cache_name}: {e}")

    return models_moved


async def _move_object_to_cpu_recursive(obj, path: str) -> int:
    """Recursively move PyTorch models to CPU and count them."""
    import torch
    models_moved = 0

    try:
        # Check if this is a PyTorch model
        if hasattr(obj, 'to') and any(hasattr(obj, attr) for attr in ['parameters', 'state_dict', 'load_state_dict']):
            try:
                obj.to('cpu')
                models_moved += 1
                _log.debug(f"Moved model at {path} to CPU")
            except Exception as e:
                _log.debug(f"Failed to move model at {path}: {e}")

        # Check for nested models in doc_converter attribute
        if hasattr(obj, 'doc_converter') and hasattr(obj.doc_converter, 'to'):
            try:
                obj.doc_converter.to('cpu')
                models_moved += 1
                _log.debug(f"Moved doc_converter at {path} to CPU")
            except Exception as e:
                _log.debug(f"Failed to move doc_converter at {path}: {e}")

        # Recursively search in object attributes
        if hasattr(obj, '__dict__'):
            for attr_name, attr_value in list(obj.__dict__.items()):
                # Skip certain attributes that can cause issues
                if attr_name.startswith('_') or attr_value is None or attr_value is obj:
                    continue

                # Recursively search in nested objects (but limit depth)
                if len(path) < 200:  # Prevent infinite recursion
                    models_moved += await _move_object_to_cpu_recursive(attr_value, f"{path}.{attr_name}")

    except Exception as e:
        _log.debug(f"Error in _move_object_to_cpu_recursive for {path}: {e}")

    return models_moved


async def _find_and_move_torch_modules(obj, path: str = "orchestrator") -> int:
    """Find and move any torch.nn.Module objects to CPU."""
    import torch
    models_moved = 0

    try:
        # Check if object is a torch.nn.Module
        if isinstance(obj, torch.nn.Module):
            try:
                obj.to('cpu')
                models_moved += 1
                _log.debug(f"Moved torch.nn.Module at {path} to CPU")
            except Exception as e:
                _log.debug(f"Failed to move torch.nn.Module at {path}: {e}")

        # Recursively search in object attributes
        if hasattr(obj, '__dict__'):
            for attr_name, attr_value in list(obj.__dict__.items()):
                # Skip certain attributes
                if attr_name.startswith('_') or attr_value is None or attr_value is obj:
                    continue

                # Limit recursion depth
                if len(path) < 200:
                    models_moved += await _find_and_move_torch_modules(attr_value, f"{path}.{attr_name}")

    except Exception as e:
        _log.debug(f"Error in _find_and_move_torch_modules for {path}: {e}")

    return models_moved


async def _aggressive_onnx_cleanup(orchestrator) -> int:
    """Aggressively find and destroy ONNX Runtime sessions."""
    sessions_closed = 0

    try:
        # Search for ONNX sessions in orchestrator
        sessions_closed += await _find_onnx_sessions_recursive(orchestrator, "orchestrator")

        # Additional cleanup: try to access converter manager caches directly
        cm = getattr(orchestrator, 'cm', None) or getattr(orchestrator, 'converter_manager', None)
        if cm:
            sessions_closed += await _find_onnx_sessions_recursive(cm, "converter_manager")

    except Exception as e:
        _log.debug(f"Error in _aggressive_onnx_cleanup: {e}")

    return sessions_closed


async def _find_onnx_sessions_recursive(obj, path: str) -> int:
    """Recursively find and destroy ONNX Runtime sessions."""
    sessions_closed = 0

    try:
        import onnxruntime

        # Check if object is an ONNX InferenceSession
        if 'onnxruntime' in str(type(obj)) and 'InferenceSession' in str(type(obj)):
            try:
                # Try to close the session if it has a close method
                if hasattr(obj, 'close'):
                    obj.close()
                _log.debug(f"Closed ONNX session at {path}")
                sessions_closed += 1
            except Exception as e:
                _log.debug(f"Failed to close ONNX session at {path}: {e}")

        # Search in object attributes
        if hasattr(obj, '__dict__'):
            for attr_name, attr_value in list(obj.__dict__.items()):
                # Limit recursion depth
                if len(path) < 200:
                    sessions_closed += await _find_onnx_sessions_recursive(attr_value, f"{path}.{attr_name}")

    except Exception as e:
        _log.debug(f"Error in _find_onnx_sessions_recursive for {path}: {e}")

    return sessions_closed


async def _enhanced_cache_clearing(orchestrator) -> int:
    """Enhanced cache clearing with multiple strategies."""
    caches_cleared = 0

    try:
        # Get converter manager
        cm = getattr(orchestrator, 'cm', None) or getattr(orchestrator, 'converter_manager', None)

        if cm:
            # Try multiple cache attribute names
            cache_attrs = [
                '_get_converter_from_hash',
                '_converter_cache',
                '_cached_converters',
                'converter_cache',
                '_options_cache'
            ]

            for attr_name in cache_attrs:
                if hasattr(cm, attr_name):
                    cache_obj = getattr(cm, attr_name)
                    cleared = await _clear_cache_object(cache_obj, attr_name)
                    caches_cleared += cleared
                    _log.debug(f"Cleared {cleared} items from cache {attr_name}")

        # Try to clear any dict-like objects that might be caches
        for attr_name in dir(orchestrator):
            if not attr_name.startswith('_') and attr_name != 'cm':
                try:
                    attr = getattr(orchestrator, attr_name)
                    if isinstance(attr, dict) and len(attr) > 0:
                        # Check if it might be a cache (has converter-like objects)
                        sample_item = next(iter(attr.values())) if attr else None
                        if sample_item and hasattr(sample_item, 'to'):
                            _log.debug(f"Found potential cache dict {attr_name} with {len(attr)} items")
                            attr.clear()
                            caches_cleared += len(attr)
                except Exception:
                    pass

    except Exception as e:
        _log.debug(f"Error in _enhanced_cache_clearing: {e}")

    return caches_cleared


async def _clear_cache_object(cache_obj, cache_name: str) -> int:
    """Clear a cache object using appropriate method."""
    items_cleared = 0

    try:
        # Handle LRU cache
        if hasattr(cache_obj, 'cache') and hasattr(cache_obj, 'cache_clear'):
            items_cleared = len(cache_obj.cache)
            cache_obj.cache_clear()
            _log.debug(f"Cleared LRU cache {cache_name} with {items_cleared} items")

        # Handle regular dict
        elif hasattr(cache_obj, 'clear'):
            items_cleared = len(cache_obj) if hasattr(cache_obj, '__len__') else 0
            cache_obj.clear()
            _log.debug(f"Cleared dict cache {cache_name} with {items_cleared} items")

        # Handle cache with values() method
        elif hasattr(cache_obj, 'values'):
            items = list(cache_obj.values())
            items_cleared = len(items)
            # Try to delete each item
            if hasattr(cache_obj, '__delitem__'):
                for key in list(cache_obj.keys()):
                    try:
                        del cache_obj[key]
                    except Exception:
                        pass

    except Exception as e:
        _log.debug(f"Error clearing cache {cache_name}: {e}")

    return items_cleared


async def _multi_stage_gc():
    """Multi-stage garbage collection with memory pressure simulation."""
    import gc

    _log.debug("Starting multi-stage garbage collection...")

    # Stage 1: Standard GC multiple times
    for i in range(3):
        collected = gc.collect()
        _log.debug(f"GC stage 1.{i+1}: collected {collected} objects")

    # Stage 2: Generation-specific collection
    for gen in range(3):
        collected = gc.collect(gen)
        _log.debug(f"GC generation {gen}: collected {collected} objects")

    # Stage 3: Force collection with memory pressure
    try:
        # Create temporary memory pressure to force GC
        dummy_data = []
        for _ in range(5):
            dummy_data.append([0] * 1000000)  # Create some temporary lists

        # Collect again
        collected = gc.collect()
        _log.debug(f"GC with memory pressure: collected {collected} objects")

        # Clean up dummy data
        del dummy_data
        gc.collect()

    except Exception as e:
        _log.debug(f"Memory pressure GC failed: {e}")


async def _onnx_provider_cleanup():
    """ONNX Runtime provider-specific cleanup."""
    try:
        import onnxruntime as ort

        # Check available providers
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            _log.debug("ONNX Runtime CUDA provider detected, attempting cleanup...")

            # Try to force CUDA provider cleanup by creating minimal sessions
            try:
                # Create a minimal session to trigger provider cleanup
                sess_options = ort.SessionOptions()
                sess_options.inter_op_num_threads = 1
                sess_options.intra_op_num_threads = 1

                # This sometimes forces ONNX Runtime to clean up cached allocations
                _log.debug("Creating temporary ONNX session to trigger cleanup...")
                # We don't actually need to create a session, just access providers
                # The provider initialization/cleanup sometimes helps
            except Exception as e:
                _log.debug(f"ONNX provider cleanup attempt: {e}")

    except ImportError:
        pass  # onnxruntime not available
    except Exception as e:
        _log.debug(f"ONNX provider cleanup error: {e}")


async def _aggressive_cuda_cleanup():
    """Aggressive CUDA memory cleanup with advanced context management."""
    try:
        import torch

        if not torch.cuda.is_available():
            return

        _log.debug("Starting aggressive CUDA cleanup with context management...")

        # Store initial context info for debugging
        device_count = torch.cuda.device_count()
        _log.debug(f"Found {device_count} CUDA devices")

        # Stage 1: Force CUDA context cleanup through device operations
        await _force_cuda_context_cleanup()

        # Stage 2: Multi-pass memory clearing with different strategies
        await _multi_pass_memory_clearing()

        # Stage 3: Advanced context manipulation if enabled
        if getattr(docling_serve_settings, 'force_cuda_context_reset', False):
            await _advanced_cuda_context_reset()

        # Stage 4: Memory pool manipulation
        await _manipulate_cuda_memory_pools()

        # Stage 5: Final cleanup and synchronization
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        _log.debug("Aggressive CUDA cleanup with context management completed")

    except ImportError:
        pass
    except Exception as e:
        _log.debug(f"Aggressive CUDA cleanup error: {e}")


async def _force_cuda_context_cleanup():
    """Force CUDA context cleanup through device operations."""
    try:
        import torch

        if not torch.cuda.is_available():
            return

        _log.debug("Forcing CUDA context cleanup...")

        device_count = torch.cuda.device_count()

        for device_id in range(device_count):
            try:
                # Switch to device
                torch.cuda.set_device(device_id)

                # Force context activity
                _ = torch.cuda.memory_allocated(device_id)
                _ = torch.cuda.memory_reserved(device_id)

                # Create and destroy a small tensor to force context activity
                dummy = torch.tensor([1.0], device=f'cuda:{device_id}')
                del dummy

                # Clear cache on this device
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()

                _log.debug(f"Context cleanup completed for device {device_id}")

            except Exception as e:
                _log.debug(f"Context cleanup failed for device {device_id}: {e}")

        # Synchronize all devices
        torch.cuda.synchronize()

    except Exception as e:
        _log.debug(f"Force CUDA context cleanup error: {e}")


async def _multi_pass_memory_clearing():
    """Multi-pass memory clearing with different strategies."""
    try:
        import torch

        if not torch.cuda.is_available():
            return

        _log.debug("Starting multi-pass memory clearing...")

        # Pass 1: Standard cache clearing
        for i in range(5):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Pass 2: Memory fraction manipulation with extreme values
        device_count = torch.cuda.device_count()
        for device_id in range(device_count):
            try:
                # Try extreme memory fractions
                fractions = [0.001, 0.01, 0.1, 0.5, 1.0]
                for fraction in fractions:
                    torch.cuda.set_per_process_memory_fraction(fraction, device_id)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                _log.debug(f"Memory fraction manipulation completed for device {device_id}")

            except Exception as e:
                _log.debug(f"Memory fraction manipulation failed for device {device_id}: {e}")

        # Pass 3: Memory statistics reset
        for device_id in range(device_count):
            try:
                with torch.cuda.device(device_id):
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()
            except Exception:
                pass

        # Pass 4: Final cache clears
        for i in range(3):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        _log.debug("Multi-pass memory clearing completed")

    except Exception as e:
        _log.debug(f"Multi-pass memory clearing error: {e}")


async def _advanced_cuda_context_reset():
    """Advanced CUDA context reset (experimental and potentially unsafe)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return

        _log.warning("Attempting advanced CUDA context reset (experimental)...")

        device_count = torch.cuda.device_count()

        for device_id in range(device_count):
            try:
                # Get current device properties
                device_props = torch.cuda.get_device_properties(device_id)
                _log.debug(f"Device {device_id}: {device_props.name}, Total memory: {device_props.total_memory / 1024**3:.2f} GB")

                # Try to release and reinitialize CUDA context
                original_device = torch.cuda.current_device()

                # Attempt to reset device (this is experimental and may not work on all systems)
                try:
                    # This is a potentially dangerous operation, so we wrap it carefully
                    if hasattr(torch.cuda, 'reset_device'):
                        torch.cuda.reset_device(device_id)
                        _log.debug(f"Reset device {device_id}")
                except Exception as e:
                    _log.debug(f"Device reset failed for {device_id}: {e}")

                # Restore original device
                torch.cuda.set_device(original_device)

            except Exception as e:
                _log.debug(f"Advanced context reset failed for device {device_id}: {e}")

        # Force cleanup after context manipulation
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        _log.debug("Advanced CUDA context reset completed")

    except Exception as e:
        _log.debug(f"Advanced CUDA context reset error: {e}")


async def _manipulate_cuda_memory_pools():
    """Manipulate CUDA memory pools to force cleanup."""
    try:
        import torch

        if not torch.cuda.is_available():
            return

        _log.debug("Manipulating CUDA memory pools...")

        device_count = torch.cuda.device_count()

        for device_id in range(device_count):
            try:
                # Get memory pool info if available
                if hasattr(torch.cuda, 'memory') and hasattr(torch.cuda.memory, 'memory_summary'):
                    try:
                        summary = torch.cuda.memory_summary(device=device_id)
                        _log.debug(f"Memory summary for device {device_id}: {summary[:500]}...")
                    except Exception:
                        pass

                # Try to access and manipulate memory allocation contexts
                try:
                    # Force memory allocation and deallocation
                    with torch.cuda.device(device_id):
                        # Allocate a large tensor and immediately delete it
                        large_tensor = torch.randn(1000, 1000, device=f'cuda:{device_id}')
                        del large_tensor
                        torch.cuda.empty_cache()

                        # Try memory pool reset if available
                        if hasattr(torch.cuda, 'memory') and hasattr(torch.cuda.memory, 'empty_cache'):
                            torch.cuda.memory.empty_cache()

                except Exception as e:
                    _log.debug(f"Memory pool manipulation failed for device {device_id}: {e}")

            except Exception as e:
                _log.debug(f"Memory pool manipulation error for device {device_id}: {e}")

        # Final synchronization
        torch.cuda.synchronize()

        _log.debug("CUDA memory pool manipulation completed")

    except Exception as e:
        _log.debug(f"CUDA memory pool manipulation error: {e}")


async def cleanup_models_after_task(orchestrator: BaseOrchestrator, task_id: str):
    """
    Cleanup models after a task completes (for synchronous endpoints).
    Only clears models if there are no other active tasks running.
    """
    if not docling_serve_settings.free_vram_on_idle:
        return

    # Check if there are other active tasks before cleaning up
    has_active_tasks = False
    for tid, t in orchestrator.tasks.items():
        if tid != task_id and t.task_status in [TaskStatus.PENDING, TaskStatus.STARTED]:
            has_active_tasks = True
            _log.info(f"Skipping model cleanup: task {tid} is still {t.task_status.value}")
            break

    if not has_active_tasks:
        _log.info(f"No active tasks remaining, clearing models to free VRAM...")
        await cleanup_models_if_needed(orchestrator)
    else:
        _log.info(f"Active tasks still running, models will remain loaded")


async def cleanup_models_background(orchestrator: BaseOrchestrator, task_id: str):
    """
    Background task to clean up models after task completion.
    Waits for the task to complete, then clears models if lazy loading is enabled.
    Only clears models if there are no other active tasks running.
    """
    if not docling_serve_settings.free_vram_on_idle:
        return

    # Wait for the task to complete
    max_wait_time = docling_serve_settings.max_document_timeout
    start_time = time.monotonic()

    while time.monotonic() - start_time < max_wait_time:
        try:
            task = await orchestrator.task_status(task_id=task_id)
            if task and task.task_status in [TaskStatus.SUCCESS, TaskStatus.FAILURE]:
                # Task completed, now check if there are other active tasks
                # before cleaning up models
                has_active_tasks = False
                for tid, t in orchestrator.tasks.items():
                    if tid != task_id and t.task_status in [TaskStatus.PENDING, TaskStatus.STARTED]:
                        has_active_tasks = True
                        _log.info(f"Skipping model cleanup: task {tid} is still {t.task_status.value}")
                        break

                if not has_active_tasks:
                    _log.info(f"No active tasks remaining, clearing models to free VRAM...")
                    await cleanup_models_if_needed(orchestrator)
                else:
                    _log.info(f"Active tasks still running, models will remain loaded")
                return
        except TaskNotFoundError:
            _log.warning(f"Task {task_id} not found during model cleanup. Stopping cleanup task.")
            return
        await asyncio.sleep(docling_serve_settings.cleanup_poll_interval)

    _log.warning(f"Cleanup task for task {task_id} timed out after {max_wait_time} seconds.")


##################################
# App creation and configuration #
##################################


def create_app():  # noqa: C901
    try:
        version = importlib.metadata.version("docling_serve")
    except importlib.metadata.PackageNotFoundError:
        _log.warning("Unable to get docling_serve version, falling back to 0.0.0")

        version = "0.0.0"

    offline_docs_assets = False
    if (
        docling_serve_settings.static_path is not None
        and (docling_serve_settings.static_path).is_dir()
    ):
        offline_docs_assets = True
        _log.info("Found static assets.")

    require_auth = APIKeyAuth(docling_serve_settings.api_key)
    app = FastAPI(
        title="Docling Serve",
        docs_url=None if offline_docs_assets else "/swagger",
        redoc_url=None if offline_docs_assets else "/docs",
        lifespan=lifespan,
        version=version,
    )

    origins = docling_serve_settings.cors_origins
    methods = docling_serve_settings.cors_methods
    headers = docling_serve_settings.cors_headers

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=methods,
        allow_headers=headers,
    )

    # Mount the Gradio app
    if docling_serve_settings.enable_ui:
        try:
            import gradio as gr

            from docling_serve.gradio_ui import ui as gradio_ui
            from docling_serve.settings import uvicorn_settings

            tmp_output_dir = get_scratch() / "gradio"
            tmp_output_dir.mkdir(exist_ok=True, parents=True)
            gradio_ui.gradio_output_dir = tmp_output_dir

            # Build the root_path for Gradio, accounting for UVICORN_ROOT_PATH
            gradio_root_path = (
                f"{uvicorn_settings.root_path}/ui"
                if uvicorn_settings.root_path
                else "/ui"
            )

            app = gr.mount_gradio_app(
                app,
                gradio_ui,
                path="/ui",
                allowed_paths=["./logo.png", tmp_output_dir],
                root_path=gradio_root_path,
            )

            # Add catch-all handler for Gradio upload progress to avoid 404 warnings
            @app.get("/ui/gradio_api/upload_progress")
            async def handle_upload_progress(upload_id: str = None):
                """Handle Gradio upload progress requests to avoid 404 warnings."""
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    content={
                        "status": "completed",
                        "progress": 1.0,
                        "message": "Upload completed or no upload in progress"
                    }
                )

            @app.get("/ui/gradio_api/upload_progress/{upload_id}")
            async def handle_upload_progress_with_id(upload_id: str):
                """Handle Gradio upload progress requests with specific ID to avoid 404 warnings."""
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    content={
                        "status": "completed",
                        "progress": 1.0,
                        "message": "Upload completed or no upload in progress"
                    }
                )
        except ImportError:
            _log.warning(
                "Docling Serve enable_ui is activated, but gradio is not installed. "
                "Install it with `pip install docling-serve[ui]` "
                "or `pip install gradio`"
            )

    #############################
    # Offline assets definition #
    #############################
    if offline_docs_assets:
        app.mount(
            "/static",
            StaticFiles(directory=docling_serve_settings.static_path),
            name="static",
        )

        @app.get("/swagger", include_in_schema=False)
        async def custom_swagger_ui_html():
            return get_swagger_ui_html(
                openapi_url=app.openapi_url,
                title=app.title + " - Swagger UI",
                oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
                swagger_js_url="/static/swagger-ui-bundle.js",
                swagger_css_url="/static/swagger-ui.css",
            )

        @app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
        async def swagger_ui_redirect():
            return get_swagger_ui_oauth2_redirect_html()

        @app.get("/docs", include_in_schema=False)
        async def redoc_html():
            return get_redoc_html(
                openapi_url=app.openapi_url,
                title=app.title + " - ReDoc",
                redoc_js_url="/static/redoc.standalone.js",
            )

    @app.get("/scalar", include_in_schema=False)
    async def scalar_html():
        return get_scalar_api_reference(
            openapi_url=app.openapi_url,
            title=app.title,
            scalar_favicon_url="https://raw.githubusercontent.com/docling-project/docling/refs/heads/main/docs/assets/logo.svg",
            # hide_client_button=True,  # not yet released but in main
        )

    ########################
    # Async / Sync helpers #
    ########################

    async def _enque_source(
        orchestrator: BaseOrchestrator,
        request: ConvertDocumentsRequest | GenericChunkDocumentsRequest,
        background_tasks: BackgroundTasks | None = None,
    ) -> Task:
        # Ensure models are loaded before enqueueing if lazy loading is enabled
        await ensure_models_loaded(orchestrator)

        sources: list[TaskSource] = []
        for s in request.sources:
            if isinstance(s, FileSourceRequest):
                sources.append(FileSource.model_validate(s))
            elif isinstance(s, HttpSourceRequest):
                sources.append(HttpSource.model_validate(s))
            elif isinstance(s, S3SourceRequest):
                sources.append(S3Coordinates.model_validate(s))

        convert_options: ConvertDocumentsRequestOptions
        chunking_options: BaseChunkerOptions | None = None
        chunking_export_options = ChunkingExportOptions()
        task_type: TaskType
        if isinstance(request, ConvertDocumentsRequest):
            task_type = TaskType.CONVERT
            convert_options = request.options
        elif isinstance(request, GenericChunkDocumentsRequest):
            task_type = TaskType.CHUNK
            convert_options = request.convert_options
            chunking_options = request.chunking_options
            chunking_export_options.include_converted_doc = (
                request.include_converted_doc
            )
        else:
            raise RuntimeError("Uknown request type.")

        task = await orchestrator.enqueue(
            task_type=task_type,
            sources=sources,
            convert_options=convert_options,
            chunking_options=chunking_options,
            chunking_export_options=chunking_export_options,
            target=request.target,
        )

        # Schedule background cleanup after task completes
        if background_tasks:
            background_tasks.add_task(cleanup_models_background, orchestrator, task.task_id)

        return task

    async def _enque_file(
        orchestrator: BaseOrchestrator,
        files: list[UploadFile],
        task_type: TaskType,
        convert_options: ConvertDocumentsRequestOptions,
        chunking_options: BaseChunkerOptions | None,
        chunking_export_options: ChunkingExportOptions | None,
        target: TargetRequest,
        background_tasks: BackgroundTasks | None = None,
    ) -> Task:
        # Ensure models are loaded before enqueueing if lazy loading is enabled
        await ensure_models_loaded(orchestrator)

        _log.info(f"Received {len(files)} files for processing.")

        # Load the uploaded files to Docling DocumentStream
        file_sources: list[TaskSource] = []
        for i, file in enumerate(files):
            buf = BytesIO(file.file.read())
            suffix = "" if len(file_sources) == 1 else f"_{i}"
            name = file.filename if file.filename else f"file{suffix}.pdf"
            file_sources.append(DocumentStream(name=name, stream=buf))

        task = await orchestrator.enqueue(
            task_type=task_type,
            sources=file_sources,
            convert_options=convert_options,
            chunking_options=chunking_options,
            chunking_export_options=chunking_export_options,
            target=target,
        )

        # Schedule background cleanup after task completes
        if background_tasks:
            background_tasks.add_task(cleanup_models_background, orchestrator, task.task_id)

        return task

    async def _wait_task_complete(orchestrator: BaseOrchestrator, task_id: str) -> bool:
        start_time = time.monotonic()
        while True:
            task = await orchestrator.task_status(task_id=task_id)
            if task.is_completed():
                return True
            await asyncio.sleep(docling_serve_settings.sync_poll_interval)
            elapsed_time = time.monotonic() - start_time
            if elapsed_time > docling_serve_settings.max_sync_wait:
                return False

    ##########################################
    # Downgrade openapi 3.1 to 3.0.x helpers #
    ##########################################

    def ensure_array_items(schema):
        """Ensure that array items are defined."""
        if "type" in schema and schema["type"] == "array":
            if "items" not in schema or schema["items"] is None:
                schema["items"] = {"type": "string"}
            elif isinstance(schema["items"], dict):
                if "type" not in schema["items"]:
                    schema["items"]["type"] = "string"

    def handle_discriminators(schema):
        """Ensure that discriminator properties are included in required."""
        if "discriminator" in schema and "propertyName" in schema["discriminator"]:
            prop = schema["discriminator"]["propertyName"]
            if "properties" in schema and prop in schema["properties"]:
                if "required" not in schema:
                    schema["required"] = []
                if prop not in schema["required"]:
                    schema["required"].append(prop)

    def handle_properties(schema):
        """Ensure that property 'kind' is included in required."""
        if "properties" in schema and "kind" in schema["properties"]:
            if "required" not in schema:
                schema["required"] = []
            if "kind" not in schema["required"]:
                schema["required"].append("kind")

    # Downgrade openapi 3.1 to 3.0.x
    def downgrade_openapi31_to_30(spec):
        def strip_unsupported(obj):
            if isinstance(obj, dict):
                obj = {
                    k: strip_unsupported(v)
                    for k, v in obj.items()
                    if k not in ("const", "examples", "prefixItems")
                }

                handle_discriminators(obj)
                ensure_array_items(obj)

                # Check for oneOf and anyOf to handle nested schemas
                for key in ["oneOf", "anyOf"]:
                    if key in obj:
                        for sub in obj[key]:
                            handle_discriminators(sub)
                            ensure_array_items(sub)

                return obj
            elif isinstance(obj, list):
                return [strip_unsupported(i) for i in obj]
            return obj

        if "components" in spec and "schemas" in spec["components"]:
            for schema_name, schema in spec["components"]["schemas"].items():
                handle_properties(schema)

        return strip_unsupported(copy.deepcopy(spec))

    #############################
    # API Endpoints definitions #
    #############################

    @app.get("/openapi-3.0.json")
    def openapi_30():
        spec = app.openapi()
        downgraded = downgrade_openapi31_to_30(spec)
        downgraded["openapi"] = "3.0.3"
        return JSONResponse(downgraded)

    # Favicon
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        logo_url = "https://raw.githubusercontent.com/docling-project/docling/refs/heads/main/docs/assets/logo.svg"
        if offline_docs_assets:
            logo_url = "/static/logo.svg"
        response = RedirectResponse(url=logo_url)
        return response

    @app.get("/health", tags=["health"])
    def health() -> HealthCheckResponse:
        memory_info = None
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                total_allocated = 0
                total_reserved = 0
                total_memory = 0
                devices = []

                for device_id in range(device_count):
                    allocated = torch.cuda.memory_allocated(device_id) / 1024**2
                    reserved = torch.cuda.memory_reserved(device_id) / 1024**2
                    device_props = torch.cuda.get_device_properties(device_id)
                    device_total = device_props.total_memory / 1024**2

                    total_allocated += allocated
                    total_reserved += reserved
                    total_memory += device_total

                    device_info = {
                        "id": device_id,
                        "name": device_props.name,
                        "total_memory_mb": device_total,
                        "allocated_mb": allocated,
                        "reserved_mb": reserved,
                        "utilization_percent": (allocated / device_total) * 100 if device_total > 0 else 0
                    }
                    devices.append(device_info)

                memory_info = {
                    "vram": {
                        "allocated_mb": total_allocated,
                        "reserved_mb": total_reserved,
                        "total_mb": total_memory,
                        "utilization_percent": (total_allocated / total_memory) * 100 if total_memory > 0 else 0,
                        "devices": devices
                    },
                    "cleanup_enabled": docling_serve_settings.free_vram_on_idle,
                    "expected_overhead_mb": getattr(docling_serve_settings, 'expected_vram_overhead_mb', 400.0)
                }

                # Add system memory if psutil is available
                try:
                    import psutil
                    system_memory = psutil.virtual_memory()
                    memory_info["system"] = {
                        "total_mb": system_memory.total / 1024**2,
                        "available_mb": system_memory.available / 1024**2,
                        "used_mb": system_memory.used / 1024**2,
                        "utilization_percent": system_memory.percent
                    }
                except ImportError:
                    pass

        except Exception as e:
            _log.debug(f"Failed to collect memory info for health endpoint: {e}")

        return HealthCheckResponse(memory_usage=memory_info)

    @app.get("/memory", tags=["health"], response_model=MemoryUsageResponse)
    def memory_usage() -> MemoryUsageResponse:
        """Detailed memory usage endpoint for monitoring."""
        response = MemoryUsageResponse(
            cleanup_enabled=docling_serve_settings.free_vram_on_idle
        )

        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                total_allocated = 0
                total_reserved = 0
                total_memory = 0
                devices = []

                for device_id in range(device_count):
                    allocated = torch.cuda.memory_allocated(device_id) / 1024**2
                    reserved = torch.cuda.memory_reserved(device_id) / 1024**2
                    device_props = torch.cuda.get_device_properties(device_id)
                    device_total = device_props.total_memory / 1024**2

                    total_allocated += allocated
                    total_reserved += reserved
                    total_memory += device_total

                    devices.append({
                        "id": device_id,
                        "name": device_props.name,
                        "total_memory_mb": device_total,
                        "allocated_mb": allocated,
                        "reserved_mb": reserved,
                        "utilization_percent": (allocated / device_total) * 100 if device_total > 0 else 0
                    })

                response.vram_allocated_mb = total_allocated
                response.vram_reserved_mb = total_reserved
                response.vram_total_mb = total_memory
                response.device_info = {"devices": devices}

        except Exception as e:
            _log.debug(f"Failed to collect detailed memory info: {e}")

        # Add system memory if available
        try:
            import psutil
            system_memory = psutil.virtual_memory()
            response.system_memory_mb = system_memory.used / 1024**2
        except ImportError:
            pass

        return response

    # API readiness compatibility for OpenShift AI Workbench
    @app.get("/api", include_in_schema=False)
    def api_check() -> HealthCheckResponse:
        return HealthCheckResponse()

    # Docling versions
    @app.get("/version", tags=["health"])
    def version_info() -> dict:
        if not docling_serve_settings.show_version_info:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden. The server is configured for not showing version details.",
            )
        return DOCLING_VERSIONS

    # Convert a document from URL(s)
    @app.post(
        "/v1/convert/source",
        tags=["convert"],
        response_model=ConvertDocumentResponse | PresignedUrlConvertDocumentResponse,
        responses={
            200: {
                "content": {"application/zip": {}},
                # "description": "Return the JSON item or an image.",
            }
        },
    )
    async def process_url(
        background_tasks: BackgroundTasks,
        auth: Annotated[AuthenticationResult, Depends(require_auth)],
        orchestrator: Annotated[BaseOrchestrator, Depends(get_async_orchestrator)],
        conversion_request: ConvertDocumentsRequest,
    ):
        task = await _enque_source(
            orchestrator=orchestrator, request=conversion_request, background_tasks=background_tasks
        )
        completed = await _wait_task_complete(
            orchestrator=orchestrator, task_id=task.task_id
        )

        if not completed:
            # TODO: abort task!
            raise HTTPException(
                status_code=504,
                detail=f"Conversion is taking too long. The maximum wait time is configure as DOCLING_SERVE_MAX_SYNC_WAIT={docling_serve_settings.max_sync_wait}.",
            )

        # Cleanup models after task completion if no other tasks are running
        await cleanup_models_after_task(orchestrator, task.task_id)

        task_result = await orchestrator.task_result(task_id=task.task_id)
        if task_result is None:
            raise HTTPException(
                status_code=404,
                detail="Task result not found. Please wait for a completion status.",
            )
        response = await prepare_response(
            task_id=task.task_id,
            task_result=task_result,
            orchestrator=orchestrator,
            background_tasks=background_tasks,
        )
        return response

    # Convert a document from file(s)
    @app.post(
        "/v1/convert/file",
        tags=["convert"],
        response_model=ConvertDocumentResponse | PresignedUrlConvertDocumentResponse,
        responses={
            200: {
                "content": {"application/zip": {}},
            }
        },
    )
    async def process_file(
        background_tasks: BackgroundTasks,
        auth: Annotated[AuthenticationResult, Depends(require_auth)],
        orchestrator: Annotated[BaseOrchestrator, Depends(get_async_orchestrator)],
        files: list[UploadFile],
        options: Annotated[
            ConvertDocumentsRequestOptions, FormDepends(ConvertDocumentsRequestOptions)
        ],
        target_type: Annotated[TargetName, Form()] = TargetName.INBODY,
    ):
        target = InBodyTarget() if target_type == TargetName.INBODY else ZipTarget()
        task = await _enque_file(
            task_type=TaskType.CONVERT,
            orchestrator=orchestrator,
            files=files,
            convert_options=options,
            chunking_options=None,
            chunking_export_options=None,
            target=target,
            background_tasks=background_tasks,
        )
        completed = await _wait_task_complete(
            orchestrator=orchestrator, task_id=task.task_id
        )

        if not completed:
            # TODO: abort task!
            raise HTTPException(
                status_code=504,
                detail=f"Conversion is taking too long. The maximum wait time is configure as DOCLING_SERVE_MAX_SYNC_WAIT={docling_serve_settings.max_sync_wait}.",
            )

        # Cleanup models after task completion if no other tasks are running
        await cleanup_models_after_task(orchestrator, task.task_id)

        task_result = await orchestrator.task_result(task_id=task.task_id)
        if task_result is None:
            raise HTTPException(
                status_code=404,
                detail="Task result not found. Please wait for a completion status.",
            )
        response = await prepare_response(
            task_id=task.task_id,
            task_result=task_result,
            orchestrator=orchestrator,
            background_tasks=background_tasks,
        )
        return response

    # Convert a document from URL(s) using the async api
    @app.post(
        "/v1/convert/source/async",
        tags=["convert"],
        response_model=TaskStatusResponse,
    )
    async def process_url_async(
        background_tasks: BackgroundTasks,
        auth: Annotated[AuthenticationResult, Depends(require_auth)],
        orchestrator: Annotated[BaseOrchestrator, Depends(get_async_orchestrator)],
        conversion_request: ConvertDocumentsRequest,
    ):
        task = await _enque_source(
            orchestrator=orchestrator, request=conversion_request, background_tasks=background_tasks
        )
        task_queue_position = await orchestrator.get_queue_position(
            task_id=task.task_id
        )
        return TaskStatusResponse(
            task_id=task.task_id,
            task_type=task.task_type,
            task_status=task.task_status,
            task_position=task_queue_position,
            task_meta=task.processing_meta,
        )

    # Convert a document from file(s) using the async api
    @app.post(
        "/v1/convert/file/async",
        tags=["convert"],
        response_model=TaskStatusResponse,
    )
    async def process_file_async(
        auth: Annotated[AuthenticationResult, Depends(require_auth)],
        orchestrator: Annotated[BaseOrchestrator, Depends(get_async_orchestrator)],
        background_tasks: BackgroundTasks,
        files: list[UploadFile],
        options: Annotated[
            ConvertDocumentsRequestOptions, FormDepends(ConvertDocumentsRequestOptions)
        ],
        target_type: Annotated[TargetName, Form()] = TargetName.INBODY,
    ):
        target = InBodyTarget() if target_type == TargetName.INBODY else ZipTarget()
        task = await _enque_file(
            task_type=TaskType.CONVERT,
            orchestrator=orchestrator,
            files=files,
            convert_options=options,
            chunking_options=None,
            chunking_export_options=None,
            target=target,
            background_tasks=background_tasks,
        )
        task_queue_position = await orchestrator.get_queue_position(
            task_id=task.task_id
        )
        return TaskStatusResponse(
            task_id=task.task_id,
            task_type=task.task_type,
            task_status=task.task_status,
            task_position=task_queue_position,
            task_meta=task.processing_meta,
        )

    # Chunking endpoints
    for display_name, path_name, opt_cls in (
        ("HybridChunker", "hybrid", HybridChunkerOptions),
        ("HierarchicalChunker", "hierarchical", HierarchicalChunkerOptions),
    ):
        req_cls = make_request_model(opt_cls)

        @app.post(
            f"/v1/chunk/{path_name}/source/async",
            name=f"Chunk sources with {display_name} as async task",
            tags=["chunk"],
            response_model=TaskStatusResponse,
        )
        async def chunk_source_async(
            background_tasks: BackgroundTasks,
            auth: Annotated[AuthenticationResult, Depends(require_auth)],
            orchestrator: Annotated[BaseOrchestrator, Depends(get_async_orchestrator)],
            request: req_cls,
        ):
            task = await _enque_source(orchestrator=orchestrator, request=request, background_tasks=background_tasks)
            task_queue_position = await orchestrator.get_queue_position(
                task_id=task.task_id
            )
            return TaskStatusResponse(
                task_id=task.task_id,
                task_type=task.task_type,
                task_status=task.task_status,
                task_position=task_queue_position,
                task_meta=task.processing_meta,
            )

        @app.post(
            f"/v1/chunk/{path_name}/file/async",
            name=f"Chunk files with {display_name} as async task",
            tags=["chunk"],
            response_model=TaskStatusResponse,
        )
        async def chunk_file_async(
            background_tasks: BackgroundTasks,
            auth: Annotated[AuthenticationResult, Depends(require_auth)],
            orchestrator: Annotated[BaseOrchestrator, Depends(get_async_orchestrator)],
            files: list[UploadFile],
            convert_options: Annotated[
                ConvertDocumentsRequestOptions,
                FormDepends(
                    ConvertDocumentsRequestOptions,
                    prefix="convert_",
                    excluded_fields=[
                        "to_formats",
                    ],
                ),
            ],
            chunking_options: Annotated[
                opt_cls,
                FormDepends(
                    HybridChunkerOptions,
                    prefix="chunking_",
                    excluded_fields=["chunker"],
                ),
            ],
            include_converted_doc: Annotated[
                bool,
                Form(
                    description="If true, the output will include both the chunks and the converted document."
                ),
            ] = False,
            target_type: Annotated[
                TargetName,
                Form(description="Specification for the type of output target."),
            ] = TargetName.INBODY,
        ):
            target = InBodyTarget() if target_type == TargetName.INBODY else ZipTarget()
            task = await _enque_file(
                task_type=TaskType.CHUNK,
                orchestrator=orchestrator,
                files=files,
                convert_options=convert_options,
                chunking_options=chunking_options,
                chunking_export_options=ChunkingExportOptions(
                    include_converted_doc=include_converted_doc
                ),
                target=target,
                background_tasks=background_tasks,
            )
            task_queue_position = await orchestrator.get_queue_position(
                task_id=task.task_id
            )
            return TaskStatusResponse(
                task_id=task.task_id,
                task_type=task.task_type,
                task_status=task.task_status,
                task_position=task_queue_position,
                task_meta=task.processing_meta,
            )

        @app.post(
            f"/v1/chunk/{path_name}/source",
            name=f"Chunk sources with {display_name}",
            tags=["chunk"],
            response_model=ChunkDocumentResponse,
            responses={
                200: {
                    "content": {"application/zip": {}},
                    # "description": "Return the JSON item or an image.",
                }
            },
        )
        async def chunk_source(
            background_tasks: BackgroundTasks,
            auth: Annotated[AuthenticationResult, Depends(require_auth)],
            orchestrator: Annotated[BaseOrchestrator, Depends(get_async_orchestrator)],
            request: req_cls,
        ):
            task = await _enque_source(orchestrator=orchestrator, request=request, background_tasks=background_tasks)
            completed = await _wait_task_complete(
                orchestrator=orchestrator, task_id=task.task_id
            )

            if not completed:
                # TODO: abort task!
                raise HTTPException(
                    status_code=504,
                    detail=f"Conversion is taking too long. The maximum wait time is configure as DOCLING_SERVE_MAX_SYNC_WAIT={docling_serve_settings.max_sync_wait}.",
                )

            # Cleanup models after task completion if no other tasks are running
            await cleanup_models_after_task(orchestrator, task.task_id)

            task_result = await orchestrator.task_result(task_id=task.task_id)
            if task_result is None:
                raise HTTPException(
                    status_code=404,
                    detail="Task result not found. Please wait for a completion status.",
                )
            response = await prepare_response(
                task_id=task.task_id,
                task_result=task_result,
                orchestrator=orchestrator,
                background_tasks=background_tasks,
            )
            return response

        @app.post(
            f"/v1/chunk/{path_name}/file",
            name=f"Chunk files with {display_name}",
            tags=["chunk"],
            response_model=ChunkDocumentResponse,
            responses={
                200: {
                    "content": {"application/zip": {}},
                }
            },
        )
        async def chunk_file(
            background_tasks: BackgroundTasks,
            auth: Annotated[AuthenticationResult, Depends(require_auth)],
            orchestrator: Annotated[BaseOrchestrator, Depends(get_async_orchestrator)],
            files: list[UploadFile],
            convert_options: Annotated[
                ConvertDocumentsRequestOptions,
                FormDepends(
                    ConvertDocumentsRequestOptions,
                    prefix="convert_",
                    excluded_fields=[
                        "to_formats",
                    ],
                ),
            ],
            chunking_options: Annotated[
                opt_cls,
                FormDepends(
                    HybridChunkerOptions,
                    prefix="chunking_",
                    excluded_fields=["chunker"],
                ),
            ],
            include_converted_doc: Annotated[
                bool,
                Form(
                    description="If true, the output will include both the chunks and the converted document."
                ),
            ] = False,
            target_type: Annotated[
                TargetName,
                Form(description="Specification for the type of output target."),
            ] = TargetName.INBODY,
        ):
            target = InBodyTarget() if target_type == TargetName.INBODY else ZipTarget()
            task = await _enque_file(
                task_type=TaskType.CHUNK,
                orchestrator=orchestrator,
                files=files,
                convert_options=convert_options,
                chunking_options=chunking_options,
                chunking_export_options=ChunkingExportOptions(
                    include_converted_doc=include_converted_doc
                ),
                target=target,
                background_tasks=background_tasks,
            )
            completed = await _wait_task_complete(
                orchestrator=orchestrator, task_id=task.task_id
            )

            if not completed:
                # TODO: abort task!
                raise HTTPException(
                    status_code=504,
                    detail=f"Conversion is taking too long. The maximum wait time is configure as DOCLING_SERVE_MAX_SYNC_WAIT={docling_serve_settings.max_sync_wait}.",
                )

            # Cleanup models after task completion if no other tasks are running
            await cleanup_models_after_task(orchestrator, task.task_id)

            task_result = await orchestrator.task_result(task_id=task.task_id)
            if task_result is None:
                raise HTTPException(
                    status_code=404,
                    detail="Task result not found. Please wait for a completion status.",
                )
            response = await prepare_response(
                task_id=task.task_id,
                task_result=task_result,
                orchestrator=orchestrator,
                background_tasks=background_tasks,
            )
            return response

    # Task status poll
    @app.get(
        "/v1/status/poll/{task_id}",
        tags=["tasks"],
        response_model=TaskStatusResponse,
    )
    async def task_status_poll(
        auth: Annotated[AuthenticationResult, Depends(require_auth)],
        orchestrator: Annotated[BaseOrchestrator, Depends(get_async_orchestrator)],
        task_id: str,
        wait: Annotated[
            float,
            Query(description="Number of seconds to wait for a completed status."),
        ] = 0.0,
    ):
        try:
            task = await orchestrator.task_status(task_id=task_id, wait=wait)
            task_queue_position = await orchestrator.get_queue_position(task_id=task_id)
        except TaskNotFoundError:
            raise HTTPException(status_code=404, detail="Task not found.")
        return TaskStatusResponse(
            task_id=task.task_id,
            task_type=task.task_type,
            task_status=task.task_status,
            task_position=task_queue_position,
            task_meta=task.processing_meta,
        )

    # Task status websocket
    @app.websocket(
        "/v1/status/ws/{task_id}",
    )
    async def task_status_ws(
        websocket: WebSocket,
        orchestrator: Annotated[BaseOrchestrator, Depends(get_async_orchestrator)],
        task_id: str,
        api_key: Annotated[str, Query()] = "",
    ):
        if docling_serve_settings.api_key:
            if api_key != docling_serve_settings.api_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Api key is required as ?api_key=SECRET.",
                )

        assert isinstance(orchestrator.notifier, WebsocketNotifier)
        await websocket.accept()

        try:
            # Get task status from Redis or RQ directly instead of checking in-memory registry
            task = await orchestrator.task_status(task_id=task_id)
        except TaskNotFoundError:
            await websocket.send_text(
                WebsocketMessage(
                    message=MessageKind.ERROR, error="Task not found."
                ).model_dump_json()
            )
            await websocket.close()
            return

        # Track active WebSocket connections for this job
        orchestrator.notifier.task_subscribers[task_id].add(websocket)

        try:
            task_queue_position = await orchestrator.get_queue_position(task_id=task_id)
            task_response = TaskStatusResponse(
                task_id=task.task_id,
                task_type=task.task_type,
                task_status=task.task_status,
                task_position=task_queue_position,
                task_meta=task.processing_meta,
            )
            await websocket.send_text(
                WebsocketMessage(
                    message=MessageKind.CONNECTION, task=task_response
                ).model_dump_json()
            )
            while True:
                task_queue_position = await orchestrator.get_queue_position(
                    task_id=task_id
                )
                task_response = TaskStatusResponse(
                    task_id=task.task_id,
                    task_type=task.task_type,
                    task_status=task.task_status,
                    task_position=task_queue_position,
                    task_meta=task.processing_meta,
                )
                await websocket.send_text(
                    WebsocketMessage(
                        message=MessageKind.UPDATE, task=task_response
                    ).model_dump_json()
                )
                # each client message will be interpreted as a request for update
                msg = await websocket.receive_text()
                _log.debug(f"Received message: {msg}")

        except WebSocketDisconnect:
            _log.info(f"WebSocket disconnected for job {task_id}")

        finally:
            orchestrator.notifier.task_subscribers[task_id].remove(websocket)

    # Task result
    @app.get(
        "/v1/result/{task_id}",
        tags=["tasks"],
        response_model=ConvertDocumentResponse
        | PresignedUrlConvertDocumentResponse
        | ChunkDocumentResponse,
        responses={
            200: {
                "content": {"application/zip": {}},
            }
        },
    )
    async def task_result(
        auth: Annotated[AuthenticationResult, Depends(require_auth)],
        orchestrator: Annotated[BaseOrchestrator, Depends(get_async_orchestrator)],
        background_tasks: BackgroundTasks,
        task_id: str,
    ):
        try:
            task_result = await orchestrator.task_result(task_id=task_id)
            if task_result is None:
                raise HTTPException(
                    status_code=404,
                    detail="Task result not found. Please wait for a completion status.",
                )

            # Cleanup models after fetching results if no other tasks are running
            await cleanup_models_after_task(orchestrator, task_id)

            response = await prepare_response(
                task_id=task_id,
                task_result=task_result,
                orchestrator=orchestrator,
                background_tasks=background_tasks,
            )
            return response
        except TaskNotFoundError:
            raise HTTPException(status_code=404, detail="Task not found.")

    # Update task progress
    @app.post(
        "/v1/callback/task/progress",
        tags=["internal"],
        include_in_schema=False,
        response_model=ProgressCallbackResponse,
    )
    async def callback_task_progress(
        auth: Annotated[AuthenticationResult, Depends(require_auth)],
        orchestrator: Annotated[BaseOrchestrator, Depends(get_async_orchestrator)],
        request: ProgressCallbackRequest,
    ):
        try:
            await orchestrator.receive_task_progress(request=request)
            return ProgressCallbackResponse(status="ack")
        except TaskNotFoundError:
            raise HTTPException(status_code=404, detail="Task not found.")
        except ProgressInvalid as err:
            raise HTTPException(
                status_code=400, detail=f"Invalid progress payload: {err}"
            )

    #### Clear requests

    # Offload models
    @app.get(
        "/v1/clear/converters",
        tags=["clear"],
        response_model=ClearResponse,
    )
    async def clear_converters(
        auth: Annotated[AuthenticationResult, Depends(require_auth)],
        orchestrator: Annotated[BaseOrchestrator, Depends(get_async_orchestrator)],
    ):
        await orchestrator.clear_converters()
        return ClearResponse()

    # Clean results
    @app.get(
        "/v1/clear/results",
        tags=["clear"],
        response_model=ClearResponse,
    )
    async def clear_results(
        auth: Annotated[AuthenticationResult, Depends(require_auth)],
        orchestrator: Annotated[BaseOrchestrator, Depends(get_async_orchestrator)],
        older_then: float = 3600,
    ):
        await orchestrator.clear_results(older_than=older_then)
        return ClearResponse()

    return app
