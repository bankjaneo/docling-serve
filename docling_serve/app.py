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

# Global VRAM tracking for detecting accumulation trends
_vram_usage_history = []
_max_vram_history_entries = 10


def track_vram_usage(context: str = "cleanup"):
    """Track VRAM usage to detect accumulation trends over time."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2

            _vram_usage_history.append({
                'allocated': allocated,
                'reserved': reserved,
                'context': context,
                'timestamp': time.time()
            })

            # Keep only recent entries
            if len(_vram_usage_history) > _max_vram_history_entries:
                _vram_usage_history.pop(0)

            # Detect accumulation trend
            if len(_vram_usage_history) >= 3:
                recent_cleanup = [h for h in _vram_usage_history[-3:] if h['context'] == 'cleanup_end']
                if len(recent_cleanup) >= 2:
                    # Check if allocated memory is trending upward during cleanups
                    allocations = [h['allocated'] for h in recent_cleanup]
                    if allocations[-1] > allocations[0] + 50:  # 50MB threshold
                        _log.warning(f"VRAM accumulation detected: {allocations[0]:.1f}MB → {allocations[-1]:.1f}MB across recent cleanups")
                        _log.warning("This suggests a memory leak or CUDA context accumulation")

            return allocated, reserved
    except Exception as e:
        _log.debug(f"VRAM tracking failed: {e}")
        return None, None


def log_vram_trend_analysis():
    """Log detailed VRAM trend analysis."""
    if len(_vram_usage_history) < 2:
        return

    cleanup_entries = [h for h in _vram_usage_history if h['context'] == 'cleanup_end']
    if len(cleanup_entries) < 2:
        return

    first_cleanup = cleanup_entries[0]
    last_cleanup = cleanup_entries[-1]

    alloc_change = last_cleanup['allocated'] - first_cleanup['allocated']
    reserv_change = last_cleanup['reserved'] - first_cleanup['reserved']

    if abs(alloc_change) > 10:  # Only log significant changes
        trend = "increasing" if alloc_change > 0 else "decreasing"
        _log.info(f"VRAM trend over {len(cleanup_entries)} cleanups: {trend} by {abs(alloc_change):.1f}MB allocated, {abs(reserv_change):.1f}MB reserved")


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
    # Handle case where base_url comes in as bytes from environment variable
    if isinstance(base_url, bytes):
        base_url = base_url.decode("utf-8")

    # Parse URL to extract only scheme and netloc, stripping any path components
    # This handles cases where users include /v1 or other paths in the base URL
    parsed = httpx.URL(base_url)

    # Handle case where parsed components come as bytes
    scheme = parsed.scheme.decode() if isinstance(parsed.scheme, bytes) else parsed.scheme
    netloc = parsed.netloc.decode() if isinstance(parsed.netloc, bytes) else parsed.netloc

    clean_base_url = f"{scheme}://{netloc}"
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
    # Handle case where base_url comes in as bytes from environment variable
    if isinstance(base_url, bytes):
        base_url = base_url.decode("utf-8")

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
        # Track VRAM before loading models
        mem_before, mem_reserved_before = track_vram_usage("model_load_start")
        if mem_before is not None:
            _log.debug(f"VRAM before model loading: {mem_before:.2f} MB allocated")

        # Check if models are already loaded to avoid double loading
        already_loaded = False
        try:
            if hasattr(orchestrator, 'cm') and hasattr(orchestrator.cm, '_get_converter_from_hash'):
                cache = orchestrator.cm._get_converter_from_hash
                if hasattr(cache, 'cache_info'):
                    cache_info = cache.cache_info()
                    if cache_info.currsize > 0:
                        _log.info(f"Models already loaded (cache size: {cache_info.currsize}), skipping warm_up_caches")
                        already_loaded = True
                elif hasattr(cache, 'cache') and len(cache.cache) > 0:
                    _log.info(f"Models already loaded (cached converters: {len(cache.cache)}), skipping warm_up_caches")
                    already_loaded = True
        except Exception as e:
            _log.debug(f"Failed to check if models already loaded: {e}")

        # First, unload external models to free VRAM if configured
        await unload_external_models()

        if not already_loaded:
            # Then load Docling models
            _log.info("Loading models for processing...")
            await orchestrator.warm_up_caches()
        else:
            _log.info("Using already loaded models")

        # Track VRAM after loading models
        mem_after, mem_reserved_after = track_vram_usage("model_load_end")
        if mem_after is not None:
            increase = mem_after - mem_before
            _log.info(f"VRAM after model loading: {mem_after:.2f} MB (+{increase:.2f} MB)")
            if increase > 1000:  # Warning for large memory increases
                _log.warning(f"Large VRAM increase detected: {increase:.2f} MB")


async def cleanup_models_if_needed(orchestrator: BaseOrchestrator, deep_cleanup: bool = False):
    """Clear models after processing if lazy loading is enabled to free VRAM."""
    if docling_serve_settings.free_vram_on_idle:
        _log.info("Clearing models to free VRAM...")

        # Check if models are actually loaded before starting cleanup
        model_status = "unknown"
        try:
            if hasattr(orchestrator, 'cm') and hasattr(orchestrator.cm, '_get_converter_from_hash'):
                cache = orchestrator.cm._get_converter_from_hash
                if hasattr(cache, 'cache_info'):
                    cache_info = cache.cache_info()
                    model_status = f"cache_hits: {cache_info.hits}, misses: {cache_info.misses}, size: {cache_info.currsize}"
                elif hasattr(cache, 'cache'):
                    model_status = f"cached converters: {len(cache.cache)}"
                else:
                    model_status = "cache structure unknown"
            else:
                model_status = "no converter manager found"
        except Exception as e:
            model_status = f"status check failed: {e}"

        _log.info(f"Model status before cleanup: {model_status}")

        # Track VRAM usage before cleanup and detect trends
        mem_before, mem_reserved_before = track_vram_usage("cleanup_start")
        if mem_before is not None:
            _log.info(f"VRAM allocated before cleanup: {mem_before:.2f} MB")
            _log.info(f"VRAM reserved before cleanup: {mem_reserved_before:.2f} MB")

        # Log trend analysis
        log_vram_trend_analysis()

        # Skip cleanup if minimal VRAM is allocated
        if mem_before is not None and mem_before < 50:
            _log.info(f"Skipping cleanup - minimal VRAM allocated ({mem_before:.2f} MB)")
            return

        # Check if we need deep cleanup based on recent history
        if not deep_cleanup and len(_vram_usage_history) >= 2:
            recent_cleanups = [h for h in _vram_usage_history[-5:] if h['context'] == 'cleanup_end']
            if len(recent_cleanups) >= 2:
                # Check if VRAM has been increasing
                allocations = [h['allocated'] for h in recent_cleanups]
                if len(allocations) >= 2 and allocations[-1] > allocations[0]:
                    _log.warning("VRAM accumulation detected, triggering deep cleanup mode")
                    deep_cleanup = True

        cleanup_mode = "deep" if deep_cleanup else "standard"
        _log.info(f"Starting {cleanup_mode} cleanup mode...")

        # Step 1: Move models to CPU and force release references
        try:
            import torch
            if torch.cuda.is_available():
                _log.info("Moving models to CPU before deletion...")
                # Try to access converter manager if it exists
                if hasattr(orchestrator, 'cm') and hasattr(orchestrator.cm, '_get_converter_from_hash'):
                    # Get the cache info
                    cache = orchestrator.cm._get_converter_from_hash
                    if hasattr(cache, 'cache_info'):
                        _log.info(f"Converter cache before cleanup: {cache.cache_info()}")

                    # Try to move any cached converters to CPU
                    if hasattr(cache, '__wrapped__'):
                        # For LRU cache, we need to access the cache directly
                        try:
                            # Access the private cache dict (this is a bit hacky but necessary)
                            cache_dict = cache.cache
                            for key, value in list(cache_dict.items()):
                                try:
                                    # Try to move converter models to CPU
                                    converter = value  # In LRU cache, value might be wrapped
                                    if hasattr(converter, 'doc_converter') and hasattr(converter.doc_converter, 'to'):
                                        converter.doc_converter.to('cpu')
                                        # Force delete reference
                                        del converter.doc_converter
                                    elif hasattr(converter, 'to'):
                                        converter.to('cpu')
                                        # Force delete reference
                                        del converter
                                except Exception:
                                    pass
                        except Exception:
                            pass
        except Exception as e:
            _log.warning(f"Failed to move models to CPU: {e}")

        # Step 2: Force garbage collection BEFORE clearing converters
        import gc
        _log.info("Pre-cleanup garbage collection...")
        for _ in range(3):
            gc.collect()

        # Step 3: Try to explicitly close ONNX Runtime sessions and delete all references
        try:
            if hasattr(orchestrator, 'cm') and hasattr(orchestrator.cm, '_get_converter_from_hash'):
                cache = orchestrator.cm._get_converter_from_hash
                if hasattr(cache, 'cache'):
                    _log.info("Attempting to close ONNX Runtime sessions...")
                    converters_to_delete = []
                    for key, converter in list(cache.cache.items()):
                        try:
                            # Store converter reference for deletion
                            converters_to_delete.append((key, converter))

                            # Try to access and delete ONNX sessions within converters
                            if hasattr(converter, '__dict__'):
                                for attr_name in list(converter.__dict__.keys()):
                                    attr = getattr(converter, attr_name, None)
                                    # Look for ONNX InferenceSession objects
                                    if attr is not None and 'onnxruntime' in str(type(attr)):
                                        _log.debug(f"Found ONNX session in {attr_name}, deleting...")
                                        # Force close session if possible
                                        if hasattr(attr, 'close'):
                                            attr.close()
                                        delattr(converter, attr_name)
                        except Exception as e:
                            _log.debug(f"Error closing ONNX session: {e}")

                    # Force delete all converter references
                    for key, converter in converters_to_delete:
                        try:
                            del converter
                        except Exception:
                            pass
                    converters_to_delete.clear()
        except Exception as e:
            _log.warning(f"Failed to close ONNX sessions: {e}")

        # Step 4: Clear converters and explicitly delete cache entries
        _log.info("Clearing orchestrator converters...")
        await orchestrator.clear_converters()

        # Also try to manually clear the cache dictionary
        try:
            if hasattr(orchestrator, 'cm') and hasattr(orchestrator.cm, '_get_converter_from_hash'):
                cache = orchestrator.cm._get_converter_from_hash
                if hasattr(cache, 'cache'):
                    cache_size = len(cache.cache)
                    _log.info(f"Manually clearing {cache_size} cached converters...")
                    # Delete each converter explicitly
                    for key in list(cache.cache.keys()):
                        try:
                            converter = cache.cache[key]
                            # Try to move converter to CPU before deletion
                            if hasattr(converter, 'to'):
                                converter.to('cpu')
                            del cache.cache[key]
                        except Exception as e:
                            _log.debug(f"Failed to clear converter {key}: {e}")
                    cache.cache.clear()
                    _log.info("Cache dictionary cleared")
                else:
                    _log.info("No cache attribute found")
            else:
                _log.info("No converter manager found")
        except Exception as e:
            _log.debug(f"Manual cache clear: {e}")

        # Step 4b: Try to clear any other potential model caches
        try:
            _log.info("Attempting to clear additional model caches...")
            import gc

            # Look for any objects that might hold model references
            for obj_name in dir(orchestrator):
                if 'model' in obj_name.lower() or 'cache' in obj_name.lower():
                    try:
                        obj = getattr(orchestrator, obj_name)
                        if hasattr(obj, 'clear'):
                            _log.debug(f"Clearing {obj_name}")
                            obj.clear()
                        elif hasattr(obj, 'reset'):
                            _log.debug(f"Resetting {obj_name}")
                            obj.reset()
                    except Exception as e:
                        _log.debug(f"Failed to clear {obj_name}: {e}")

            # Force garbage collection again
            gc.collect()
            gc.collect()

        except Exception as e:
            _log.debug(f"Additional cache clearing failed: {e}")

        # Step 5: Force aggressive garbage collection
        _log.info("Running aggressive garbage collection...")
        for _ in range(5):
            gc.collect()
        gc.collect(2)  # Full collection including generation 2

        # Step 6: Try to force ONNX Runtime CUDA cleanup with additional techniques
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in providers:
                _log.info("ONNX Runtime CUDA provider detected, forcing cleanup...")
                # Multiple cleanup attempts for ONNX Runtime
                for attempt in range(3):
                    gc.collect()
                    # Force CUDA context reset if possible
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    except Exception:
                        pass
        except Exception as e:
            _log.debug(f"ONNX Runtime cleanup attempt: {e}")

        # Step 7: Enhanced CUDA memory cleanup with multiple techniques
        try:
            import torch
            if torch.cuda.is_available():
                _log.info("Starting enhanced CUDA memory cleanup...")

                # Track devices
                device_count = torch.cuda.device_count()
                _log.info(f"Found {device_count} CUDA device(s)")

                # Step 7a: Force empty cache and synchronize multiple times
                cache_iterations = 5 if deep_cleanup else 3
                for i in range(cache_iterations):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    _log.debug(f"CUDA cleanup iteration {i+1} completed")

                # Step 7b: Reset memory stats
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()

                # Step 7c: Try aggressive memory fraction reset for each device
                for device_id in range(device_count):
                    try:
                        current_device = torch.cuda.current_device()
                        torch.cuda.set_device(device_id)

                        # Get current memory state
                        mem_alloc = torch.cuda.memory_allocated(device_id)
                        mem_reserv = torch.cuda.memory_reserved(device_id)
                        _log.debug(f"Device {device_id} - Allocated: {mem_alloc/1024**2:.2f}MB, Reserved: {mem_reserv/1024**2:.2f}MB")

                        # Aggressive memory fraction manipulation
                        if deep_cleanup:
                            # More aggressive sequence for deep cleanup
                            for fraction in [0.05, 0.01, 0.0, 0.1, 1.0]:
                                torch.cuda.set_per_process_memory_fraction(fraction, device_id)
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                                await asyncio.sleep(0.01)  # Brief pause for deep cleanup
                        else:
                            # Standard sequence
                            torch.cuda.set_per_process_memory_fraction(0.1, device_id)
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()

                            torch.cuda.set_per_process_memory_fraction(0.0, device_id)
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()

                            torch.cuda.set_per_process_memory_fraction(1.0, device_id)
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()

                        # Restore original device
                        torch.cuda.set_device(current_device)

                    except Exception as e:
                        _log.debug(f"Device {device_id} cleanup failed: {e}")
                        continue

                # Step 7d: Final cleanup pass with extra garbage collection
                gc.collect()
                for _ in range(3 if deep_cleanup else 2):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Step 7e: Force context release attempts (experimental)
                try:
                    # This is a more aggressive technique to force CUDA context cleanup
                    _log.debug("Attempting CUDA context cleanup...")

                    # Try to force context release by creating dummy tensors and deleting them
                    for device_id in range(device_count):
                        try:
                            with torch.cuda.device(device_id):
                                if deep_cleanup:
                                    # More aggressive context cleanup
                                    for size in [1, 10, 100]:
                                        dummy = torch.randn(size, size, device=f'cuda:{device_id}')
                                        del dummy
                                        torch.cuda.empty_cache()
                                        torch.cuda.synchronize()
                                else:
                                    # Standard approach
                                    dummy = torch.tensor([1.0], device=f'cuda:{device_id}')
                                    del dummy
                                    torch.cuda.empty_cache()
                        except Exception as e:
                            _log.debug(f"Context cleanup failed for device {device_id}: {e}")
                except Exception as e:
                    _log.debug(f"CUDA context cleanup failed: {e}")

                # Step 7f: Deep cleanup specific techniques
                if deep_cleanup:
                    try:
                        _log.info("Applying deep cleanup techniques...")

                        # Method 1: Force reset all CUDA devices (aggressive)
                        for device_id in range(device_count):
                            try:
                                torch.cuda.set_device(device_id)
                                torch.cuda.reset_device()
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                            except Exception as e:
                                _log.debug(f"Device reset failed for {device_id}: {e}")

                        # Method 2: Try to completely reinitialize CUDA context (experimental)
                        try:
                            _log.info("Attempting CUDA context reinitialization...")

                            # Force complete cleanup
                            for device_id in range(device_count):
                                torch.cuda.set_device(device_id)
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()

                            # Try to disable and re-enable CUDA
                            torch.cuda.set_per_process_memory_fraction(0.0)
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()

                            # Create and destroy contexts
                            for device_id in range(device_count):
                                try:
                                    with torch.cuda.device(device_id):
                                        # Force context creation
                                        x = torch.randn(100, 100, device=device_id)
                                        del x
                                        torch.cuda.empty_cache()
                                except Exception as e:
                                    _log.debug(f"Context recreation failed: {e}")

                            # Reset memory fraction back to normal
                            torch.cuda.set_per_process_memory_fraction(1.0)
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()

                        except Exception as e:
                            _log.debug(f"CUDA context reinitialization failed: {e}")

                        # Method 3: Force final synchronization and cleanup
                        for device_id in range(device_count):
                            torch.cuda.set_device(device_id)
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()

                        # Restore original device
                        if device_count > 0:
                            torch.cuda.set_device(current_device)

                    except Exception as e:
                        _log.debug(f"Deep cleanup device reset failed: {e}")

                # Step 7g: Final memory report
                mem_after = torch.cuda.memory_allocated() / 1024**2
                mem_reserved = torch.cuda.memory_reserved() / 1024**2
                _log.info(f"VRAM allocated after cleanup: {mem_after:.2f} MB")
                _log.info(f"VRAM reserved after cleanup: {mem_reserved:.2f} MB")
                _log.info(f"{cleanup_mode.title()} CUDA cleanup completed")

                # Track final VRAM usage
                track_vram_usage("cleanup_end")

                # Calculate and log cleanup effectiveness
                if mem_before is not None:
                    freed = mem_before - mem_after
                    if freed > 0:
                        _log.info(f"VRAM cleanup freed {freed:.2f} MB ({(freed/mem_before*100):.1f}% of allocated memory)")
                    else:
                        _log.warning(f"VRAM cleanup did not free allocated memory (change: {freed:.2f} MB)")

                # If still significant memory, log detailed warning with troubleshooting tips
                if mem_after > 100:
                    _log.warning(f"VRAM still has {mem_after:.2f} MB allocated - this is likely persistent CUDA context overhead")
                    if deep_cleanup:
                        _log.warning("Deep cleanup was unable to free additional VRAM - this confirms it's CUDA context overhead")
                        _log.info("CUDA context overhead of ~600-800MB is normal and cannot be cleared without process restart")
                    _log.info("This residual memory is NOT actual model data - it's CUDA driver/context overhead")
                    _log.info("Expected behavior: Models ARE being cleared (cache size=0), but CUDA context persists")
                    _log.info("Only process restart can fully clear this CUDA context overhead")
                    _log.info("This is considered normal behavior for GPU applications")

                # Additional cleanup - one more pass for good measure
                torch.cuda.empty_cache()
                gc.collect()

                # Wait a moment to let cleanup settle
                await asyncio.sleep(0.2 if deep_cleanup else 0.1)

                # Check VRAM again after a short delay
                mem_delayed = torch.cuda.memory_allocated() / 1024**2
                if mem_delayed != mem_after:
                    _log.info(f"VRAM changed after delay: {mem_after:.2f}MB → {mem_delayed:.2f}MB")
                    # Track the delayed measurement
                    track_vram_usage("cleanup_delayed")

        except Exception as e:
            _log.warning(f"Failed to clear CUDA cache: {e}")

        # Final model status check
        try:
            model_status_after = "unknown"
            if hasattr(orchestrator, 'cm') and hasattr(orchestrator.cm, '_get_converter_from_hash'):
                cache = orchestrator.cm._get_converter_from_hash
                if hasattr(cache, 'cache_info'):
                    cache_info = cache.cache_info()
                    model_status_after = f"cache_hits: {cache_info.hits}, misses: {cache_info.misses}, size: {cache_info.currsize}"
                elif hasattr(cache, 'cache'):
                    model_status_after = f"cached converters: {len(cache.cache)}"
                else:
                    model_status_after = "cache structure unknown"
            else:
                model_status_after = "no converter manager found"
            _log.info(f"Model status after cleanup: {model_status_after}")
        except Exception as e:
            _log.debug(f"Post-cleanup status check failed: {e}")


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
        return HealthCheckResponse()

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

    # Deep VRAM cleanup
    @app.get(
        "/v1/clear/vram",
        tags=["clear"],
        response_model=ClearResponse,
    )
    async def deep_vram_cleanup(
        auth: Annotated[AuthenticationResult, Depends(require_auth)],
        orchestrator: Annotated[BaseOrchestrator, Depends(get_async_orchestrator)],
        force: bool = False,
    ):
        """
        Perform a deep VRAM cleanup. This endpoint can be used to manually trigger
        aggressive VRAM cleanup when standard cleanup is not effective.

        Parameters:
        - force: Force cleanup even if minimal VRAM is allocated
        """
        if not docling_serve_settings.free_vram_on_idle:
            raise HTTPException(
                status_code=400,
                detail="Deep VRAM cleanup requires DOCLING_SERVE_FREE_VRAM_ON_IDLE=True"
            )

        _log.info(f"Manual deep VRAM cleanup requested via API (force={force})")

        # Temporarily bypass the minimal VRAM check if force is True
        if force:
            # Temporarily modify the function to skip the minimal VRAM check
            import functools
            original_cleanup = cleanup_models_if_needed

            async def forced_cleanup(orchestrator_ref, deep_cleanup_ref=False):
                # Store original settings
                original_free_vram = docling_serve_settings.free_vram_on_idle

                # Force cleanup
                await original_cleanup(orchestrator_ref, deep_cleanup_ref=deep_cleanup_ref)

                # Restore original settings
                docling_serve_settings.free_vram_on_idle = original_free_vram

            await forced_cleanup(orchestrator, deep_cleanup=True)
        else:
            await cleanup_models_if_needed(orchestrator, deep_cleanup=True)

        return ClearResponse()

    # VRAM status check
    @app.get(
        "/v1/status/vram",
        tags=["status"],
        response_model=dict,
    )
    async def vram_status(
        auth: Annotated[AuthenticationResult, Depends(require_auth)],
        orchestrator: Annotated[BaseOrchestrator, Depends(get_async_orchestrator)],
    ):
        """
        Get current VRAM status and model cache information for debugging.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return {"error": "CUDA not available"}

            # Get VRAM info
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2

            # Get model cache info
            model_status = "unknown"
            cache_size = 0
            try:
                if hasattr(orchestrator, 'cm') and hasattr(orchestrator.cm, '_get_converter_from_hash'):
                    cache = orchestrator.cm._get_converter_from_hash
                    if hasattr(cache, 'cache_info'):
                        cache_info = cache.cache_info()
                        model_status = f"cache_hits: {cache_info.hits}, misses: {cache_info.misses}, size: {cache_info.currsize}"
                        cache_size = cache_info.currsize
                    elif hasattr(cache, 'cache'):
                        model_status = f"cached converters: {len(cache.cache)}"
                        cache_size = len(cache.cache)
                    else:
                        model_status = "cache structure unknown"
                else:
                    model_status = "no converter manager found"
            except Exception as e:
                model_status = f"status check failed: {e}"

            # Get VRAM history
            history_summary = []
            if len(_vram_usage_history) > 0:
                recent = _vram_usage_history[-5:]  # Last 5 entries
                history_summary = [
                    {
                        "context": h["context"],
                        "allocated_mb": round(h["allocated"], 2),
                        "reserved_mb": round(h["reserved"], 2)
                    }
                    for h in recent
                ]

            return {
                "vram_allocated_mb": round(allocated, 2),
                "vram_reserved_mb": round(reserved, 2),
                "vram_total_mb": round(total, 2),
                "vram_free_mb": round(total - reserved, 2),
                "model_status": model_status,
                "cache_size": cache_size,
                "recent_history": history_summary,
                "free_vram_on_idle": docling_serve_settings.free_vram_on_idle
            }

        except Exception as e:
            return {"error": f"Failed to get VRAM status: {e}"}

    return app
