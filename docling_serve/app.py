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
    _log.warning(f"DEBUG: Received base_url: {base_url}, type: {type(base_url)}")

    # Handle case where base_url comes in as bytes from environment variable
    if isinstance(base_url, bytes):
        _log.warning(f"DEBUG: Decoding bytes to string")
        base_url = base_url.decode("utf-8")
        _log.warning(f"DEBUG: After decoding: {base_url}")

    # Parse URL to extract only scheme and netloc, stripping any path components
    # This handles cases where users include /v1 or other paths in the base URL
    parsed = httpx.URL(base_url)
    clean_base_url = f"{parsed.scheme}://{parsed.netloc}"
    url = f"{clean_base_url}/unload"
    _log.warning(f"DEBUG: Final URL: {url}")

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
        # First, unload external models to free VRAM if configured
        await unload_external_models()
        # Then load Docling models
        _log.info("Loading models for processing...")
        await orchestrator.warm_up_caches()


async def cleanup_models_if_needed(orchestrator: BaseOrchestrator):
    """Clear models after processing if lazy loading is enabled to free VRAM."""
    if docling_serve_settings.free_vram_on_idle:
        _log.info("Clearing models to free VRAM...")

        # Log VRAM usage before cleanup
        try:
            import torch
            if torch.cuda.is_available():
                mem_before = torch.cuda.memory_allocated() / 1024**2
                _log.info(f"VRAM allocated before cleanup: {mem_before:.2f} MB")
        except Exception:
            pass

        # Step 1: Move models to CPU before deletion (critical for VRAM release)
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
                                    elif hasattr(converter, 'to'):
                                        converter.to('cpu')
                                except Exception:
                                    pass
                        except Exception:
                            pass
        except Exception as e:
            _log.warning(f"Failed to move models to CPU: {e}")

        # Step 2: Try to explicitly close ONNX Runtime sessions
        try:
            if hasattr(orchestrator, 'cm') and hasattr(orchestrator.cm, '_get_converter_from_hash'):
                cache = orchestrator.cm._get_converter_from_hash
                if hasattr(cache, 'cache'):
                    _log.info("Attempting to close ONNX Runtime sessions...")
                    for key, converter in list(cache.cache.items()):
                        try:
                            # Try to access and delete ONNX sessions within converters
                            if hasattr(converter, '__dict__'):
                                for attr_name in list(converter.__dict__.keys()):
                                    attr = getattr(converter, attr_name, None)
                                    # Look for ONNX InferenceSession objects
                                    if attr is not None and 'onnxruntime' in str(type(attr)):
                                        _log.info(f"Found ONNX session in {attr_name}, deleting...")
                                        delattr(converter, attr_name)
                        except Exception as e:
                            _log.debug(f"Error closing ONNX session: {e}")
        except Exception as e:
            _log.warning(f"Failed to close ONNX sessions: {e}")

        # Step 3: Clear converters and explicitly delete cache entries
        await orchestrator.clear_converters()

        # Also try to manually clear the cache dictionary
        try:
            if hasattr(orchestrator, 'cm') and hasattr(orchestrator.cm, '_get_converter_from_hash'):
                cache = orchestrator.cm._get_converter_from_hash
                if hasattr(cache, 'cache'):
                    _log.info(f"Manually clearing {len(cache.cache)} cached converters...")
                    # Delete each converter explicitly
                    for key in list(cache.cache.keys()):
                        try:
                            del cache.cache[key]
                        except:
                            pass
                    cache.cache.clear()
        except Exception as e:
            _log.debug(f"Manual cache clear: {e}")

        # Step 4: Force aggressive garbage collection
        import gc
        _log.info("Running aggressive garbage collection...")
        for _ in range(5):
            gc.collect()
        gc.collect(2)  # Full collection including generation 2

        # Step 5: Try to force ONNX Runtime CUDA cleanup
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in providers:
                _log.info("ONNX Runtime CUDA provider detected, forcing cleanup...")
                # Trick: Creating and immediately destroying a session sometimes triggers
                # ONNX Runtime to release cached CUDA allocations
                # This is a workaround since ONNX Runtime has no official cleanup API
                import gc
                gc.collect()
                gc.collect()
        except Exception as e:
            _log.debug(f"ONNX Runtime cleanup attempt: {e}")

        # Step 6: Explicitly free CUDA memory
        try:
            import torch
            if torch.cuda.is_available():
                _log.info("Clearing CUDA cache...")
                # Clear CUDA cache multiple times
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()

                # Try to set memory fraction to minimal (releases reserved memory)
                try:
                    for device_id in range(torch.cuda.device_count()):
                        torch.cuda.set_per_process_memory_fraction(0.0, device_id)
                        torch.cuda.empty_cache()
                        torch.cuda.set_per_process_memory_fraction(1.0, device_id)
                except Exception:
                    pass

                # Final empty cache calls
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                mem_after = torch.cuda.memory_allocated() / 1024**2
                mem_reserved = torch.cuda.memory_reserved() / 1024**2
                _log.info(f"VRAM allocated after cleanup: {mem_after:.2f} MB")
                _log.info(f"VRAM reserved after cleanup: {mem_reserved:.2f} MB")
                _log.info("CUDA cache cleared and synchronized")

                # If still significant memory, log warning
                if mem_after > 100:
                    _log.warning(f"VRAM still has {mem_after:.2f} MB allocated - this may be CUDA context overhead")
        except Exception as e:
            _log.warning(f"Failed to clear CUDA cache: {e}")


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

    return app
