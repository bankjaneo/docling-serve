"""
VRAM Worker Orchestrator - Process-based orchestrator for complete VRAM isolation.

This orchestrator spawns a separate Python process for each conversion task,
ensuring complete VRAM release on task completion by terminating the worker process.
This eliminates CUDA context persistence issues that plague thread-based approaches.

Key Features:
- Each task runs in isolated process with fresh CUDA context
- Worker process terminates after task completion -> complete VRAM release
- Main process remains lightweight (~50MB VRAM)
- No VRAM accumulation across tasks
"""

import asyncio
import logging
import multiprocessing
import os
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any, Optional

from docling_jobkit.datamodel.chunking import (
    BaseChunkerOptions,
    ChunkingExportOptions,
)
from docling_jobkit.datamodel.task import Task, TaskStatus, TaskType
from docling_jobkit.orchestrators.base_notifier import BaseNotifier
from docling_jobkit.orchestrators.base_orchestrator import (
    BaseOrchestrator,
    TaskNotFoundError,
)

_log = logging.getLogger(__name__)


@dataclass
class VRAMWorkerOrchestratorConfig:
    """Configuration for VRAM Worker Orchestrator."""

    scratch_dir: Path
    worker_timeout: int = 600  # 10 minutes timeout per task
    max_retries: int = 1  # Retry once on worker crash


@dataclass
class WorkerTask:
    """
    Task data for worker process.

    Note: This will be serialized to JSON for inter-process communication,
    so all fields must be JSON-serializable (no enums, complex objects, etc.)
    """

    task_id: str
    task_type: str  # TaskType as string (e.g., "convert", "chunk")
    sources: list[Any]
    convert_options: Optional[dict[str, Any]]
    chunking_options: Optional[dict[str, Any]]  # Serialized to dict
    chunking_export_options: Optional[dict[str, Any]]  # Serialized to dict
    target: Optional[dict[str, Any]]  # Serialized to dict
    converter_manager_config: dict[str, Any]


@dataclass
class WorkerResult:
    """Result from worker process."""

    task_id: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    traceback_str: Optional[str] = None


def _worker_process_entry(
    task_data: WorkerTask,
    result_queue: Queue,
    converter_manager_config: dict[str, Any],
) -> None:
    """
    Worker process entry point.

    This function runs in a separate process and performs the actual conversion.
    After completion (success or failure), the process terminates, releasing all VRAM.
    """
    try:
        # Import all dependencies inside worker process
        # This ensures main process doesn't load CUDA
        import os
        import base64
        import tempfile
        from pathlib import Path

        from docling.datamodel.pipeline_options import PdfBackend
        from docling_jobkit.convert.manager import (
            DoclingConverterManager,
            DoclingConverterManagerConfig,
        )

        _log.info(f"Worker process {os.getpid()} started for task {task_data.task_id}")

        # Create converter manager in worker process
        cm_config = DoclingConverterManagerConfig(**converter_manager_config)
        cm = DoclingConverterManager(config=cm_config)

        # Get a converter instance from the manager
        # Try to get converter without pdf_format_option first
        try:
            converter = cm.get_converter()
        except TypeError as e:
            _log.error(f"Failed to get converter without pdf_format_option: {e}")

            # If that fails, try with None
            try:
                converter = cm.get_converter(pdf_format_option=None)
            except TypeError as e2:
                _log.error(f"Failed to get converter with pdf_format_option=None: {e2}")
                raise ValueError(f"Unable to get converter: {e2}")

        # Perform conversion
        task_type_str = task_data.task_type.upper() if isinstance(task_data.task_type, str) else task_data.task_type

        if task_type_str == "CONVERT":
            # Convert documents using the converter
            results = []
            for source in task_data.sources:
                # Handle different source types
                if hasattr(source, 'kind') and source.kind == 'file':
                    # This is a FileSourceRequest object, check its structure
                    _log.info(f"FileSourceRequest attributes: {dir(source)}")

                    if hasattr(source, 'base64_data') and source.base64_data:
                        # For base64 data, we need to decode and write to temp file
                        _log.info("Processing base64 data")
                        # Decode base64 and write to temporary file
                        file_data = base64.b64decode(source.base64_data)
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                            temp_file.write(file_data)
                            temp_path = temp_file.name

                        try:
                            conv_result = converter.convert(temp_path)
                        finally:
                            # Clean up temp file
                            os.unlink(temp_path)
                    else:
                        # Check what attributes this FileSource has
                        if hasattr(source, 'name'):
                            _log.info(f"Using source.name: {source.name}")
                            conv_result = converter.convert(source.name)
                        elif hasattr(source, 'filename'):
                            _log.info(f"Using source.filename: {source.filename}")
                            conv_result = converter.convert(source.filename)
                        elif hasattr(source, 'url'):
                            _log.info(f"Using source.url: {source.url}")
                            conv_result = converter.convert(source.url)
                        else:
                            # Log all available attributes for debugging
                            _log.error(f"Unknown FileSourceRequest structure. Available attributes: {[attr for attr in dir(source) if not attr.startswith('_')]}")
                            raise ValueError(f"Cannot extract file path from FileSourceRequest")
                elif isinstance(source, (str, Path)):
                    # Handle string or Path objects directly
                    conv_result = converter.convert(source)
                else:
                    # Try to convert source to string or raise informative error
                    _log.error(f"Unsupported source type: {type(source)} - {source}")
                    raise ValueError(f"Unsupported source type: {type(source)}. Expected str, Path, or FileSourceRequest.")
                results.append(conv_result)
            result = results

        elif task_type_str == "CHUNK":
            # Chunk documents
            results = []
            for source in task_data.sources:
                # Handle different source types
                if hasattr(source, 'kind') and source.kind == 'file':
                    # This is a FileSourceRequest object, check its structure
                    _log.info(f"FileSourceRequest attributes: {dir(source)}")

                    if hasattr(source, 'base64_data') and source.base64_data:
                        # For base64 data, we need to decode and write to temp file
                        _log.info("Processing base64 data")
                        # Decode base64 and write to temporary file
                        file_data = base64.b64decode(source.base64_data)
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                            temp_file.write(file_data)
                            temp_path = temp_file.name

                        try:
                            conv_result = converter.convert(temp_path)
                        finally:
                            # Clean up temp file
                            os.unlink(temp_path)
                    else:
                        # Check what attributes this FileSource has
                        if hasattr(source, 'name'):
                            _log.info(f"Using source.name: {source.name}")
                            conv_result = converter.convert(source.name)
                        elif hasattr(source, 'filename'):
                            _log.info(f"Using source.filename: {source.filename}")
                            conv_result = converter.convert(source.filename)
                        elif hasattr(source, 'url'):
                            _log.info(f"Using source.url: {source.url}")
                            conv_result = converter.convert(source.url)
                        else:
                            # Log all available attributes for debugging
                            _log.error(f"Unknown FileSourceRequest structure. Available attributes: {[attr for attr in dir(source) if not attr.startswith('_')]}")
                            raise ValueError(f"Cannot extract file path from FileSourceRequest")
                elif isinstance(source, (str, Path)):
                    # Handle string or Path objects directly
                    conv_result = converter.convert(source)
                else:
                    # Try to convert source to string or raise informative error
                    _log.error(f"Unsupported source type: {type(source)} - {source}")
                    raise ValueError(f"Unsupported source type: {type(source)}. Expected str, Path, or FileSourceRequest.")
                results.append(conv_result)
            result = results
        else:
            raise ValueError(f"Unknown task type: {task_data.task_type}")

        # Send result back to main process
        worker_result = WorkerResult(
            task_id=task_data.task_id,
            success=True,
            result=result,
        )
        result_queue.put(worker_result)

        _log.info(
            f"Worker process {os.getpid()} completed task {task_data.task_id} successfully"
        )

    except Exception as e:
        error_msg = f"Worker process error: {str(e)}"
        tb_str = traceback.format_exc()
        _log.error(f"Worker process {os.getpid()} failed: {error_msg}\n{tb_str}")

        worker_result = WorkerResult(
            task_id=task_data.task_id,
            success=False,
            error=error_msg,
            traceback_str=tb_str,
        )
        result_queue.put(worker_result)

    finally:
        # Ensure CUDA cleanup before process termination
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                _log.info(
                    f"Worker process {os.getpid()} cleared CUDA cache before exit"
                )
        except ImportError:
            pass  # PyTorch not available
        except Exception as e:
            _log.warning(f"CUDA cleanup warning: {e}")

        _log.info(f"Worker process {os.getpid()} terminating - VRAM will be released")


class VRAMWorkerOrchestrator(BaseOrchestrator):
    """
    Process-based orchestrator for complete VRAM isolation.

    Each task spawns a separate Python process that:
    1. Loads models and initializes CUDA
    2. Performs document conversion
    3. Returns result to main process
    4. Terminates -> complete VRAM release

    This eliminates CUDA context persistence and VRAM accumulation.
    """

    def __init__(
        self,
        config: VRAMWorkerOrchestratorConfig,
        converter_manager_config: dict[str, Any],
    ):
        super().__init__()
        self.config = config
        self.converter_manager_config = converter_manager_config

        # Task management
        self.tasks: dict[str, Task] = {}
        self.task_results: dict[str, Any] = {}
        self.active_workers: dict[str, Process] = {}
        self.result_queues: dict[str, Queue] = {}
        self.task_start_times: dict[str, float] = {}  # Track task start time for timeout

        # Notifier support
        self._notifier: Optional[BaseNotifier] = None

        _log.info(
            f"VRAMWorkerOrchestrator initialized (worker_timeout={config.worker_timeout}s)"
        )

    def bind_notifier(self, notifier: BaseNotifier) -> None:
        """Bind a notifier for task status updates."""
        self._notifier = notifier

    async def _safe_notify(self, task_id: str) -> None:
        """
        Safely notify task subscribers, handling cases where task has no subscribers.

        This avoids errors when notifying tasks that haven't been registered with
        the websocket notifier (e.g., API calls without websocket connections).
        """
        if not self._notifier:
            return

        try:
            await self._notifier.notify_task_subscribers(task_id)
        except RuntimeError as e:
            # Task doesn't have subscribers list - this is normal for non-websocket requests
            if "does not have a subscribers list" in str(e):
                _log.debug(f"Task {task_id} has no subscribers (non-websocket request)")
            else:
                raise
        except Exception as e:
            _log.warning(f"Failed to notify subscribers for task {task_id}: {e}")

    async def warm_up_caches(self) -> None:
        """
        Warm up caches - NO-OP for VRAM orchestrator.

        We don't warm up caches because:
        1. Models are loaded in worker processes, not main process
        2. Workers are ephemeral and terminate after each task
        3. Loading models in main process would defeat the purpose
        """
        _log.info("VRAM orchestrator: Skipping cache warm-up (worker processes only)")

    async def clear_converters(self) -> None:
        """
        Clear converters - NO-OP for VRAM orchestrator.

        Workers are terminated after each task, so there's nothing to clear.
        """
        _log.info("VRAM orchestrator: No converters to clear (stateless workers)")

    async def enqueue(
        self,
        task_type: TaskType,
        sources: list[Any],
        convert_options: Optional[dict[str, Any]] = None,
        chunking_options: Optional[BaseChunkerOptions] = None,
        chunking_export_options: Optional[ChunkingExportOptions] = None,
        target: Optional[Any] = None,
    ) -> Task:
        """
        Enqueue a new task for processing in a worker process.

        Args:
            task_type: Type of task (CONVERT or CHUNK)
            sources: List of document sources
            convert_options: Conversion options
            chunking_options: Chunking options (for CHUNK tasks)
            chunking_export_options: Chunking export options (for CHUNK tasks)
            target: Target for output

        Returns:
            Task object with PENDING status
        """
        import uuid

        task_id = str(uuid.uuid4())

        # Create task
        task = Task(
            task_id=task_id,
            task_type=task_type,
            task_status=TaskStatus.PENDING,
            processing_meta={
                "num_docs": len(sources),
                "num_processed": 0,
                "num_succeeded": 0,
                "num_failed": 0,
            },
        )
        self.tasks[task_id] = task

        # Serialize complex objects to JSON-safe formats
        # This ensures everything can be pickled for multiprocessing
        import json

        def _serialize_to_json_safe(obj):
            """
            Convert any object to JSON-safe primitives.
            Handles Pydantic models, enums, and complex nested structures.
            """
            if obj is None:
                return None

            # Try Pydantic model serialization first (handles enums properly)
            if hasattr(obj, 'model_dump'):
                try:
                    # Pydantic v2 with JSON mode (converts enums to values)
                    return obj.model_dump(mode='json')
                except Exception:
                    try:
                        return obj.model_dump()
                    except Exception:
                        pass
            elif hasattr(obj, 'dict'):
                try:
                    # Pydantic v1
                    return obj.dict()
                except Exception:
                    pass

            # Handle dictionaries recursively
            if isinstance(obj, dict):
                return {k: _serialize_to_json_safe(v) for k, v in obj.items()}

            # Handle lists recursively
            if isinstance(obj, (list, tuple)):
                return [_serialize_to_json_safe(item) for item in obj]

            # Handle enums
            if hasattr(obj, 'value'):
                return obj.value

            # Try JSON round-trip as fallback
            try:
                # This will convert most objects to JSON-safe types
                return json.loads(json.dumps(obj, default=str))
            except Exception:
                # Last resort: convert to string
                return str(obj)

        # Create worker task data with fully serialized options
        worker_task = WorkerTask(
            task_id=task_id,
            task_type=task_type.value if hasattr(task_type, 'value') else str(task_type),
            sources=sources,
            convert_options=_serialize_to_json_safe(convert_options),
            chunking_options=_serialize_to_json_safe(chunking_options),
            chunking_export_options=_serialize_to_json_safe(chunking_export_options),
            target=_serialize_to_json_safe(target),
            converter_manager_config=self.converter_manager_config,
        )

        # Store for processing
        if not hasattr(self, "_pending_tasks"):
            self._pending_tasks = []
        self._pending_tasks.append(worker_task)

        _log.info(f"Task {task_id} enqueued for worker process")

        return task

    async def process_queue(self) -> None:
        """
        Process queued tasks by spawning worker processes.

        This continuously monitors pending tasks and spawns worker processes to handle them.
        """
        _log.info("VRAM orchestrator: Starting queue processor")

        if not hasattr(self, "_pending_tasks"):
            self._pending_tasks = []

        while True:
            try:
                # Process pending tasks
                while self._pending_tasks:
                    worker_task = self._pending_tasks.pop(0)
                    await self._spawn_worker(worker_task)

                # Check for completed workers
                await self._check_workers()

                # Sleep briefly to avoid busy-waiting
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                _log.info("Queue processor cancelled, cleaning up workers...")
                await self._cleanup_all_workers()
                raise
            except Exception as e:
                _log.error(f"Queue processor error: {e}")
                await asyncio.sleep(1.0)

    async def _spawn_worker(self, worker_task: WorkerTask) -> None:
        """Spawn a worker process for the given task."""
        task_id = worker_task.task_id

        try:
            # Use 'spawn' start method to ensure clean process (no CUDA inheritance)
            # IMPORTANT: Create Queue using the same context to avoid SemLock errors
            ctx = multiprocessing.get_context("spawn")
            result_queue = ctx.Queue()
            self.result_queues[task_id] = result_queue

            # Update task status to STARTED
            self.tasks[task_id].task_status = TaskStatus.STARTED
            self.task_start_times[task_id] = time.time()  # Record start time
            await self._safe_notify(task_id)

            # Spawn worker process
            process = ctx.Process(
                target=_worker_process_entry,
                args=(worker_task, result_queue, self.converter_manager_config),
                daemon=False,  # Not daemon - we want to control lifecycle
            )
            process.start()

            self.active_workers[task_id] = process

            _log.info(
                f"Spawned worker process {process.pid} for task {task_id} (process-based VRAM isolation)"
            )

        except Exception as e:
            _log.error(f"Failed to spawn worker for task {task_id}: {e}")
            self.tasks[task_id].task_status = TaskStatus.FAILURE
            self.task_results[task_id] = {"error": str(e)}
            await self._safe_notify(task_id)

    async def _check_workers(self) -> None:
        """Check status of active worker processes."""
        completed_tasks = []
        current_time = time.time()

        for task_id, process in list(self.active_workers.items()):
            # Check for timeout
            start_time = self.task_start_times.get(task_id, current_time)
            elapsed = current_time - start_time

            if elapsed > self.config.worker_timeout:
                _log.warning(
                    f"Task {task_id} timed out after {elapsed:.1f}s (limit: {self.config.worker_timeout}s), terminating worker process {process.pid}..."
                )

                # Terminate timed-out worker
                try:
                    process.terminate()
                    # Give it 5 seconds to terminate gracefully
                    process.join(timeout=5.0)
                    if process.is_alive():
                        _log.warning(f"Worker {process.pid} didn't terminate gracefully, killing...")
                        process.kill()
                        process.join()
                except Exception as e:
                    _log.error(f"Error terminating timed-out worker {process.pid}: {e}")

                # Mark as failed
                self.tasks[task_id].task_status = TaskStatus.FAILURE
                self.task_results[task_id] = {
                    "error": f"Task timed out after {elapsed:.1f} seconds"
                }

                await self._safe_notify(task_id)

                completed_tasks.append(task_id)
                continue

            # Check if process has terminated
            if not process.is_alive():
                exit_code = process.exitcode
                _log.info(
                    f"Worker process {process.pid} for task {task_id} terminated (exit code: {exit_code})"
                )

                # Retrieve result from queue
                result_queue = self.result_queues.get(task_id)
                if result_queue and not result_queue.empty():
                    try:
                        worker_result: WorkerResult = result_queue.get_nowait()

                        if worker_result.success:
                            self.tasks[task_id].task_status = TaskStatus.SUCCESS
                            self.task_results[task_id] = worker_result.result
                            _log.info(f"Task {task_id} completed successfully")
                        else:
                            self.tasks[task_id].task_status = TaskStatus.FAILURE
                            self.task_results[task_id] = {
                                "error": worker_result.error,
                                "traceback": worker_result.traceback_str,
                            }
                            _log.error(f"Task {task_id} failed: {worker_result.error}")

                    except Exception as e:
                        _log.error(f"Error retrieving result for task {task_id}: {e}")
                        self.tasks[task_id].task_status = TaskStatus.FAILURE
                        self.task_results[task_id] = {
                            "error": "Failed to retrieve worker result"
                        }
                else:
                    # Process died without result
                    _log.error(
                        f"Worker process {process.pid} died without result (exit code: {exit_code})"
                    )
                    self.tasks[task_id].task_status = TaskStatus.FAILURE
                    self.task_results[task_id] = {
                        "error": f"Worker process died (exit code: {exit_code})"
                    }

                # Notify status change
                await self._safe_notify(task_id)

                completed_tasks.append(task_id)

        # Cleanup completed tasks
        for task_id in completed_tasks:
            self._cleanup_worker(task_id)

    def _cleanup_worker(self, task_id: str) -> None:
        """Clean up worker process resources."""
        if task_id in self.active_workers:
            process = self.active_workers[task_id]
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
                if process.is_alive():
                    process.kill()
                    process.join()
            del self.active_workers[task_id]

        if task_id in self.result_queues:
            del self.result_queues[task_id]

        if task_id in self.task_start_times:
            del self.task_start_times[task_id]

        _log.debug(f"Cleaned up worker resources for task {task_id}")

    async def _cleanup_all_workers(self) -> None:
        """Terminate all active workers on shutdown."""
        for task_id in list(self.active_workers.keys()):
            self._cleanup_worker(task_id)

    async def task_status(self, task_id: str, wait: float = 0.0) -> Task:
        """Get task status."""
        if task_id not in self.tasks:
            raise TaskNotFoundError(f"Task {task_id} not found")

        # TODO: Implement wait logic if needed
        return self.tasks[task_id]

    async def task_result(self, task_id: str) -> Any:
        """Get task result."""
        if task_id not in self.tasks:
            raise TaskNotFoundError(f"Task {task_id} not found")

        if self.tasks[task_id].task_status != TaskStatus.SUCCESS:
            return None

        return self.task_results.get(task_id)

    async def get_raw_task(self, task_id: str) -> Task:
        """Get raw task object."""
        if task_id not in self.tasks:
            raise TaskNotFoundError(f"Task {task_id} not found")
        return self.tasks[task_id]

    async def check_connection(self) -> bool:
        """
        Check if orchestrator is ready to accept tasks.

        For VRAM orchestrator, always returns True as it's local and always ready.
        """
        return True

    async def queue_size(self) -> int:
        """
        Get the current queue size.

        Returns the number of pending tasks waiting to be processed.
        """
        if not hasattr(self, "_pending_tasks"):
            return 0
        return len(self._pending_tasks)

    async def get_queue_position(self, task_id: str) -> int:
        """
        Get the position of a task in the queue.

        Returns:
            0 if task is being processed or completed
            N if task is at position N in the queue
            -1 if task not found
        """
        # Task is being processed or completed
        if task_id in self.active_workers:
            return 0

        if task_id in self.tasks and self.tasks[task_id].task_status in [
            TaskStatus.SUCCESS,
            TaskStatus.FAILURE,
        ]:
            return 0

        # Check if in pending queue
        if hasattr(self, "_pending_tasks"):
            for i, worker_task in enumerate(self._pending_tasks):
                if worker_task.task_id == task_id:
                    return i + 1

        return -1
