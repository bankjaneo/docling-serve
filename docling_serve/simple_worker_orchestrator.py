"""
Simple Worker Orchestrator for complete VRAM cleanup.

This orchestrator spawns completely separate processes for each OCR task,
ensuring that VRAM is completely released when the task finishes.
Uses file-based storage instead of Redis for simplicity.
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional
import uuid

from docling_jobkit.datamodel.task import Task, TaskSource, TaskStatus, TaskType
from docling_jobkit.orchestrators.base_orchestrator import BaseOrchestrator, TaskNotFoundError

from docling_serve.settings import docling_serve_settings
from docling_serve.storage import get_scratch

_log = logging.getLogger(__name__)


class SimpleWorkerOrchestrator(BaseOrchestrator):
    """
    Simplified Worker Orchestrator that uses file-based storage.

    This orchestrator spawns separate processes for each OCR task,
    ensuring complete VRAM cleanup by terminating the process.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._active_workers: dict[str, subprocess.Popen] = {}
        self._worker_timeout = docling_serve_settings.max_document_timeout
        self._scratch_dir = get_scratch()
        self._tasks_dir = self._scratch_dir / "worker_tasks"
        self._tasks_dir.mkdir(exist_ok=True)

    async def enqueue(self, task_type: TaskType, sources: list[TaskSource], convert_options: Any, target: Any, chunking_options: Any = None, chunking_export_options: Any = None) -> Task:
        """Enqueue a task to be processed in a separate process."""

        task_id = str(uuid.uuid4())

        # Create task object
        task = Task(
            task_id=task_id,
            task_type=task_type,
            task_status=TaskStatus.PENDING,
            processing_meta={
                "num_docs": len(sources),
                "num_processed": 0,
                "num_succeeded": 0,
                "num_failed": 0,
                "start_time": time.time(),
            },
        )

        # Store task metadata in file
        await self._store_task_metadata(task)

        # Prepare input file for worker
        input_data = {
            "task_id": task_id,
            "task_type": task_type.value if hasattr(task_type, "value") else str(task_type),
            "sources": [],
            "convert_options": self._serialize_options(convert_options),
            "target": self._serialize_options(target),
            "chunking_options": self._serialize_options(chunking_options) if chunking_options else None,
            "chunking_export_options": self._serialize_options(chunking_export_options) if chunking_export_options else None,
            "scratch_dir": str(self._scratch_dir),
        }

        # Prepare sources
        for source in sources:
            source_data = {
                "type": source.__class__.__name__,
                "data": source.model_dump() if hasattr(source, "model_dump") else str(source),
            }
            input_data["sources"].append(source_data)

        # Write input file
        input_file = self._tasks_dir / f"task_{task_id}_input.json"
        with open(input_file, 'w') as f:
            json.dump(input_data, f, indent=2)

        # Start worker process
        await self._start_worker(task_id, input_file)

        _log.info(f"Task {task_id} queued for worker processing")

        return task

    def _serialize_options(self, options: Any) -> dict:
        """Serialize options to dictionary."""
        if options is None:
            return None
        if hasattr(options, 'model_dump'):
            return options.model_dump()
        elif hasattr(options, '__dict__'):
            return options.__dict__
        else:
            return {"value": str(options)}

    async def _start_worker(self, task_id: str, input_file: Path) -> None:
        """Start a worker process for the given task."""

        # Create worker script
        worker_script = self._create_worker_script()

        # Process environment
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(__file__).parent.parent)

        # Start the worker process
        cmd = [
            sys.executable,
            '-c',
            worker_script,
            str(input_file),
            str(self._tasks_dir / f"task_{task_id}_output.json"),
        ]

        try:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid if os.name != 'nt' else None,  # Create new process group
            )

            self._active_workers[task_id] = process

            # Update task status to STARTED
            task = await self.task_status(task_id)
            task.task_status = TaskStatus.STARTED
            await self._store_task_metadata(task)

            # Start monitoring the worker
            asyncio.create_task(self._monitor_worker(task_id, process))

        except Exception as e:
            _log.error(f"Failed to start worker for task {task_id}: {e}")
            task = await self.task_status(task_id)
            task.task_status = TaskStatus.FAILURE
            task.processing_meta["error"] = f"Failed to start worker: {e}"
            await self._store_task_metadata(task)

    def _create_worker_script(self) -> str:
        """Create the Python script that runs in the worker process."""

        return '''
import json
import logging
import sys
import traceback
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

def main():
    if len(sys.argv) != 3:
        print("Usage: worker_script.py <input_file> <output_file>")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    try:
        # Load task data
        with open(input_file, 'r') as f:
            data = json.load(f)

        task_id = data['task_id']
        _log.info(f"Worker starting task {task_id}")

        # Import docling modules (this will load CUDA contexts in this process)
        from docling.datamodel.base_models import DocumentStream
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        # Create converter
        pipeline_options = PdfPipelineOptions()
        converter = DocumentConverter(format_options={
            "application/pdf": PdfFormatOption(pipeline_options=pipeline_options),
        })

        # Load sources - simplified handling
        sources = []
        for source_data in data['sources']:
            try:
                if source_data['type'] == 'FileSource':
                    # Handle FileSource
                    source_info = source_data['data']
                    if 'path' in source_info:
                        sources.append(DocumentStream.from_path(source_info['path']))
                    else:
                        # Skip sources without path for now
                        _log.warning(f"Skipping source without path: {source_info}")
                elif source_data['type'] == 'DocumentStream':
                    # Handle DocumentStream
                    sources.append(DocumentStream(**source_data['data']))
                else:
                    _log.warning(f"Unknown source type: {source_data['type']}")
            except Exception as e:
                _log.error(f"Error loading source: {e}")

        if not sources:
            raise ValueError("No valid sources found")

        _log.info(f"Processing {len(sources)} documents for task {task_id}")

        # Process documents
        results = []
        for i, doc_stream in enumerate(sources):
            try:
                _log.info(f"Processing document {i+1}/{len(sources)}")
                result = converter.convert(doc_stream)
                results.append({
                    "success": True,
                    "document": result.document.export_to_dict(),
                })
                _log.info(f"Document {i+1} processed successfully")
            except Exception as e:
                _log.error(f"Error processing document {i+1}: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })

        # Save results
        output_data = {
            "task_id": task_id,
            "status": "success" if all(r["success"] for r in results) else "partial_failure",
            "results": results,
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        _log.info(f"Task {task_id} completed successfully")

    except Exception as e:
        _log.error(f"Worker failed for task: {e}")
        _log.error(traceback.format_exc())

        # Save error result
        output_data = {
            "status": "failure",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
        except Exception:
            pass

if __name__ == "__main__":
    main()
'''

    async def _monitor_worker(self, task_id: str, process: subprocess.Popen) -> None:
        """Monitor the worker process and handle completion."""

        try:
            # Wait for process to complete with timeout
            try:
                stdout, stderr = process.communicate(timeout=self._worker_timeout)
                return_code = process.returncode

            except subprocess.TimeoutExpired:
                _log.warning(f"Worker process for task {task_id} timed out, terminating...")
                # Kill the entire process group
                if os.name != 'nt':
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:
                    process.terminate()

                stdout, stderr = process.communicate(timeout=5)  # Give it time to terminate
                return_code = -1

            # Check output file
            output_file = self._tasks_dir / f"task_{task_id}_output.json"

            if return_code != 0 or not output_file.exists():
                _log.error(f"Worker process failed for task {task_id}")
                if stderr:
                    _log.error(f"Worker stderr: {stderr}")

                # Update task status
                task = await self.task_status(task_id)
                task.task_status = TaskStatus.FAILURE
                task.processing_meta["error"] = "Worker process failed"
                if stderr:
                    task.processing_meta["stderr"] = stderr
                await self._store_task_metadata(task)

            else:
                # Load results
                try:
                    with open(output_file, 'r') as f:
                        results = json.load(f)

                    # Update task status
                    task = await self.task_status(task_id)

                    if results.get("status") == "success":
                        task.task_status = TaskStatus.SUCCESS
                        task.processing_meta["num_succeeded"] = len(results.get("results", []))
                    elif results.get("status") == "partial_failure":
                        task.task_status = TaskStatus.SUCCESS  # Still success overall
                        task.processing_meta["num_succeeded"] = sum(1 for r in results.get("results", []) if r.get("success"))
                        task.processing_meta["num_failed"] = sum(1 for r in results.get("results", []) if not r.get("success"))
                    else:
                        task.task_status = TaskStatus.FAILURE
                        if "error" in results:
                            task.processing_meta["error"] = results["error"]

                    task.processing_meta["num_processed"] = len(results.get("results", []))
                    await self._store_task_metadata(task)

                    _log.info(f"Task {task_id} completed with status: {task.task_status.value}")

                except Exception as e:
                    _log.error(f"Failed to load results for task {task_id}: {e}")
                    task = await self.task_status(task_id)
                    task.task_status = TaskStatus.FAILURE
                    task.processing_meta["error"] = f"Failed to load results: {e}"
                    await self._store_task_metadata(task)

            # Cleanup worker from active list
            if task_id in self._active_workers:
                del self._active_workers[task_id]

            # Cleanup files
            try:
                input_file = self._tasks_dir / f"task_{task_id}_input.json"
                output_file = self._tasks_dir / f"task_{task_id}_output.json"

                if input_file.exists():
                    input_file.unlink()
                if output_file.exists():
                    output_file.unlink()

            except Exception as e:
                _log.debug(f"Failed to cleanup files for task {task_id}: {e}")

        except Exception as e:
            _log.error(f"Error monitoring worker for task {task_id}: {e}")

    async def _store_task_metadata(self, task: Task) -> None:
        """Store task metadata in file."""
        try:
            data = {
                "task_id": task.task_id,
                "task_type": task.task_type.value if hasattr(task.task_type, "value") else str(task.task_type),
                "task_status": task.task_status.value,
                "processing_meta": task.processing_meta,
            }

            metadata_file = self._tasks_dir / f"task_{task.task_id}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            _log.error(f"Failed to store task {task.task_id} metadata: {e}")

    async def task_status(self, task_id: str, wait: float = 0.0) -> Task:
        """Get task status from file."""

        try:
            metadata_file = self._tasks_dir / f"task_{task_id}_metadata.json"

            if not metadata_file.exists():
                raise TaskNotFoundError(f"Task {task_id} not found")

            with open(metadata_file, 'r') as f:
                data = json.load(f)

            return Task(
                task_id=data["task_id"],
                task_type=TaskType(data["task_type"]) if data["task_type"] in [t.value for t in TaskType] else TaskType.CONVERT,
                task_status=TaskStatus(data["task_status"]),
                processing_meta=data.get("processing_meta", {}),
            )

        except Exception as e:
            if isinstance(e, TaskNotFoundError):
                raise
            _log.error(f"Failed to get task status for {task_id}: {e}")
            raise TaskNotFoundError(f"Task {task_id} not found")

    async def task_result(self, task_id: str) -> Optional[Any]:
        """Get task results from file."""

        try:
            result_file = self._tasks_dir / f"task_{task_id}_output.json"

            if result_file.exists():
                with open(result_file, 'r') as f:
                    return json.load(f)

            return None

        except Exception as e:
            _log.error(f"Failed to get task result for {task_id}: {e}")
            return None

    async def warm_up_caches(self) -> None:
        """Warm up caches (no-op for worker orchestrator)."""
        _log.info("Simple Worker Orchestrator doesn't require cache warming")
        pass

    async def clear_converters(self) -> None:
        """Clear converters (no-op for worker orchestrator)."""
        _log.info("Simple Worker Orchestrator doesn't require explicit converter clearing")
        pass

    async def clear_results(self, older_than: float = 3600) -> None:
        """Clear old task files."""
        try:
            current_time = time.time()
            for file_path in self._tasks_dir.glob("task_*.json"):
                try:
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > older_than:
                        file_path.unlink()
                except Exception as e:
                    _log.debug(f"Failed to delete old file {file_path}: {e}")
        except Exception as e:
            _log.error(f"Failed to clear results: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup any running workers."""

        # Terminate all active workers
        for task_id, process in list(self._active_workers.items()):
            try:
                _log.info(f"Terminating worker for task {task_id}")
                if os.name != 'nt':
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:
                    process.terminate()

                # Wait a bit for graceful termination
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name != 'nt':
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    else:
                        process.kill()

            except Exception as e:
                _log.error(f"Failed to terminate worker for task {task_id}: {e}")

        self._active_workers.clear()