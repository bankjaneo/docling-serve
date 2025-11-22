"""
VRAM Worker Orchestrator - Minimal implementation for complete VRAM cleanup.

This orchestrator spawns completely separate processes for each OCR task,
ensuring that VRAM is completely released when the task finishes.
Minimal dependencies - doesn't require Redis or docling_jobkit.
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass

from docling_serve.settings import docling_serve_settings
from docling_serve.storage import get_scratch

_log = logging.getLogger(__name__)


@dataclass
class TaskStatus:
    PENDING = "pending"
    STARTED = "started"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass
class Task:
    task_id: str
    task_type: str
    task_status: str
    processing_meta: Dict[str, Any]


class VRAMWorkerOrchestrator:
    """
    Minimal Worker Orchestrator for complete VRAM cleanup.

    This orchestrator spawns separate processes for each OCR task,
    ensuring complete VRAM cleanup by terminating the process.
    """

    def __init__(self, *args, **kwargs):
        self._active_workers: dict[str, subprocess.Popen] = {}
        self._worker_timeout = getattr(docling_serve_settings, 'max_document_timeout', 300)
        self._scratch_dir = get_scratch()
        self._tasks_dir = self._scratch_dir / "vram_workers"
        self._tasks_dir.mkdir(exist_ok=True)

    async def enqueue(self, task_type: str, sources: list, convert_options: Any, target: Any, chunking_options: Any = None, chunking_export_options: Any = None) -> Task:
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
            "task_type": task_type,
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

        _log.info(f"Task {task_id} queued for VRAM worker processing")

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

        # Create a simple worker script that calls the docling CLI
        worker_script = '''
import json
import sys
import subprocess
from pathlib import Path

def main():
    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    try:
        with open(input_file, 'r') as f:
            data = json.load(f)

        task_id = data["task_id"]
        print(f"VRAM Worker starting task {task_id}")

        # Extract source files from the data
        source_files = []
        for source in data.get("sources", []):
            if source.get("type") == "FileSource":
                source_data = source.get("data", {})
                path = source_data.get("path") if source_data else None
                if path:
                    source_files.append(path)

        if not source_files:
            raise ValueError("No source files found")

        # Use docling CLI to process the files
        output_dir = output_file.parent / f"task_{task_id}_output"
        output_dir.mkdir(exist_ok=True)

        cmd = [
            "python", "-m", "docling",
            "--to", "json",
            "--output", str(output_dir),
        ] + source_files

        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            # Success - collect the JSON outputs
            json_outputs = []
            for json_file in output_dir.glob("*.json"):
                with open(json_file, 'r') as f:
                    json_outputs.append(json.load(f))

            output_data = {
                "task_id": task_id,
                "status": "success",
                "results": json_outputs,
                "stdout": result.stdout,
            }
        else:
            output_data = {
                "task_id": task_id,
                "status": "failure",
                "error": result.stderr,
                "stdout": result.stdout,
            }

        # Save output
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Task {task_id} completed with status: {output_data['status']}")

    except Exception as e:
        print(f"Worker failed: {e}")
        output_data = {
            "status": "failure",
            "error": str(e),
        }
        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
        except:
            pass

if __name__ == "__main__":
    main()
'''

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
            _log.error(f"Failed to start VRAM worker for task {task_id}: {e}")
            task = await self.task_status(task_id)
            task.task_status = TaskStatus.FAILURE
            task.processing_meta["error"] = f"Failed to start worker: {e}"
            await self._store_task_metadata(task)

    async def _monitor_worker(self, task_id: str, process: subprocess.Popen) -> None:
        """Monitor the worker process and handle completion."""

        try:
            # Wait for process to complete with timeout
            try:
                stdout, stderr = process.communicate(timeout=self._worker_timeout)
                return_code = process.returncode

            except subprocess.TimeoutExpired:
                _log.warning(f"VRAM worker process for task {task_id} timed out, terminating...")
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
                _log.error(f"VRAM worker process failed for task {task_id}")
                if stderr:
                    _log.error(f"Worker stderr: {stderr}")

                # Update task status
                task = await self.task_status(task_id)
                task.task_status = TaskStatus.FAILURE
                task.processing_meta["error"] = "VRAM worker process failed"
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
                    else:
                        task.task_status = TaskStatus.FAILURE
                        if "error" in results:
                            task.processing_meta["error"] = results["error"]

                    task.processing_meta["num_processed"] = len(results.get("results", []))
                    await self._store_task_metadata(task)

                    _log.info(f"VRAM Task {task_id} completed with status: {task.task_status}")

                except Exception as e:
                    _log.error(f"Failed to load results for VRAM task {task_id}: {e}")
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
                _log.debug(f"Failed to cleanup files for VRAM task {task_id}: {e}")

        except Exception as e:
            _log.error(f"Error monitoring VRAM worker for task {task_id}: {e}")

    async def _store_task_metadata(self, task: Task) -> None:
        """Store task metadata in file."""
        try:
            data = {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "task_status": task.task_status,
                "processing_meta": task.processing_meta,
            }

            metadata_file = self._tasks_dir / f"task_{task.task_id}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            _log.error(f"Failed to store VRAM task {task.task_id} metadata: {e}")

    async def task_status(self, task_id: str, wait: float = 0.0) -> Task:
        """Get task status from file."""

        try:
            metadata_file = self._tasks_dir / f"task_{task_id}_metadata.json"

            if not metadata_file.exists():
                raise Exception(f"VRAM Task {task_id} not found")

            with open(metadata_file, 'r') as f:
                data = json.load(f)

            return Task(
                task_id=data["task_id"],
                task_type=data["task_type"],
                task_status=data["task_status"],
                processing_meta=data.get("processing_meta", {}),
            )

        except Exception as e:
            _log.error(f"Failed to get VRAM task status for {task_id}: {e}")
            raise Exception(f"VRAM Task {task_id} not found")

    async def task_result(self, task_id: str) -> Optional[Any]:
        """Get task results from file."""

        try:
            result_file = self._tasks_dir / f"task_{task_id}_output.json"

            if result_file.exists():
                with open(result_file, 'r') as f:
                    return json.load(f)

            return None

        except Exception as e:
            _log.error(f"Failed to get VRAM task result for {task_id}: {e}")
            return None

    async def warm_up_caches(self) -> None:
        """Warm up caches (no-op for VRAM worker orchestrator)."""
        _log.info("VRAM Worker Orchestrator doesn't require cache warming")
        pass

    async def clear_converters(self) -> None:
        """Clear converters (no-op for VRAM worker orchestrator)."""
        _log.info("VRAM Worker Orchestrator doesn't require explicit converter clearing")
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
            _log.error(f"Failed to clear VRAM results: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    def bind_notifier(self, notifier):
        """Bind a notifier for compatibility with existing system."""
        _log.info("VRAMWorkerOrchestrator: notifier binding not required (uses file-based status)")
        pass

    def get_queue_position(self, task_id: str) -> int:
        """Get queue position (not applicable for worker orchestrator)."""
        return 0

    async def process_queue(self):
        """Process queue (not applicable for worker orchestrator)."""
        _log.debug("VRAMWorkerOrchestrator: queue processing not required")
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup any running workers."""

        # Terminate all active workers
        for task_id, process in list(self._active_workers.items()):
            try:
                _log.info(f"Terminating VRAM worker for task {task_id}")
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
                _log.error(f"Failed to terminate VRAM worker for task {task_id}: {e}")

        self._active_workers.clear()