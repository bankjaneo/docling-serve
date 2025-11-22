"""
Subprocess worker for OCR tasks to ensure complete VRAM cleanup.

When DOCLINGSERVEFREEVRAMON_IDLE=True, this worker runs OCR tasks in isolated
subprocesses that are terminated after each task, guaranteeing complete VRAM release.
"""

import asyncio
import logging
import multiprocessing as mp
import pickle
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

_log = logging.getLogger(__name__)


def _worker_process_main(task_data: bytes, result_queue: mp.Queue) -> None:
    """
    Main function for worker process. Loads models, processes task, returns result.

    This runs in an isolated subprocess that will be terminated after completion,
    ensuring all CUDA memory is released back to the OS.
    """
    try:
        # Unpickle task data
        task_dict = pickle.loads(task_data)

        # Import docling components inside subprocess to avoid parent process pollution
        from docling_jobkit.convert.manager import (
            DoclingConverterManager,
            DoclingConverterManagerConfig,
        )

        # Create converter manager with same config as parent
        cm_config = DoclingConverterManagerConfig(
            artifacts_path=task_dict.get("artifacts_path"),
            options_cache_size=task_dict.get("options_cache_size", 10),
            enable_remote_services=task_dict.get("enable_remote_services", False),
            allow_external_plugins=task_dict.get("allow_external_plugins", False),
            max_num_pages=task_dict.get("max_num_pages"),
            max_file_size=task_dict.get("max_file_size"),
            queue_max_size=task_dict.get("queue_max_size", 10),
            ocr_batch_size=task_dict.get("ocr_batch_size", 1),
            layout_batch_size=task_dict.get("layout_batch_size", 1),
            table_batch_size=task_dict.get("table_batch_size", 1),
            batch_polling_interval_seconds=task_dict.get("batch_polling_interval_seconds", 0.1),
        )

        cm = DoclingConverterManager(config=cm_config)

        # Process the task
        # Note: This is a simplified version - you'll need to adapt this
        # based on your actual task processing logic
        sources = task_dict.get("sources", [])
        convert_options = task_dict.get("convert_options")

        # Get converter for this task
        converter = cm.get_converter(convert_options)

        # Process documents
        results = []
        for source in sources:
            result = converter.convert(source)
            results.append(result)

        # Serialize and return results
        result_data = {
            "status": "success",
            "results": results,
        }

        result_queue.put(pickle.dumps(result_data))

    except Exception as e:
        # Return error
        error_data = {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        result_queue.put(pickle.dumps(error_data))

    finally:
        # Explicit cleanup before process termination
        try:
            import gc
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass


class SubprocessWorkerPool:
    """
    Manages a pool of worker subprocesses for OCR tasks.

    Each task runs in a fresh subprocess that is terminated after completion,
    ensuring complete VRAM cleanup.
    """

    def __init__(self, max_workers: int = 1):
        """
        Initialize worker pool.

        Args:
            max_workers: Maximum number of concurrent worker processes.
                        For VRAM cleanup, typically use 1 worker.
        """
        self.max_workers = max_workers
        self._semaphore = asyncio.Semaphore(max_workers)

    async def process_task(
        self,
        task_dict: Dict[str, Any],
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """
        Process a task in an isolated subprocess.

        Args:
            task_dict: Task configuration and data
            timeout: Maximum time to wait for task completion (seconds)

        Returns:
            Result dictionary with 'status' and either 'results' or 'error'

        Raises:
            TimeoutError: If task exceeds timeout
            RuntimeError: If subprocess fails
        """
        async with self._semaphore:
            return await self._run_task_in_subprocess(task_dict, timeout)

    async def _run_task_in_subprocess(
        self,
        task_dict: Dict[str, Any],
        timeout: float,
    ) -> Dict[str, Any]:
        """Run a single task in an isolated subprocess."""

        # Create queue for results
        mp_ctx = mp.get_context("spawn")  # Use spawn for CUDA compatibility
        result_queue = mp_ctx.Queue()

        # Serialize task data
        task_data = pickle.dumps(task_dict)

        # Create and start worker process
        process = mp_ctx.Process(
            target=_worker_process_main,
            args=(task_data, result_queue),
            daemon=False,
        )

        try:
            _log.info(f"Starting subprocess worker (PID will be {process.pid})")
            process.start()
            _log.debug(f"Subprocess worker started with PID: {process.pid}")

            # Wait for result with timeout
            loop = asyncio.get_event_loop()
            result_data = await asyncio.wait_for(
                loop.run_in_executor(None, result_queue.get),
                timeout=timeout,
            )

            # Wait for process to complete
            await loop.run_in_executor(None, process.join, 5.0)

            if process.is_alive():
                _log.warning(f"Worker process {process.pid} still alive, terminating...")
                process.terminate()
                await loop.run_in_executor(None, process.join, 2.0)

                if process.is_alive():
                    _log.error(f"Worker process {process.pid} still alive, killing...")
                    process.kill()
                    process.join()

            _log.info(f"Subprocess worker {process.pid} terminated successfully")

            # Deserialize result
            result = pickle.loads(result_data)
            return result

        except asyncio.TimeoutError:
            _log.error(f"Worker process {process.pid} timed out after {timeout}s")
            process.terminate()
            process.join(timeout=2.0)

            if process.is_alive():
                process.kill()
                process.join()

            raise TimeoutError(f"Task processing timed out after {timeout} seconds")

        except Exception as e:
            _log.error(f"Error in subprocess worker: {e}")

            # Ensure process is cleaned up
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)

                if process.is_alive():
                    process.kill()
                    process.join()

            raise RuntimeError(f"Subprocess worker failed: {e}")

        finally:
            # Final cleanup check
            if process.is_alive():
                _log.warning(f"Cleaning up zombie process {process.pid}")
                try:
                    process.kill()
                    process.join(timeout=1.0)
                except Exception as e:
                    _log.error(f"Failed to clean up process {process.pid}: {e}")

            # Give OS time to reclaim CUDA memory
            await asyncio.sleep(0.1)


# Global worker pool instance
_worker_pool: Optional[SubprocessWorkerPool] = None


def get_worker_pool(max_workers: int = 1) -> SubprocessWorkerPool:
    """Get or create the global worker pool."""
    global _worker_pool
    if _worker_pool is None:
        _worker_pool = SubprocessWorkerPool(max_workers=max_workers)
    return _worker_pool


async def process_in_subprocess(
    task_dict: Dict[str, Any],
    timeout: float = 600.0,
) -> Dict[str, Any]:
    """
    Convenience function to process a task in a subprocess.

    Args:
        task_dict: Task configuration and data
        timeout: Maximum time to wait for task completion (seconds)

    Returns:
        Result dictionary
    """
    pool = get_worker_pool()
    return await pool.process_task(task_dict, timeout)
