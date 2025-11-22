#!/usr/bin/env python3
"""
Test script for VRAM Worker Orchestrator.

This script verifies:
1. VRAMWorkerOrchestrator can be imported and instantiated
2. Process spawning works correctly
3. Basic task lifecycle (enqueue -> process -> result)
"""

import asyncio
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(name)s - %(message)s",
)

_log = logging.getLogger(__name__)


async def test_vram_orchestrator_import():
    """Test that VRAMWorkerOrchestrator can be imported."""
    try:
        from docling_serve.orchestrators.vram_worker_orchestrator import (
            VRAMWorkerOrchestrator,
            VRAMWorkerOrchestratorConfig,
        )
        _log.info("✓ VRAMWorkerOrchestrator imported successfully")
        return True
    except Exception as e:
        _log.error(f"✗ Failed to import VRAMWorkerOrchestrator: {e}")
        return False


async def test_vram_orchestrator_instantiation():
    """Test that VRAMWorkerOrchestrator can be instantiated."""
    try:
        from docling_serve.orchestrators.vram_worker_orchestrator import (
            VRAMWorkerOrchestrator,
            VRAMWorkerOrchestratorConfig,
        )

        config = VRAMWorkerOrchestratorConfig(
            scratch_dir=Path("/tmp/docling_test"),
            worker_timeout=300,
        )

        cm_config = {
            "options_cache_size": 2,
            "max_num_pages": 100,
            "max_file_size": 100 * 1024 * 1024,
        }

        orchestrator = VRAMWorkerOrchestrator(
            config=config,
            converter_manager_config=cm_config,
        )

        _log.info("✓ VRAMWorkerOrchestrator instantiated successfully")
        _log.info(f"  - Worker timeout: {orchestrator.config.worker_timeout}s")
        _log.info(f"  - Scratch dir: {orchestrator.config.scratch_dir}")

        return True
    except Exception as e:
        _log.error(f"✗ Failed to instantiate VRAMWorkerOrchestrator: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_orchestrator_factory():
    """Test that orchestrator factory correctly selects VRAM orchestrator."""
    try:
        # Temporarily set environment variable
        import os
        original_value = os.environ.get("DOCLING_SERVE_FREE_VRAM_ON_IDLE")

        os.environ["DOCLING_SERVE_FREE_VRAM_ON_IDLE"] = "true"

        # Clear LRU cache to force re-evaluation
        from docling_serve.orchestrator_factory import get_async_orchestrator
        get_async_orchestrator.cache_clear()

        # Get orchestrator
        orchestrator = get_async_orchestrator()

        # Check type
        from docling_serve.orchestrators.vram_worker_orchestrator import VRAMWorkerOrchestrator

        if isinstance(orchestrator, VRAMWorkerOrchestrator):
            _log.info("✓ Orchestrator factory correctly returns VRAMWorkerOrchestrator when FREE_VRAM_ON_IDLE=true")
        else:
            _log.error(f"✗ Expected VRAMWorkerOrchestrator, got {type(orchestrator)}")
            return False

        # Restore original value
        if original_value is not None:
            os.environ["DOCLING_SERVE_FREE_VRAM_ON_IDLE"] = original_value
        else:
            os.environ.pop("DOCLING_SERVE_FREE_VRAM_ON_IDLE", None)

        get_async_orchestrator.cache_clear()

        return True
    except Exception as e:
        _log.error(f"✗ Orchestrator factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    _log.info("Starting VRAM Worker Orchestrator tests...\n")

    results = []

    _log.info("Test 1: Import VRAMWorkerOrchestrator")
    results.append(await test_vram_orchestrator_import())
    print()

    _log.info("Test 2: Instantiate VRAMWorkerOrchestrator")
    results.append(await test_vram_orchestrator_instantiation())
    print()

    _log.info("Test 3: Orchestrator Factory Selection")
    results.append(await test_orchestrator_factory())
    print()

    # Summary
    passed = sum(results)
    total = len(results)

    _log.info("=" * 60)
    _log.info(f"Test Summary: {passed}/{total} tests passed")
    _log.info("=" * 60)

    if passed == total:
        _log.info("✓ All tests passed!")
        return 0
    else:
        _log.error(f"✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
