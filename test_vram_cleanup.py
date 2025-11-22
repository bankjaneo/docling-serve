#!/usr/bin/env python3
"""
Test script for improved VRAM cleanup functionality in docling-serve.

This script simulates the VRAM cleanup process and tests the effectiveness
of the enhanced cleanup methods.
"""

import asyncio
import logging
import sys
import os

# Add the current directory to Python path to import docling_serve modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from docling_serve.app import cleanup_models_if_needed, force_complete_cuda_cleanup

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:\t%(asctime)s - %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)

_log = logging.getLogger(__name__)


class MockOrchestrator:
    """Mock orchestrator for testing VRAM cleanup."""

    def __init__(self):
        self.cm = MockConverterManager()
        self.tasks = {}

    async def clear_converters(self):
        """Mock clear_converters method."""
        _log.info("Mock: Clearing converters...")


class MockConverterManager:
    """Mock converter manager for testing."""

    def __init__(self):
        self._get_converter_from_hash = MockCache()


class MockCache:
    """Mock cache with some simulated data."""

    def __init__(self):
        self.cache_info = MockCacheInfo()
        self.cache = {
            "key1": MockConverter(),
            "key2": MockConverter(),
        }

    def cache_info(self):
        return self.cache_info


class MockCacheInfo:
    """Mock cache info."""

    def __init__(self):
        self.hits = 10
        self.misses = 5
        self.maxsize = 128
        self.currsize = 2

    def __str__(self):
        return f"CacheInfo(hits={self.hits}, misses={self.misses}, maxsize={self.maxsize}, currsize={self.currsize})"


class MockConverter:
    """Mock converter with simulated ONNX sessions."""

    def __init__(self):
        self.doc_converter = MockDocConverter()
        self.__dict__['onnx_session'] = "MockONNXSession"


class MockDocConverter:
    """Mock doc converter."""

    def to(self, device):
        """Mock device movement."""
        _log.info(f"Mock: Moving converter to {device}")


async def test_enhanced_vram_cleanup():
    """Test the enhanced VRAM cleanup functionality."""

    _log.info("=" * 60)
    _log.info("TESTING ENHANCED VRAM CLEANUP")
    _log.info("=" * 60)

    # Test with mock orchestrator (this won't actually use CUDA)
    mock_orchestrator = MockOrchestrator()

    try:
        _log.info("Testing cleanup_models_if_needed with mock orchestrator...")

        # Mock the free_vram_on_idle setting
        import docling_serve.settings
        original_setting = getattr(docling_serve.settings.docling_serve_settings, 'free_vram_on_idle', False)
        docling_serve.settings.docling_serve_settings.free_vram_on_idle = True

        try:
            await cleanup_models_if_needed(mock_orchestrator)
            _log.info("✓ cleanup_models_if_needed completed successfully")
        except Exception as e:
            _log.error(f"✗ cleanup_models_if_needed failed: {e}")
        finally:
            # Restore original setting
            docling_serve.settings.docling_serve_settings.free_vram_on_idle = original_setting

    except Exception as e:
        _log.error(f"✗ Mock test failed: {e}")

    # Test actual CUDA cleanup if CUDA is available
    try:
        import torch

        if torch.cuda.is_available():
            _log.info("\n" + "=" * 60)
            _log.info("TESTING ACTUAL CUDA CLEANUP")
            _log.info("=" * 60)

            current_device = torch.cuda.current_device()

            # Get initial memory state
            initial_memory = torch.cuda.memory_allocated(current_device) / 1024**2
            initial_reserved = torch.cuda.memory_reserved(current_device) / 1024**2

            _log.info(f"Initial VRAM state:")
            _log.info(f"  Allocated: {initial_memory:.2f} MB")
            _log.info(f"  Reserved:  {initial_reserved:.2f} MB")

            # Allocate some test memory
            _log.info("\nAllocating test tensors...")
            test_tensors = []
            for i in range(5):
                tensor = torch.randn(1000, 1000, device=f'cuda:{current_device}')
                test_tensors.append(tensor)

            allocated_memory = torch.cuda.memory_allocated(current_device) / 1024**2
            _log.info(f"After allocating test tensors: {allocated_memory:.2f} MB")

            # Delete tensors and run enhanced cleanup
            _log.info("\nDeleting test tensors and running enhanced cleanup...")
            del test_tensors

            await force_complete_cuda_cleanup()

            final_memory = torch.cuda.memory_allocated(current_device) / 1024**2
            final_reserved = torch.cuda.memory_reserved(current_device) / 1024**2

            _log.info(f"\nFinal VRAM state:")
            _log.info(f"  Allocated: {final_memory:.2f} MB")
            _log.info(f"  Reserved:  {final_reserved:.2f} MB")
            _log.info(f"  Freed:     {(initial_reserved - final_reserved):.2f} MB")

            if final_memory < 50:
                _log.info("✓ Enhanced cleanup successful - near-zero VRAM usage achieved")
            else:
                _log.warning(f"⚠ Still have {final_memory:.2f} MB allocated after cleanup")

        else:
            _log.info("CUDA not available - skipping actual CUDA cleanup test")

    except ImportError:
        _log.warning("PyTorch not available - skipping CUDA cleanup test")
    except Exception as e:
        _log.error(f"✗ CUDA cleanup test failed: {e}")

    _log.info("\n" + "=" * 60)
    _log.info("VRAM CLEANUP TEST COMPLETE")
    _log.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_enhanced_vram_cleanup())