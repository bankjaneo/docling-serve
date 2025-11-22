#!/usr/bin/env python3
"""
Test script to validate the improved VRAM cleanup process.

This script simulates the model loading and cleanup cycle to test
the effectiveness of the enhanced VRAM cleanup implementation.
"""

import asyncio
import logging
import sys
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(asctime)s - %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)

_log = logging.getLogger(__name__)

try:
    import torch
    if not torch.cuda.is_available():
        _log.error("CUDA is not available. This test requires a GPU.")
        sys.exit(1)

    _log.info(f"Using GPU: {torch.cuda.get_device_name()}")
    _log.info(f"CUDA Version: {torch.version.cuda}")
    _log.info(f"PyTorch Version: {torch.__version__}")

except ImportError:
    _log.error("PyTorch is not installed. Please install it to run this test.")
    sys.exit(1)


# Import the tracking functions from our implementation
sys.path.insert(0, '/private/var/folders/fm/6lcsfk190z15dwlj5ws0849r0000gn/T/vibe-kanban/worktrees/29ff-improve-doclings/docling_serve')
from app import track_vram_usage, log_vram_trend_analysis, _vram_usage_history


def simulate_model_allocation(size_mb=500):
    """Simulate allocating GPU memory similar to model loading."""
    _log.info(f"Simulating model allocation of {size_mb} MB...")

    # Allocate a tensor of approximately the specified size
    # Each float32 is 4 bytes, so we need size_mb * 1024*1024 / 4 elements
    num_elements = int(size_mb * 1024 * 1024 / 4)

    try:
        # Create tensor on GPU
        dummy_model = torch.randn(num_elements // 1000, 1000, device='cuda', dtype=torch.float32)
        _log.info(f"Allocated tensor of shape {dummy_model.shape}")

        # Force synchronization to ensure allocation completes
        torch.cuda.synchronize()

        return dummy_model
    except torch.cuda.OutOfMemoryError:
        _log.error(f"Failed to allocate {size_mb} MB - out of memory")
        return None
    except Exception as e:
        _log.error(f"Failed to allocate tensor: {e}")
        return None


def simulate_model_cleanup(dummy_model):
    """Simulate the enhanced cleanup process."""
    _log.info("Starting enhanced cleanup simulation...")

    # Track VRAM before cleanup
    mem_before, _ = track_vram_usage("cleanup_start")

    # Step 1: Move model to CPU and delete
    if dummy_model is not None:
        try:
            dummy_model.cpu()
            del dummy_model
            _log.info("Moved model to CPU and deleted reference")
        except Exception as e:
            _log.warning(f"Failed to move model to CPU: {e}")

    # Step 2: Garbage collection
    import gc
    _log.info("Running garbage collection...")
    for _ in range(3):
        gc.collect()

    # Step 3: Enhanced CUDA cleanup
    _log.info("Starting enhanced CUDA cleanup...")

    # Multiple empty cache calls
    for i in range(3):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        _log.debug(f"CUDA cleanup iteration {i+1} completed")

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()

    # Aggressive memory fraction manipulation
    device_count = torch.cuda.device_count()
    _log.info(f"Found {device_count} CUDA device(s)")

    for device_id in range(device_count):
        try:
            current_device = torch.cuda.current_device()
            torch.cuda.set_device(device_id)

            # Get current memory state
            mem_alloc = torch.cuda.memory_allocated(device_id)
            mem_reserv = torch.cuda.memory_reserved(device_id)
            _log.debug(f"Device {device_id} - Allocated: {mem_alloc/1024**2:.2f}MB, Reserved: {mem_reserv/1024**2:.2f}MB")

            # Memory fraction manipulation
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
            _log.warning(f"Device {device_id} cleanup failed: {e}")
            continue

    # Final cleanup pass
    gc.collect()
    for _ in range(2):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Try experimental context cleanup
    try:
        _log.debug("Attempting CUDA context cleanup...")
        for device_id in range(device_count):
            try:
                with torch.cuda.device(device_id):
                    dummy = torch.tensor([1.0], device=f'cuda:{device_id}')
                    del dummy
                    torch.cuda.empty_cache()
            except Exception as e:
                _log.debug(f"Context cleanup failed for device {device_id}: {e}")
    except Exception as e:
        _log.debug(f"CUDA context cleanup failed: {e}")

    # Track VRAM after cleanup
    mem_after, mem_reserved_after = track_vram_usage("cleanup_end")

    # Calculate effectiveness
    if mem_before is not None:
        freed = mem_before - mem_after
        if freed > 0:
            _log.info(f"VRAM cleanup freed {freed:.2f} MB ({(freed/mem_before*100):.1f}% of allocated memory)")
        else:
            _log.warning(f"VRAM cleanup did not free allocated memory (change: {freed:.2f} MB)")

        _log.info(f"Final VRAM: {mem_after:.2f} MB allocated, {mem_reserved_after:.2f} MB reserved")

    return mem_before, mem_after


def test_cleanup_cycles(num_cycles=3):
    """Test multiple cleanup cycles to detect accumulation."""
    _log.info(f"Starting {num_cycles} cleanup cycle test...")

    # Clear history
    _vram_usage_history.clear()

    cycle_results = []

    for cycle in range(num_cycles):
        _log.info(f"\n=== Cycle {cycle + 1}/{num_cycles} ===")

        # Simulate model loading
        dummy_model = simulate_model_allocation(size_mb=300 + cycle * 100)  # Vary size slightly

        if dummy_model is None:
            _log.error("Failed to allocate model, stopping test")
            break

        # Wait a moment to simulate processing
        time.sleep(0.5)

        # Simulate cleanup
        mem_before, mem_after = simulate_model_cleanup(dummy_model)

        if mem_before is not None and mem_after is not None:
            cycle_results.append({
                'cycle': cycle + 1,
                'before': mem_before,
                'after': mem_after,
                'freed': mem_before - mem_after
            })

        # Log trend analysis
        log_vram_trend_analysis()

        # Wait between cycles
        time.sleep(1)

    # Analyze results
    _log.info("\n=== Test Results Analysis ===")
    if len(cycle_results) >= 2:
        first_after = cycle_results[0]['after']
        last_after = cycle_results[-1]['after']

        accumulation = last_after - first_after
        if accumulation > 50:  # 50MB threshold
            _log.warning(f"VRAM accumulation detected: {first_after:.1f}MB → {last_after:.1f}MB (+{accumulation:.1f}MB)")
            _log.warning("This suggests the cleanup process may not be fully effective")
        elif accumulation < -50:
            _log.info(f"VRAM decreased over cycles: {first_after:.1f}MB → {last_after:.1f}MB ({accumulation:.1f}MB)")
        else:
            _log.info(f"VRAM remained stable over cycles: {first_after:.1f}MB → {last_after:.1f}MB")

    for result in cycle_results:
        efficiency = (result['freed'] / result['before'] * 100) if result['before'] > 0 else 0
        _log.info(f"Cycle {result['cycle']}: {result['before']:.1f}MB → {result['after']:.1f}MB "
                   f"(freed {result['freed']:.1f}MB, {efficiency:.1f}% efficiency)")


if __name__ == "__main__":
    try:
        _log.info("Starting VRAM cleanup test...")

        # Initial VRAM state
        initial_allocated = torch.cuda.memory_allocated() / 1024**2
        initial_reserved = torch.cuda.memory_reserved() / 1024**2
        _log.info(f"Initial VRAM: {initial_allocated:.2f} MB allocated, {initial_reserved:.2f} MB reserved")

        # Run test cycles
        test_cleanup_cycles(num_cycles=3)

        # Final VRAM state
        final_allocated = torch.cuda.memory_allocated() / 1024**2
        final_reserved = torch.cuda.memory_reserved() / 1024**2
        _log.info(f"Final VRAM: {final_allocated:.2f} MB allocated, {final_reserved:.2f} MB reserved")

        total_change = final_allocated - initial_allocated
        _log.info(f"Total VRAM change: {total_change:+.2f} MB allocated")

        if total_change < 0:
            _log.info("✅ Test completed successfully - VRAM was cleaned up effectively")
        elif total_change < 100:
            _log.info("✅ Test completed - minimal VRAM accumulation (likely CUDA context overhead)")
        else:
            _log.warning("⚠️  Test completed - significant VRAM accumulation detected")

    except Exception as e:
        _log.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)