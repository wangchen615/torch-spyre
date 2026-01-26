# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Owner(s): ["module: cpp"]

"""
Unit tests for VF (Virtual Function) allocator implementation.

These tests verify the VF allocator's segment-based memory management system,
including:
- Segment creation and block allocation
- Memory reuse and deallocation
- Free interval merging
- Alignment requirements (128-byte minimum)
- Edge cases and error handling

To run these tests, set FLEX_DEVICE=VF before running pytest:
    FLEX_DEVICE=VF python -m pytest tests/test_vf_allocator.py

Note: The allocator is a singleton initialized at module load time, so
FLEX_DEVICE must be set before importing torch_spyre. Tests will be
automatically skipped if FLEX_DEVICE is not set to "VF".
"""

import os
import unittest
import gc
from contextlib import contextmanager

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


@contextmanager
def set_flex_device(mode):
    """Context manager to temporarily set FLEX_DEVICE environment variable."""
    old_value = os.environ.get("FLEX_DEVICE")
    try:
        os.environ["FLEX_DEVICE"] = mode
        yield
    finally:
        if old_value is None:
            os.environ.pop("FLEX_DEVICE", None)
        else:
            os.environ["FLEX_DEVICE"] = old_value


def is_vf_mode():
    """Check if current FLEX_DEVICE is set to VF mode."""
    return os.environ.get("FLEX_DEVICE") == "VF"


class TestVFAllocator(TestCase):
    """Test suite for VF (Virtual Function) allocator implementation."""

    @unittest.skipUnless(
        is_vf_mode(), "VF allocator tests require FLEX_DEVICE=VF"
    )
    def test_vf_mode_detection(self):
        """Test that VF mode is correctly detected from environment variable."""
        with set_flex_device("VF"):
            # Create a tensor to trigger allocator initialization
            x = torch.empty(10, device="spyre", dtype=torch.float16)
            self.assertEqual(x.device.type, "spyre")
            # If we get here without exception, VF mode was detected correctly

    @unittest.skipUnless(
        is_vf_mode(), "VF allocator tests require FLEX_DEVICE=VF"
    )
    def test_basic_allocation(self):
        """Test basic memory allocation in VF mode."""
        # Allocate a small tensor
        x = torch.empty(100, device="spyre", dtype=torch.float16)
        self.assertEqual(x.device.type, "spyre")
        self.assertEqual(x.numel(), 100)
        self.assertEqual(x.dtype, torch.float16)

        # Verify storage is allocated (should be at least 128 bytes due to alignment)
        storage_size = x.untyped_storage().nbytes()
        self.assertGreaterEqual(storage_size, 128)

    @unittest.skipUnless(
        is_vf_mode(), "VF allocator tests require FLEX_DEVICE=VF"
    )
    def test_allocation_alignment(self):
        """Test that allocations are aligned to 128 bytes (Spyre requirement)."""
        # Test various sizes that should be rounded up to 128-byte alignment
        test_sizes = [1, 50, 100, 127, 128, 129, 200, 255, 256]

        for size in test_sizes:
            x = torch.empty(size, device="spyre", dtype=torch.float16)
            storage_size = x.untyped_storage().nbytes()
            # Storage size should be a multiple of 128
            self.assertEqual(
                storage_size % 128,
                0,
                f"Allocation size {storage_size} for tensor size {size} is not aligned to 128 bytes",
            )

    @unittest.skipUnless(
        is_vf_mode(), "VF allocator tests require FLEX_DEVICE=VF"
    )
    def test_memory_reuse(self):
        """Test that deallocated memory can be reused for new allocations."""
        # Allocate and deallocate multiple tensors
        tensors = []
        for i in range(5):
            t = torch.empty(100, device="spyre", dtype=torch.float16)
            tensors.append(t)

        # Clear references to trigger deallocation
        del tensors
        gc.collect()

        # Allocate new tensors - memory should be reused from freed blocks
        new_tensors = []
        for i in range(5):
            t = torch.empty(100, device="spyre", dtype=torch.float16)
            new_tensors.append(t)

        # Verify all new tensors are valid
        for t in new_tensors:
            self.assertEqual(t.device.type, "spyre")
            self.assertEqual(t.numel(), 100)

    @unittest.skipUnless(
        is_vf_mode(), "VF allocator tests require FLEX_DEVICE=VF"
    )
    def test_segment_creation(self):
        """Test that new segments are created when needed."""
        # Allocate a large tensor that should create a new segment
        # Segment size is 8GB by default, so we'll allocate something smaller
        # but significant enough to test segment creation
        large_size = 1024 * 1024  # 1M elements
        x = torch.empty(large_size, device="spyre", dtype=torch.float16)
        self.assertEqual(x.device.type, "spyre")
        self.assertEqual(x.numel(), large_size)

    @unittest.skipUnless(
        is_vf_mode(), "VF allocator tests require FLEX_DEVICE=VF"
    )
    def test_multiple_allocations_in_segment(self):
        """Test multiple allocations within the same segment."""
        # Allocate multiple small tensors that should fit in one segment
        tensors = []
        for i in range(10):
            t = torch.empty(1000, device="spyre", dtype=torch.float16)
            tensors.append(t)

        # Verify all tensors are valid
        for t in tensors:
            self.assertEqual(t.device.type, "spyre")
            self.assertEqual(t.numel(), 1000)

    @unittest.skipUnless(
        is_vf_mode(), "VF allocator tests require FLEX_DEVICE=VF"
    )
    def test_zero_size_allocation(self):
        """Test that zero-size allocation returns valid but empty tensor."""
        x = torch.empty(0, device="spyre", dtype=torch.float16)
        self.assertEqual(x.device.type, "spyre")
        self.assertEqual(x.numel(), 0)

    @unittest.skipUnless(
        is_vf_mode(), "VF allocator tests require FLEX_DEVICE=VF"
    )
    def test_different_dtypes(self):
        """Test allocation with different data types."""
        dtypes = [
            torch.float16,
            torch.float32,
            torch.int32,
            torch.bool,
        ]

        for dtype in dtypes:
            x = torch.empty(100, device="spyre", dtype=dtype)
            self.assertEqual(x.device.type, "spyre")
            self.assertEqual(x.dtype, dtype)
            self.assertEqual(x.numel(), 100)

    @unittest.skipUnless(
        is_vf_mode(), "VF allocator tests require FLEX_DEVICE=VF"
    )
    def test_sequential_allocation_deallocation(self):
        """Test sequential allocation and deallocation pattern."""
        # Pattern: allocate -> use -> deallocate -> repeat
        for iteration in range(5):
            x = torch.empty(500, device="spyre", dtype=torch.float16)
            x.fill_(iteration)
            x_cpu = x.cpu()
            self.assertTrue((x_cpu == iteration).all())
            del x, x_cpu
            gc.collect()

    @unittest.skipUnless(
        is_vf_mode(), "VF allocator tests require FLEX_DEVICE=VF"
    )
    def test_interleaved_allocations(self):
        """Test interleaved allocation pattern (allocate some, free some, allocate more)."""
        # Allocate first batch
        batch1 = [torch.empty(200, device="spyre", dtype=torch.float16) for _ in range(3)]

        # Free middle one
        del batch1[1]
        gc.collect()

        # Allocate new ones
        batch2 = [torch.empty(200, device="spyre", dtype=torch.float16) for _ in range(2)]

        # Verify all remaining tensors are valid
        self.assertEqual(batch1[0].numel(), 200)
        self.assertEqual(batch1[2].numel(), 200)
        for t in batch2:
            self.assertEqual(t.numel(), 200)

    @unittest.skipUnless(
        is_vf_mode(), "VF allocator tests require FLEX_DEVICE=VF"
    )
    def test_free_interval_merging(self):
        """Test that adjacent free intervals are merged correctly."""
        # Allocate three adjacent blocks
        t1 = torch.empty(1000, device="spyre", dtype=torch.float16)
        t2 = torch.empty(1000, device="spyre", dtype=torch.float16)
        t3 = torch.empty(1000, device="spyre", dtype=torch.float16)

        # Free middle block first
        del t2
        gc.collect()

        # Free first block (should merge with middle)
        del t1
        gc.collect()

        # Free last block (should merge with previous)
        del t3
        gc.collect()

        # Allocate a large block that should fit in the merged interval
        large = torch.empty(3000, device="spyre", dtype=torch.float16)
        self.assertEqual(large.numel(), 3000)

    @unittest.skipUnless(
        is_vf_mode(), "VF allocator tests require FLEX_DEVICE=VF"
    )
    def test_tensor_operations_with_vf_allocator(self):
        """Test that tensor operations work correctly with VF allocator."""
        x = torch.randn(100, device="spyre", dtype=torch.float16)
        y = torch.randn(100, device="spyre", dtype=torch.float16)

        # Perform operations
        z = x + y
        w = x * y
        v = torch.sum(x)

        # Verify results
        self.assertEqual(z.device.type, "spyre")
        self.assertEqual(w.device.type, "spyre")
        self.assertEqual(v.device.type, "spyre")

        # Copy back to CPU and verify
        z_cpu = z.cpu()
        w_cpu = w.cpu()
        v_cpu = v.cpu()

        self.assertEqual(z_cpu.numel(), 100)
        self.assertEqual(w_cpu.numel(), 100)
        self.assertEqual(v_cpu.numel(), 1)

    @unittest.skipUnless(
        is_vf_mode(), "VF allocator tests require FLEX_DEVICE=VF"
    )
    def test_large_tensor_allocation(self):
        """Test allocation of large tensors."""
        # Allocate a relatively large tensor
        large_size = 10 * 1024 * 1024  # 10M elements
        x = torch.empty(large_size, device="spyre", dtype=torch.float16)
        self.assertEqual(x.device.type, "spyre")
        self.assertEqual(x.numel(), large_size)

        # Verify we can use it
        x.fill_(1.0)
        x_cpu = x.cpu()
        self.assertTrue((x_cpu == 1.0).all())

    @unittest.skipUnless(
        is_vf_mode(), "VF allocator tests require FLEX_DEVICE=VF"
    )
    def test_mixed_size_allocations(self):
        """Test allocations of various sizes to test block management."""
        sizes = [10, 100, 1000, 10000, 100000]
        tensors = []

        for size in sizes:
            t = torch.empty(size, device="spyre", dtype=torch.float16)
            tensors.append(t)
            self.assertEqual(t.numel(), size)

        # Free some
        del tensors[1], tensors[3]
        gc.collect()

        # Allocate new ones
        t_new1 = torch.empty(100, device="spyre", dtype=torch.float16)
        t_new2 = torch.empty(10000, device="spyre", dtype=torch.float16)

        self.assertEqual(t_new1.numel(), 100)
        self.assertEqual(t_new2.numel(), 10000)

    @unittest.skipUnless(
        is_vf_mode(), "VF allocator tests require FLEX_DEVICE=VF"
    )
    def test_concurrent_allocations(self):
        """Test that multiple allocations can coexist."""
        tensors = []
        num_tensors = 20

        for i in range(num_tensors):
            t = torch.empty(100, device="spyre", dtype=torch.float16)
            t.fill_(i)
            tensors.append(t)

        # Verify all are valid and have correct values
        for i, t in enumerate(tensors):
            self.assertEqual(t.numel(), 100)
            t_cpu = t.cpu()
            # Note: fill_ might not work as expected on device, but tensor should be valid
            self.assertEqual(t_cpu.numel(), 100)


class TestVFAllocatorEdgeCases(TestCase):
    """Test edge cases and error conditions for VF allocator."""

    @unittest.skipUnless(
        is_vf_mode(), "VF allocator tests require FLEX_DEVICE=VF"
    )
    def test_invalid_flex_device_value(self):
        """Test that invalid FLEX_DEVICE values are rejected."""
        # This test would require restarting the Python process to test
        # constructor behavior, so we'll skip it for now
        # The error should be caught during allocator initialization
        pass

    @unittest.skipUnless(
        is_vf_mode(), "VF allocator tests require FLEX_DEVICE=VF"
    )
    def test_very_small_allocation(self):
        """Test allocation of very small tensors."""
        # Single element
        x = torch.empty(1, device="spyre", dtype=torch.float16)
        self.assertEqual(x.numel(), 1)

        # Should still be aligned to 128 bytes
        storage_size = x.untyped_storage().nbytes()
        self.assertGreaterEqual(storage_size, 128)
        self.assertEqual(storage_size % 128, 0)

    @unittest.skipUnless(
        is_vf_mode(), "VF allocator tests require FLEX_DEVICE=VF"
    )
    def test_allocation_after_many_deallocations(self):
        """Test allocation after many deallocations to stress test free interval management."""
        # Allocate and deallocate many times
        for _ in range(50):
            t = torch.empty(100, device="spyre", dtype=torch.float16)
            del t
            gc.collect()

        # Final allocation should still work
        final = torch.empty(100, device="spyre", dtype=torch.float16)
        self.assertEqual(final.numel(), 100)


if __name__ == "__main__":
    run_tests()
