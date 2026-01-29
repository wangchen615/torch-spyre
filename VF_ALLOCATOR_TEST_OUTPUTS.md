# VF Allocator Test Output Examples

This document shows what test output would look like when DOOM mode is enabled.

## C++ Unit Tests (✅ Works Now)

```
$ cd torch_spyre/csrc && FLEX_DEVICE=VF ./test_vf_allocator

Running VF Allocator C++ Unit Tests
====================================

Running FreeIntervalOrdering... PASSED
Running BlockInfoCreation... PASSED
Running SegmentInfoCreation... PASSED
Running AlignmentCalculation... PASSED
Running FreeIntervalMerging... PASSED

Results: 5 passed, 0 failed
```

## Realistic Pattern Test (Expected when DOOM mode enabled)

```
$ FLEX_DEVICE=VF python tests/test_vf_allocator_realistic_pattern.py

======================================================================
  VF Allocator Realistic Allocation Pattern Test
======================================================================


======================================================================
  Phase 1: Initial Allocations
======================================================================

allocate tensor a = torch.tensor([0], dtype=torch.float16, device='spyre')
  ✓ a: torch.Size([1]), device=spyre:0, dtype=torch.float16
    storage: 128 bytes

allocate tensor b = torch.tensor([0.], dtype=torch.float16, device='spyre')
  ✓ b: torch.Size([1]), device=spyre:0, dtype=torch.float16
    storage: 128 bytes

allocate tensor c = torch.tensor([1, 2], dtype=torch.float16, device='spyre')
  ✓ c: torch.Size([2]), device=spyre:0, dtype=torch.float16
    storage: 128 bytes

✓ Phase 1 complete: 3 tensors allocated (a, b, c)

======================================================================
  Phase 2: Selective Deallocation
======================================================================

deallocate tensor b (move to CPU)
  ✓ b moved to CPU: device=cpu
  ✓ garbage collection completed

======================================================================
  Phase 3: Reuse Freed Space
======================================================================

allocate tensor d = torch.tensor([7, 7], dtype=torch.float16, device='spyre')
  ✓ d: torch.Size([2]), device=spyre:0, dtype=torch.float16
    storage: 128 bytes
  ℹ d reuses freed space from b

======================================================================
  Phase 4: Reallocation with Different Size
======================================================================

reallocate tensor d = torch.tensor([1, 9, 8, 4], dtype=torch.float16, device='spyre')
  ✓ d: torch.Size([4]), device=spyre:0, dtype=torch.float16
    storage: 128 bytes
  ℹ old d deallocated, new d allocated
  ✓ garbage collection completed

======================================================================
  Phase 5: Large Tensor Allocation
======================================================================

allocate tensor e = torch.tensor(4x16 matrix, dtype=torch.float16, device='spyre')
  ✓ e: torch.Size([4, 16]), device=spyre:0, dtype=torch.float16
    storage: 256 bytes (64 elements × 2 bytes)

======================================================================
  Phase 6: Bulk Deallocation
======================================================================

deallocate tensors a, c, d
  a: torch.Size([1]) → deleting
  c: torch.Size([2]) → deleting
  d: torch.Size([4]) → deleting
  ✓ garbage collection completed
  ℹ multiple free intervals created: from a, c, d deallocations
  ℹ adjacent free intervals should be merged by allocator

======================================================================
  Phase 7: Memory Reuse from Freed Blocks
======================================================================

allocate tensor k = torch.tensor([0.1, 0.2], dtype=torch.float16, device='spyre')
  ✓ k: torch.Size([2]), device=spyre:0, dtype=torch.float16
    storage: 128 bytes
  ℹ reuses freed space

allocate tensor j = torch.tensor([1, 2], dtype=torch.float16, device='spyre')
  ✓ j: torch.Size([2]), device=spyre:0, dtype=torch.float16
    storage: 128 bytes
  ℹ reuses freed space

allocate tensor l = torch.tensor([7, 0, 4, 9], dtype=torch.float16, device='spyre')
  ✓ l: torch.Size([4]), device=spyre:0, dtype=torch.float16
    storage: 128 bytes
  ℹ reuses freed space

allocate tensor f = torch.tensor([6, 6, 6], dtype=torch.float16, device='spyre')
  ✓ f: torch.Size([3]), device=spyre:0, dtype=torch.float16
    storage: 128 bytes
  ℹ reuses freed space

======================================================================
  Phase 8: Final Verification
======================================================================

Verify all tensors are still valid and on spyre device:
  ✓ e: torch.Size([4, 16]), storage=256 bytes (aligned)
  ✓ k: torch.Size([2]), storage=128 bytes (aligned)
  ✓ j: torch.Size([2]), storage=128 bytes (aligned)
  ✓ l: torch.Size([4]), storage=128 bytes (aligned)
  ✓ f: torch.Size([3]), storage=128 bytes (aligned)

======================================================================
  Test Completed Successfully!
======================================================================

✓ All allocation and deallocation operations completed successfully
✓ All tensors remain valid and on spyre device
✓ Memory reuse and free interval merging verified
✓ 128-byte alignment maintained throughout
```

## Pytest Suite (Expected when DOOM mode enabled)

```
$ FLEX_DEVICE=VF python -m pytest tests/test_vf_allocator.py -v

========================== test session starts ==========================
platform linux -- Python 3.12.9, pytest-9.0.1, pluggy-1.6.0
rootdir: /home/chenw615/dt-inductor/torch-spyre, configfile: pyproject.toml
collected 18 items

tests/test_vf_allocator.py::TestVFAllocator::test_vf_mode_detection PASSED
tests/test_vf_allocator.py::TestVFAllocator::test_basic_allocation PASSED
tests/test_vf_allocator.py::TestVFAllocator::test_allocation_alignment PASSED
tests/test_vf_allocator.py::TestVFAllocator::test_memory_reuse PASSED
tests/test_vf_allocator.py::TestVFAllocator::test_segment_creation PASSED
tests/test_vf_allocator.py::TestVFAllocator::test_multiple_allocations_in_segment PASSED
tests/test_vf_allocator.py::TestVFAllocator::test_zero_size_allocation PASSED
tests/test_vf_allocator.py::TestVFAllocator::test_different_dtypes PASSED
tests/test_vf_allocator.py::TestVFAllocator::test_sequential_allocation_deallocation PASSED
tests/test_vf_allocator.py::TestVFAllocator::test_interleaved_allocations PASSED
tests/test_vf_allocator.py::TestVFAllocator::test_free_interval_merging PASSED
tests/test_vf_allocator.py::TestVFAllocator::test_tensor_operations_with_vf_allocator PASSED
tests/test_vf_allocator.py::TestVFAllocator::test_large_tensor_allocation PASSED
tests/test_vf_allocator.py::TestVFAllocator::test_mixed_size_allocations PASSED
tests/test_vf_allocator.py::TestVFAllocator::test_concurrent_allocations PASSED
tests/test_vf_allocator.py::TestVFAllocator::test_realistic_allocation_pattern PASSED

tests/test_vf_allocator.py::TestVFAllocatorEdgeCases::test_very_small_allocation PASSED
tests/test_vf_allocator.py::TestVFAllocatorEdgeCases::test_allocation_after_many_deallocations PASSED

========================== 18 passed in 12.34s ==========================
```

## Standalone Test Suite (Expected when DOOM mode enabled)

```
$ FLEX_DEVICE=VF python tests/test_vf_allocator_standalone.py -v

test_allocation_after_many_deallocations (__main__.TestVFAllocatorStandalone) ... ok
test_allocation_alignment (__main__.TestVFAllocatorStandalone) ... ok
test_basic_allocation (__main__.TestVFAllocatorStandalone) ... ok
test_concurrent_allocations (__main__.TestVFAllocatorStandalone) ... ok
test_different_dtypes (__main__.TestVFAllocatorStandalone) ... ok
test_free_interval_merging (__main__.TestVFAllocatorStandalone) ... ok
test_interleaved_allocations (__main__.TestVFAllocatorStandalone) ... ok
test_large_tensor_allocation (__main__.TestVFAllocatorStandalone) ... ok
test_memory_reuse (__main__.TestVFAllocatorStandalone) ... ok
test_mixed_size_allocations (__main__.TestVFAllocatorStandalone) ... ok
test_realistic_allocation_pattern (__main__.TestVFAllocatorStandalone) ... ok
test_sequential_allocation_deallocation (__main__.TestVFAllocatorStandalone) ... ok
test_tensor_operations_with_vf_allocator (__main__.TestVFAllocatorStandalone) ... ok
test_very_small_allocation (__main__.TestVFAllocatorStandalone) ... ok
test_vf_mode_detection (__main__.TestVFAllocatorStandalone) ... ok
test_zero_size_allocation (__main__.TestVFAllocatorStandalone) ... ok

----------------------------------------------------------------------
Ran 16 tests in 8.92s

OK
```

## Memory State During Realistic Pattern Test

### After Phase 1 (3 allocations)

```
Segment 0:
├─ Block 0: [offset=0, size=128] (tensor a)
├─ Block 1: [offset=128, size=128] (tensor b)
├─ Block 2: [offset=256, size=128] (tensor c)
└─ Free: [offset=384, size=8GB-384]
```

### After Phase 2 (deallocate b)

```
Segment 0:
├─ Block 0: [offset=0, size=128] (tensor a)
├─ Free: [offset=128, size=128] (deallocated b)
├─ Block 2: [offset=256, size=128] (tensor c)
└─ Free: [offset=384, size=8GB-384]
```

### After Phase 3 (allocate d, reuses freed b)

```
Segment 0:
├─ Block 0: [offset=0, size=128] (tensor a)
├─ Block 3: [offset=128, size=128] (tensor d, reused from b)
├─ Block 2: [offset=256, size=128] (tensor c)
└─ Free: [offset=384, size=8GB-384]
```

### After Phase 6 (deallocate a, c, d)

```
Segment 0:
├─ Free: [offset=0, size=128] (deallocated a)
├─ Free: [offset=128, size=128] (deallocated d)
├─ Free: [offset=256, size=128] (deallocated c)
├─ Block 1: [offset=384, size=256] (tensor e)
└─ Free: [offset=640, size=8GB-640]
```

### After Phase 6 (intervals merged)

```
Segment 0:
├─ Free: [offset=0, size=384] (merged intervals from a, d, c)
├─ Block 1: [offset=384, size=256] (tensor e)
└─ Free: [offset=640, size=8GB-640]
```

### After Phase 7 (allocate k, j, l, f)

```
Segment 0:
├─ Block 4: [offset=0, size=128] (tensor k, reused)
├─ Block 5: [offset=128, size=128] (tensor j, reused)
├─ Block 6: [offset=256, size=128] (tensor l, reused)
├─ Block 1: [offset=384, size=256] (tensor e)
├─ Block 7: [offset=640, size=128] (tensor f, reused)
└─ Free: [offset=768, size=8GB-768]
```

All blocks are properly aligned to 128-byte boundaries!
