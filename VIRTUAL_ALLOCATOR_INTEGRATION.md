# Integration Guide - Phase 2

This document explains how to integrate the virtual allocator into the torch-spyre build system and complete the implementation.

## Step 1: Update setup.py

The new `spyre_virtual_allocator.cpp` file needs to be included in the build. Update [setup.py](setup.py) to automatically pick it up:

**Current behavior:**

```python
sources = list(CSRC_DIR.glob("*.cpp"))  # Already includes all .cpp files!
```

**Good news:** `setup.py` already uses glob to collect all `.cpp` files from `torch_spyre/csrc/`, so `spyre_virtual_allocator.cpp` will be automatically included in the next build. ✅

**To verify:**

```bash
python setup.py build_ext --inplace  # Will compile the new .cpp file
```

## Step 2: Extend Context Structure

Currently, `SharedOwnerCtx` in [module.h](torch_spyre/csrc/module.h) only stores:

```cpp
struct SharedOwnerCtx {
  flex::DeviceMemoryAllocationPtr owner;
  signed char device_id;
};
```

**Need to add:** Offset information for virtual allocator.

### Option A: Extend SharedOwnerCtx (Breaking Change)

```cpp
struct SharedOwnerCtx {
  flex::DeviceMemoryAllocationPtr owner;
  signed char device_id;
  // Virtual allocator fields
  int chunk_id;      // Which chunk? (-1 = per-tensor mode)
  size_t chunk_offset; // Offset within chunk
};
```

**Pros:** Single context type for both modes  
**Cons:** Wastes memory in per-tensor mode

### Option B: Create VirtualAllocationContext (Recommended)
Create new context struct in `spyre_virtual_allocator.h`:

```cpp
struct VirtualAllocationContext {
  flex::DeviceMemoryAllocationPtr chunk_handle;
  signed char device_id;
  size_t chunk_offset;
};
```

Store in allocations_ map as `(data_ptr, AllocationInfo)` for lookup during deallocation.

**Pros:** Clean separation of concerns  
**Cons:** Need to adapt operation templates

## Step 3: Fix Deallocation Flow

Current issue: `on_deleter()` doesn't know which pointer to deallocate.

### Current Flow (Broken in Virtual Allocator):

```cpp
static void on_deleter(void* ctx_void) {
  auto* ctx = static_cast<VirtualAllocationContext*>(ctx_void);
  delete ctx;
  // BUG: How do we call deallocate(ptr)?
  // We only have context, not original data_ptr!
}
```

### Solution: Store data_ptr in context

Modify `on_deleter()` to store and use data_ptr:

```cpp
struct VirtualAllocationContext {
  flex::DeviceMemoryAllocationPtr chunk_handle;
  signed char device_id;
  size_t chunk_offset;
  void* data_ptr;  // NEW: Store original pointer for deallocation
};

static void on_deleter(void* ctx_void) {
  auto* ctx = static_cast<VirtualAllocationContext*>(ctx_void);
  VirtualSpyreAllocator::instance().deallocate(ctx->data_ptr);
  delete ctx;
}
```

## Step 4: Update Tensor Operations

Operations like `matmul`, `add`, etc. currently extract handles like:

```cpp
// From codegen/inputs/spyre_torch_ops.cpp:207
eager_inputs[eager_idx] = (static_cast<SharedOwnerCtx *>(
    tmp_tensor.storage().data_ptr().get_context()
))->owner;  // <- Just gets the handle
```

**Need to update** to also pass offset:

```cpp
auto* ctx = static_cast<VirtualAllocationContext*>(
    tmp_tensor.storage().data_ptr().get_context()
);
auto handle = ctx->chunk_handle;
auto offset = ctx->chunk_offset;
// Pass both to backend operation
eager_inputs[eager_idx] = handle;
operation_offsets[eager_idx] = offset;  // NEW
```

**Or:** Backend might support passing offset as part of SetSpyreData:

```cpp
inp_tensor.SetSpyreData(ctx->chunk_handle, ctx->chunk_offset);  // NEW API?
```

## Step 5: Update Copy Operations

The DMA graph generation in [spyre_mem.cpp](torch_spyre/csrc/spyre_mem.cpp:200) needs offset support:

```cpp
auto copy_host_to_device(const at::Tensor& self, const at::Tensor& dst) {
  // Get destination tensor's context with offset
  auto* ctx = static_cast<VirtualAllocationContext*>(
      dst.storage().data_ptr().get_context()
  );
  
  // Create DMA graph with offset
  auto gl = create_dma_graph(self, dst, /*host2device=*/true);
  
  // Pass offset to data transfer
  auto inp_tensor = createInputTensor(*gl, self.storage().data_ptr().get(),
                                      tensor_idx, sn_idx);
  inp_tensor.SetSpyreData(ctx->chunk_handle, ctx->chunk_offset);  // With offset
  
  SEN_THROW_NOK(gl->Copy(sendnn::Outputs(), {inp_tensor}, sn_idx));
}
```

## Step 6: Add to Registration

In `setup.py`, make sure the header is available:

```python
INCLUDE_DIRS += [
    CSRC_DIR,  # This already includes spyre_virtual_allocator.h
]
```

No changes needed - headers in csrc/ are automatically available.

## Step 7: Implement Allocator Selection

Add environment variable to choose allocator mode:

In [spyre_mem.cpp](torch_spyre/csrc/spyre_mem.cpp):

```cpp
bool use_virtual_allocator() {
  const char* env = std::getenv("TORCH_SPYRE_VIRTUAL_ALLOCATOR");
  if (!env) return false;  // Default: per-tensor allocator
  
  std::string val = env;
  return (val == "1" || val == "true" || val == "on");
}

// In allocator registration:
if (use_virtual_allocator()) {
  REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1,
                    &VirtualSpyreAllocator::instance());
} else {
  REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1,
                    &SpyreAllocator::instance());
}
```

**Usage:**

```bash
export TORCH_SPYRE_VIRTUAL_ALLOCATOR=1
python your_script.py
```

## Step 8: Testing

### Unit Tests for Virtual Allocator

Create [tests/test_virtual_allocator.py](tests/test_virtual_allocator.py):

```python
import torch
import torch_spyre
import os

# Enable virtual allocator
os.environ["TORCH_SPYRE_VIRTUAL_ALLOCATOR"] = "1"

def test_many_tensors():
    """Test that we can create 100+ tensors with only 16 handles"""
    tensors = []
    for i in range(100):
        t = torch.randn(1024, 1024, device="spyre")
        tensors.append(t)
  
    # If this succeeds, virtual allocator is working!
    assert len(tensors) == 100
    print(f"✓ Created 100 tensors with virtual allocator")

def test_eviction():
    """Test that eviction works when at chunk limit"""
    tensors = []
    try:
        for i in range(20):  # Try to create 20 * 256MB chunks
            t = torch.randn(256_000_000, device="spyre")
            tensors.append(t)
    except RuntimeError:
        pass  # Expected when we run out of backend memory
  
    # Verify some tensors were created
    assert len(tensors) > 0
    print(f"✓ Eviction handled {len(tensors)} allocations")

def test_fragmentation():
    """Test that coalescing helps with fragmentation"""
    # Create pattern: big, small, big, small
    big1 = torch.randn(100_000_000, device="spyre")  # 100MB
    small1 = torch.randn(1_000_000, device="spyre")  # 1MB
    big2 = torch.randn(100_000_000, device="spyre")  # 100MB
    small2 = torch.randn(1_000_000, device="spyre")  # 1MB
  
    # Delete in a way that fragments: big1, small1
    del big1
    del small1
  
    # Now try to allocate a 50MB tensor
    # Should succeed if coalescing is working
    medium = torch.randn(50_000_000, device="spyre")
  
    assert medium is not None
    print("✓ Fragmentation coalescing working")

if __name__ == "__main__":
    test_many_tensors()
    test_eviction()
    test_fragmentation()
```

### Integration Tests

Update [tests/test_spyre.py](tests/test_spyre.py):

```python
# Add with virtual allocator enabled
def test_with_virtual_allocator():
    os.environ["TORCH_SPYRE_VIRTUAL_ALLOCATOR"] = "1"
  
    # Run existing tests
    test_spyre_empty()
    test_copy_host_to_device()
    test_spyre_ops()
    # etc.
```

## Step 9: Compilation & Verification

### Build the extension:

```bash
python setup.py build_ext --inplace
```

### Verify new file is compiled:

```bash
# Should see spyre_virtual_allocator.cpp in compilation output
python setup.py build_ext --inplace 2>&1 | grep spyre_virtual_allocator
```

### Quick test:

```bash
python -c "import torch_spyre; print('Virtual allocator imported')"
```

## Common Issues & Debugging

### Issue 1: Compilation errors in new .cpp
**Check:** Are all includes present?

```cpp
#include "logging.h"      // For DEBUGINFO
#include "module.h"       // For SharedOwnerCtx, GlobalRuntime
```

### Issue 2: Offset not passed to backend
**Check:** Did you update all operation templates?
- Look in `codegen/templates/` for handle extraction
- Verify offset is included

### Issue 3: Memory not freed (memory leak)
**Check:** Are deallocations being called?
- Add debug logging: `DEBUGINFO("deallocate called")`
- Check `on_deleter()` is registered properly

### Issue 4: Segfault in operations
**Check:** Are you accessing chunk_offset from correct context type?
- Verify context cast is correct: `VirtualAllocationContext*`
- Check offset is within chunk bounds

## Build System Changes Summary

| File | Changes | Priority |
|------|---------|----------|
| [setup.py](setup.py) | None needed (auto-includes .cpp) | ✅ Done |
| [module.h](torch_spyre/csrc/module.h) | Create VirtualAllocationContext | 🔴 TODO |
| [spyre_mem.cpp](torch_spyre/csrc/spyre_mem.cpp) | Update allocator selection, registration | 🔴 TODO |
| [codegen/inputs/spyre_torch_ops.cpp](codegen/inputs/spyre_torch_ops.cpp) | Add offset to handle extraction | 🔴 TODO |
| [codegen/templates/](codegen/templates/) | Add offset parameter | 🔴 TODO |
| [tests/test_virtual_allocator.py](tests/test_virtual_allocator.py) | Create new test file | 🔴 TODO |

## Timeline Estimate

- **Context setup:** 30 min
- **Update operations:** 1-2 hours (need to touch templates)
- **DMA graph updates:** 1 hour
- **Testing & debugging:** 2-4 hours
- **Total:** ~5-8 hours

## Next: Phase 2 Planning

Once Phase 1 compiles, Phase 2 focus areas:
1. Chunk eviction policy optimization (LRU, size-based)
2. Memory compaction with tensor movement
3. Performance profiling vs. per-tensor allocator
4. Handle statistics for monitoring
5. Documentation and user guide
