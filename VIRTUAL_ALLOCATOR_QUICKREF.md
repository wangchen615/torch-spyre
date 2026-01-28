# Virtual Allocator Quick Reference

## What Was Created

Two new files implementing the chunked virtual memory allocator for VF mode:

1. **`torch_spyre/csrc/spyre_virtual_allocator.h`** (470 lines)
   - Complete header with all class/struct definitions
   - Full documentation and design rationale

2. **`torch_spyre/csrc/spyre_virtual_allocator.cpp`** (450 lines)
   - All implementations with algorithms
   - Debug logging and error handling

3. **Documentation** (This folder):
   - `VIRTUAL_ALLOCATOR_PHASE1.md` - Design overview
   - `VIRTUAL_ALLOCATOR_INTEGRATION.md` - Integration steps
   - `VIRTUAL_ALLOCATOR_CHECKLIST.md` - Task tracking

## Key Classes

### `Chunk`
Represents one large backend allocation (one handle).

```cpp
Chunk {
  handle: DeviceMemoryAllocationPtr    // Backend handle (limited!)
  total_size: 256MB                    // Chunk size
  allocations: map<ptr, AllocationInfo> // Sub-allocations in chunk
  free_blocks: vector<FreeBlock>       // Free space tracking
  
  try_allocate(nbytes) -> offset
  deallocate(offset, size)
  coalesce_free_blocks()
  get_largest_free_block() -> size
}
```

### `VirtualSpyreAllocator`
Main allocator managing all chunks (max 16 for VF mode).

```cpp
VirtualSpyreAllocator : at::Allocator {
  chunks_: vector<Chunk>               // Max 16 chunks
  allocations_: map<ptr, AllocationInfo> // ptr -> allocation info
  total_allocated_: size_t             // Stats
  
  allocate(nbytes) -> DataPtr          // Main API
  deallocate(ptr)                       // Called via on_deleter
  get_stats_string() -> string         // Debugging
}
```

## Key Algorithms

### First-Fit Allocation (Chunk::try_allocate)
```
For each free block in free_blocks:
  If block.size >= nbytes:
    return block.offset
    remove/shrink block
Return -1 (no space)
```

### Coalescing (Chunk::coalesce_free_blocks)
```
1. Sort free_blocks by offset
2. Merge adjacent blocks (offset1 + size1 == offset2)
3. Reduces fragmentation
```

### Allocation Flow (VirtualSpyreAllocator::allocate)
```
1. Try find_chunk_with_space()
2. If none, allocate_new_chunk() (if < 16 chunks)
3. If at limit, try_evict_chunk()
4. Call chunk->try_allocate()
5. Track in allocations_ map
6. Return DataPtr with context
```

## Current Status

### ✅ Completed (Phase 1)
- Core allocator implementation
- Chunk and allocation management
- First-fit algorithm with coalescing
- Statistics and debugging
- Full documentation

### ⚠️ Needs Integration (Phase 2)
- **Context structure**: Need to track chunk offset
  - Create `VirtualAllocationContext` with offset field
  - Update allocator registration
  
- **Operations**: Extract handle and offset
  - Update codegen templates
  - Modify operation handle extraction
  
- **Copy operations**: Support offsets in DMA
  - Update copy_host_to_device()
  - Update copy_device_to_host()

## How to Use (After Integration)

### Enable Virtual Allocator
```bash
export TORCH_SPYRE_VIRTUAL_ALLOCATOR=1
python your_script.py
```

### Check Stats (in Python)
```python
from torch_spyre.csrc import VirtualSpyreAllocator
allocator = VirtualSpyreAllocator.instance()
print(allocator.get_stats_string())
# Output:
# VirtualSpyreAllocator Stats:
#   Chunks: 5 / 16
#   Allocations: 150
#   Total Allocated: 512MB
#   ...
```

## Important Constants

```cpp
CHUNK_SIZE = 256 * 1024 * 1024  // 256MB per chunk
MAX_CHUNKS = 16                  // VF mode limit
```

## Build Integration

### Automatic
- `setup.py` already globs all `.cpp` files → `spyre_virtual_allocator.cpp` included ✅
- Headers in `csrc/` are automatically available ✅

### Manual Test
```bash
cd /home/chenw615/dt-inductor/torch-spyre
python setup.py build_ext --inplace 2>&1 | grep spyre_virtual_allocator
# Should see compilation output for the new file
```

## Next Immediate Steps

1. **Create context struct** (15 min)
   - Add `VirtualAllocationContext` to module.h
   - Include chunk_offset field

2. **Update allocator registration** (20 min)
   - Add environment variable check in spyre_mem.cpp
   - Register virtual vs. per-tensor based on mode

3. **Test compilation** (10 min)
   - Run setup.py build_ext
   - Fix any compilation errors

4. **Run pytest** (10 min)
   - `python -m pytest tests/test_spyre.py -v`
   - Should pass with new allocator

## Design Benefits Summary

| Aspect | Per-Tensor | Virtual |
|--------|-----------|---------|
| **Handles** | 1 per tensor | 1 per chunk |
| **Max Tensors** | 12-16 | Unlimited (via chunks) |
| **Mode** | ✅ PF Mode | ❌ PF (now), ✅ VF (future) |
| **Fragmentation** | None | Managed by coalescing |
| **Complexity** | Simple | Medium |

## Debug Output

Enable full logging:
```bash
export TORCH_SPYRE_DEBUG=1
export TORCH_SPYRE_VIRTUAL_ALLOCATOR=1
python script.py 2>&1 | grep "VirtualSpyreAllocator"
```

Sample output:
```
VirtualSpyreAllocator::allocate - 1048576 bytes on device 0
VirtualSpyreAllocator::find_chunk_with_space - looking for 1048576 bytes
VirtualSpyreAllocator::find_chunk_with_space - found in chunk 0
Chunk::try_allocate - requesting 1048576 bytes
Chunk::try_allocate - success at offset 0
VirtualSpyreAllocator::allocate - allocated chunk=0 offset=0 size=1048576
```

## File Locations

```
torch-spyre/
├── torch_spyre/csrc/
│   ├── spyre_virtual_allocator.h       ← NEW
│   ├── spyre_virtual_allocator.cpp     ← NEW
│   ├── module.h                        ← TO MODIFY
│   ├── spyre_mem.cpp                   ← TO MODIFY
│   └── ...
├── codegen/
│   ├── inputs/spyre_torch_ops.cpp      ← TO MODIFY
│   ├── templates/                      ← TO MODIFY
│   └── ...
├── tests/
│   ├── test_spyre.py                   ← TO MODIFY
│   └── test_virtual_allocator.py       ← TO CREATE
└── VIRTUAL_ALLOCATOR_*.md              ← NEW (docs)
```

## Performance Expectations

### Allocation Speed
- Virtual allocator: Similar to per-tensor (one Chunk::try_allocate call)
- Operation speed: Same (just different handle source)

### Memory Efficiency
- Per-tensor: Zero fragmentation, but handle-limited
- Virtual: Some fragmentation, but unlimited tensors

### Coalescing Overhead
- Only happens on deallocation (not critical path)
- Reduces future allocation failures

## Common Pitfalls

1. ❌ Forgetting to update context struct
   - ✅ Must add offset field to VirtualAllocationContext

2. ❌ Not updating operation templates
   - ✅ All handle extractions need offset support

3. ❌ Mixing context types
   - ✅ Use consistent context type throughout

4. ❌ Not testing with 100+ tensors
   - ✅ Verify handle count <= 16 with stress test

## Success Indicators

After integration is complete:
- ✅ Builds without errors
- ✅ Can create 100+ tensors without failing
- ✅ All existing tests pass
- ✅ Handle count stays <= 16
- ✅ Results match per-tensor allocator
- ✅ Ready for VF mode

## References

- **Design Analysis**: ALLOCATOR_ANALYSIS.md (provided)
- **Code Examples**: ALLOCATOR_CODE_REFERENCE.md (provided)
- **This Allocator**: VIRTUAL_ALLOCATOR_PHASE1.md (in this folder)
- **Integration Guide**: VIRTUAL_ALLOCATOR_INTEGRATION.md (in this folder)
- **Checklist**: VIRTUAL_ALLOCATOR_CHECKLIST.md (in this folder)

---

**Phase 1 Complete!** ✅

Ready for Phase 2 integration work. Start with creating VirtualAllocationContext.
