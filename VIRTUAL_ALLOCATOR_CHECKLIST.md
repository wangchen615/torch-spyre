# Implementation Checklist - Phase 1 & 2

## Phase 1: Core Virtual Allocator (COMPLETED ✅)

### Header File: `spyre_virtual_allocator.h`
- [x] `FreeBlock` struct with offset and size
- [x] `AllocationInfo` struct with chunk_id, offset, size, is_pinned
- [x] `Chunk` struct with:
  - [x] Backend handle storage
  - [x] `try_allocate()` method (first-fit algorithm)
  - [x] `deallocate()` method with free block tracking
  - [x] `get_free_space()` and `get_largest_free_block()`
  - [x] `coalesce_free_blocks()` for fragmentation reduction
  - [x] `is_empty()` check for eviction
- [x] `VirtualSpyreAllocator` class with:
  - [x] Singleton pattern
  - [x] Configuration constants (CHUNK_SIZE=256MB, MAX_CHUNKS=16)
  - [x] `allocate(nbytes)` virtual method override
  - [x] `deallocate(ptr)` method
  - [x] `on_deleter()` static callback
  - [x] `raw_deleter()` returning nullptr
  - [x] `copy_data()` stub
  - [x] Statistics methods
  - [x] Full documentation with design rationale

### Implementation File: `spyre_virtual_allocator.cpp`
- [x] `Chunk::try_allocate()` with first-fit search
- [x] `Chunk::deallocate()` with free block management
- [x] `Chunk::coalesce_free_blocks()` with adjacency merging
- [x] `Chunk::get_free_space()` aggregation
- [x] `Chunk::get_largest_free_block()` for fragmentation metric
- [x] `VirtualSpyreAllocator::allocate()` full allocation flow
  - [x] Find existing chunk with space
  - [x] Allocate new chunk if needed
  - [x] Eviction attempt when at MAX_CHUNKS
  - [x] Allocation info tracking
  - [x] DataPtr creation with context
- [x] `VirtualSpyreAllocator::deallocate()` for freeing memory
- [x] `VirtualSpyreAllocator::on_deleter()` callback
- [x] Helper methods:
  - [x] `get_backend_allocator()`
  - [x] `allocate_new_chunk()`
  - [x] `find_chunk_with_space()`
  - [x] `try_evict_chunk()`
  - [x] `compact_chunk()`
  - [x] `remove_chunk()`
- [x] Statistics via `get_stats_string()`
- [x] `reset()` for testing
- [x] Error handling with TORCH_CHECK
- [x] Debug logging with DEBUGINFO

### Documentation
- [x] VIRTUAL_ALLOCATOR_PHASE1.md with design overview
- [x] VIRTUAL_ALLOCATOR_INTEGRATION.md with next steps

---

## Phase 2: Integration with Existing Code (TODO)

### A. Context Structure Update
- [ ] **module.h**: Create `VirtualAllocationContext` struct
  - [ ] Fields: chunk_handle, device_id, chunk_offset, data_ptr
  - [ ] Add static cast helper for safe conversion
  - [ ] Keep `SharedOwnerCtx` for backward compatibility

### B. Allocator Registration
- [ ] **spyre_mem.cpp**: Add allocator mode selection
  - [ ] Read `TORCH_SPYRE_VIRTUAL_ALLOCATOR` environment variable
  - [ ] Conditional registration based on mode
  - [ ] Add logging for which allocator is active
  - [ ] Update `spyre_empty()` to work with both modes

### C. Tensor Operation Updates
- [ ] **codegen/inputs/spyre_torch_ops.cpp**: Update handle extraction
  - [ ] Check context type and extract offset
  - [ ] Pass offset to backend operations
  - [ ] Add fallback for per-tensor mode (offset=0)
  
- [ ] **codegen/templates/*.jinja2**: Update templates
  - [ ] base.jinja2: Handle extraction with offset support
  - [ ] native_call.jinja2: Pass offset to operations
  - [ ] Any other templates using handle extraction

### D. Copy Operations
- [ ] **spyre_mem.cpp**: Update `copy_host_to_device()`
  - [ ] Extract offset from destination context
  - [ ] Pass offset to DMA setup
  - [ ] Verify offset is handled in tensor setup
  
- [ ] **spyre_mem.cpp**: Update `copy_device_to_host()`
  - [ ] Extract offset from source context
  - [ ] Pass offset to DMA setup
  - [ ] Verify reverse direction works

- [ ] **spyre_mem.cpp**: Implement `VirtualSpyreAllocator::copy_data()`
  - [ ] Handle device-to-device copies with offsets
  - [ ] Or verify it's not needed

### E. DMA Graph Generation
- [ ] **spyre_mem.cpp**: Review `create_dma_graph()`
  - [ ] Check if offset parameter needed
  - [ ] Update data transfer nodes if needed
  - [ ] Test with offset != 0

### F. Build System
- [ ] **setup.py**: Verify compilation includes new .cpp
  - [ ] Confirm in build output
  - [ ] Check no linker errors
  - [ ] Verify imports work

- [ ] **setup.py**: No changes needed (already uses glob pattern ✅)

---

## Phase 3: Testing (TODO)

### Unit Tests
- [ ] **tests/test_virtual_allocator.py**: Create new test file
  - [ ] Test allocate in empty allocator
  - [ ] Test deallocate and free block reuse
  - [ ] Test chunk allocation when needed
  - [ ] Test try_allocate algorithm (first-fit)
  - [ ] Test coalescing after deallocations
  - [ ] Test fragmentation metrics
  - [ ] Test statistics reporting

### Integration Tests
- [ ] **tests/test_spyre.py**: Add virtual allocator mode
  - [ ] Run all existing tests with VIRTUAL_ALLOCATOR=1
  - [ ] Verify no regression vs. per-tensor mode
  - [ ] Check handle count stays <= 16

- [ ] **tests/test_inductor_ops.py**: Test inductor operations
  - [ ] Test with virtual allocator enabled
  - [ ] Verify correctness of results
  - [ ] Check memory usage

### Stress Tests
- [ ] Create 100+ resident tensors (should not fail)
- [ ] Verify handle count metrics
- [ ] Test repeated allocation/deallocation cycles
- [ ] Test with different chunk sizes
- [ ] Test eviction under memory pressure

### Performance Tests
- [ ] Benchmark vs. per-tensor allocator
  - [ ] Allocation speed
  - [ ] Operation speed (should be same)
  - [ ] Memory fragmentation ratio
  - [ ] Handle pressure metrics

---

## Phase 4: Advanced Features (TODO)

### Memory Compaction
- [ ] Implement chunk compaction strategy
  - [ ] Identify movable allocations (non-pinned)
  - [ ] Move allocations to consolidate chunks
  - [ ] Free compacted chunks
  - [ ] Requires tracking allocation identity

### Eviction Policies
- [ ] Implement LRU eviction (beyond current simple strategy)
  - [ ] Track last access time per allocation
  - [ ] Prioritize older allocations for eviction
  - [ ] Consider tensor size in priority

- [ ] Implement size-aware eviction
  - [ ] Prefer evicting large chunks first
  - [ ] Consider ratio of free to allocated

### Statistics & Profiling
- [ ] Detailed profiling output
  - [ ] Allocation timeline
  - [ ] Eviction history
  - [ ] Failure reason tracking
  - [ ] Per-chunk statistics

- [ ] Integration with torch.utils.memory
  - [ ] Report to PyTorch memory tracking
  - [ ] Integrate with torch.cuda.memory_allocated() style API

### Environment Variable Configuration
- [ ] TORCH_SPYRE_CHUNK_SIZE (bytes, default 256MB)
- [ ] TORCH_SPYRE_MAX_CHUNKS (count, default 16)
- [ ] TORCH_SPYRE_ALLOCATOR_DEBUG (verbosity level)
- [ ] TORCH_SPYRE_ALLOCATION_MODE (virtual/per-tensor)

---

## Validation Checklist

### Correctness
- [ ] All tensor allocations succeed with virtual allocator
- [ ] All deallocations properly free space
- [ ] No memory leaks detected (valgrind/asan)
- [ ] Free blocks properly coalesced
- [ ] Handle count never exceeds 16

### Compatibility
- [ ] Existing tests pass with virtual allocator
- [ ] Results match per-tensor allocator
- [ ] No regression in operation speeds
- [ ] Works with all data types and shapes

### Robustness
- [ ] Handles zero-size allocations
- [ ] Handles allocation failures gracefully
- [ ] Handles eviction edge cases
- [ ] Handles concurrent access (if applicable)

### Documentation
- [ ] Code is well-commented
- [ ] Design rationale documented
- [ ] API documented with examples
- [ ] Known limitations documented
- [ ] Integration guide complete

---

## Known Issues & Deferred Items

### Critical (Must Fix)
1. **Context tracking** - Currently using SharedOwnerCtx without offset storage
   - Solution: Create VirtualAllocationContext
   - Blocks: Operations, copy operations

2. **on_deleter() callback** - Doesn't have data_ptr to deallocate
   - Solution: Store data_ptr in context
   - Blocks: Proper deallocation

### Important (Should Fix)
1. **copy_data() not implemented** - Device-to-device copies
   - Solution: Extract offset and handle offsets in DMA
   - Blocks: Full copy operation support

2. **Eviction policy too simple** - Only evicts empty chunks
   - Solution: Implement LRU or size-based eviction
   - Affects: Memory pressure scenarios

### Nice to Have (Can Defer)
1. **Chunk size configuration** - Hardcoded at 256MB
   - Solution: Environment variable
   - Affects: Performance tuning

2. **Detailed profiling** - Basic stats only
   - Solution: Add allocation timeline, history
   - Affects: Debugging and tuning

---

## Implementation Order (Recommended)

1. **Phase 1** ✅ DONE - Core allocator code written
2. **Phase 2a** - Create VirtualAllocationContext (15 min)
3. **Phase 2b** - Update allocator registration (20 min)
4. **Phase 2c** - Update handle extraction in operations (1-2 hours)
5. **Phase 2d** - Update copy operations (1 hour)
6. **Build & Test** - Verify compilation and basic functionality (1 hour)
7. **Phase 3** - Comprehensive testing (2-4 hours)
8. **Phase 4** - Advanced features as time permits

---

## Success Criteria

- [x] Phase 1: Core allocator code is complete and well-documented
- [ ] Phase 2: Integration is complete and builds without errors
- [ ] Phase 3: All tests pass with virtual allocator enabled
- [ ] Can create 100+ resident tensors without handle exhaustion
- [ ] No performance regression vs. per-tensor allocator
- [ ] Handle count metrics available for monitoring
- [ ] Ready for VF mode deployment

---

## Files Modified/Created

### Created
- [x] `torch_spyre/csrc/spyre_virtual_allocator.h` - Header (470 lines)
- [x] `torch_spyre/csrc/spyre_virtual_allocator.cpp` - Implementation (450 lines)
- [x] `VIRTUAL_ALLOCATOR_PHASE1.md` - Design documentation
- [x] `VIRTUAL_ALLOCATOR_INTEGRATION.md` - Integration guide
- [x] `VIRTUAL_ALLOCATOR_CHECKLIST.md` - This file

### To Modify
- [ ] `torch_spyre/csrc/module.h` - Add VirtualAllocationContext
- [ ] `torch_spyre/csrc/spyre_mem.cpp` - Allocator selection + registration
- [ ] `codegen/inputs/spyre_torch_ops.cpp` - Handle extraction with offset
- [ ] `codegen/templates/*.jinja2` - Template updates
- [ ] `tests/test_virtual_allocator.py` - New test file

---

## Contact & Questions

For questions about the implementation:
1. Review the detailed documentation in VIRTUAL_ALLOCATOR_PHASE1.md
2. Check code comments in .h and .cpp files
3. Refer to ALLOCATOR_CODE_REFERENCE.md for current allocator patterns
4. Test incrementally and review debug output via DEBUGINFO logs
