# Control Block Stream Design Proposal

## Overview

This document proposes a structure for Control Block (CB) Streams that enables direct runtime interaction, dynamic shape handling, and efficient execution on the Spyre device.

## Executive Summary

This design replaces the current `GraphLoader`-based execution model with a more efficient `CBStream`-based approach. The key changes are:

- **New Components**: 4 new C++ files implementing CB streams, stream pool, cache, and runtime interface
- **Modified Components**: 7 existing files need updates to integrate CB streams
- **Key Benefits**: Direct runtime interaction, async execution, better memory management, dynamic shape support
- **Migration**: Phased approach over 7 phases, maintaining backward compatibility during transition

### Quick Reference: Files to Change

**New Files (4)**:
- `torch_spyre/csrc/cb_stream.h/cpp` - Core CB stream implementation
- `torch_spyre/csrc/cb_stream_pool.h/cpp` - Stream pool management
- `torch_spyre/csrc/cb_cache.h/cpp` - Control block caching
- `torch_spyre/csrc/cb_runtime_interface.h/cpp` - Direct runtime interface

**Modified Files (7)**:
- `torch_spyre/csrc/spyre_hooks.cpp` - Stream hook integration
- `torch_spyre/csrc/spyre_mem.cpp/h` - VF allocator stream support
- `torch_spyre/csrc/module.cpp` - Replace launchKernel()
- `torch_spyre/csrc/module.h` - Add GlobalCBStreamPool
- `torch_spyre/_inductor/runtime/kernel_runner.py` - Use CB streams
- `codegen/templates/base.jinja2` - Replace GraphLoader
- `codegen/inputs/spyre_torch_ops.cpp` - Replace GraphLoader

## Core Components

### 1. ControlBlock

A pre-compiled operation unit that can handle a range of shapes up to a maximum size.

```cpp
class ControlBlock {
public:
    // Unique identifier for this control block
    std::string id;
    
    // Operation this CB represents (e.g., "aten::mm", "aten::add")
    std::string operation;
    
    // Maximum shape this CB can handle
    std::vector<int64_t> max_shape;
    
    // Compiled binary/artifact for execution
    std::shared_ptr<CompiledArtifact> artifact;
    
    // Runtime handle for this CB
    std::shared_ptr<RuntimeCBHandle> runtime_handle;
    
    // Metadata for shape adaptation
    struct ShapeMetadata {
        std::vector<int64_t> base_shape;
        std::vector<int64_t> stride_info;
        bool supports_masking;
        bool supports_tiling;
    } shape_metadata;
    
    // Execute this CB with given inputs/outputs
    Status execute(
        const std::vector<sendnn::ConstTensor>& inputs,
        const std::vector<sendnn::Tensor>& outputs,
        const std::vector<int64_t>& actual_shape,
        const MaskInfo& mask = MaskInfo::none()
    );
};
```

### 2. CBStream

A sequence of Control Blocks that execute asynchronously.

```cpp
class CBStream {
public:
    // Stream identifier (maps to c10::Stream)
    uint64_t stream_id;
    
    // Device this stream belongs to
    c10::Device device;
    
    // Queue of control blocks to execute
    std::deque<CBExecutionItem> execution_queue;
    
    // State management
    enum class State {
        OPEN,        // Can append more CBs
        FINALIZED,   // No more appends, ready to execute
        EXECUTING,   // Currently executing
        COMPLETED,   // Execution finished
        ERROR        // Error state
    } state;
    
    // Memory tracking for VF allocator integration
    struct MemoryTracking {
        std::set<void*> allocated_blocks;  // Blocks allocated on this stream
        std::vector<int> preferred_segments; // Preferred segments for this stream
        std::shared_ptr<StreamMemoryContext> mem_ctx;
    } memory;
    
    // Synchronization
    std::shared_ptr<StreamEvent> completion_event;
    
    // Methods
    void append(const ControlBlock& cb, 
                const std::vector<sendnn::ConstTensor>& inputs,
                const std::vector<sendnn::Tensor>& outputs,
                const std::vector<int64_t>& shape);
    
    void finalize();  // Mark stream as ready for execution
    
    Status execute();  // Launch async execution
    
    void synchronize();  // Wait for completion
    
    bool isReady() const;  // Check if ready to execute
    
    // Dynamic shape composition
    Status composeForShape(
        const std::string& operation,
        const std::vector<int64_t>& target_shape,
        const std::vector<sendnn::ConstTensor>& inputs,
        const std::vector<sendnn::Tensor>& outputs
    );
};
```

### 3. CBExecutionItem

Represents a single CB execution within a stream.

```cpp
struct CBExecutionItem {
    std::shared_ptr<ControlBlock> cb;
    std::vector<sendnn::ConstTensor> inputs;
    std::vector<sendnn::Tensor> outputs;
    std::vector<int64_t> actual_shape;
    MaskInfo mask;  // For handling oversized CBs
    
    // Execution metadata
    uint64_t execution_id;
    std::chrono::time_point<std::chrono::steady_clock> enqueue_time;
};
```

### 4. MaskInfo

Handles cases where CB is larger than needed.

```cpp
struct MaskInfo {
    bool is_active;
    std::vector<std::pair<int64_t, int64_t>> mask_ranges;  // [start, end) per dim
    
    static MaskInfo none() { return {false, {}}; }
    
    static MaskInfo create(
        const std::vector<int64_t>& cb_shape,
        const std::vector<int64_t>& target_shape
    );
};
```

## Caching Strategy

### Control Block Cache

```cpp
class ControlBlockCache {
public:
    // Key: (operation, max_shape, dtype)
    using CacheKey = std::tuple<std::string, std::vector<int64_t>, c10::ScalarType>;
    
    std::optional<std::shared_ptr<ControlBlock>> get(
        const std::string& operation,
        const std::vector<int64_t>& max_shape,
        c10::ScalarType dtype
    );
    
    void put(
        const std::string& operation,
        const std::vector<int64_t>& max_shape,
        c10::ScalarType dtype,
        std::shared_ptr<ControlBlock> cb
    );
    
    // Get or compile a CB
    std::shared_ptr<ControlBlock> getOrCompile(
        const std::string& operation,
        const std::vector<int64_t>& max_shape,
        const std::vector<int64_t>& target_shape,
        c10::ScalarType dtype
    );
    
private:
    std::unordered_map<CacheKey, std::shared_ptr<ControlBlock>, CacheKeyHash> cache_;
    std::mutex mutex_;
};
```

### Stream Pool

```cpp
class CBStreamPool {
public:
    // Get a stream from the pool (or create new one)
    std::shared_ptr<CBStream> acquire(c10::Device device);
    
    // Return stream to pool (after synchronization)
    void release(std::shared_ptr<CBStream> stream);
    
    // Get default stream for device
    std::shared_ptr<CBStream> getDefault(c10::Device device);
    
private:
    std::unordered_map<c10::Device, std::vector<std::shared_ptr<CBStream>>> pool_;
    std::unordered_map<c10::Device, std::shared_ptr<CBStream>> default_streams_;
    std::mutex mutex_;
};
```

## Integration with VF Allocator

### Stream-Aware Memory Allocation

```cpp
// Extend SpyreAllocator to support stream-aware allocation
class SpyreAllocator {
    // ... existing code ...
    
    // Allocate memory tied to a specific stream
    at::DataPtr allocateForStream(
        size_t nbytes,
        c10::Stream stream,
        c10::Device device
    ) {
        // Get CB stream from stream ID
        auto cb_stream = getCBStreamFromC10Stream(stream);
        
        // Prefer segments associated with this stream
        auto preferred_segments = cb_stream->memory.preferred_segments;
        
        // Allocate with stream context
        auto data_ptr = vf_allocation_with_preferences(
            nbytes, device, preferred_segments
        );
        
        // Track allocation on stream
        cb_stream->memory.allocated_blocks.insert(data_ptr.get());
        recordAllocationOnStream(data_ptr.get(), stream);
        
        return data_ptr;
    }
    
    // Track memory lifetime tied to stream
    void recordAllocationOnStream(void* ptr, c10::Stream stream);
    
    // Defer deallocation until stream completes
    void deferDeallocationUntilStreamComplete(void* ptr, c10::Stream stream);
};
```

## Dynamic Shape Composition

### Shape Composition Logic

```cpp
Status CBStream::composeForShape(
    const std::string& operation,
    const std::vector<int64_t>& target_shape,
    const std::vector<sendnn::ConstTensor>& inputs,
    const std::vector<sendnn::Tensor>& outputs
) {
    // 1. Get or compile CB for max practical shape
    auto max_shape = calculateMaxPracticalShape(operation, target_shape);
    auto cb = cb_cache_->getOrCompile(operation, max_shape, target_shape, dtype);
    
    // 2. Check if CB fits target shape
    if (shapeFits(cb->max_shape, target_shape)) {
        // CB is larger or equal - may need masking
        if (shapeEquals(cb->max_shape, target_shape)) {
            // Perfect fit
            append(*cb, inputs, outputs, target_shape);
        } else {
            // CB is larger - use masking
            auto mask = MaskInfo::create(cb->max_shape, target_shape);
            append(*cb, inputs, outputs, target_shape, mask);
        }
    } else {
        // CB is smaller - need to compose multiple CBs
        auto composition = composeMultipleCBs(cb, target_shape);
        for (const auto& item : composition) {
            append(*item.cb, item.inputs, item.outputs, item.shape);
        }
    }
    
    return Status::OK();
}
```

## Runtime Interface

### Direct Runtime Interaction

```cpp
class CBRuntimeInterface {
public:
    // Execute a CB stream directly on runtime (bypassing GraphLoader)
    Status executeStream(
        std::shared_ptr<CBStream> stream,
        std::shared_ptr<flex::Runtime> runtime
    );
    
    // Execute a single CB
    Status executeControlBlock(
        const ControlBlock& cb,
        const std::vector<sendnn::ConstTensor>& inputs,
        const std::vector<sendnn::Tensor>& outputs,
        const std::vector<int64_t>& shape,
        const MaskInfo& mask,
        std::shared_ptr<flex::Runtime> runtime
    );
    
private:
    // Direct runtime calls (no GraphLoader)
    Status launchCBOnRuntime(
        const ControlBlock& cb,
        const ExecutionParams& params,
        std::shared_ptr<flex::Runtime> runtime
    );
};
```

## Stream State Machine

```text
OPEN → [append CBs] → FINALIZED → [execute()] → EXECUTING → COMPLETED
  ↓                                                              ↑
  └────────────────────────────────────────────────────────────┘
  (can reset and reuse)
```

## Usage Example

```cpp
// Get a stream
auto stream = stream_pool->acquire(device);

// Compose operation for dynamic shape
stream->composeForShape(
    "aten::mm",
    {batch_size, M, N},  // Dynamic shape
    {input1, input2},
    {output}
);

// Finalize and execute
stream->finalize();
stream->execute();  // Async

// Synchronize if needed
stream->synchronize();

// Release stream
stream_pool->release(stream);
```

## Key Design Decisions

1. **Two-Level Caching**: Cache both Control Blocks (compiled operations) and potentially composed streams
2. **Open vs Finalized**: Streams can be open (appendable) or finalized (ready to execute)
3. **Stream Pool**: Reuse streams to avoid allocation overhead
4. **Memory Integration**: Streams track their memory allocations for proper lifetime management
5. **Direct Runtime**: Bypass GraphLoader for direct runtime interaction
6. **Shape Flexibility**: Support masking (CB too large) and composition (CB too small)

## Implementation Details

### File Structure and Locations

#### New Files to Create

1. **`torch_spyre/csrc/cb_stream.h`** and **`torch_spyre/csrc/cb_stream.cpp`**
   - Core CB stream implementation
   - Contains `CBStream`, `ControlBlock`, `CBExecutionItem`, `MaskInfo` classes
   - Location: `torch_spyre/csrc/`

2. **`torch_spyre/csrc/cb_stream_pool.h`** and **`torch_spyre/csrc/cb_stream_pool.cpp`**
   - Stream pool implementation
   - Contains `CBStreamPool` class
   - Location: `torch_spyre/csrc/`

3. **`torch_spyre/csrc/cb_cache.h`** and **`torch_spyre/csrc/cb_cache.cpp`**
   - Control block cache implementation
   - Contains `ControlBlockCache` class
   - Location: `torch_spyre/csrc/`

4. **`torch_spyre/csrc/cb_runtime_interface.h`** and **`torch_spyre/csrc/cb_runtime_interface.cpp`**
   - Direct runtime interface (bypassing GraphLoader)
   - Contains `CBRuntimeInterface` class
   - Location: `torch_spyre/csrc/`

#### Files to Modify

1. **`torch_spyre/csrc/spyre_hooks.cpp`**
   - **Changes**: Update stream methods to use `CBStreamPool`
   - **Lines**: 117-150 (stream-related methods)
   - **Details**: Replace hardcoded stream ID 0 with actual CB stream retrieval

2. **`torch_spyre/csrc/spyre_mem.cpp`** and **`torch_spyre/csrc/spyre_mem.h`**
   - **Changes**: Add stream-aware allocation methods to `SpyreAllocator`
   - **Location**: Around line 416 (SpyreAllocator class)
   - **Details**: 
     - Add `allocateForStream()` method
     - Add `recordAllocationOnStream()` method
     - Add `deferDeallocationUntilStreamComplete()` method
     - Modify `vf_allocation()` to accept optional stream parameter

3. **`torch_spyre/csrc/module.cpp`**
   - **Changes**: Replace `launchKernel()` to use CB streams instead of GraphLoader
   - **Lines**: 94-208 (launchKernel function)
   - **Details**: Refactor to use `CBRuntimeInterface` instead of `GraphLoader`

4. **`torch_spyre/csrc/module.h`**
   - **Changes**: Add CB stream pool as global singleton (similar to GlobalRuntime)
   - **Location**: After GlobalRuntime class (around line 53)
   - **Details**: Add `GlobalCBStreamPool` class

5. **`torch_spyre/_inductor/runtime/kernel_runner.py`**
   - **Changes**: Update `SpyreSDSCKernelRunner.run()` to use CB streams
   - **Lines**: 36-40
   - **Details**: Replace `launch_kernel()` call with CB stream execution

6. **`codegen/templates/base.jinja2`**
   - **Changes**: Replace GraphLoader usage with CB stream composition
   - **Lines**: 135-207 (GraphLoader cache and execution)
   - **Details**: Use `composeForShape()` instead of GraphLoader

7. **`codegen/inputs/spyre_torch_ops.cpp`**
   - **Changes**: Replace GraphLoader usage with CB streams
   - **Lines**: Various (all GraphLoader usages)
   - **Details**: Similar to base.jinja2 changes

### Implementation Details by Component

#### 1. ControlBlock Implementation

**File**: `torch_spyre/csrc/cb_stream.h` / `cb_stream.cpp`

```cpp
// cb_stream.h
#include <string>
#include <vector>
#include <memory>
#include <sendnn/tensor/sentensor_info.hpp>
#include <flex/runtime.hpp>

namespace spyre {

// Forward declarations
struct MaskInfo;
class RuntimeCBHandle;

// Compiled artifact wrapper (replaces g2 graph)
class CompiledArtifact {
public:
    std::string artifact_path;  // Path to g2.graph.cbor or compiled binary
    std::shared_ptr<sendnn::Graph> graph;  // Deserialized graph (if needed)
    bool is_loaded;
    
    CompiledArtifact(const std::string& path);
    Status load();
};

class ControlBlock {
public:
    std::string id;
    std::string operation;
    std::vector<int64_t> max_shape;
    c10::ScalarType dtype;
    
    std::shared_ptr<CompiledArtifact> artifact;
    std::shared_ptr<RuntimeCBHandle> runtime_handle;
    
    struct ShapeMetadata {
        std::vector<int64_t> base_shape;
        std::vector<int64_t> stride_info;
        bool supports_masking;
        bool supports_tiling;
    } shape_metadata;
    
    // Factory method: create from compiled artifact
    static std::shared_ptr<ControlBlock> fromArtifact(
        const std::string& operation,
        const std::string& artifact_path,
        const std::vector<int64_t>& max_shape,
        c10::ScalarType dtype
    );
    
    // Execute this CB
    Status execute(
        const std::vector<sendnn::ConstTensor>& inputs,
        const std::vector<sendnn::Tensor>& outputs,
        const std::vector<int64_t>& actual_shape,
        const MaskInfo& mask = MaskInfo::none(),
        std::shared_ptr<flex::Runtime> runtime = nullptr
    );
    
private:
    Status loadArtifact();
    Status prepareRuntimeHandle(std::shared_ptr<flex::Runtime> runtime);
};
```

**Key Implementation Notes**:
- `CompiledArtifact` wraps the existing g2 graph format (compatible with current compilation)
- `RuntimeCBHandle` will be a new interface to the runtime (replacing GraphLoader's internal state)
- Shape metadata extracted from compiled artifact or computed during compilation

#### 2. CBStream Implementation

**File**: `torch_spyre/csrc/cb_stream.h` / `cb_stream.cpp`

```cpp
// cb_stream.h (continued)
#include <deque>
#include <mutex>
#include <atomic>
#include <c10/core/Stream.h>

class ControlBlockCache;  // Forward declaration

class CBStream {
public:
    uint64_t stream_id;
    c10::Device device;
    
    enum class State {
        OPEN,
        FINALIZED,
        EXECUTING,
        COMPLETED,
        ERROR
    };
    
    // Thread-safe state management
    std::atomic<State> state;
    std::mutex queue_mutex;
    std::deque<CBExecutionItem> execution_queue;
    
    // Memory tracking
    struct MemoryTracking {
        std::set<void*> allocated_blocks;
        std::vector<int> preferred_segments;
        std::shared_ptr<void> mem_ctx;  // Opaque memory context
    } memory;
    
    // Synchronization
    std::shared_ptr<std::condition_variable> completion_cv;
    std::mutex completion_mutex;
    bool is_completed;
    
    // Dependencies
    std::shared_ptr<ControlBlockCache> cb_cache;
    std::shared_ptr<flex::Runtime> runtime;
    
    CBStream(uint64_t id, c10::Device dev, 
             std::shared_ptr<ControlBlockCache> cache,
             std::shared_ptr<flex::Runtime> rt);
    
    // Append CB to stream (thread-safe)
    Status append(
        std::shared_ptr<ControlBlock> cb,
        const std::vector<sendnn::ConstTensor>& inputs,
        const std::vector<sendnn::Tensor>& outputs,
        const std::vector<int64_t>& shape,
        const MaskInfo& mask = MaskInfo::none()
    );
    
    void finalize();
    Status execute();  // Async execution
    void synchronize();  // Block until complete
    bool isReady() const;
    void reset();  // Reset for reuse
    
    // Dynamic shape composition
    Status composeForShape(
        const std::string& operation,
        const std::vector<int64_t>& target_shape,
        const std::vector<sendnn::ConstTensor>& inputs,
        const std::vector<sendnn::Tensor>& outputs,
        c10::ScalarType dtype
    );
    
private:
    Status executeItem(const CBExecutionItem& item);
    void markCompleted();
    void markError(const std::string& error_msg);
};
```

**Key Implementation Notes**:
- Thread-safe queue operations (multiple threads can append)
- State machine with atomic state for lock-free reads
- Memory tracking integrated with VF allocator
- Async execution using background thread or runtime async API

#### 3. CBStreamPool Implementation

**File**: `torch_spyre/csrc/cb_stream_pool.h` / `cb_stream_pool.cpp`

```cpp
// cb_stream_pool.h
#include <unordered_map>
#include <vector>
#include <mutex>
#include <memory>
#include <c10/core/Stream.h>
#include "cb_stream.h"

namespace spyre {

class GlobalCBStreamPool {
public:
    static GlobalCBStreamPool& instance() {
        static GlobalCBStreamPool pool;
        return pool;
    }
    
    // Get or create stream for c10::Stream
    std::shared_ptr<CBStream> getStream(const c10::Stream& stream);
    
    // Get default stream for device
    std::shared_ptr<CBStream> getDefaultStream(c10::Device device);
    
    // Acquire stream from pool (or create new)
    std::shared_ptr<CBStream> acquire(c10::Device device);
    
    // Release stream back to pool
    void release(std::shared_ptr<CBStream> stream);
    
    // Map c10::Stream to CBStream
    void registerStream(const c10::Stream& c10_stream, 
                       std::shared_ptr<CBStream> cb_stream);
    
private:
    GlobalCBStreamPool();
    ~GlobalCBStreamPool();
    
    std::mutex mutex_;
    std::unordered_map<c10::Device, std::vector<std::shared_ptr<CBStream>>> pool_;
    std::unordered_map<c10::Device, std::shared_ptr<CBStream>> default_streams_;
    std::unordered_map<uint64_t, std::weak_ptr<CBStream>> stream_id_map_;
    
    uint64_t next_stream_id_;
    std::shared_ptr<ControlBlockCache> cb_cache_;
    
    std::shared_ptr<CBStream> createStream(c10::Device device);
};

// Helper function for spyre_hooks.cpp
std::shared_ptr<CBStream> getCBStreamFromC10Stream(const c10::Stream& stream);

}  // namespace spyre
```

**Key Implementation Notes**:
- Singleton pattern (similar to GlobalRuntime)
- Thread-safe pool management
- Weak pointers in stream_id_map_ to avoid circular references
- Automatic stream creation on first use

#### 4. ControlBlockCache Implementation

**File**: `torch_spyre/csrc/cb_cache.h` / `cb_cache.cpp`

```cpp
// cb_cache.h
#include <unordered_map>
#include <mutex>
#include <memory>
#include <string>
#include <vector>
#include <c10/core/ScalarType.h>
#include "cb_stream.h"

namespace spyre {

struct CacheKeyHash {
    std::size_t operator()(const std::tuple<std::string, std::vector<int64_t>, c10::ScalarType>& key) const;
};

class ControlBlockCache {
public:
    using CacheKey = std::tuple<std::string, std::vector<int64_t>, c10::ScalarType>;
    
    static ControlBlockCache& instance() {
        static ControlBlockCache cache;
        return cache;
    }
    
    std::optional<std::shared_ptr<ControlBlock>> get(
        const std::string& operation,
        const std::vector<int64_t>& max_shape,
        c10::ScalarType dtype
    );
    
    void put(
        const std::string& operation,
        const std::vector<int64_t>& max_shape,
        c10::ScalarType dtype,
        std::shared_ptr<ControlBlock> cb
    );
    
    // Get or compile a CB
    // This will need to integrate with torch.compile compilation pipeline
    std::shared_ptr<ControlBlock> getOrCompile(
        const std::string& operation,
        const std::vector<int64_t>& max_shape,
        const std::vector<int64_t>& target_shape,
        c10::ScalarType dtype
    );
    
    // Calculate max practical shape for an operation
    std::vector<int64_t> calculateMaxPracticalShape(
        const std::string& operation,
        const std::vector<int64_t>& target_shape
    );
    
private:
    ControlBlockCache();
    ~ControlBlockCache();
    
    std::mutex mutex_;
    std::unordered_map<CacheKey, std::shared_ptr<ControlBlock>, CacheKeyHash> cache_;
    
    // Compilation integration
    std::string compileOperation(
        const std::string& operation,
        const std::vector<int64_t>& max_shape,
        c10::ScalarType dtype
    );
};

}  // namespace spyre
```

**Key Implementation Notes**:
- Cache key includes operation, max_shape, and dtype
- `getOrCompile()` integrates with existing torch.compile pipeline
- `calculateMaxPracticalShape()` uses heuristics based on operation type
- Compilation happens lazily (on cache miss)

#### 5. CBRuntimeInterface Implementation

**File**: `torch_spyre/csrc/cb_runtime_interface.h` / `cb_runtime_interface.cpp`

```cpp
// cb_runtime_interface.h
#include <memory>
#include <vector>
#include <sendnn/tensor/sentensor_info.hpp>
#include <flex/runtime.hpp>
#include "cb_stream.h"

namespace spyre {

class CBRuntimeInterface {
public:
    // Execute a CB stream directly on runtime
    static Status executeStream(
        std::shared_ptr<CBStream> stream,
        std::shared_ptr<flex::Runtime> runtime
    );
    
    // Execute a single CB
    static Status executeControlBlock(
        const ControlBlock& cb,
        const std::vector<sendnn::ConstTensor>& inputs,
        const std::vector<sendnn::Tensor>& outputs,
        const std::vector<int64_t>& shape,
        const MaskInfo& mask,
        std::shared_ptr<flex::Runtime> runtime
    );
    
private:
    // Direct runtime calls (no GraphLoader)
    static Status launchCBOnRuntime(
        const ControlBlock& cb,
        const std::vector<sendnn::ConstTensor>& inputs,
        const std::vector<sendnn::Tensor>& outputs,
        const std::vector<int64_t>& shape,
        const MaskInfo& mask,
        std::shared_ptr<flex::Runtime> runtime
    );
    
    // Convert CB artifact to runtime-executable format
    static Status prepareRuntimeExecution(
        const ControlBlock& cb,
        std::shared_ptr<flex::Runtime> runtime
    );
};

}  // namespace spyre
```

**Key Implementation Notes**:
- Static methods (no instance needed)
- Direct calls to `flex::Runtime` API (bypassing GraphLoader)
- Will need to understand flex::Runtime API for direct execution
- May need to convert g2 graph format to runtime's native format

#### 6. VF Allocator Integration

**File**: `torch_spyre/csrc/spyre_mem.h` / `spyre_mem.cpp`

**Changes to SpyreAllocator**:

```cpp
// Add to spyre_mem.h
#include <c10/core/Stream.h>
#include "cb_stream_pool.h"  // Forward declaration

// Add to SpyreAllocator class in spyre_mem.cpp
class SpyreAllocator {
    // ... existing code ...
    
    // New method: stream-aware allocation
    at::DataPtr allocateForStream(
        size_t nbytes,
        c10::Stream stream,
        c10::Device device
    );
    
    // Track allocation on stream
    void recordAllocationOnStream(void* ptr, c10::Stream stream);
    
    // Defer deallocation
    void deferDeallocationUntilStreamComplete(void* ptr, c10::Stream stream);
    
    // Modified: vf_allocation with stream preference
    at::DataPtr vf_allocation(
        flex::DeviceMemoryAllocatorPtr allocator,
        size_t nbytes,
        c10::Device curr_device,
        unsigned int device_id,
        std::optional<std::vector<int>> preferred_segments = std::nullopt
    );
    
private:
    // Track pending deallocations per stream
    std::unordered_map<uint64_t, std::vector<void*>> pending_deallocations_;
    std::mutex dealloc_mutex_;
};
```

**Implementation Details**:
- `allocateForStream()` gets CB stream, extracts preferred segments
- `recordAllocationOnStream()` adds pointer to stream's memory tracking
- `deferDeallocationUntilStreamComplete()` queues deallocation until stream completes
- Stream completion callback triggers deferred deallocations

### Integration Points

#### 1. Stream Hook Integration

**File**: `torch_spyre/csrc/spyre_hooks.cpp`

**Changes**:
```cpp
// Replace lines 117-150
c10::Stream getStream(c10::Device d) const noexcept override {
    py::gil_scoped_acquire acquire;
    auto cb_stream = GlobalCBStreamPool::instance().getDefaultStream(d);
    return c10::Stream(c10::Stream::UNSAFE, current_device, cb_stream->stream_id);
}

c10::Stream getDefaultStream(c10::Device d) const override {
    py::gil_scoped_acquire acquire;
    auto cb_stream = GlobalCBStreamPool::instance().getDefaultStream(d);
    return c10::Stream(c10::Stream::UNSAFE, current_device, cb_stream->stream_id);
}

c10::Stream getStreamFromGlobalPool(c10::Device d, bool isHighPriority = false) const override {
    py::gil_scoped_acquire acquire;
    auto cb_stream = GlobalCBStreamPool::instance().acquire(d);
    return c10::Stream(c10::Stream::UNSAFE, current_device, cb_stream->stream_id);
}

c10::Stream getNewStream(c10::Device d, int priority = 0) const override {
    py::gil_scoped_acquire acquire;
    auto cb_stream = GlobalCBStreamPool::instance().acquire(d);
    GlobalCBStreamPool::instance().registerStream(
        c10::Stream(c10::Stream::UNSAFE, d, cb_stream->stream_id),
        cb_stream
    );
    return c10::Stream(c10::Stream::UNSAFE, current_device, cb_stream->stream_id);
}
```

#### 2. Kernel Launch Integration

**File**: `torch_spyre/csrc/module.cpp`

**Changes to `launchKernel()`**:
```cpp
void launchKernel(std::string g2_path, std::vector<at::Tensor> args) {
    // Get current stream
    auto device = args[0].device();
    auto c10_stream = c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)
                         ->getStream(device);
    auto cb_stream = GlobalCBStreamPool::instance().getStream(c10_stream);
    
    // Create ControlBlock from artifact
    auto cb = ControlBlock::fromArtifact("kernel", g2_path, 
                                         calculateMaxShape(args), 
                                         args[0].scalar_type());
    
    // Prepare inputs/outputs
    std::vector<sendnn::ConstTensor> inputs;
    std::vector<sendnn::Tensor> outputs;
    // ... convert args to sendnn tensors ...
    
    // Append to stream and execute
    cb_stream->append(cb, inputs, outputs, calculateShape(args));
    cb_stream->finalize();
    cb_stream->execute();
}
```

#### 3. Eager Operation Integration

**File**: `codegen/templates/base.jinja2`

**Changes**: Replace GraphLoader usage with CB stream composition
```cpp
// Instead of:
// std::optional<sendnn::GraphLoader> opt_gl = getCachedGraphLoader(...);
// sendnn::GraphLoader gl = ...;

// Use:
auto device = {{ output_variable_name }}.device();
auto c10_stream = c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)
                     ->getStream(device);
auto cb_stream = GlobalCBStreamPool::instance().getStream(c10_stream);

cb_stream->composeForShape(
    {{ template_data.op_label }},
    {{ output_variable_name }}.sizes().vec(),
    input_sendnn_tensor_vector,
    {output_sendnn_tensor},
    {{ output_variable_name }}.scalar_type()
);

cb_stream->finalize();
cb_stream->execute();
```

### Compilation Artifact Integration

The existing compilation pipeline generates:
- `g2.graph.cbor` files (in `torch_spyre/_inductor/runtime/async_compile.py`)
- These are compatible with ControlBlock's `CompiledArtifact`

**No changes needed** to compilation pipeline initially - CB streams will use existing artifacts.

### Testing Strategy

1. **Unit Tests**: `torch_spyre/csrc/test_cb_stream.cpp`
   - Test CBStream state machine
   - Test ControlBlock execution
   - Test stream pool

2. **Integration Tests**: `tests/test_cb_stream.py`
   - Test end-to-end kernel execution
   - Test eager operations
   - Test memory tracking

3. **Performance Tests**: Compare GraphLoader vs CB stream performance

## Migration Path

1. **Phase 1**: Implement core components
   - Create `cb_stream.h/cpp` with basic CBStream and ControlBlock
   - Create `cb_stream_pool.h/cpp` with stream pool
   - Create `cb_cache.h/cpp` with cache (initially empty, no compilation)

2. **Phase 2**: Integrate with hooks
   - Update `spyre_hooks.cpp` to use CBStreamPool
   - Test stream creation/retrieval

3. **Phase 3**: VF allocator integration
   - Add stream-aware allocation to `SpyreAllocator`
   - Test memory tracking

4. **Phase 4**: Runtime interface
   - Implement `CBRuntimeInterface` with direct runtime calls
   - Replace `launchKernel()` to use CB streams

5. **Phase 5**: Eager operations
   - Update codegen templates to use CB streams
   - Replace GraphLoader in eager ops

6. **Phase 6**: Dynamic shapes
   - Implement `composeForShape()` logic
   - Add masking and composition support

7. **Phase 7**: Optimization
   - Add stream pooling optimizations
   - Add CB caching optimizations
   - Performance tuning
