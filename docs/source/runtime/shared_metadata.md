# Persistent shared KV pools

Torch-Spyre exposes Flex's persistent, node-local KV pool as one composite
resource. A generation contains one global metadata directory and fixed,
rank-owned data arenas. The generation can remain published with no attached
processes; it is removed only by an explicit, generation-checked destroy.

The Python boundary exposes identifiers and protocol records, not raw host or
device pointers.

## Configuration

`PoolConfig` describes the whole generation before it is created:

- `name` is the logical node-local pool name.
- `metadata.max_chunks` limits the chunk descriptors stored for one value.
- `data_pools` is the complete, fixed arena list.

Each `SharedDataPoolConfig` supplies an arena name, backend, rank-homogeneous
partition list, and `PoolCompatibility`. Every partition in one arena must
belong to the same TP rank. `HOST_DRAM` is the currently supported backend. The
concrete compatibility fields are `model`, `kv_dtype`, `block_size`, and
`tensor_parallel_size`. They prevent workers with incompatible KV layouts from
sharing bytes accidentally.

All metadata and data children are created before the root generation becomes
`READY`. Child registration and retirement are not dynamic: changing the
membership or geometry requires a new generation.

```python
from torch_spyre import _C

compatibility = _C.PoolCompatibility(
    model="granite-example",
    kv_dtype="float16",
    block_size=64,
    tensor_parallel_size=2,
)
rank_0 = _C.SharedDataPoolConfig(
    "arena-r0",
    _C.PoolBackend.HOST_DRAM,
    [_C.SharedDataPartitionConfig("host-r0", 0, 0, 128, 4096)],
    compatibility,
)
rank_1 = _C.SharedDataPoolConfig(
    "arena-r1",
    _C.PoolBackend.HOST_DRAM,
    [_C.SharedDataPartitionConfig("host-r1", 1, 0, 128, 4096)],
    compatibility,
)
config = _C.PoolConfig(
    "kv-cache",
    _C.MetadataConfig(max_chunks=2),
    [rank_0, rank_1],
)
```

## Management and attachment

Pool management and worker attachment are separate operations:

- `SharedPool.create_pool(config)` creates and publishes a new generation.
- `SharedPool.open_pool(name)` inspects the published generation without
  attaching to it.
- `SharedPool.destroy_pool(descriptor)` removes that exact generation after
  all leases have drained.
- `SharedPool.attach_pool(name, generation, tp_rank)` is the normal worker API. It
  opens the local descriptor, verifies the controller-provided 16-byte
  generation, and returns the global metadata plus all mapped data partitions
  for that rank.
- `SharedPool.attach_pool(descriptor, tp_rank)` provides the same rank-scoped
  operation for a complete descriptor.
- `SharedPool.detach_pool(attachment)` closes that aggregate. It never
  destroys or unlinks the generation.

These methods return structured results. Check `result.code` before reading
the descriptor or attachment. `PoolResultCode` contains `OK`, `NOT_FOUND`,
`ALREADY_EXISTS`, `INITIALIZING`, `DESTROYING`, `POOL_BUSY`,
`STALE_GENERATION`, `CONFIGURATION_MISMATCH`, and `RESOURCE_ERROR`; `detail`
contains diagnostic text.

Management is intended to move to a dedicated delegator. The distributed
tests use rank 0 as a temporary stand-in: rank 0 creates or opens and later
destroys, while every rank only attaches and detaches during normal work.

```python
import torch.distributed as dist

identity_box = [None]
if dist.get_rank() == 0:
    result = _C.SharedPool.create_pool(config)
    if result.code is not _C.PoolResultCode.OK:
        raise RuntimeError(f"create failed: {result.code}: {result.detail}")
    identity_box[0] = (
        result.descriptor.name,
        result.descriptor.generation.bytes,
    )

dist.broadcast_object_list(identity_box, src=0)
name, generation = identity_box[0]
tp_rank = dist.get_rank()
result = _C.SharedPool.attach_pool(name, generation, tp_rank)
if result.code is not _C.PoolResultCode.OK:
    raise RuntimeError(f"attach failed: {result.code}: {result.detail}")
attachment = result.attachment
metadata = attachment.metadata
local_pools = {
    item.registration.name: item.data_pool
    for item in attachment.data_pools
}
```

A successful call means the metadata mapping and every data/IOMMU mapping for
the requested rank are ready. No later data-pool resolution step is required.

`PoolDescriptor` remains pickleable for compatibility. Normal workers only
need the name and opaque generation returned by the delegator. Neither form
holds an attachment or keeps the generation busy. Reusing a logical name
produces a new generation, so an old identity cannot attach to or destroy its
replacement.

## Lease ownership

An attachment holds a shared lock on the generation root. Independently
retained metadata, `SharedDataPool`, and `SlotReadPin` handles also retain that
root lease. This lets a data operation finish safely after the aggregate is
closed. Conversely, keeping any of those objects alive makes `destroy_pool`
return `POOL_BUSY`.

Release objects in dependency order:

1. Finish all DMA and destroy read pins.
2. Drop every independently retained metadata and data-pool handle.
3. Call `detach_pool(attachment)` or `attachment.close()` for every worker.
4. After a host-wide barrier, let the manager call
   `destroy_pool(descriptor)`.

The last detach leaves the generation and its bytes intact. The manager may
open and attach it again later. `destroy_pool` unlinks data children first and
the metadata/root objects last; a missing generation reports `NOT_FOUND`.

## Metadata and data handles

The metadata directory describes every rank's fixed partitions and versioned
slot tenancies. Use the registration paired with an eager local data handle for
claims and DMA:

```python
local = {
    item.registration.name: item
    for item in attachment.data_pools
}
registered = local[f"host-r{tp_rank}"].registration
pool = local[f"host-r{tp_rank}"].data_pool
```

Use `metadata.find_pool(name)` only when directory lookup itself is needed;
the directory also contains registrations owned by other ranks, whose data
arenas are intentionally not mapped by this attachment.

Every slot is identified by a versioned `PoolRef`, `slot_id`, and
`slot_version`. `metadata_version`, `pool_version`, and `slot_version` reject
stale records after reuse. A slot version is a wrapping 64-bit tenancy
detector, not a lifetime-unique identifier.

## Write protocol

The caller composes a write from three operations:

```text
claim -> blocking D2H copy -> publish
```

`claim(pool_ref, key)` returns one of:

- `Reservation`: this caller owns the unpublished slot and must eventually
  call `publish` or `abort`.
- `ExistingClaim`: another caller already owns the key; `valid` says whether
  it has been published.
- `NoSpace`: the selected child has no free slot.
- `Unavailable`: the supplied versioned pool reference is stale or unusable.

After receiving a `Reservation`, copy the complete KV value into its slot with
a blocking device-to-host call such as `copy_tensor_raw(...,
to_device=False)`. Only then call `publish(reservation, chunks)`. Until
publication, `lookup(key)` returns `None`, even if DMA has changed the shared
bytes.

`abort(reservation)` releases an unpublished reservation. Abort only before
submitting DMA or after all submitted DMA is quiescent. Aborting an in-flight
write could make the slot reusable while hardware is still writing it.

No binding holds the metadata directory lock across DMA. Reservation state,
slot versioning, and the publication gate protect the caller-composed
sequence.

## Read protocol

Readers compose the corresponding sequence:

```text
lookup -> pin_read -> blocking H2D copy -> release pin
```

`lookup(key)` returns only published values. `pin_read(entry)` revalidates the
pool and slot versions and returns `None` for a stale or reused entry. Keep the
pin alive until the blocking host-to-device copy completes, then destroy it on
the same thread that acquired it.

```python
entry = metadata.lookup(key)
if entry is not None:
    pin = metadata.pin_read(entry)
    if pin is not None:
        _C.copy_tensor_raw(
            destination,
            pool,
            entry.slot.slot_id,
            to_device=True,
        )
        del pin
```

Eviction waits for outstanding read pins before recycling a slot. Operations
that can wait on a process-shared lock or DMA release the Python GIL, so an
unrelated Python thread can continue running.

## Two-card verification

From the repository's `tests` directory, run the lifecycle-managed data copy
and metadata tests on two cards with:

```bash
SPYRE_DEVICES=0,1 bash run_test.sh \
  configs/distributed_tests/test_shared_host_pool.yaml
SPYRE_DEVICES=0,1 bash run_test.sh \
  configs/distributed_tests/test_kv_offload.yaml
SPYRE_DEVICES=0,1 bash run_test.sh \
  configs/distributed_tests/test_shared_metadata.yaml
```

The metadata suite verifies cross-process claims, publication, global peer
metadata visibility, rank-local DMA, read-pin eviction blocking, slot reuse,
and that a rank holding the only live attachment still makes destruction
report `POOL_BUSY`.
