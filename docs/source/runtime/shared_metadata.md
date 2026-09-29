# Shared Metadata

Torch-Spyre exposes Flex's process-shared KV metadata directory and shared host
pool as thin Python bindings. Torch-Spyre does not create a separate directory
algorithm, choose eviction victims, or add locks. Each method maps to one
internally atomic Flex operation.

The Python boundary contains identifiers and protocol records only. It does not
expose a raw host pointer, device address, or slot-pointer accessor.

## Creating or attaching the shared store

Every process uses the same metadata name and configuration. The first process
creates the directory and pools; later processes attach them.

```python
from torch_spyre import _C

compatibility = _C.CompatibilityDescriptor(1, [1, 2, 3])
pool_config = _C.SharedDataPoolConfig(
    "kv-pool",
    _C.SharedPoolKind.HOST,
    128,
    4096,
    compatibility,
)
metadata = _C.SharedMetadata.create_or_attach(
    "kv-directory",
    _C.SharedMetadataConfig(1, [pool_config], None),
)
registered = metadata.find_pool("kv-pool")
pool = metadata.resolve_pool(registered.pool_ref)
```

`create_or_attach` initializes the torch-spyre runtime internally. It does not
take a Python stream. A directory can contain multiple pools, and every slot is
identified by a versioned `PoolRef`, `slot_id`, and `slot_version`.

## Write protocol

The caller composes a write from three separate operations:

```text
claim -> blocking D2H copy -> publish
```

`claim(pool_ref, key)` selects a free slot in the caller-selected pool and
returns one of these results:

- `Reservation`: this caller owns the new write tenancy and must eventually
  call `publish` or `abort`.
- `ExistingClaim`: another claim already owns the key. Its `valid` field says
  whether the entry is already published.
- `NoSpace`: the selected pool has no free slot. The caller must select an
  eviction victim, evict it, and retry.
- `Unavailable`: the supplied versioned pool reference is no longer usable.

After receiving a `Reservation`, copy the complete value into its slot with a
blocking device-to-host operation such as `copy_tensor_raw(...,
to_device=False)`. Only then call `publish(reservation, chunks)`. Until
publication, `lookup(key)` returns `None`, even if the DMA has already changed
the shared bytes.

`abort(reservation)` releases an unpublished reservation. Abort only before
submitting DMA or after all submitted DMA is quiescent; aborting an in-flight
write can make the slot reusable while hardware is still writing it.

No binding holds the directory lock across DMA. Correctness across the
caller-composed sequence comes from the reservation state, slot version, and
publication gate.

## Read protocol

Readers also compose three operations:

```text
lookup -> pin_read -> blocking H2D copy -> destroy pin
```

`lookup(key)` returns a `LookupEntry` only for a published value. Pass that
entry to `pin_read(entry)` to validate its pool and slot versions and acquire a
read pin. A stale or reused entry returns `None`. Keep the pin alive until the
blocking host-to-device copy completes, then destroy it on the same thread that
acquired it.

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

An eviction waits for outstanding read pins before recycling the slot. Calls
that may wait on a process-shared lock or DMA release the Python GIL, so an
unrelated Python thread can continue running.

## Versions and lifetime

`metadata_version`, `pool_version`, and `slot_version` detect stale references
when directories, pool-table rows, or slots are reused. A slot version is a
wrapping 64-bit tenancy detector, not a lifetime-unique identifier.

Keep `SharedMetadata` alive while using objects returned from it. A Python
`SlotReadPin` retains its originating metadata object automatically, but it
must still be destroyed on its acquiring thread after the reload DMA finishes.

`pool_count()` reports the current number of registered pools. Flex does not
expose an aggregate `capacity()` accessor through this API; configuration and
per-pool geometry are available through the versioned pool records.
