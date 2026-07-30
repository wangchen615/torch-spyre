**Summary:** Merge the RFC that specifies the KV-offload **mechanism** at the hardware-runtime layer: a byte-exact raw device↔host DMA primitive, a DMA-able shared host memory pool (slot-addressed, pinned internally per IOMMU Function), and a block-hash→slot directory with its concurrency protocol. This is the bottom layer the other two design docs consume.

### What

- Land the RFC document in the hardware-runtime repo (RFC + any figures).
- Public API surface defined: `copyRaw(host_addr, CompositeAddress*, to_device)`, `SharedHostPool::create_or_attach(stream, name, num_slots, slot_bytes)`, `SharedHostMetadata::create_or_attach(...)`, the concurrency protocol, and the multi-chunk pack/rebuild contract.
- Ownership split stated: the hardware runtime owns the mechanism; torch-spyre owns only the tensor→address step and thin bindings; spyre-inference owns cache policy.

### Acceptance

- [ ] RFC merged.
- [ ] API signatures match what E-D2 (torch-spyre) and E-D3 (spyre-inference) consume — one canonical `copy_tensor_raw(dev_tensor, pool, slot_id, to_device)` seam, integer `slot_id`, no caller-owned host-buffer path.

**Blocked by:** none

_Part of the KV-cache offload design epic (__EDESIGN__)._
