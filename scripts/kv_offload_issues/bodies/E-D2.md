**Summary:** Merge `docs/source/architecture/raw_copy_kv_offload.md` (plus figures) — the torch-spyre Python surface: the one tensor-aware address step (`get_composite_address`) and thin passthrough bindings over the hardware runtime's raw copy, pool, and directory.

### What

- Land the design doc and its call-path figure.
- Specify a **single** `copy_tensor_raw` signature: `(dev_tensor, pool, slot_id, to_device, non_blocking)` — **no host-tensor overload**; the host destination is always a pool slot.
- State the invariants: pinning is internal to the pool (no host-buffer-registration binding); `slot_ptr` is not exposed to Python; raw host pointers never cross into Python.

### Acceptance

- [ ] Doc merged.
- [ ] `copy_tensor_raw` signature, `slot_id` seam, and "pinning internal / no host-buffer-registration" statements are identical to the hardware-runtime RFC (E-D1).

**Blocked by:** none

_Part of the KV-cache offload design epic (__EDESIGN__)._
