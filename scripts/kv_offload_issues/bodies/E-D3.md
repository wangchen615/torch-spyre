**Summary:** Merge the RFC (plus figures) that ports the upstream vLLM `OffloadingConnector` experience to spyre-inference — `SpyreOffloadingSpec`, handlers, and the M1/M2 milestone ladder.

### What

- Land the RFC and its architecture figures.
- M1 copier must call the **canonical** `copy_tensor_raw(dev_tensor, pool, slot_id, to_device)` into a **private** pool — not a converting copy into a host tensor.
- M2 differs from M1 only by attaching a **shared** named pool + directory (same copy signature).

### Acceptance

- [ ] Doc merged.
- [ ] The M1 **and** M2 copier code both use the one canonical `copy_tensor_raw` signature (the "reuses M1 unchanged" claim is literally true — only pool name/sharing differs).
- [ ] No `copy_tensor_raw(dev_tensor, host_tensor, …)` host-tensor form remains in the doc.

**Blocked by:** none

_Part of the KV-cache offload design epic (__EDESIGN__)._
