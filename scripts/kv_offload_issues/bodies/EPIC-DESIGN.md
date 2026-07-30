**Summary:** Land the three design documents that specify a three-layer KV-cache offload stack, as one coherent deliverable. The **hardware-runtime** RFC owns the mechanism (a byte-exact raw device↔host DMA, a DMA-able shared host memory pool addressed by integer slot, and a block-hash→slot directory with a concurrency protocol). The **torch-spyre** design specifies the thin Python surface (`torch_spyre._C`): the one tensor-aware address step and passthrough bindings. The **spyre-inference** RFC ports the upstream vLLM `OffloadingConnector` experience and defines the M1/M2 milestone ladder. The three docs must be **mutually consistent**: one canonical copy signature `copy_tensor_raw(dev_tensor, pool, slot_id, to_device)`, one integer `slot_id` seam, one ownership split — verified by a cross-doc consistency pass.

### Design documents

- [ ] __ED1__ — Merge the hardware-runtime shared host memory KV pool RFC (layer: hardware runtime)
- [ ] __ED2__ — Merge the torch-spyre Python-surface design doc (layer: torch-spyre)
- [ ] __ED3__ — Merge the spyre-inference upstream-connector-port RFC (layer: spyre-inference)

### Closes with

- [ ] All three RFCs merged.
- [ ] Cross-doc consistency note recorded: the `copy_tensor_raw(dev_tensor, pool, slot_id, to_device)` signature, the integer `slot_id` seam, and the "pinning internal to the pool / no host-buffer-registration binding" statement appear **identically** in all three docs.
