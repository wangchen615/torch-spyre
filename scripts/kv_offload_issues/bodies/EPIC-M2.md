**Summary:** Make the host tier a single **shared host memory pool** shared by every co-located Spyre instance, so a KV block offloaded by one instance is reloaded by another with **one raw DMA and no serialization** — at memory speed, no disk. Reuses M1's private pool object and canonical copy path **unchanged**; adds the cross-instance layer: a `SharedHostMetadata` block-hash→slot directory, a per-slot concurrency/generation protocol with a publish-on-DMA-completion gate (a stale or mid-write slot degrades to a **cache miss, never torn bytes**), the multi-chunk (1p5) raw-copy path, and the `SpyreSharedOffloadingSpec` connector wiring. Gated on the directory, the concurrency protocol, and the multi-chunk copy landing. M2 targets the correctness **baseline**; a dedicated DMA stream (overlap optimization) is **deferred to the backlog** (M2-T2), not part of M2.

### Hardware runtime

- [ ] __M2F1__ — `SharedHostMetadata` — block-hash → slot directory
- [ ] __M2F2__ — Concurrency protocol (locks + generation + publish gate) — full race coverage
- [ ] __M2F3__ — `copyRaw` multi-chunk (1p5) + cross-process slot round-trip

### torch-spyre

- [ ] __M2T1__ — `SharedHostMetadata` (+ shared-pool attach) pybind passthroughs

### spyre-inference

- [ ] __M2S1__ — `SpyreSharedOffloadingSpec` + registration
- [ ] __M2S2__ — Shared-pool round-trip + torn-read test
- [ ] __M2S3__ — Cross-instance test + M2 acceptance & benchmark

### Closes with (concrete end-to-end acceptance)

- [ ] **Cross-instance peer hit:** two `vllm serve` instances on one host attach the same shared pool; the second gets a host-tier hit **on its first request** on a block the first offloaded — served by a device←host DMA, no recompute, no disk.
- [ ] **Generation accuracy with peer-reloaded KVC:** with `temperature=0`, B's tokens are **byte-identical** to a no-cache baseline.
- [ ] **Full data-race coverage** passes (the M2-F2 matrix) — zero torn reads, zero double-allocations.
- [ ] **Performance benchmark recorded:** cross-instance peer-hit latency (B's TTFT on a shared-pool hit) vs. full recompute, and shared-pool reload throughput vs. M1.
- [ ] M1 path unaffected.

### Deferred (backlog — not gating M2)

- __M2T2__ — `get_dma_stream` dedicated DMA stream (overlap optimization). Pulled out of M2 per
  review; schedule after the M2 baseline lands. No milestone.
