**Summary:** Deliver **byte-exact** KV-cache offload end-to-end: a user runs `vllm serve … spec_name: SpyreOffloadingSpec` and gets host-RAM offload that survives across requests, offloading into a **shared** host memory pool. The device↔host copy is byte-exact via the raw slot-addressed path `copy_tensor_raw(dev_tensor, pool, slot_id, to_device)` — never the converting copy (which drifts ~1 ULP and is a correctness defect for KV data). Spans three layers plus prerequisites: the hardware runtime adds the public raw copy primitive and the **shared** pool; torch-spyre adds the tensor-address accessor, the shared-pool binding, and the canonical copy binding; spyre-inference builds the production connector (copier → handlers → spec).

**Why shared-pool-first:** M1 and M2 have a compact combined schedule, and deferring the shared pool to M2 would mean a large, hard-to-review M2 rewrite. We must build the pool (M1-F2) anyway, so we build it **shared from the start** — only pool creation (a named, shared segment) and its cross-process acceptance tests differ from a single-instance pool; everything else is identical. The block-hash→slot **directory** (M2-F1) and the per-slot **concurrency protocol** (M2-F2) stay in M2; M1's shared pool uses externally-coordinated slot ids without the safe-concurrency layer.

**Prerequisites (per discussion with @frankeh):** before any offload code, M1 requires a reproducible custom-built env, a **recomputation baseline** on the *latest* spyre-inference with a **pinned** version, and a documented **host CPU buffer model** for the raw copy. An earlier proof-of-concept connector (converting copy into a host tensor, on an old spyre-inference commit, capped at ~4K-token prompts) is **discarded** — production code is built fresh to the latest design.

### Prerequisites (block all of M1)

- [ ] __M1P1__ — Env: custom-built hardware runtime + torch-spyre + spyre-inference (reproducible, pinned)
- [ ] __M1P2__ — spyre-inference recomputation baseline on latest code + pin a version
- [ ] __M1P3__ — Define the host CPU buffer/tensor model for raw copy (before M1-F1)

### Hardware runtime

- [ ] __M1F1__ — Public raw (byte-exact) host↔device DMA
- [ ] __M1F2__ — `SharedHostPool` — shared host memory pool (cross-process, attach-by-name)

### torch-spyre

- [ ] __M1T1__ — `get_composite_address` accessor
- [ ] __M1T2__ — `SharedHostPool` pybind passthrough (incl. cross-process sharing)
- [ ] __M1T3__ — `copy_tensor_raw(dev_tensor, pool, slot_id, …)` binding (canonical)

### spyre-inference

- [ ] __M1S1__ — `SpyreKvDmaCopier` + `kv_offload` package scaffold
- [ ] __M1S2__ — `SpyreCpuOffloadingHandlers`
- [ ] __M1S3__ — `SpyreOffloadingSpec` + registration + M1 acceptance & benchmark

### Closes with (concrete end-to-end acceptance)

- [ ] Env + baseline in place: the pinned custom stack builds, and the **recomputation baseline** (no offload) runs on latest spyre-inference with its TTFT-vs-prompt-length curve recorded (M1-P1/P2).
- [ ] `vllm serve … spec_name: SpyreOffloadingSpec` boots and reaches `register_kv_caches` without raising.
- [ ] **Load/restore KVC from device HBM:** a device→host→device round-trip of a KV block is **byte-for-byte** identical (byte-exact raw copy).
- [ ] **Cross-process pool sharing:** a KV page written to a shared-pool slot by one process is reloaded byte-exact by a second process attaching the same named pool.
- [ ] A prefix-extending second prompt reports a host-tier hit (`N > 0` blocks loaded from host).
- [ ] **Generation accuracy with reloaded KVC:** with `temperature=0`, generated tokens are **byte-identical** to a no-offload baseline.
- [ ] **Performance benchmark recorded:** per-block offload (D2H) / reload (H2D) latency and throughput (GB/s), and TTFT reduction on a cache hit vs. the recomputation baseline.
- [ ] No source changes to the Spyre worker or platform (verified by diff).
