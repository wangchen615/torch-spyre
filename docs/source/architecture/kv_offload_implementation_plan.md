# KV-Cache Offload — Implementation Plan (Epics, Sub-Issues, Milestones)

| Field | Value |
|---|---|
| Status | Planning |
| Created | 2026-07-23 |
| Updated | 2026-07-28 |
| Project board | [torch-spyre project #2, view 23](https://github.com/orgs/torch-spyre/projects/2/views/23) |
| Label (all issues) | `kvc-offloading` |
| Issue home | **All issues are created in the `torch-spyre` repo** and added to the project board. |
| Milestone 1 due | **July 2026** |
| Milestone 2 due | **August 2026** |
| Design docs | torch-spyre `docs/source/architecture/raw_copy_kv_offload.md`; the hardware-runtime shared-host-pool RFC; the spyre-inference upstream-connector-port RFC |

> **Terminology.** This plan and every issue it describes use **neutral terms**. The
> per-process C++ device runtime is the **hardware runtime**. The cross-process shared
> host memory it provides is the **shared host memory pool** (a **secondary memory
> pool**). Issue titles and bodies must not name internal component or library
> code-names — say "hardware runtime" and "secondary / shared host memory pool"
> instead.

> **Production code follows the latest design, not the current prototype.** An earlier
> proof-of-concept connector exists (a converting `copy_tensor` into a process-local host
> tensor, with pluggable experimental backends). That PoC is **not** what this plan
> tracks. The production implementation is built fresh against the three latest design
> docs — a single canonical byte-exact raw-copy signature
> (`copy_tensor_raw(dev_tensor, pool, slot_id, to_device)`), a shared host memory pool as
> the **only** host-memory source (no caller-owned host-buffer path), and the integer
> `slot_id` seam. Where the PoC diverges (converting copy, host-tensor destination), the
> production code replaces it.

## 1. What this plan covers

Three design documents describe a three-layer KV-cache offload stack:

1. **Hardware-runtime layer** — owns the mechanism: a raw (byte-exact) device↔host
   DMA primitive (`copyRaw`), a DMA-able shared host memory pool
   (`SharedHostPool`) addressed by integer slot, and a shared directory
   (`SharedHostMetadata`, block-hash→slot with a concurrency protocol).
2. **torch-spyre layer** (this repo) — a thin Python surface: turn a
   `device("spyre")` tensor into the hardware-runtime device address, and expose
   the runtime's raw-copy + pool + directory objects to Python through
   `torch_spyre._C`.
3. **spyre-inference layer** — the vLLM connector and cache policy: a
   `SpyreOffloadingSpec` and handlers that ride the upstream `OffloadingConnector`.

This document turns those designs into a concrete, dependency-ordered issue backlog:
**3 epics (1 design + 2 milestone) + 18 sub-issues (3 design-doc + 15 implementation)
= 21 issues**, all created in `torch-spyre`, all labeled `kvc-offloading`, all added to
project board view 23. The three design docs are one deliverable, so they sit as three
sub-issues under a single **design epic** (one RFC-merge PR per layer).

## 2. Current code status (grounds the scoping)

Assessed against the latest `main` of each layer on 2026-07-28.

| Layer | Already on `main` | Missing — the actual work |
|---|---|---|
| **Hardware runtime** | `CompositeAddress` (with `total_size()`, `is_single_chunk()`, multi-chunk `chunks_`); `DmaParams` where **`dci == nullptr` is already a straight byte copy**; `createDmaParams(void*, size_t, bool, const CompositeAddress*, dci)`; `launchOperationH2D/D2H`; `fillAsync(CompositeAddress*)` | Public `copyRaw`; `SharedHostPool`; `SharedHostMetadata`; internal per-Function pinning. **The DMA engine and device-address model exist; the shared-pool building blocks do not.** |
| **torch-spyre** | `SpyreStream::copyAsync` / `copyAsyncImpl` — **already accepts `dci == nullptr` and sizes the DMA by `CompositeAddress::total_size()`**; `SharedOwnerCtx::composite_addr` used across `job_plan.cpp` / `spyre_ccl.cpp`; `copy_tensor` pybind; `getStreamFromPool` | `copy_tensor_raw` (slot-addressed) binding; `get_composite_address` accessor; `SharedHostPool` / `SharedHostMetadata` pybind passthroughs; `get_dma_stream` accessor |
| **spyre-inference** | `spyre_worker.py`, `spyre_model_runner.py`, `platform.py`; an **experimental PoC** `kv_offload/` (converting copy, not the production design) | Production `kv_offload/` package built to the latest design — connector, handlers, `register_kv_caches` wiring, private-pool M1 path, shared-pool M2 path |

**Consequences that shape the milestone split:**

- **The device-address model is already in place.** torch-spyre `main` already
  resolves `ctx->composite_addr` everywhere, and `copyAsyncImpl` already does a
  `dci == nullptr` byte copy sized by `total_size()`. The raw copy is *not* blocked on
  any address-handle migration; the torch-spyre accessor + binding work is small.
- **The host memory pool is an M1 prerequisite, not an M2-only building block.** The
  latest design makes the `SharedHostPool` the **only** source of host memory for a KV
  page — there is no caller-owned host-buffer path — and the canonical copy signature is
  `copy_tensor_raw(dev_tensor, pool, slot_id, to_device)` in **both** milestones. So M1
  already needs the pool (a **private, single-instance** pool) and the slot-addressed
  raw copy. M1 is therefore **not** "almost no runtime dependency": it needs the raw
  primitive **and** the pool. What M1 does **not** need is the shared **directory**, the
  cross-process **concurrency protocol**, or a dedicated DMA stream — those are M2.
- **M2 is the cross-instance layer.** M2 adds the `SharedHostMetadata` directory, the
  per-slot concurrency/generation protocol and publish gate, the multi-chunk (1p5)
  `copyRaw` path, and the connector wiring that lets co-located instances share one pool.

## 3. Milestones and their boundaries

- **Milestone 1 — per-instance host-RAM offload (due July 2026).** A user runs
  `vllm serve … spec_name: SpyreOffloadingSpec` and gets **byte-exact** host-RAM offload
  that survives across requests. Each instance offloads into its **own private**
  `SharedHostPool` (single-instance, unshared name); the copy is byte-exact via the raw
  slot-addressed path (`copy_tensor_raw(dev_tensor, pool, slot_id, to_device)`), never
  the converting `copy_tensor` (which drifts ~1 ULP and is a correctness defect for KV
  data). M1 needs the runtime raw primitive **and** the pool building block, but **not**
  the directory, the cross-process concurrency protocol, or a dedicated DMA stream.
- **Milestone 2 — cross-instance shared host pool (due August 2026).** The host tier
  becomes a single **shared host memory pool** shared by co-located instances via the
  `SharedHostMetadata` directory: a block offloaded by one instance is reloaded by
  another with one raw DMA and no serialization. Gated on the directory, the concurrency
  protocol, and the multi-chunk raw-copy path landing.

A milestone spans multiple layers; **every individual issue is closeable within one
PR in one repo.** Cross-layer ordering is expressed as dependencies, not as multi-repo
issues.

## 4. Epics

Three epics. **The design work is one epic (E-DESIGN)** whose three sub-issues each
merge one RFC (one PR, one layer) — the three design docs are a single deliverable, not
three separate epics. The two milestone epics track the implementation sub-issues.

Each epic's **Closes with** below is a concrete, runnable end-to-end check (not a
paperwork condition) so the definition of done is unambiguous — per review feedback that
"Closes with" needs a target test case, e.g. *successfully load/restore KVC from device
HBM* and *verify generation accuracy with the reloaded KVC*.

| ID | Title | Milestone | Closes with (concrete acceptance) |
|---|---|---|---|
| **E-DESIGN** | `[Epic] KV-cache offload design docs (runtime + torch-spyre + spyre-inference)` | design | All three E-D* RFCs merged **and mutually consistent**: the same canonical `copy_tensor_raw(dev_tensor, pool, slot_id, to_device)` signature, the same integer `slot_id` seam, and the same ownership split appear identically in all three (a cross-doc consistency pass is part of the epic). |
| **M1** | `[Epic] Milestone 1 — per-instance host-RAM KV offload end-to-end` | M1 (Jul 2026) | `vllm serve … spec_name: SpyreOffloadingSpec` (a) boots and reaches `register_kv_caches` without raising; (b) **loads/restores a KV block from device HBM** into a private host pool and back — a device→host→device round-trip is **byte-for-byte** identical; (c) a prefix-extending second prompt reports a host-tier hit (`N>0` blocks loaded); (d) **generation-accuracy check**: with `temperature=0`, tokens are **byte-identical** to a no-offload baseline (reloaded KVC produces the exact same output); (e) an offload/reload **latency + throughput** micro-benchmark is recorded vs. recompute. |
| **M2** | `[Epic] Milestone 2 — cross-instance shared host-memory KV pool` | M2 (Aug 2026) | Two co-located `vllm serve` instances attach one shared pool; (a) instance B gets a **host-tier peer hit on its first request** on a block instance A offloaded — served by a device←host DMA, no recompute, no disk; (b) **generation-accuracy check**: B's `temperature=0` tokens are byte-identical to a no-cache baseline (peer-reloaded KVC is exact); (c) **full data-race coverage** passes (see M2-F2 matrix) — no torn read is ever consumable; (d) a **cross-instance peer-hit latency** benchmark vs. recompute is recorded. |

## 5. Sub-issue backlog

Every sub-issue below is one PR in one repo. **Layer** names the repo the PR lands in
(hardware-runtime / torch-spyre / spyre-inference). All issues are *filed* in
`torch-spyre` and labeled `kvc-offloading`; the "layer" tells the implementer which
codebase the PR targets.

### 5.0 Design epic (E-DESIGN) — 3 sub-issues

The three design docs are a single deliverable under **E-DESIGN**; each sub-issue is one
RFC-merge PR in one layer. They are independent of each other and of the milestone work.

#### E-D1 — merge the hardware-runtime shared host memory KV pool RFC

- **Layer:** hardware runtime
- **Epic:** E-DESIGN
- **Label:** `kvc-offloading`
- **Blocked by:** none
- **Description:** Merge the RFC that specifies the mechanism — the byte-exact raw copy
  primitive, the DMA-able shared host memory pool (slot-addressed, pinned internally),
  and the block-hash → slot directory with its concurrency protocol.
- **Closes with:** the RFC is merged and its API surface (`copyRaw`, `SharedHostPool`,
  `SharedHostMetadata`) matches the signatures the torch-spyre and spyre-inference RFCs
  consume — verified by a cross-doc consistency note.

#### E-D2 — merge the torch-spyre Python-surface design doc

- **Layer:** torch-spyre
- **Epic:** E-DESIGN
- **Label:** `kvc-offloading`
- **Blocked by:** none
- **Description:** Merge `docs/source/architecture/raw_copy_kv_offload.md` (plus its
  figures) — the torch-spyre surface: the one tensor-aware address step
  (`get_composite_address`) and the thin bindings over the runtime's raw copy, pool, and
  directory. There is a **single** `copy_tensor_raw` signature:
  `(dev_tensor, pool, slot_id, to_device, non_blocking)` — no host-tensor overload.
- **Closes with:** the doc is merged; its `copy_tensor_raw` signature, `slot_id` seam,
  and "pinning internal to the pool / no host-buffer-registration binding" statements are
  identical to the runtime RFC's.

#### E-D3 — merge the spyre-inference upstream-connector-port RFC

- **Layer:** spyre-inference
- **Epic:** E-DESIGN
- **Label:** `kvc-offloading`
- **Blocked by:** none
- **Description:** Merge the RFC (plus figures) that ports the upstream
  `OffloadingConnector` experience to spyre-inference — `SpyreOffloadingSpec`, handlers,
  and the M1/M2 milestone ladder. The M1 copier must call the **canonical**
  `copy_tensor_raw(dev_tensor, pool, slot_id, to_device)` into a **private** pool (not a
  converting copy into a host tensor); M2 differs only by attaching a **shared** named
  pool + directory.
- **Closes with:** the doc is merged and its M1 **and** M2 copier code both use the one
  canonical `copy_tensor_raw` signature (the "reuses M1 unchanged" claim is literally
  true — only the pool name/sharing differs, not the copy signature).

### 5.1 Milestone 1 — 7 sub-issues

M1 needs the raw primitive **and** the (private) pool, plus the torch-spyre accessor and
binding, plus the connector. It does **not** need the directory, the cross-process
concurrency protocol, or a dedicated DMA stream (those are M2).

#### M1-F1 — hardware runtime: public raw (byte-exact) host↔device DMA

- **Layer:** hardware runtime
- **Milestone:** M1 (Jul 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** none
- **Blocks:** M1-F2, M1-T2, M2-F3
- **Description:** Add a public raw-copy entrypoint on the runtime stream that performs
  a byte-exact host↔device DMA with **no dtype/layout conversion**. It wraps the
  existing straight-byte-copy path (the DMA params where the conversion descriptor is
  null) and the existing H2D/D2H launch. The copy length is the device allocation's
  physical size (`total_size()` — the padded/tiled byte count, **not**
  `numel × itemsize`). Single-chunk is the common case; the multi-chunk (1p5) path is
  M2-F3. This is the named, testable primitive both milestones build on.
- **Closes with / acceptance:** a device page filled with a known pattern **loads to a
  host buffer and restores byte-for-byte** (device→host→device round-trip, byte-exact);
  the copy length is derived from `total_size()`, never the logical byte count; the
  converting copy path is untouched.

#### M1-F2 — hardware runtime: `SharedHostPool` (private / single-instance host memory pool)

- **Layer:** hardware runtime
- **Milestone:** M1 (Jul 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M1-F1
- **Blocks:** M1-T2, M1-T3, M2-F1, M2-F3
- **Description:** A DMA-able host memory pool of fixed-size slots, index (slot-id)
  addressed: `create_or_attach(stream, name, num_slots, slot_bytes)`, `slot_count()`,
  `slot_bytes()`, internal `slot_ptr(i)` for the raw copy, and attach-refcount lifecycle
  (unlink on last-out). **Pinning is internal to the pool** — the pool is pinned once per
  IOMMU Function inside `create_or_attach`; no raw host pointer crosses out of the
  runtime. M1 uses it **single-instance** (a private, unshared name); the same object is
  reused, unchanged, as the shared pool in M2. This is the reason M1 is byte-exact: the
  KV page's host destination is a pool slot, the sole host-memory source (there is no
  caller-owned host-buffer path).
- **Closes with / acceptance:** a single process creates a pool, offloads a device KV
  page into a slot and reloads it byte-for-byte; slot addressing is stable across the
  process's lifetime; pinning is handled internally with **no external pointer
  exposure**.

#### M1-T1 — torch-spyre: `get_composite_address` accessor

- **Layer:** torch-spyre
- **Milestone:** M1 (Jul 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** none
- **Blocks:** M1-T2
- **Description:** Add a read-only Python accessor returning an opaque handle over the
  device address that backs a `device("spyre")` tensor's storage (the address already
  held on the tensor's owner context — no new bookkeeping). This is the one
  tensor-aware step the design assigns to torch-spyre. Handle holds no ownership and is
  invalidated when the tensor's storage is freed. Bind next to the existing copy/stream
  bindings.
- **Closes with / acceptance:** returns a handle whose reported chunk shape matches the
  tensor's allocation; rejected after the tensor storage is freed; no change to existing
  copy paths.

#### M1-T2 — torch-spyre: `copy_tensor_raw(dev_tensor, pool, slot_id, …)` binding (canonical)

- **Layer:** torch-spyre
- **Milestone:** M1 (Jul 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M1-T1, M1-F1, M1-F2
- **Blocks:** M1-S1, M2-S1
- **Description:** Bind the **single canonical** byte-exact raw copy:
  `copy_tensor_raw(dev_tensor, pool, slot_id, to_device, non_blocking=False)`. Resolve
  the device address via M1-T1, obtain the slot's host address from the pool **inside**
  the call (`pool.slot_ptr(slot_id)`, never surfaced to Python), and issue the runtime
  raw copy with a **null conversion descriptor** so it reproduces the device page's bytes
  exactly. The runtime owns the copy size (`total_size()`), the chunking, and the
  byte-identical-layout invariant; torch-spyre computes no byte count. `non_blocking=False`
  synchronizes after enqueue; `True` returns after enqueue and the caller synchronizes.
  **There is no host-tensor overload** — the host destination is always a pool slot
  (M1 uses a private pool; M2 a shared one), which is what makes "M2 reuses M1's copy path
  unchanged" literally true.
- **Closes with / acceptance:** allocate a device tensor with a known fp16 pattern,
  offload D2H into a private-pool slot, zero the device tensor, reload H2D into it (and
  into a **different** same-`(shape,dtype)` tensor), assert **byte-equal** both ways; the
  converting `copy_tensor` path is unaffected; no raw host pointer surfaces to Python.

#### M1-S1 — spyre-inference: `SpyreKvDmaCopier` + `kv_offload` package scaffold

- **Layer:** spyre-inference
- **Milestone:** M1 (Jul 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M1-T2
- **Blocks:** M1-S2
- **Description:** Build the production `spyre_inference/v1/kv_offload/` package
  (`__init__.py`, `copier.py`) fresh to the latest design (the PoC copier is discarded).
  `SpyreKvDmaCopier` is a thin, stateless wrapper around the canonical
  `copy_tensor_raw(dev_tensor, pool, slot_id, to_device)` exposing `copy_d2h` / `copy_h2d`
  — **byte-exact**, no converting copy, no host-tensor destination. Neither method
  allocates. Bump the torch-spyre pin to one exposing `copy_tensor_raw` + `SharedHostPool`.
  Add `test_copier_round_trip.py` (Spyre-gated): known fp16 pattern → offload into a
  private-pool slot → mutate device page → reload → assert byte-exact content.
- **Closes with / acceptance:** the round-trip test passes on a Spyre runner and is
  **byte-exact** (not tolerance-based); the copier never allocates; the pin bump is the
  only dependency change.

#### M1-S2 — spyre-inference: `SpyreCpuOffloadingHandlers`

- **Layer:** spyre-inference
- **Milestone:** M1 (Jul 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M1-S1
- **Blocks:** M1-S3
- **Description:** Implement `handlers.py` — `SpyreCpuOffloadingHandlers` and a
  `_SingleDirectionSpyreHandler` implementing the upstream `OffloadingHandler` contract
  (`transfer_async` / `get_finished` / `shutdown`). Each direction walks block-id pairs
  and calls the copier against a pool `slot_id`. Host destinations are slots in a
  **private** `SharedHostPool` (single-instance) — the `pool` is always present (there is
  no host-tensor path); M2 supplies a **shared** pool through the same parameter. Add
  `test_handler_dispatch.py` (Spyre-gated): exercise both directions and assert content
  lands byte-exact and `get_finished` reports success.
- **Closes with / acceptance:** handler dispatch test green; content round-trips
  byte-exact through the private pool; handler names mirror the upstream shape so the
  spec yields them unchanged.

#### M1-S3 — spyre-inference: `SpyreOffloadingSpec` + registration + M1 acceptance & benchmark

- **Layer:** spyre-inference
- **Milestone:** M1 (Jul 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M1-S2
- **Blocks:** M2-S1
- **Description:** Implement `spec.py` — `SpyreOffloadingSpec` subclassing the upstream
  CPU offloading spec, overriding the handler-creation hook to return Spyre handlers +
  create a private `SharedHostPool`, and dropping the CUDA/XPU platform gate, inheriting
  the manager and block-count math. Add the lazy factory registration in
  `spyre_inference/__init__.py`. Add `test_spec_registration.py` (CPU-only). Run and
  record the **M1 acceptance** and a **micro-benchmark**.
- **Closes with / acceptance:**
  - Registration resolves via the factory; no changes required to the Spyre worker or
    platform (verified by inspecting the diff).
  - `vllm serve … spec_name: SpyreOffloadingSpec` boots and reaches `register_kv_caches`
    without raising.
  - **Load/restore KVC from HBM:** a prefix-extending second prompt reports a host-tier
    hit (`N>0` blocks loaded from host).
  - **Generation-accuracy with reloaded KVC:** with `temperature=0`, generated tokens are
    **byte-identical** to a no-offload baseline (the reloaded KV produces the exact same
    output — no drift, because the copy is byte-exact).
  - **Performance benchmark (recorded, not just pass/fail):** per-block offload (D2H) and
    reload (H2D) **latency** and **throughput** (GB/s), and end-to-end **TTFT reduction on
    a cache hit vs. full recompute**, on a representative model — recorded in the PR so
    later milestones can compare.

### 5.2 Milestone 2 — 8 sub-issues

M2 adds the shared directory, the cross-process concurrency protocol, the multi-chunk
raw-copy path, the dedicated DMA stream (deferred out of M1), and the cross-instance
connector wiring. The pool itself (M1-F2) and the canonical copy binding (M1-T2) are
reused unchanged.

#### M2-F1 — hardware runtime: `SharedHostMetadata` (block-hash → slot directory)

- **Layer:** hardware runtime
- **Milestone:** M2 (Aug 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M1-F2
- **Blocks:** M2-F2, M2-T1
- **Description:** A shared directory mapping block-hash → slot-id with per-slot
  lifecycle state (empty → reserved → valid → empty) and a chunk descriptor
  (`{num_chunks, [{domain_id, size}]}`) for the multi-chunk (1p5) path.
  `lookup(hash) -> slot | miss`, `claim(hash) -> slot`, `publish(hash, slot)`,
  `evict(hash)`. The metadata segment is mapped at a common virtual base so its internals
  can be pointer-based; the data pool (M1-F2) stays index-addressed. This is the
  bookkeeping that makes cross-instance reuse possible.
- **Closes with / acceptance:** claim/publish/lookup/evict behave across two attached
  processes; lifecycle transitions are enforced; the chunk descriptor round-trips a
  multi-chunk allocation's shape.

#### M2-F2 — hardware runtime: concurrency protocol (locks + generation + publish gate) — full race coverage

- **Layer:** hardware runtime
- **Milestone:** M2 (Aug 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M2-F1
- **Blocks:** M2-F3
- **Description:** The correctness protocol over the directory: a process-shared
  directory lock, a per-slot read/write pin (read = reload, write = evict), a per-slot
  generation counter for reuse/ABA detection, and a publish-on-DMA-completion gate so a
  reader only ever observes a slot whose write has completed. A stale or mid-write slot
  must degrade to a **cache miss, never torn bytes**.
- **Closes with / acceptance — full data-race matrix (per review: full coverage on the
  racing cases):** each of the following is exercised under concurrent multi-process load
  and asserts either correct data or a clean miss (never a torn read):
  1. **Reader vs. evictor:** a reader reloads a slot while another instance evicts and
     re-DMAs it — the reader either sees the pre-evict bytes (pin held) or a miss, never a
     mixture.
  2. **ABA / generation reuse:** a slot is evicted and re-claimed for a *different* hash
     between a reader's `lookup` and its DMA — the generation check fails and the reader
     gets a miss.
  3. **Publish-gate visibility:** a reader that `lookup`s a slot mid-write (claimed,
     DMA not yet complete, not yet published) gets a miss — the mapping is invisible until
     `publish`.
  4. **Concurrent claim of the same hash:** two instances `claim` the same
     previously-absent hash simultaneously — exactly one slot is allocated, the other
     observes the winner (no double-allocation, no duplicate slot for one hash).
  5. **Concurrent claim under slot exhaustion:** claims race when the pool is full —
     eviction picks a victim safely (victim's pin respected), no slot is handed out twice.
  6. **Writer vs. writer:** two instances offload *different* hashes concurrently — no
     cross-contamination of slots; both publish correctly.
  7. **Reader during re-DMA of the same hash:** the same hash is re-offloaded (e.g. after
     eviction) while a reader holds a read pin — reader completes on the old generation or
     misses; the writer waits or picks a different slot per the pin discipline.
  A stress test running the matrix for a sustained duration under N processes reports
  **zero torn reads** and **zero double-allocations**.

#### M2-F3 — hardware runtime: `copyRaw` multi-chunk (1p5) + cross-process slot round-trip

- **Layer:** hardware runtime
- **Milestone:** M2 (Aug 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M1-F1, M1-F2, M2-F2
- **Description:** Extend the raw copy for the multi-chunk (1p5) case: on offload, walk
  the source `CompositeAddress` and pack the chunks contiguously into the slot, recording
  the `{num_chunks, [{domain_id, size}]}` descriptor into the directory; on reload, read
  the descriptor, place the chunks on their recorded domains to build a **fresh**
  `CompositeAddress`, and DMA each chunk from its sub-offset in the slot. Single-chunk is
  the degenerate `num_chunks == 1` case. The `slot_id` seam and copy signature are
  unchanged from M1.
- **Closes with / acceptance:** a **multi-chunk** device page stored into a slot from one
  process reloads **byte-for-byte** from another process into a fresh same-`(shape,dtype)`
  page; the copy size is owned by the runtime (`total_size()`); single-chunk continues to
  round-trip byte-exact.

#### M2-T1 — torch-spyre: `SharedHostMetadata` (+ shared-pool attach) pybind passthroughs

- **Layer:** torch-spyre
- **Milestone:** M2 (Aug 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M2-F1
- **Blocks:** M2-S1
- **Description:** Add the pybind passthrough for the runtime's `SharedHostMetadata`
  directory (the `SharedHostPool` binding already lands in M1-T2's dependency M1-F2 +
  M1-T2; here we expose the directory and the shared-attach usage). torch-spyre adds
  nothing — no shared-segment creation, no locking, no directory logic. `slot_ptr` stays
  **unexposed** to Python (the seam is the integer slot-id). Method sets track the runtime
  headers as a passthrough; the directory's common-base mapping is the runtime's concern.
- **Closes with / acceptance:** create-or-attach the pool + directory from two processes
  and observe the **same slots and the same block-hash → slot mappings** (driven from
  Python); no raw host pointer or device address crosses into Python.

#### M2-T2 — torch-spyre: `get_dma_stream` accessor (deferred from M1; gated on multi-stream)

- **Layer:** torch-spyre
- **Milestone:** M2 (Aug 2026) — **optional / non-gating**
- **Label:** `kvc-offloading`
- **Blocked by:** the torch-spyre multi-stream support landing (external; **may slip past
  M1 — deliberately not on the M1 critical path**)
- **Blocks:** none (used by the S-layer as an *optional* dedicated stream)
- **Description:** Thin wrapper over the pooled-stream accessor so the connector can keep
  a **dedicated** DMA stream for offload/reload, letting it overlap the compute stream.
  This was intentionally **removed from M1**: the multi-stream PR is not guaranteed to
  merge by the July M1 date, and M1 must not depend on it. M1 copies on the current/default
  stream (synchronous). When `get_dma_stream` lands, copy bindings accept an optional
  explicit stream; when omitted they use the current stream for the device — so this is a
  pure enhancement, gating nothing.
- **Closes with / acceptance:** returns a usable stream handle for a valid device; copies
  issued on it complete correctly; with it absent, the connector still works on the
  default stream (single-stream fallback verified).

#### M2-S1 — spyre-inference: `SpyreSharedOffloadingSpec` + registration

- **Layer:** spyre-inference
- **Milestone:** M2 (Aug 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M2-T1, M1-T2, M2-F3, M1-S3
- **Blocks:** M2-S2
- **Description:** Implement `shared_spec.py` — `SpyreSharedOffloadingSpec` subclassing
  M1's `SpyreOffloadingSpec`, reusing its handlers and copier **unchanged** (same
  canonical `copy_tensor_raw(dev_tensor, pool, slot_id, to_device)` — the copy signature
  does not change between milestones). The only difference from M1 is the host
  destination: a slot in a **shared** host memory pool named by integer `slot_id` via a
  **shared** `SharedHostMetadata` directory (block-hash → slot), instead of a private
  pool. On store: `claim` a slot, D2H raw copy, then `publish` after the copy
  synchronizes. On load: `lookup` the hash, H2D raw copy from the slot. Add the third
  lazy factory registration; keep it inert when not selected and on builds without the M2
  runtime surface.
- **Closes with / acceptance:** the spec resolves via the factory; importing the plugin
  on a build without the M2 surface does **not** error; the M1 path is unaffected; the
  copier/handler code is **byte-for-byte the M1 code** (diff shows only pool
  construction/sharing changes).

#### M2-S2 — spyre-inference: shared-pool round-trip + torn-read test

- **Layer:** spyre-inference
- **Milestone:** M2 (Aug 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M2-S1
- **Blocks:** M2-S3
- **Description:** `test_shared_pool_round_trip.py` (Spyre-gated): store a known-pattern
  device page into a slot (`claim` + D2H raw copy + `publish`), then `lookup` + H2D raw
  copy into a fresh page and assert byte-exact content. Also drive, from Python, the
  torn-read cases from the M2-F2 matrix that are observable at the connector layer
  (mid-write slot ⇒ miss; ABA/generation reuse ⇒ miss).
- **Closes with / acceptance:** byte-exact round-trip through a shared slot; a mid-write
  or reused slot yields a **miss, never torn bytes**, as observed through the connector.

#### M2-S3 — spyre-inference: cross-instance test + M2 acceptance & benchmark

- **Layer:** spyre-inference
- **Milestone:** M2 (Aug 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M2-S2
- **Description:** `test_cross_instance.py` (two-process): process A stores+publishes a
  block into the shared pool; process B, attaching the same named pool + directory, looks
  up the same content hash and reloads it — assert a cross-instance hit and byte-identical
  reload. Run and record the **M2 acceptance** and a **cross-instance benchmark**.
- **Closes with / acceptance:**
  - **Cross-instance peer hit:** two `vllm serve` instances on one host with a shared
    pool; the second instance gets a host-tier hit **on its first request** on a block the
    first offloaded (no warmup on B) — served by a device←host DMA, no recompute, no disk.
  - **Generation-accuracy with peer-reloaded KVC:** with `temperature=0`, B's tokens are
    **byte-identical** to a no-cache baseline.
  - **Performance benchmark (recorded):** cross-instance **peer-hit latency** (B's TTFT on
    a shared-pool hit) vs. full recompute on B, and shared-pool reload **throughput** —
    recorded and compared against the M1 (private-pool) numbers.
  - M1 path unaffected (`pytest … kv_offload` green).

## 6. Dependency graph

```text
E-DESIGN epic (3 sub-issues, independent, one RFC-merge PR each):
  E-D1 (runtime RFC)   E-D2 (torch-spyre design)   E-D3 (spyre-inference RFC)

MILESTONE 1  (due Jul 2026) — per-instance PRIVATE pool, byte-exact
  M1-F1 ─► M1-F2 ─┐                    (runtime: raw copy, then private pool)
  M1-T1 ──────────┴─► M1-T2 ─► M1-S1 ─► M1-S2 ─► M1-S3
                       (torch-spyre canonical copy binding → connector)
  (get_dma_stream is NOT in M1 — deferred to M2-T2, gated on multi-stream)

MILESTONE 2  (due Aug 2026) — SHARED pool + directory + concurrency
  M1-F2 ─► M2-F1 ─► M2-F2 ─► M2-F3 ─┐          (directory → concurrency → multi-chunk)
  M2-F1 ─► M2-T1 ───────────────────┼─► M2-S1 ─► M2-S2 ─► M2-S3
  M1-T2 ─────────────────────────────┤            ▲
  M1-S3 ──────────────────────────────┘           │
  M2-T2 (get_dma_stream, optional, non-gating) ────┘  (enhancement only)
```

**Cross-layer seams** (dependencies that cross a repo boundary):

- `M1-F2` (runtime) → `M1-T2` (torch-spyre): the private pool + raw primitive gate the
  canonical copy binding.
- `M1-T2` (torch-spyre) → `M1-S1` (spyre-inference): the raw-copy binding + pin bump gate
  the connector.
- `M2-F1` (runtime) → `M2-T1` (torch-spyre); `M2-F3` (runtime) → `M2-S1`
  (spyre-inference): the directory and multi-chunk copy gate the shared bindings/spec.
- `M2-T1` (torch-spyre) → `M2-S1` (spyre-inference): the torch-spyre metadata surface
  gates the shared spec.

## 7. Issue counts

| Layer | Design subs | M1 subs | M2 subs |
|---|---|---|---|
| hardware runtime | 1 (E-D1) | 2 (M1-F1, M1-F2) | 3 (M2-F1…F3) |
| torch-spyre | 1 (E-D2) | 2 (M1-T1, M1-T2) | 2 (M2-T1, M2-T2) |
| spyre-inference | 1 (E-D3) | 3 (M1-S1…S3) | 3 (M2-S1…S3) |

Plus **3 epics**: E-DESIGN, M1, M2.

**Total: 21 issues** — 3 epics (1 design + 2 milestone) + 3 design subs + 7 M1 subs +
8 M2 subs (18 sub-issues). All created in `torch-spyre`, all labeled `kvc-offloading`,
all added to project board view 23.

> **What changed vs. the earlier draft.** The host memory **pool moved into M1**
> (`M1-F2`) because the latest design makes the pool the sole host-memory source and uses
> one canonical `copy_tensor_raw(dev_tensor, pool, slot_id, to_device)` signature in both
> milestones (no host-tensor overload). `get_dma_stream` **moved out of M1** to `M2-T2`
> as an optional, non-gating enhancement, since the multi-stream PR may not merge by July.
> M1 stays 7 subs (dropped the stream accessor, gained the pool); M2 stays 8 (the former
> `SharedHostPool` M2-F1 is now M1-F2, and `get_dma_stream` arrives as M2-T2). Every
> epic's "Closes with" and every sub-issue's acceptance now names concrete test cases
> (load/restore KVC from HBM, generation-accuracy with reloaded KVC), M2-F2 enumerates a
> full data-race matrix, and M1-S3 / M2-S3 add performance benchmarks.

## 8. Creation procedure (for when we proceed)

1. Create the 3 epics (E-DESIGN, M1, M2) in `torch-spyre` with the `kvc-offloading`
   label; spot-check.
2. Create the 18 sub-issues (3 design + 15 implementation), each referencing its epic
   (E-DESIGN for E-D*, the milestone epic for M1-*/M2-*) and listing its `Blocked by` /
   `Blocks` from §5 in the body (GitHub issues do not enforce dependencies natively —
   encode them as `Blocked by #NNN` lines and/or project-board relations).
3. Set the two milestones (M1 → July 2026, M2 → August 2026) and assign every sub-issue
   to its milestone.
4. Add all 21 issues to [project board view 23](https://github.com/orgs/torch-spyre/projects/2/views/23).
```
