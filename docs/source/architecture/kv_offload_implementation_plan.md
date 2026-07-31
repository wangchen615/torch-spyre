# KV-Cache Offload — Implementation Plan (Epics, Sub-Issues, Milestones)

| Field | Value |
|---|---|
| Status | Planning |
| Created | 2026-07-23 |
| Updated | 2026-07-30 |
| Project board | [torch-spyre project #2, view 23](https://github.com/orgs/torch-spyre/projects/2/views/23) |
| Label (all issues) | `kvc-offloading` |
| Issue home | **All issues are created in the `torch-spyre` repo** and added to the project board. |
| Milestone 1 due | **end of August 2026** (GitHub milestone `2026 Q3`) |
| Milestone 2 due | **end of September 2026** (GitHub milestone `2026 Q3`) |
| Design docs | torch-spyre `docs/source/architecture/raw_copy_kv_offload.md`; the hardware-runtime shared-host-pool RFC; the spyre-inference upstream-connector-port RFC |

> **Terminology.** This plan and every issue it describes use **neutral terms**. The
> per-process C++ device runtime is the **hardware runtime**. The cross-process shared
> host memory it provides is the **shared host memory pool** (a **secondary memory
> pool**). Issue titles and bodies must not name internal component or library
> code-names — say "hardware runtime" and "secondary / shared host memory pool"
> instead. (Public API surface names like `SharedHostPool` / `copy_tensor_raw` are the
> design-doc surface names developers implement against, not internal code-names.)

> **Production code follows the latest design, not the current prototype.** An earlier
> proof-of-concept connector exists (a converting `copy_tensor` into a process-local host
> tensor, with pluggable experimental backends), built on an **old spyre-inference commit**
> that could only reach **~4K-token prompts**. That PoC is **not** what this plan tracks and
> is **discarded**. The production implementation is built fresh against the three latest
> design docs — a single canonical byte-exact raw-copy signature
> (`copy_tensor_raw(dev_tensor, pool, slot_id, to_device)`), a shared host memory pool as
> the **only** host-memory source (no caller-owned host-buffer path), and the integer
> `slot_id` seam — on the **latest** spyre-inference with a **pinned** version (M1-P2).

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
3. **spyre-inference layer** — the vLLM connector: a `SpyreOffloadingSpec` and handlers
   that ride the upstream `OffloadingConnector`. **The cache policy is owned upstream**
   (the vLLM `OffloadingManager`), not by us — our specs override only the transfer
   mechanism (see M2-S1).

This document turns those designs into a concrete, dependency-ordered issue backlog:
**3 epics (1 design + 2 milestone) + 21 sub-issues (3 design-doc + 18 implementation)
= 24 issues**, all created in `torch-spyre`, all labeled `kvc-offloading`, all added to
project board view 23. One sub-issue (M2-T2) is a **deferred backlog item with no
milestone**. The three design docs are one deliverable, so they sit as three sub-issues
under a single **design epic** (one RFC-merge PR per layer).

## 2. Current code status (grounds the scoping)

Assessed against the latest `main` of each layer on 2026-07-28.

| Layer | Already on `main` | Missing — the actual work |
|---|---|---|
| **Hardware runtime** | `CompositeAddress` (with `total_size()`, `is_single_chunk()`, multi-chunk `chunks_`); `DmaParams` where **`dci == nullptr` is already a straight byte copy**; `createDmaParams(void*, size_t, bool, const CompositeAddress*, dci)`; `launchOperationH2D/D2H`; `fillAsync(CompositeAddress*)` | Public `copyRaw`; `SharedHostPool`; `SharedHostMetadata`; internal per-Function pinning. **The DMA engine and device-address model exist; the shared-pool building blocks do not.** |
| **torch-spyre** | `SpyreStream::copyAsync` / `copyAsyncImpl` — **already accepts `dci == nullptr` and sizes the DMA by `CompositeAddress::total_size()`**; `SharedOwnerCtx::composite_addr` used across `job_plan.cpp` / `spyre_ccl.cpp`; `copy_tensor` pybind; `getStreamFromPool` | `copy_tensor_raw` (slot-addressed) binding; `get_composite_address` accessor; `SharedHostPool` / `SharedHostMetadata` pybind passthroughs; `get_dma_stream` accessor (deferred backlog) |
| **spyre-inference** | `spyre_worker.py`, `spyre_model_runner.py`, `platform.py`; an **experimental PoC** `kv_offload/` (converting copy on an old commit, not the production design) | Production `kv_offload/` package built to the latest design — connector, handlers, `register_kv_caches` wiring, **shared-pool** M1 path, shared-directory M2 path |

**Consequences that shape the milestone split:**

- **The device-address model is already in place.** torch-spyre `main` already
  resolves `ctx->composite_addr` everywhere, and `copyAsyncImpl` already does a
  `dci == nullptr` byte copy sized by `total_size()`. The raw copy is *not* blocked on
  any address-handle migration; the torch-spyre accessor + binding work is small.
- **The host memory pool is an M1 prerequisite, not an M2-only building block.** The
  latest design makes the `SharedHostPool` the **only** source of host memory for a KV
  page — there is no caller-owned host-buffer path — and the canonical copy signature is
  `copy_tensor_raw(dev_tensor, pool, slot_id, to_device)` in **both** milestones. So M1
  already needs the pool and the slot-addressed raw copy. **We build the pool shared from
  the start** (see below). What M1 does **not** need is the shared **directory**, the
  cross-process **concurrency protocol**, or a dedicated DMA stream — those are M2 (the
  stream is deferred further, to a backlog item).
- **Shared-pool-first (per review).** M1 and M2 have a compact combined schedule, and
  deferring the shared pool to M2 would force a large, hard-to-review M2 rewrite. We must
  build the pool (M1-F2) anyway, so we build it **shared from the start** (cross-process,
  attach-by-name). Only pool *creation* (a named, shared segment) and its *cross-process
  acceptance tests* differ from a single-instance pool; everything else — slot addressing,
  internal pinning, the raw copy — is identical. The block-hash→slot **directory** (M2-F1)
  and the per-slot **concurrency protocol** (M2-F2) stay in M2; M1's shared pool uses
  externally-coordinated slot ids without the safe-concurrency layer.
- **M1 has three prerequisites (per discussion with @frankeh).** Before any offload code,
  M1 requires a reproducible custom-built env (M1-P1), a **recomputation baseline** on the
  *latest* spyre-inference with a **pinned** version (M1-P2), and a documented **host CPU
  buffer model** for the raw copy (M1-P3, before M1-F1).
- **M2 is the cross-instance layer.** M2 adds the `SharedHostMetadata` directory, the
  per-slot concurrency/generation protocol and publish gate, the multi-chunk
  `copyRaw` path, and the connector wiring that lets co-located instances share one pool.
  M2 targets the correctness **baseline**; a dedicated DMA stream (overlap optimization) is
  **deferred to the backlog** (M2-T2), not part of M2.

## 3. Milestones and their boundaries

- **Milestone 1 — host-RAM offload into a shared pool (due end of August 2026).** A user
  runs `vllm serve … spec_name: SpyreOffloadingSpec` and gets **byte-exact** host-RAM
  offload that survives across requests, offloading into a **shared** `SharedHostPool`
  (cross-process, attach-by-name); the copy is byte-exact via the raw slot-addressed path
  (`copy_tensor_raw(dev_tensor, pool, slot_id, to_device)`), never the converting
  `copy_tensor` (which drifts ~1 ULP and is a correctness defect for KV data). M1 needs the
  runtime raw primitive **and** the shared pool building block, plus three prerequisites
  (env, recomputation baseline + pin, host-buffer model), but **not** the directory, the
  cross-process concurrency protocol, or a dedicated DMA stream.
- **Milestone 2 — cross-instance shared host pool (due end of September 2026).** The host
  tier becomes a single **shared host memory pool** shared by co-located instances via the
  `SharedHostMetadata` directory: a block offloaded by one instance is reloaded by another
  with one raw DMA and no serialization. Reuses M1's shared pool and canonical copy path
  **unchanged**; adds the directory, the concurrency protocol, and the multi-chunk
  raw-copy path. M2 targets the correctness baseline; a dedicated DMA stream is deferred to
  the backlog (M2-T2).

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
| **[E-DESIGN](../../../scripts/kv_offload_issues/bodies/EPIC-DESIGN.md)** | `[Epic] KV-cache offload design docs (runtime + torch-spyre + spyre-inference)` | design | All three E-D* RFCs merged **and mutually consistent**: the same canonical `copy_tensor_raw(dev_tensor, pool, slot_id, to_device)` signature, the same integer `slot_id` seam, and the same ownership split appear identically in all three (a cross-doc consistency pass is part of the epic). |
| **[M1](../../../scripts/kv_offload_issues/bodies/EPIC-M1.md)** | `[Epic] Milestone 1 — host-RAM KV offload end-to-end (shared pool)` | M1 (end Aug 2026) | Env + baseline in place (M1-P1/P2: pinned custom stack builds, recomputation baseline recorded); `vllm serve … spec_name: SpyreOffloadingSpec` (a) boots and reaches `register_kv_caches` without raising; (b) **loads/restores a KV block from device HBM** into a shared host pool and back — a device→host→device round-trip is **byte-for-byte** identical; (c) **cross-process pool sharing**: a page written to a shared-pool slot by one process is reloaded byte-exact by a second process attaching the same named pool; (d) a prefix-extending second prompt reports a host-tier hit (`N>0` blocks loaded); (e) **generation-accuracy check**: with `temperature=0`, tokens are **byte-identical** to a no-offload baseline; (f) an offload/reload **latency + throughput** benchmark is recorded vs. recompute. |
| **[M2](../../../scripts/kv_offload_issues/bodies/EPIC-M2.md)** | `[Epic] Milestone 2 — cross-instance shared host-memory KV pool` | M2 (end Sep 2026) | Two co-located `vllm serve` instances attach one shared pool; (a) instance B gets a **host-tier peer hit on its first request** on a block instance A offloaded — served by a device←host DMA, no recompute, no disk; (b) **generation-accuracy check**: B's `temperature=0` tokens are byte-identical to a no-cache baseline; (c) **full data-race coverage** passes (see M2-F2 matrix) — no torn read is ever consumable; (d) a **cross-instance peer-hit latency** benchmark vs. recompute is recorded. (A dedicated DMA stream is **deferred to the backlog**, M2-T2 — not part of M2.) |

## 5. Sub-issue backlog

Every sub-issue below is one PR in one repo. **Layer** names the repo the PR lands in
(hardware-runtime / torch-spyre / spyre-inference). All issues are *filed* in
`torch-spyre` and labeled `kvc-offloading`; the "layer" tells the implementer which
codebase the PR targets.

### 5.0 Design epic (E-DESIGN) — 3 sub-issues

The three design docs are a single deliverable under **E-DESIGN**; each sub-issue is one
RFC-merge PR in one layer. They are independent of each other and of the milestone work.

#### E-D1 — merge the hardware-runtime shared host memory KV pool RFC

- **Detailed body:** [`bodies/E-D1.md`](../../../scripts/kv_offload_issues/bodies/E-D1.md)
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

- **Detailed body:** [`bodies/E-D2.md`](../../../scripts/kv_offload_issues/bodies/E-D2.md)
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

- **Detailed body:** [`bodies/E-D3.md`](../../../scripts/kv_offload_issues/bodies/E-D3.md)
- **Layer:** spyre-inference
- **Epic:** E-DESIGN
- **Label:** `kvc-offloading`
- **Blocked by:** none
- **Description:** Merge the RFC (plus figures) that ports the upstream
  `OffloadingConnector` experience to spyre-inference — `SpyreOffloadingSpec`, handlers,
  and the M1/M2 milestone ladder. The M1 copier must call the **canonical**
  `copy_tensor_raw(dev_tensor, pool, slot_id, to_device)` into the **shared** pool (not a
  converting copy into a host tensor); M2 differs only by adding the shared block-hash→slot
  directory + concurrency on top of the same shared pool.
- **Closes with:** the doc is merged and its M1 **and** M2 copier code both use the one
  canonical `copy_tensor_raw` signature (the "reuses M1 unchanged" claim is literally
  true — only the directory/concurrency layer is added in M2, not a new copy signature).

### 5.1 Milestone 1 — 3 prerequisites + 7 sub-issues

M1 needs the raw primitive **and** the shared pool, plus the torch-spyre accessor,
pool-binding, and copy-binding, plus the connector. It does **not** need the directory,
the cross-process concurrency protocol, or a dedicated DMA stream (those are M2 / the
backlog). Three **prerequisites** block all of M1.

#### M1-P1 — prereq: reproducible custom-built hardware runtime + torch-spyre + spyre-inference env

- **Detailed body:** [`bodies/M1-P1.md`](../../../scripts/kv_offload_issues/bodies/M1-P1.md)
- **Layer:** all (env)
- **Milestone:** M1 (end Aug 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** none
- **Blocks:** all of M1
- **Description:** Stand up a reproducible development/test environment with **custom
  builds** of all three layers (hardware runtime, torch-spyre, spyre-inference), pinned
  together so KV-offload work is developed and tested against a known-good stack. A single
  command/script (or CI recipe) produces the environment from scratch on a Spyre-capable
  host; the exact commit triple is recorded.
- **Closes with / acceptance:** a documented, repeatable build of the three custom layers;
  a recorded commit/version triple; a clean rebuild reproduces a working stack; a trivial
  spyre op and a `vllm serve` boot both succeed on it.

#### M1-P2 — prereq: spyre-inference recomputation baseline on latest code + pin a version

- **Detailed body:** [`bodies/M1-P2.md`](../../../scripts/kv_offload_issues/bodies/M1-P2.md)
- **Layer:** spyre-inference
- **Milestone:** M1 (end Aug 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M1-P1
- **Blocks:** all of M1
- **Description:** Establish a **recomputation baseline** on the **latest** spyre-inference
  (no KV offload) and **pin a spyre-inference version** for KV-offload development (per
  @frankeh). The earlier PoC ran on an **old commit** and could only reach **~4K-token**
  prompts; the target model `ibm-ai-platform/micro-g3.3-8b-instruct-1b` has a **32K** max
  length. Sweep prompt lengths (1K…toward 32K), record which run cleanly and where/why it
  breaks and the runtime configs tried, and record baseline **TTFT vs. prompt length** as
  the reference for later offload comparisons. Pin a spyre-inference commit as the baseline.
- **Closes with / acceptance:** recomputation baseline runs on **latest** spyre-inference
  with the target model, TTFT-vs-prompt-length curve recorded; max stable prompt length
  characterized (target: push toward the 32K model max; document the ceiling/cause if not
  reached); a spyre-inference commit is **pinned**.

#### M1-P3 — prereq: define the host CPU buffer/tensor model for raw copy

- **Detailed body:** [`bodies/M1-P3.md`](../../../scripts/kv_offload_issues/bodies/M1-P3.md)
- **Layer:** hardware runtime + torch-spyre (contract)
- **Milestone:** M1 (end Aug 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M1-P1
- **Blocks:** M1-F1, M1-F2
- **Description:** Decide and document **how host CPU buffers/tensors are created for the
  raw copy** — the host side of every `copyRaw` — **before M1-F1** (M1-F1's DMA and tests
  are written against this model). Cover: a **regular contiguous CPU tensor**
  (`torch.empty(total_size_bytes, dtype=uint8)`, `data_ptr()` as `host_addr`); a **tensor
  over a pre-allocated buffer at an offset** (`base_ptr + slot_offset`, i.e. the M1-F2
  slot layout); and the **pinned vs. pageable** decision (pinning is internal to the pool
  per M1-F2). Define how a device tensor's `total_size()` maps to the host byte region
  (contiguous and padded/tiled cases).
- **Closes with / acceptance:** host-buffer/tensor model documented (regular CPU tensor,
  offset-into-pre-allocated buffer, pinned/pageable decision); `total_size()` → host-region
  mapping specified for contiguous and padded/tiled tensors; M1-F1 and M1-F2 test plans
  reference this model (no divergent buffer assumptions).

#### M1-F1 — hardware runtime: public raw (byte-exact) host↔device DMA

- **Detailed body:** [`bodies/M1-F1.md`](../../../scripts/kv_offload_issues/bodies/M1-F1.md)
- **Layer:** hardware runtime
- **Milestone:** M1 (end Aug 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M1-P3
- **Blocks:** M1-F2, M1-T3, M2-F3
- **Description:** Add a public raw-copy entrypoint on the runtime stream that performs
  a byte-exact host↔device DMA with **no dtype/layout conversion**. It wraps the
  existing straight-byte-copy path (the DMA params where the conversion descriptor is
  null) and the existing H2D/D2H launch. The copy length is the device allocation's
  physical size (`total_size()` — the padded/tiled byte count, **not**
  `numel × itemsize`). Single-chunk is the common case; the multi-chunk path is
  M2-F3. This is the named, testable primitive both milestones build on.
- **Closes with / acceptance:** a device page filled with a known pattern **loads to a
  host region and restores byte-for-byte** (device→host→device round-trip, byte-exact),
  verified across the **host-buffer types** the KV path actually uses (per M1-P3) and at
  realistic sizes — **not** just a small buffer:
  - **Host buffer type:** (a) a **regular contiguous CPU tensor**
    (`torch.empty(total_size_bytes, dtype=uint8)`, `data_ptr()` as `host_addr`);
    (b) a **tensor viewed over a pre-allocated host buffer at an offset**
    (`base_ptr + slot_offset`, i.e. the M1-F2 pool-slot layout) so `copyRaw` is exercised
    against an arbitrary offset region, not only an allocation's start; (c) if applicable,
    a **pinned vs. pageable** host buffer (same bytes either way — `copyRaw` must not
    silently require pinned memory).
  - **Tensor→region mapping:** both a **contiguous** and a **padded/tiled** device tensor
    (`total_size()` > `numel × itemsize`) round-trip byte-for-byte, padding bytes included;
    the copy length is derived from `total_size()`, never the logical byte count.
  - **Size:** a small page (fast CI) **and** a realistic KV-page size **and** at least one
    **large multi-MB** allocation, so the DMA is exercised beyond a trivial length.
  - The converting copy path is untouched.

#### M1-F2 — hardware runtime: `SharedHostPool` (shared host memory pool, cross-process)

- **Detailed body:** [`bodies/M1-F2.md`](../../../scripts/kv_offload_issues/bodies/M1-F2.md)
- **Layer:** hardware runtime
- **Milestone:** M1 (end Aug 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M1-F1, M1-P3
- **Blocks:** M1-T2, M1-T3, M2-F1, M2-F3
- **Description:** A DMA-able **shared** host memory pool of fixed-size slots, index
  (slot-id) addressed, **attachable by name across processes**. We build the **shared**
  pool directly in M1 (not a private pool we later rewrite): starting shared avoids a
  large, hard-to-review M2 rewrite, and only pool *creation* (a shared, named segment) and
  its *acceptance-test coverage* (cross-process) differ from a single-instance pool.
  `create_or_attach(stream, name, num_slots, slot_bytes)`, `slot_count()`, `slot_bytes()`,
  internal `slot_ptr(i)` for the raw copy, and attach-refcount lifecycle (unlink on
  last-out) across processes. **Pinning is internal to the pool** — pinned once per IOMMU
  Function inside `create_or_attach`; no raw host pointer crosses out of the runtime. The
  block-hash→slot **directory** (M2-F1) and locking/generation/publish protocol (M2-F2) are
  **not** in scope here; M1 uses the shared pool with simple, externally-coordinated slot
  ids.
- **Closes with / acceptance:** a single-process offload→reload through a slot is
  byte-exact; **cross-process** — a page written to slot `i` by one process is reloaded
  byte-exact by a second process attaching the same named pool; consistent slot addressing
  across processes; attach-refcount lifecycle correct; internal pinning with **no external
  pointer exposure**.

#### M1-T1 — torch-spyre: `get_composite_address` accessor

- **Detailed body:** [`bodies/M1-T1.md`](../../../scripts/kv_offload_issues/bodies/M1-T1.md)
- **Layer:** torch-spyre
- **Milestone:** M1 (end Aug 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** none
- **Blocks:** M1-T3
- **Description:** Add a read-only Python accessor returning an opaque handle over the
  device address that backs a `device("spyre")` tensor's storage (the address already
  held on the tensor's owner context — no new bookkeeping). This is the one
  tensor-aware step the design assigns to torch-spyre. Handle holds no ownership and is
  invalidated when the tensor's storage is freed. Bind next to the existing copy/stream
  bindings.
- **Closes with / acceptance:** returns a handle whose reported chunk shape matches the
  tensor's allocation; rejected after the tensor storage is freed; no change to existing
  copy paths.

#### M1-T2 — torch-spyre: `SharedHostPool` pybind passthrough (incl. cross-process sharing)

- **Detailed body:** [`bodies/M1-T2.md`](../../../scripts/kv_offload_issues/bodies/M1-T2.md)
- **Layer:** torch-spyre
- **Milestone:** M1 (end Aug 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M1-F2
- **Blocks:** M1-T3
- **Description:** Expose the runtime's **shared** `SharedHostPool` to Python via a pybind
  passthrough, **including the cross-process sharing path** (attach-by-name): expose
  `create_or_attach(stream, name, num_slots, slot_bytes)`, `slot_count()`, `slot_bytes()`,
  with the attach-by-name / shared path working from Python (two Python processes attaching
  the same `name` see the same slots). `slot_ptr` stays **unexposed** to Python (the seam
  is the integer slot-id). No directory here — that binding is M2-T1; M1 exposes only the
  pool.
- **Closes with / acceptance:** shared pool create-or-attach works from Python
  (`slot_count`/`slot_bytes` correct); **two Python processes attaching the same named pool
  see the same slots** (cross-process sharing verified from Python, driven end-to-end with
  M1-T3's `copy_tensor_raw`); `slot_ptr` not exposed, no pointer/address leaks;
  **slot geometry is derived from a real model's KV dimensions** (number of KV heads, head
  dim, block/page size, layer count for `ibm-ai-platform/micro-g3.3-8b-instruct-1b` →
  per-block `total_size()` as `slot_bytes`) — not arbitrary sizes — and a real-shaped KV
  page round-trips byte-exact through the pool (single- and cross-process), including at
  least one **large multi-MB** slot.

#### M1-T3 — torch-spyre: `copy_tensor_raw(dev_tensor, pool, slot_id, …)` binding (canonical)

- **Detailed body:** [`bodies/M1-T3.md`](../../../scripts/kv_offload_issues/bodies/M1-T3.md)
- **Layer:** torch-spyre
- **Milestone:** M1 (end Aug 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M1-T1, M1-T2, M1-F1, M1-F2
- **Blocks:** M1-S1, M2-S1
- **Description:** Bind the **single canonical** byte-exact raw copy:
  `copy_tensor_raw(dev_tensor, pool, slot_id, to_device, non_blocking=False)`. Resolve
  the device address via M1-T1, obtain the slot's host address from the pool **inside**
  the call (`pool.slot_ptr(slot_id)`, never surfaced to Python), and issue the runtime
  raw copy with a **null conversion descriptor** so it reproduces the device page's bytes
  exactly. The runtime owns the copy size (`total_size()`), the chunking, and the
  byte-identical-layout invariant; torch-spyre computes no byte count. `non_blocking=False`
  synchronizes after enqueue; `True` returns after enqueue and the caller synchronizes.
  **There is no host-tensor overload** — the host destination is always a pool slot (the
  M1 shared pool; unchanged in M2), which is what makes "M2 reuses M1's copy path
  unchanged" literally true.
- **Closes with / acceptance:** allocate a device tensor with a known fp16 pattern,
  offload D2H into a shared-pool slot, zero the device tensor, reload H2D into it (and
  into a **different** same-`(shape,dtype)` tensor), assert **byte-equal** both ways;
  **cross-process** — reload from another process attaching the same named pool (via M1-T2)
  is byte-equal; the converting `copy_tensor` path is unaffected; no raw host pointer
  surfaces to Python.

#### M1-S1 — spyre-inference: `SpyreKvDmaCopier` + `kv_offload` package scaffold

- **Detailed body:** [`bodies/M1-S1.md`](../../../scripts/kv_offload_issues/bodies/M1-S1.md)
- **Layer:** spyre-inference
- **Milestone:** M1 (end Aug 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M1-T3
- **Blocks:** M1-S2
- **Description:** Build the production `spyre_inference/v1/kv_offload/` package
  (`__init__.py`, `copier.py`) fresh to the latest design (the PoC copier is discarded).
  `SpyreKvDmaCopier` is a thin, stateless wrapper around the canonical
  `copy_tensor_raw(dev_tensor, pool, slot_id, to_device)` exposing `copy_d2h` / `copy_h2d`
  — **byte-exact**, no converting copy, no host-tensor destination. Neither method
  allocates. Bump the torch-spyre pin to one exposing `copy_tensor_raw` + `SharedHostPool`.
  Add `test_copier_round_trip.py` (Spyre-gated): known fp16 pattern → offload into a
  **shared-pool** slot (the M1-F2 shared `SharedHostPool`, attached by name) → mutate device
  page → reload → assert byte-exact content.
- **Closes with / acceptance:** the round-trip test passes on a Spyre runner and is
  **byte-exact** (not tolerance-based), offloading into a **shared-pool** slot; **byte-exact
  at a large / realistic KV-page size** (real model geometry, incl. a multi-MB page), not
  only a small buffer; the copier never allocates; the pin bump is the only dependency
  change.

#### M1-S2 — spyre-inference: `SpyreCpuOffloadingHandlers`

- **Detailed body:** [`bodies/M1-S2.md`](../../../scripts/kv_offload_issues/bodies/M1-S2.md)
- **Layer:** spyre-inference
- **Milestone:** M1 (end Aug 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M1-S1
- **Blocks:** M1-S3
- **Description:** Implement `handlers.py` — `SpyreCpuOffloadingHandlers` and a
  `_SingleDirectionSpyreHandler` implementing the upstream `OffloadingHandler` contract
  (`transfer_async` / `get_finished` / `shutdown`). Each direction walks block-id pairs
  and calls the copier against a pool `slot_id`. Host destinations are slots in the
  **shared** `SharedHostPool` (M1-F2, attached by name) — the `pool` is always present
  (there is no host-tensor path); M2 adds the block-hash→slot directory + concurrency on
  top of the same pool without changing the handler signature. Add
  `test_handler_dispatch.py` (Spyre-gated): exercise both directions and assert content
  lands byte-exact and `get_finished` reports success.
- **Closes with / acceptance:** handler dispatch test green; content round-trips
  byte-exact through the shared pool; handler names mirror the upstream shape so the
  spec yields them unchanged.

#### M1-S3 — spyre-inference: `SpyreOffloadingSpec` + registration + M1 acceptance & benchmark

- **Detailed body:** [`bodies/M1-S3.md`](../../../scripts/kv_offload_issues/bodies/M1-S3.md)
- **Layer:** spyre-inference
- **Milestone:** M1 (end Aug 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M1-S2
- **Blocks:** M2-S1
- **Description:** Implement `spec.py` — `SpyreOffloadingSpec` subclassing the upstream
  CPU offloading spec, overriding the handler-creation hook to return Spyre handlers +
  create/attach the **shared** `SharedHostPool` (M1-F2), and dropping the CUDA/XPU platform
  gate, inheriting the manager and block-count math (the cache policy stays the upstream
  manager's — see M2-S1). Add the lazy factory registration in
  `spyre_inference/__init__.py`. Add `test_spec_registration.py` (CPU-only). Run and
  record the **M1 acceptance** and a **benchmark**.
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
    a cache hit vs. the recomputation baseline** (M1-P2), recorded in the PR so later
    milestones can compare.

### 5.2 Milestone 2 — 6 sub-issues (+ 1 deferred backlog item)

M2 adds the shared directory, the cross-process concurrency protocol, the multi-chunk
raw-copy path, and the cross-instance connector wiring. The shared pool (M1-F2) and the
canonical copy binding (M1-T3) are reused unchanged. M2 targets the correctness
**baseline**; the dedicated DMA stream (M2-T2) is **deferred to the backlog**, not part of
M2.

#### M2-F1 — hardware runtime: `SharedHostMetadata` (block-hash → slot directory)

- **Detailed body:** [`bodies/M2-F1.md`](../../../scripts/kv_offload_issues/bodies/M2-F1.md)
- **Layer:** hardware runtime
- **Milestone:** M2 (end Sep 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M1-F2
- **Blocks:** M2-F2, M2-T1
- **Description:** A shared directory mapping block-hash → slot-id with per-slot
  lifecycle state (empty → reserved → valid → empty) and a chunk descriptor
  (`{num_chunks, [{domain_id, size}]}`) for the multi-chunk path.
  `lookup(hash) -> slot | miss`, `claim(hash) -> slot`, `publish(hash, slot)`,
  `evict(hash)`. The metadata segment is mapped at a common virtual base so its internals
  can be pointer-based; the data pool (M1-F2) stays index-addressed. This is the
  bookkeeping that makes cross-instance reuse possible.
- **Closes with / acceptance:** claim/publish/lookup/evict behave across two attached
  processes; lifecycle transitions are enforced; the chunk descriptor round-trips a
  multi-chunk allocation's shape; **directory + M1-F2 data pool compose (single-process)** —
  a KV page stored under a block-hash via `claim`/`publish` is retrieved byte-exact via
  `lookup` → `copyRaw` H2D (hash-addressed round-trip verified end-to-end;
  single-process is sufficient since concurrency is M2-F2).

#### M2-F2 — hardware runtime: concurrency protocol (locks + generation + publish gate) — full race coverage

- **Detailed body:** [`bodies/M2-F2.md`](../../../scripts/kv_offload_issues/bodies/M2-F2.md)
- **Layer:** hardware runtime
- **Milestone:** M2 (end Sep 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M2-F1
- **Blocks:** M2-F3
- **Description:** The correctness protocol over the directory: a process-shared
  directory lock, a per-slot read/write pin (read = reload, write = evict), a per-slot
  generation counter for reuse/ABA detection, and a publish-on-DMA-completion gate so a
  reader only ever observes a slot whose write has completed. A stale or mid-write slot
  must degrade to a **cache miss, never torn bytes**. This is the layer that **establishes**
  torn-read freedom (proven exhaustively here; the Python bindings prove faithful
  passthrough in M2-T1, and the connector proves miss→recompute in M2-S2).
- **Atomicity / critical-section contract (RFC §4.4).** The upper layer drives a transfer as
  a **sequence** of directory ops (`claim` → DMA → `publish`, or `lookup` → pin → reload), so
  this issue must state which steps a single lock hold covers vs. what the caller composes:
  each individual op (`lookup`/`claim`/`publish`/`evict`) is internally atomic against the
  shared directory, but the `claim`→DMA→`publish` span is **not** held under one lock for the
  DMA's duration — correctness across it comes from the pin + generation + publish gate, not
  from holding the lock. The directory lock is held only for the short mutations and **never
  across a DMA**. This contract is what M2-T1 exposes to Python and what defines its
  GIL-release policy (multi-op sequences called under the GIL — Takeshi's review point).
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
  **zero torn reads** and **zero double-allocations**. The matrix is **also driven from
  concurrent Python threads through the M2-T1 bindings**, so the multi-op sequences race
  across both processes and Python threads: a binding blocked on the directory lock must not
  stall unrelated Python threads (validating M2-T1's GIL-release policy), and the
  caller-composed span still yields a clean miss, never torn bytes.

#### M2-F3 — hardware runtime: `copyRaw` multi-chunk + cross-process slot round-trip

- **Detailed body:** [`bodies/M2-F3.md`](../../../scripts/kv_offload_issues/bodies/M2-F3.md)
- **Layer:** hardware runtime
- **Milestone:** M2 (end Sep 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M1-F1, M1-F2, M2-F2
- **Description:** Extend the raw copy for the multi-chunk case: on offload, walk
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

- **Detailed body:** [`bodies/M2-T1.md`](../../../scripts/kv_offload_issues/bodies/M2-T1.md)
- **Layer:** torch-spyre
- **Milestone:** M2 (end Sep 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M2-F1, M2-F2
- **Blocks:** M2-S1
- **Description:** Add the pybind passthrough for the runtime's `SharedHostMetadata`
  directory (the `SharedHostPool` object itself is bound in M1-T2; here we expose the
  directory and the shared-attach usage). torch-spyre adds nothing — no shared-segment
  creation, no locking, no directory logic. `slot_ptr` and any raw host/device address stay
  **unexposed** (the seam is exclusively the integer `slot_id` / `block_hash`). **Exact
  Python calls to expose:** `SharedHostMetadata.create_or_attach(stream, name, capacity)`;
  `md.lookup(block_hash: int) -> int` (miss sentinel, e.g. `-1`); `md.claim(block_hash:
  int) -> int`; `md.publish(block_hash: int, slot_id: int) -> None`; `md.evict(block_hash:
  int) -> None`; `md.capacity() -> int`; plus `SharedHostPool.create_or_attach(stream,
  name, num_slots, slot_bytes)` used with a **shared name**. Names/arities are pinned to the
  merged runtime header at implementation time.
- **GIL / multi-op sequencing (Takeshi's review point).** The upper layer calls these ops as
  a *sequence* (`claim` → DMA → `publish`, or `lookup` → pin → reload) from Python, and each
  op may block on the runtime's process-shared directory lock (RFC §4.4). This binding issue
  must **release the GIL** (`py::gil_scoped_release`) around any call that can block on the
  lock or a DMA, so a Spyre instance stalled on a peer's lock does not freeze every other
  Python thread in the process (vLLM's scheduler, other connectors). There is **no** combined
  `claim_dma_publish` binding that holds the lock across the DMA — the critical section is
  **caller-composed**; correctness across the span comes from M2-F2's pin + generation +
  publish gate, not from holding the GIL or the directory lock across the DMA. Each binding
  maps to exactly one internally-atomic directory op (per the M2-F2 contract) and adds no
  locking of its own.
- **Closes with / acceptance:** all listed calls exposed with the stated signatures and
  integer-only seam; `lookup`/`claim`/`publish`/`evict` semantics verified from Python (miss
  → claim → publish → hit → evict → miss); two-process attach sees the **same slots and
  mappings** from Python; **torn-read passthrough** — a mid-write / reused slot yields a
  **miss from Python**, never torn bytes (the bindings do not weaken the M2-F2 protocol;
  this is the right layer for the "torn-read visible from Python" assertion — it needs the
  bindings, not the connector); **GIL released** for any binding that can block on the lock or
  a DMA — a call blocked on a peer's lock does not stall unrelated Python threads (test drives
  two Python threads, one blocked, and asserts the other progresses); the `claim`→DMA→`publish`
  critical section is **caller-composed** (no single atomic binding holds the lock across a
  DMA); no pointer/address leaks to Python.

#### M2-S1 — spyre-inference: `SpyreSharedOffloadingSpec` + registration

- **Detailed body:** [`bodies/M2-S1.md`](../../../scripts/kv_offload_issues/bodies/M2-S1.md)
- **Layer:** spyre-inference
- **Milestone:** M2 (end Sep 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M2-T1, M1-T3, M2-F3, M1-S3
- **Blocks:** M2-S2
- **Description:** Implement `shared_spec.py` — `SpyreSharedOffloadingSpec` subclassing
  M1's `SpyreOffloadingSpec`, reusing its handlers and copier **unchanged** (same
  canonical `copy_tensor_raw(dev_tensor, pool, slot_id, to_device)` — the copy signature
  does not change between milestones). The only difference from M1 is the host
  destination: a slot in a **shared** pool named by integer `slot_id` via a **shared**
  `SharedHostMetadata` directory, instead of a private-per-process slot bookkeeping. On
  store: `claim` a slot, D2H raw copy, then `publish` after the copy synchronizes. On load:
  `lookup` the hash, H2D raw copy from the slot. Add the third lazy factory registration;
  keep it inert when not selected and on builds without the M2 runtime surface.
  - **Who owns the cache policy.** The **cache policy is owned upstream, not by us.**
    `SpyreOffloadingSpec` (M1-S3) subclasses the upstream vLLM CPU `OffloadingSpec`, which
    brings the upstream `OffloadingManager` — that manager owns **admission, eviction
    (LRU), block-hash bookkeeping, and the hit/miss decision**. Our Spyre specs override
    only the **transfer mechanism**, never the policy, so `SpyreSharedOffloadingSpec`
    **inherits the same upstream cache policy** transitively. What M2 changes is **where a
    decided hit resolves to storage**: the block-hash → slot mapping is externalized to the
    **shared** `SharedHostMetadata` directory so a hit can resolve to a slot another
    instance published (eviction still runs upstream; on evict we call
    `SharedHostMetadata.evict(hash)`). Any Spyre-specific admission/eviction behavior would
    be a **separate, explicitly-scoped** manager override — out of scope for M1/M2, which
    reuse the upstream policy verbatim.
- **Closes with / acceptance:** the spec resolves via the factory; importing the plugin
  on a build without the M2 surface does **not** error; the M1 path is unaffected; the
  copier/handler code is **byte-for-byte the M1 code** (diff shows only pool
  construction/sharing changes); **cache policy not overridden** — admission/eviction/
  hit-miss remain the upstream `OffloadingManager`'s (verified by diff — no policy
  override), M2 changes only pool construction/sharing and the shared directory that backs
  slot resolution.

#### M2-S2 — spyre-inference: shared-pool round-trip + connector miss→recompute test

- **Detailed body:** [`bodies/M2-S2.md`](../../../scripts/kv_offload_issues/bodies/M2-S2.md)
- **Layer:** spyre-inference
- **Milestone:** M2 (end Sep 2026)
- **Label:** `kvc-offloading`
- **Blocked by:** M2-S1
- **Blocks:** M2-S3
- **Description:** A **connector-level** shared-pool round-trip test, plus a test that the
  connector handles a **miss correctly** (falls back to recompute) when the lower layers
  report one. This issue does **not** re-prove concurrency correctness — that is owned and
  tested where the protocol lives:
  - **M2-F2 (hardware runtime)** owns and exhaustively tests the full data-race matrix —
    where torn-read freedom is *established*.
  - **M2-T1 (torch-spyre)** proves the **Python bindings faithfully pass the protocol
    through** — a mid-write `lookup` observed **from Python** returns a miss (the right
    place for the "torn-read visible from Python" assertion; needs only the bindings).
  - **M2-S2 (here)** verifies only the **connector's** behavior given those guarantees: a
    shared-slot round-trip succeeds, and when the layers below return a **miss**, the
    connector degrades to **recompute** (no crash, no corrupt output) rather than serving
    stale bytes. It does not construct races itself.
  - Tests: `test_shared_pool_round_trip.py` (Spyre-gated: `claim` + D2H + `publish` →
    `lookup` + H2D into a fresh page → byte-exact) and `test_connector_miss_recompute.py`
    (with `lookup` forced to report a miss at the binding boundary — **not** by racing the
    pool — the connector takes the recompute path and produces correct output).
- **Closes with / acceptance:** byte-exact round-trip through a shared slot (connector
  level); on a reported miss the connector **recomputes** cleanly (correct output, no
  stale/torn bytes consumed) — the race-correctness proof itself lives in M2-F2 (protocol)
  and M2-T1 (binding passthrough), not duplicated here.

#### M2-S3 — spyre-inference: cross-instance test + M2 acceptance & benchmark

- **Detailed body:** [`bodies/M2-S3.md`](../../../scripts/kv_offload_issues/bodies/M2-S3.md)
- **Layer:** spyre-inference
- **Milestone:** M2 (end Sep 2026)
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
    recorded and compared against the M1 numbers.
  - M1 path unaffected (`pytest … kv_offload` green).

### 5.3 Deferred backlog (not part of M1 or M2)

#### M2-T2 — torch-spyre: `get_dma_stream` accessor (DEFERRED backlog — no milestone)

- **Detailed body:** [`bodies/M2-T2.md`](../../../scripts/kv_offload_issues/bodies/M2-T2.md)
- **Layer:** torch-spyre
- **Milestone:** **none — deferred backlog** (per review: an optimization, not the M2
  baseline)
- **Label:** `kvc-offloading`
- **Blocked by:** the torch-spyre multi-stream support landing (external), **and** the M2
  baseline (shared pool + directory + concurrency) being in place so overlap can be
  measured against it
- **Blocks:** none (a pure enhancement; both M1 and M2 copy on the current/default stream)
- **Description:** *(Deferred / backlog — an optimization, **not** part of M1 or M2.)* A
  thin wrapper over the pooled-stream accessor so the connector can keep a **dedicated** DMA
  stream for offload/reload, overlapping the compute stream. Per review (Yue): M2 focuses on
  the correctness **baseline** (shared pool + directory + concurrency); a dedicated DMA
  stream is a throughput optimization layered on a working baseline, so it is pulled out of
  M2 and tracked here for later scheduling. It gates nothing: both M1 and M2 copy on the
  current/default stream. `get_dma_stream(device=None) -> SpyreStreamHandle`; copy bindings
  accept an optional explicit stream; when omitted they use the current stream for the
  device (the M1/M2 single-stream fallback).
- **Closes with / acceptance:** returns a usable stream handle for a valid device; copies
  issued on it complete correctly; with it absent, the connector still works on the default
  stream (single-stream fallback verified); measured **overlap benefit recorded** vs. the
  single-stream baseline (the optimization's justification). **Schedule only after the M2
  baseline lands.**

## 6. Dependency graph

```text
E-DESIGN epic (3 sub-issues, independent, one RFC-merge PR each):
  E-D1 (runtime RFC)   E-D2 (torch-spyre design)   E-D3 (spyre-inference RFC)

MILESTONE 1  (due end Aug 2026) — SHARED pool from the start, byte-exact
  prerequisites (block all of M1):
    M1-P1 (env) ─► M1-P2 (recompute baseline + pin)
    M1-P1 ──────► M1-P3 (host CPU buffer model)
  M1-P3 ─► M1-F1 ─► M1-F2 ─┐              (runtime: raw copy, then shared pool)
  M1-T1 ───────────────────┤
  M1-F2 ─► M1-T2 ──────────┴─► M1-T3 ─► M1-S1 ─► M1-S2 ─► M1-S3
           (pool pybind)       (canonical copy binding → connector)
  (get_dma_stream is NOT in M1 or M2 — deferred to the backlog, M2-T2)

MILESTONE 2  (due end Sep 2026) — directory + concurrency on the SAME shared pool
  M1-F2 ─► M2-F1 ─► M2-F2 ─► M2-F3 ─┐          (directory → concurrency → multi-chunk)
  M2-F1 ─► M2-T1 ───────────────────┼─► M2-S1 ─► M2-S2 ─► M2-S3
  M2-F2 ─► M2-T1                     │            ▲
  M1-T3 ─────────────────────────────┤            │
  M1-S3 ──────────────────────────────┘           │
  (M2-T2 get_dma_stream is deferred to backlog, gates nothing) ─── ✗ not in M2
```

**Cross-layer seams** (dependencies that cross a repo boundary):

- `M1-F2` (runtime) → `M1-T2` (torch-spyre): the shared pool gates the pool pybind
  passthrough (incl. cross-process sharing).
- `M1-F2` + `M1-T1` (runtime + torch-spyre) → `M1-T3` (torch-spyre): the shared pool + raw
  primitive + address accessor gate the canonical copy binding.
- `M1-T3` (torch-spyre) → `M1-S1` (spyre-inference): the raw-copy binding + pin bump gate
  the connector.
- `M2-F1`/`M2-F2` (runtime) → `M2-T1` (torch-spyre); `M2-F3` (runtime) → `M2-S1`
  (spyre-inference): the directory + concurrency and the multi-chunk copy gate the shared
  bindings/spec.
- `M2-T1` (torch-spyre) → `M2-S1` (spyre-inference): the torch-spyre metadata surface
  gates the shared spec.

## 7. Issue counts

| Layer | Design subs | M1 subs | M2 subs | Deferred backlog |
|---|---|---|---|---|
| hardware runtime | 1 (E-D1) | 3 (M1-F1, M1-F2, and M1-P3 shared contract) | 3 (M2-F1…F3) | — |
| torch-spyre | 1 (E-D2) | 3 (M1-T1, M1-T2, M1-T3) | 1 (M2-T1) | 1 (M2-T2) |
| spyre-inference | 1 (E-D3) | 3 (M1-S1…S3) + M1-P2 baseline | 3 (M2-S1…S3) | — |
| env / cross-cutting | — | M1-P1 | — | — |

Prerequisites: **M1-P1** (env, cross-cutting), **M1-P2** (spyre-inference recomputation
baseline + pin), **M1-P3** (hardware-runtime + torch-spyre host-buffer contract).

Plus **3 epics**: E-DESIGN, M1, M2.

**Total: 24 issues** — 3 epics (1 design + 2 milestone) + 3 design subs + 10 M1 subs
(3 prereqs + 7) + 7 M2 subs (6 milestoned + 1 deferred backlog). All created in
`torch-spyre`, all labeled `kvc-offloading`, all added to project board view 23. M2-T2 is
created and cross-linked but carries **no milestone**.

> **What changed vs. the earlier draft.** (1) **Dates pushed back one month**: M1 → end of
> August 2026, M2 → end of September 2026 (both still in `2026 Q3`). (2) **Shared-pool-first**:
> `M1-F2` builds the **shared** `SharedHostPool` (cross-process, attach-by-name) from the
> start — not a private pool later rewritten — so M2 reuses it unchanged and only adds the
> directory (`M2-F1`) + concurrency (`M2-F2`). (3) **Three M1 prerequisites added**: M1-P1
> (reproducible custom-built env), M1-P2 (recomputation baseline on **latest** spyre-inference,
> plus pinning a version), M1-P3 (host CPU buffer model, before M1-F1). (4) **torch-spyre
> renumber**:
> `M1-T2` is now the `SharedHostPool` pybind passthrough (incl. cross-process); `M1-T3` is the
> canonical `copy_tensor_raw` binding (formerly M1-T2); downstream blocked-by points at M1-T3.
> (5) **`get_dma_stream` deferred to the backlog** (`M2-T2`, **no milestone**) — an
> optimization, not the M2 baseline. (6) Per-comment test/acceptance additions folded in:
> M1-F1 host-buffer taxonomy + large tensor, M1-T2 real-model KV geometry, M1-S1 large/realistic
> tensor + shared-pool wording, M2-F1 directory↔pool single-process connection test, M2-T1 exact
> exposed Python calls + torn-read passthrough, M2-S1 cache-policy-ownership section, M2-S2
> narrowed to connector-level (race proof pushed to M2-F2/M2-T1). Every epic's "Closes with"
> and every sub-issue's acceptance names concrete test cases (load/restore KVC from HBM,
> generation-accuracy with reloaded KVC, cross-process sharing), M2-F2 enumerates a full
> data-race matrix, and M1-S3 / M2-S3 record performance benchmarks.

## 8. Creation procedure (for when we proceed)

1. Create the `kvc-offloading` label first (`gh label create kvc-offloading`); the `epic`
   label already exists.
2. Create the 3 epics (E-DESIGN, M1, M2) in `torch-spyre` with the `kvc-offloading` +
   `epic` labels; spot-check.
3. Create the 21 sub-issues (3 design + 18 implementation), each referencing its epic
   (E-DESIGN for E-D*, the milestone epic for M1-*/M2-*) and listing its `Blocked by` /
   `Blocks` from §5 in the body (GitHub issues do not enforce dependencies natively —
   encode them as `Blocked by #NNN` lines and/or native sub-issue nesting).
4. Set the GitHub milestone `2026 Q3` on every milestoned issue (M1 targets end of August,
   M2 end of September — both inside `2026 Q3`); **leave M2-T2 with no milestone** (deferred
   backlog).
5. Add all 24 issues to [project board view 23](https://github.com/orgs/torch-spyre/projects/2/views/23).

The scripts under `scripts/kv_offload_issues/` automate this: `create_issues.sh` (full:
label, `2026 Q3` milestone on all but M2-T2, native sub-issue nesting, board) and
`create_issues_triage.sh` (triage-only: bodies + `epic` label + cross-links, no milestone/
nesting/board). Both read the per-issue bodies from `scripts/kv_offload_issues/bodies/`,
which are the source of truth this doc mirrors.
